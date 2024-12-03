import glob
import os
import pickle
import random
import shutil
import unittest
import warnings

import ray

os.environ["RUN_ENV"] = "local"
import numpy as np
import tensorflow as tf

from human_aware_rl.data_dir import DATA_DIR
from human_aware_rl.imitation.behavior_cloning_tf2 import (
    get_bc_params,
    train_bc_model,
)
from human_aware_rl.ppo.ppo_rllib_client import ex
from human_aware_rl.ppo.ppo_rllib_from_params_client import ex_fp
from human_aware_rl.rllib.rllib import load_agent, load_agent_pair
from human_aware_rl.static import PPO_EXPECTED_DATA_PATH
from human_aware_rl.utils import get_last_episode_rewards
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld


# Note: using the same seed across architectures can still result in differing values
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(seed)


def _clear_pickle():
    # Write an empty dictionary to our static "expected" results location
    with open(PPO_EXPECTED_DATA_PATH, "wb") as f:
        pickle.dump({}, f)


class TestPPORllib(unittest.TestCase):

    """
    Unittests for rllib PPO training loop

    compute_pickle (bool):      Whether the results of this test should be stored as the expected values for future tests
    strict (bool):              Whether the results of this test should be compared against expected values for exact match
    min_performance (int):      Minimum sparse reward that must be achieved during training for test to count as "success"

    Note, this test always performs a basic sanity check to verify some learning is happening, even if the `strict` param is false
    """

    def __init__(self, test_name):
        super(TestPPORllib, self).__init__(test_name)
        # default parameters, feel free to change
        self.compute_pickle = False
        # Reproducibility test
        self.strict = False
        # this value exists as sanity checks since random actions can still obtain at least this much dense reward
        # some performance thresholds are hardcoded and are determined by comparing empirical performances with and with actual trainings/gradient updates
        self.min_performance = 5
        assert not (
            self.compute_pickle and self.strict
        ), "Cannot compute pickle and run strict reproducibility tests at same time"
        if self.compute_pickle:
            _clear_pickle()

    def setUp(self):
        set_global_seed(0)
        print(
            "\nIn Class {}, in Method {}".format(
                self.__class__.__name__, self._testMethodName
            )
        )
        # unittest generates a lot of warning msgs due to third-party dependencies (e.g. ray[rllib] using outdated np methods)
        # not a problem when directly ran, but when using -m unittest this helps filter out the warnings
        warnings.filterwarnings("ignore")
        # Setting CWD
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        # Temporary disk space to store logging results from tests
        self.temp_results_dir = os.path.join(
            os.path.abspath("."), "results_temp"
        )
        self.temp_model_dir = os.path.join(os.path.abspath("."), "model_temp")

        # Make all necessary directories
        if not os.path.exists(self.temp_model_dir):
            os.makedirs(self.temp_model_dir)

        if not os.path.exists(self.temp_results_dir):
            os.makedirs(self.temp_results_dir)

        # Load in expected values (this is an empty dict if compute_pickle=True)
        with open(PPO_EXPECTED_DATA_PATH, "rb") as f:
            self.expected = pickle.load(f)

    def tearDown(self):
        # Write results of this test to disk for future reproducibility tests
        # Note: This causes unit tests to have a side effect (generally frowned upon) and only works because
        #   unittest is single threaded. If tests were run concurrently this could result in a race condition!
        if self.compute_pickle:
            with open(PPO_EXPECTED_DATA_PATH, "wb") as f:
                pickle.dump(self.expected, f)

        # Cleanup
        shutil.rmtree(self.temp_results_dir)
        shutil.rmtree(self.temp_model_dir)
        ray.shutdown()

    def test_save_load(self):
        # Train a quick self play agent for 2 iterations
        ex.run(
            config_updates={
                # Please feel free to modify the parameters below
                "results_dir": self.temp_results_dir,
                "experiment_name": "save_load_test",
                "layout_name": "cramped_room",
                "num_workers": 1,
                "train_batch_size": 800,
                "sgd_minibatch_size": 800,
                "num_training_iters": 2,
                "evaluation_interval": 10,
                "entropy_coeff_start": 0.0,
                "entropy_coeff_end": 0.0,
                "use_phi": False,
                "evaluation_display": False,
                "verbose": False,
            },
            options={"--loglevel": "ERROR"},
        )

        # Kill all ray processes to ensure loading works in a vaccuum
        ray.shutdown()

        # Where the agent is stored (this is kind of hardcoded, would like for it to be more easily obtainable)
        # 2 checkpoints(checkpoint_000001 and checkpoint_000002) are saved
        # since we are only interested in reproducing the same actions, either one should be fine
        load_path = os.path.join(
            glob.glob(os.path.join(self.temp_results_dir, "save_load_test*"))[
                0
            ],
            "checkpoint_000002",
        )

        # Load a dummy state
        mdp = OvercookedGridworld.from_layout_name("cramped_room")
        state = mdp.get_standard_start_state()

        # Ensure simple single-agent loading works
        agent_0 = load_agent(load_path)
        agent_0.reset()

        agent_1 = load_agent(load_path)
        agent_1.reset()

        # Ensure forward pass of policy network still works
        _, _ = agent_0.action(state)
        _, _ = agent_1.action(state)

        # Now let's load an agent pair and evaluate it
        agent_pair = load_agent_pair(load_path)
        ae = AgentEvaluator.from_layout_name(
            mdp_params={"layout_name": "cramped_room"},
            env_params={"horizon": 400},
        )

        # We assume no runtime errors => success, no performance consistency check for now
        ae.evaluate_agent_pair(agent_pair, 1, info=False)

    def test_ppo_sp_no_phi(self):
        # Train a self play agent for 20 iterations
        results = ex.run(
            config_updates={
                # Please feel free to modify the parameters below
                "results_dir": self.temp_results_dir,
                "num_workers": 2,
                "train_batch_size": 800,
                "sgd_minibatch_size": 800,
                "num_training_iters": 30,
                "evaluation_interval": 10,
                "entropy_coeff_start": 0.0,
                "entropy_coeff_end": 0.0,
                "use_phi": False,
                "evaluation_display": False,
                "verbose": False,
            },
            options={"--loglevel": "ERROR"},
        ).result
        # Sanity check (make sure it begins to learn to receive dense reward)
        self.assertGreaterEqual(
            results["average_total_reward"], self.min_performance
        )

        if self.compute_pickle:
            self.expected["test_ppo_sp_no_phi"] = results

        # Reproducibility test
        if self.strict:
            self.assertDictEqual(results, self.expected["test_ppo_sp_no_phi"])

    def test_ppo_sp_yes_phi(self):
        # Train a self play agent for 20 iterations
        results = ex.run(
            config_updates={
                # Please feel free to modify the parameters below
                "results_dir": self.temp_results_dir,
                "num_workers": 2,
                "train_batch_size": 1600,
                "sgd_minibatch_size": 800,
                "num_training_iters": 30,
                "evaluation_interval": 10,
                "entropy_coeff_start": 0.0,
                "entropy_coeff_end": 0.0,
                "use_phi": True,
                "evaluation_display": False,
                "verbose": False,
                "lr": 5e-3,
            },
            options={"--loglevel": "ERROR"},
        ).result
        # Sanity check (make sure it begins to learn to receive dense reward)
        # This value is determined by comparing emperical performances with and without actual training updates
        self.assertGreaterEqual(results["average_total_reward"], 13)

        if self.compute_pickle:
            self.expected["test_ppo_sp_yes_phi"] = results

        # Reproducibility test
        if self.strict:
            self.assertDictEqual(results, self.expected["test_ppo_sp_yes_phi"])

    def test_ppo_fp_sp_no_phi(self):
        # Train a self play agent for 20 iterations
        results = ex_fp.run(
            config_updates={
                "results_dir": self.temp_results_dir,
                "num_workers": 2,
                "train_batch_size": 2400,
                "sgd_minibatch_size": 800,
                "num_training_iters": 30,
                "evaluation_interval": 10,
                "use_phi": False,
                "entropy_coeff_start": 0.0002,
                "entropy_coeff_end": 0.00005,
                "lr": 5e-3,
                "seeds": [0],
                "outer_shape": (5, 4),
                "evaluation_display": False,
                "verbose": False,
            },
            options={"--loglevel": "ERROR"},
        ).result
        # Sanity check (make sure it begins to learn to receive dense reward)
        self.assertGreaterEqual(results["average_total_reward"], 7)

        if self.compute_pickle:
            self.expected["test_ppo_fp_sp_no_phi"] = results

        # Reproducibility test
        if self.strict:
            self.assertDictEqual(
                results, self.expected["test_ppo_fp_sp_no_phi"]
            )

    def test_ppo_fp_sp_yes_phi(self):
        # Train a self play agent for 20 iterations
        results = ex_fp.run(
            config_updates={
                "results_dir": self.temp_results_dir,
                "num_workers": 2,
                "train_batch_size": 1600,
                "sgd_minibatch_size": 800,
                "num_training_iters": 30,
                "evaluation_interval": 10,
                "use_phi": True,
                "entropy_coeff_start": 0.0002,
                "entropy_coeff_end": 0.00005,
                "lr": 7e-4,
                "seeds": [0],
                "outer_shape": (5, 4),
                "evaluation_display": False,
                "verbose": False,
            },
            options={"--loglevel": "ERROR"},
        ).result

        # Sanity check (make sure it begins to learn to receive dense reward)
        self.assertGreaterEqual(
            results["average_total_reward"], self.min_performance
        )

        if self.compute_pickle:
            self.expected["test_ppo_fp_sp_yes_phi"] = results

        # Reproducibility test
        if self.strict:
            self.assertDictEqual(
                results, self.expected["test_ppo_fp_sp_yes_phi"]
            )

    def test_ppo_bc(self):
        # Train bc model
        model_dir = self.temp_model_dir
        params_to_override = {
            "layouts": ["asymmetric_advantages_tomato"],
            "data_path": None,
            "epochs": 10,
        }
        bc_params = get_bc_params(**params_to_override)
        train_bc_model(model_dir, bc_params)

        # Train rllib model
        config_updates = {
            "results_dir": self.temp_results_dir,
            "bc_schedule": [(0.0, 0.0), (8e3, 1.0)],
            "num_training_iters": 20,
            "bc_model_dir": model_dir,
            "evaluation_interval": 5,
            "verbose": False,
            "layout_name": "asymmetric_advantages_tomato",
        }
        results = ex.run(
            config_updates=config_updates, options={"--loglevel": "ERROR"}
        ).result
        # Sanity check
        # This value is determined by comparing emperical performances with and without actual training updates
        self.assertGreaterEqual(results["average_total_reward"], 30)

        if self.compute_pickle:
            self.expected["test_ppo_bc"] = results

        # Reproducibility test
        if self.strict:
            self.assertDictEqual(results, self.expected["test_ppo_bc"])

    # def test_resume_functionality(self):
    #     load_path = os.path.join(
    #         os.path.abspath("."),
    #         "trained_example/checkpoint_000500",
    #     )
    #     # Load and train an agent for another iteration
    #     results = ex_fp.run(
    #         config_updates={
    #             "results_dir": self.temp_results_dir,
    #             "num_workers": 1,
    #             "num_training_iters": 1,
    #             "resume_checkpoint_path": load_path,
    #             "verbose": False,
    #             "evaluation_display": False,
    #         },
    #         options={"--loglevel": "ERROR"},
    #     ).result

    #     # Test that the rewards from 1 additional iteration are not too different from the original model
    #     # performance

    #     threshold = 0.1

    #     rewards = get_last_episode_rewards("trained_example/result.json")

    #     # Test total reward
    #     self.assertAlmostEqual(
    #         rewards["episode_reward_mean"],
    #         results["average_total_reward"],
    #         delta=threshold * rewards["episode_reward_mean"],
    #     )
    #     # Test sparse reward
    #     self.assertAlmostEqual(
    #         rewards["sparse_reward_mean"],
    #         results["average_sparse_reward"],
    #         delta=threshold * rewards["sparse_reward_mean"],
    #     )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    unittest.main()
