import glob
import os
import pickle
import random
import shutil
import warnings

import pytest
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


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(seed)


def _clear_pickle():
    with open(PPO_EXPECTED_DATA_PATH, "wb") as f:
        pickle.dump({}, f)


class TestPPORllib:
    """
    Tests for rllib PPO training loop

    compute_pickle (bool):      Whether the results of this test should be stored as the expected values for future tests
    strict (bool):              Whether the results of this test should be compared against expected values for exact match
    min_performance (int):      Minimum sparse reward that must be achieved during training for test to count as "success"
    """

    compute_pickle = False
    strict = False
    min_performance = 5

    @pytest.fixture(autouse=True)
    def _setup(self):
        assert not (
            self.compute_pickle and self.strict
        ), "Cannot compute pickle and run strict reproducibility tests at same time"
        if self.compute_pickle:
            _clear_pickle()

        set_global_seed(0)
        warnings.filterwarnings("ignore")
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        self.temp_results_dir = os.path.join(
            os.path.abspath("."), "results_temp"
        )
        self.temp_model_dir = os.path.join(os.path.abspath("."), "model_temp")

        if not os.path.exists(self.temp_model_dir):
            os.makedirs(self.temp_model_dir)

        if not os.path.exists(self.temp_results_dir):
            os.makedirs(self.temp_results_dir)

        with open(PPO_EXPECTED_DATA_PATH, "rb") as f:
            self.expected = pickle.load(f)

        yield

        if self.compute_pickle:
            with open(PPO_EXPECTED_DATA_PATH, "wb") as f:
                pickle.dump(self.expected, f)

        shutil.rmtree(self.temp_results_dir)
        shutil.rmtree(self.temp_model_dir)
        ray.shutdown()

    def test_save_load(self):
        ex.run(
            config_updates={
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

        ray.shutdown()

        load_path = os.path.join(
            glob.glob(os.path.join(self.temp_results_dir, "save_load_test*"))[
                0
            ],
            "checkpoint_000002",
        )

        mdp = OvercookedGridworld.from_layout_name("cramped_room")
        state = mdp.get_standard_start_state()

        agent_0 = load_agent(load_path)
        agent_0.reset()

        agent_1 = load_agent(load_path)
        agent_1.reset()

        _, _ = agent_0.action(state)
        _, _ = agent_1.action(state)

        agent_pair = load_agent_pair(load_path)
        ae = AgentEvaluator.from_layout_name(
            mdp_params={"layout_name": "cramped_room"},
            env_params={"horizon": 400},
        )

        ae.evaluate_agent_pair(agent_pair, 1, info=False)

    def test_ppo_sp_no_phi(self):
        results = ex.run(
            config_updates={
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
        assert results["average_total_reward"] >= self.min_performance

        if self.compute_pickle:
            self.expected["test_ppo_sp_no_phi"] = results
        if self.strict:
            assert results == self.expected["test_ppo_sp_no_phi"]

    def test_ppo_sp_yes_phi(self):
        results = ex.run(
            config_updates={
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
        assert results["average_total_reward"] >= 13

        if self.compute_pickle:
            self.expected["test_ppo_sp_yes_phi"] = results
        if self.strict:
            assert results == self.expected["test_ppo_sp_yes_phi"]

    def test_ppo_fp_sp_no_phi(self):
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
        assert results["average_total_reward"] >= 7

        if self.compute_pickle:
            self.expected["test_ppo_fp_sp_no_phi"] = results
        if self.strict:
            assert results == self.expected["test_ppo_fp_sp_no_phi"]

    def test_ppo_fp_sp_yes_phi(self):
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
        assert results["average_total_reward"] >= self.min_performance

        if self.compute_pickle:
            self.expected["test_ppo_fp_sp_yes_phi"] = results
        if self.strict:
            assert results == self.expected["test_ppo_fp_sp_yes_phi"]

    def test_ppo_bc(self):
        model_dir = self.temp_model_dir
        params_to_override = {
            "layouts": ["asymmetric_advantages_tomato"],
            "data_path": None,
            "epochs": 10,
        }
        bc_params = get_bc_params(**params_to_override)
        train_bc_model(model_dir, bc_params)

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
        assert results["average_total_reward"] >= 30

        if self.compute_pickle:
            self.expected["test_ppo_bc"] = results
        if self.strict:
            assert results == self.expected["test_ppo_bc"]
