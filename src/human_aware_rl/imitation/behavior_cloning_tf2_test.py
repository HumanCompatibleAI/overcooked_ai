import argparse
import os
import pickle
import shutil
import sys
import unittest
import warnings

import numpy as np
import tensorflow as tf

from human_aware_rl.human.process_dataframes import get_trajs_from_data
from human_aware_rl.imitation.behavior_cloning_tf2 import (
    BC_SAVE_DIR,
    build_bc_model,
    evaluate_bc_model,
    get_bc_params,
    load_bc_model,
    save_bc_model,
    train_bc_model,
)
from human_aware_rl.static import (
    BC_EXPECTED_DATA_PATH,
    DUMMY_2019_CLEAN_HUMAN_DATA_PATH,
)
from human_aware_rl.utils import set_global_seed


def _clear_pickle():
    with open(BC_EXPECTED_DATA_PATH, "wb") as f:
        pickle.dump({}, f)


class TestBCTraining(unittest.TestCase):

    """
    Unittests for behavior cloning training and utilities

    compute_pickle (bool):      Whether the results of this test should be stored as the expected values for future tests
    strict (bool):              Whether the results of this test should be compared against expected values for exact match
    min_performance (int):      Minimum reward achieved in BC-BC rollout after training to consider training successfull

    Note, this test always performs a basic sanity check to verify some learning is happening, even if the `strict` param is false
    """

    def __init__(self, test_name):
        super(TestBCTraining, self).__init__(test_name)
        self.compute_pickle = False
        self.strict = False
        self.min_performance = 0
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
        warnings.simplefilter("ignore", ResourceWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        self.bc_params = get_bc_params(
            **{"data_path": DUMMY_2019_CLEAN_HUMAN_DATA_PATH}
        )
        self.bc_params["mdp_params"]["layout_name"] = "cramped_room"
        self.bc_params["training_params"]["epochs"] = 1
        self.model_dir = os.path.join(BC_SAVE_DIR, "test_model")

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        processed_trajs, _ = get_trajs_from_data(
            **self.bc_params["data_params"], silent=True
        )
        self.dummy_input = np.vstack(processed_trajs["ep_states"])[:1, :]
        self.initial_states = [
            np.zeros((1, self.bc_params["cell_size"])),
            np.zeros((1, self.bc_params["cell_size"])),
        ]
        with open(BC_EXPECTED_DATA_PATH, "rb") as f:
            self.expected = pickle.load(f)

        # Disable TF warnings and infos
        tf.get_logger().setLevel("ERROR")

    def tearDown(self):
        if self.compute_pickle:
            with open(BC_EXPECTED_DATA_PATH, "wb") as f:
                pickle.dump(self.expected, f)

        shutil.rmtree(self.model_dir)

    def test_model_construction(self):
        model = build_bc_model(**self.bc_params)

        if self.compute_pickle:
            self.expected["test_model_construction"] = model(self.dummy_input)
        if self.strict:
            self.assertTrue(
                np.allclose(
                    model(self.dummy_input),
                    self.expected["test_model_construction"],
                )
            )

    def test_save_and_load(self):
        model = build_bc_model(**self.bc_params)
        save_bc_model(self.model_dir, model, self.bc_params)
        loaded_model, loaded_params = load_bc_model(self.model_dir)
        self.assertDictEqual(self.bc_params, loaded_params)
        self.assertTrue(
            np.allclose(
                model(self.dummy_input), loaded_model(self.dummy_input)
            )
        )

    def test_training(self):
        model = train_bc_model(self.model_dir, self.bc_params)

        if self.compute_pickle:
            self.expected["test_training"] = model(self.dummy_input)
        if self.strict:
            self.assertTrue(
                np.allclose(
                    model(self.dummy_input), self.expected["test_training"]
                )
            )

    def test_agent_evaluation(self):
        self.bc_params["training_params"]["epochs"] = 20
        model = train_bc_model(self.model_dir, self.bc_params)
        results = evaluate_bc_model(model, self.bc_params)

        # Sanity Check
        self.assertGreaterEqual(results, self.min_performance)

        if self.compute_pickle:
            self.expected["test_agent_evaluation"] = results
        if self.strict:
            self.assertAlmostEqual(
                results, self.expected["test_agent_evaluation"]
            )


class TestBCTrainingLSTM(TestBCTraining):
    # LSTM tests break on older versions of tensorflow so be careful with this
    def test_lstm_construction(self):
        self.bc_params["use_lstm"] = True
        model = build_bc_model(**self.bc_params)

        if self.compute_pickle:
            self.expected["test_lstm_construction"] = model(self.dummy_input)
        if self.strict:
            self.assertTrue(
                np.allclose(
                    model(self.dummy_input),
                    self.expected["test_lstm_construction"],
                )
            )

    def test_lstm_training(self):
        self.bc_params["use_lstm"] = True
        model = train_bc_model(self.model_dir, self.bc_params)

        if self.compute_pickle:
            self.expected["test_lstm_training"] = model(self.dummy_input)
        if self.strict:
            self.assertTrue(
                np.allclose(
                    model(self.dummy_input),
                    self.expected["test_lstm_training"],
                )
            )

    def test_lstm_evaluation(self):
        self.bc_params["use_lstm"] = True
        self.bc_params["training_params"]["epochs"] = 1
        model = train_bc_model(self.model_dir, self.bc_params)
        results = evaluate_bc_model(model, self.bc_params)

        # Sanity Check
        self.assertGreaterEqual(results, self.min_performance)

        if self.compute_pickle:
            self.expected["test_lstm_evaluation"] = results
        if self.strict:
            self.assertAlmostEqual(
                results, self.expected["test_lstm_evaluation"]
            )

    def test_lstm_save_and_load(self):
        self.bc_params["use_lstm"] = True
        model = build_bc_model(**self.bc_params)
        save_bc_model(self.model_dir, model, self.bc_params)
        loaded_model, loaded_params = load_bc_model(self.model_dir)
        self.assertDictEqual(self.bc_params, loaded_params)
        self.assertTrue(
            np.allclose(
                self._lstm_forward(model, self.dummy_input)[0],
                self._lstm_forward(loaded_model, self.dummy_input)[0],
            )
        )

    def _lstm_forward(self, model, obs_batch, states=None):
        obs_batch = np.expand_dims(obs_batch, 1)
        seq_lens = np.ones(len(obs_batch))
        states_batch = states if states else self.initial_states
        model_out = model.predict([obs_batch, seq_lens] + states_batch)
        logits, states = model_out[0], model_out[1:]
        logits = logits.reshape((logits.shape[0], -1))
        return logits, states


if __name__ == "__main__":
    unittest.main()
