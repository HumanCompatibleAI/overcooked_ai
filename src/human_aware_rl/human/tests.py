import copy
import os
import pickle
import shutil
import sys
import unittest

import numpy as np
from numpy.testing._private.utils import assert_raises

from human_aware_rl.human.process_dataframes import (
    csv_to_df_pickle,
    get_trajs_from_data,
)
from human_aware_rl.human.process_human_trials import (
    main as process_human_trials_main,
)
from human_aware_rl.static import *
from human_aware_rl.utils import equal_dicts
from overcooked_ai_py.agents.agent import AgentPair, GreedyHumanModel
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import (
    OvercookedGridworld,
    OvercookedState,
)
from overcooked_ai_py.planning.planners import (
    NO_COUNTERS_PARAMS,
    MediumLevelActionManager,
)


class TestProcessDataFrames(unittest.TestCase):
    temp_data_dir = "this_is_a_temp"
    data_len_2019 = 3546
    data_len_2020 = 1189

    base_csv_to_df_params = {
        "csv_path": DUMMY_2020_RAW_HUMAN_DATA_PATH,
        "out_dir": "this_is_a_temp",
        "out_file_prefix": "unittest",
        "button_presses_threshold": 0.25,
        "perform_train_test_split": False,
        "silent": True,
    }

    base_get_trajs_from_data_params = {
        "data_path": DUMMY_2019_CLEAN_HUMAN_DATA_PATH,
        "featurize_states": False,
        "check_trajectories": False,
        "silent": True,
        "layouts": ["cramped_room"],
    }

    def setUp(self):
        print(
            "\nIn Class {}, in Method {}".format(
                self.__class__.__name__, self._testMethodName
            )
        )
        if not os.path.exists(self.temp_data_dir):
            os.makedirs(self.temp_data_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_data_dir)

    def test_csv_to_df_pickle_2019(self):
        params = copy.deepcopy(self.base_csv_to_df_params)
        params["csv_path"] = DUMMY_2019_RAW_HUMAN_DATA_PATH
        params["button_presses_threshold"] = 0.0
        data = csv_to_df_pickle(**params)
        self.assertEqual(len(data), self.data_len_2019)

        params = copy.deepcopy(self.base_csv_to_df_params)
        params["csv_path"] = DUMMY_2019_RAW_HUMAN_DATA_PATH
        params["button_presses_threshold"] = 0.7
        data = csv_to_df_pickle(**params)
        self.assertLess(len(data), self.data_len_2019)

    def test_csv_to_df_pickle_2020(self):
        params = copy.deepcopy(self.base_csv_to_df_params)
        params["button_presses_threshold"] = 0.0
        data = csv_to_df_pickle(**params)
        self.assertEqual(len(data), self.data_len_2020)

        params = copy.deepcopy(self.base_csv_to_df_params)
        params["button_presses_threshold"] = 0.7
        data = csv_to_df_pickle(**params)
        self.assertLess(len(data), self.data_len_2020)

    def test_csv_to_df_pickle(self):
        # Try various button thresholds (hand-picked to lie between different values for dummy data games)
        button_thresholds = [0.2, 0.6, 0.7]
        lengths = []
        for threshold in button_thresholds:
            # dummy dataset is too small to partion so we set train_test_split=False
            params = copy.deepcopy(self.base_csv_to_df_params)
            params["button_presses_threshold"] = threshold
            data = csv_to_df_pickle(**params)
            lengths.append(len(data))

        # Filtered data size should be monotonically decreasing wrt button_threshold
        for i in range(len(lengths) - 1):
            self.assertGreaterEqual(lengths[i], lengths[i + 1])

        # Picking a threshold that's suficiently high discards all data, should result in value error
        params = copy.deepcopy(self.base_csv_to_df_params)
        params["button_presses_threshold"] = 0.8
        self.assertRaises(ValueError, csv_to_df_pickle, **params)

    def test_get_trajs_from_data_2019(self):
        params = copy.deepcopy(self.base_get_trajs_from_data_params)
        trajectories, _ = get_trajs_from_data(**params)

    def test_get_trajs_from_data_2019_featurize(self):
        params = copy.deepcopy(self.base_get_trajs_from_data_params)
        params["featurize_states"] = True
        trajectories, _ = get_trajs_from_data(**params)

    def test_get_trajs_from_data_2020(self):
        # Ensure we can properly deserialize states with updated objects (i.e tomatoes)
        params = copy.deepcopy(self.base_get_trajs_from_data_params)
        params["layouts"] = ["inverse_marshmallow_experiment"]
        params["data_path"] = DUMMY_2020_CLEAN_HUMAN_DATA_PATH
        trajectories, _ = get_trajs_from_data(**params)

    def test_get_trajs_from_data_2020_featurize(self):
        # Ensure we can properly featurize states with updated dynamics and updated objects (i.e tomatoes)
        params = copy.deepcopy(self.base_get_trajs_from_data_params)
        params["layouts"] = ["inverse_marshmallow_experiment"]
        params["data_path"] = DUMMY_2020_CLEAN_HUMAN_DATA_PATH
        params["featurize_states"] = True
        trajectories, _ = get_trajs_from_data(**params)

    def test_csv_to_df_to_trajs_integration(self):
        # Ensure the output of 'csv_to_df_pickle' works as valid input to 'get_trajs_from_data'
        params = copy.deepcopy(self.base_csv_to_df_params)
        _ = csv_to_df_pickle(**params)

        params = copy.deepcopy(self.base_get_trajs_from_data_params)
        params["data_path"] = os.path.join(
            self.temp_data_dir, "unittest_all.pickle"
        )
        params["layouts"] = ["inverse_marshmallow_experiment"]
        _ = get_trajs_from_data(**params)


class TestHumanDataConversion(unittest.TestCase):
    temp_dir = "this_is_also_a_temp"
    infile = DUMMY_2019_CLEAN_HUMAN_DATA_PATH
    horizon = 400
    DATA_TYPE = "train"
    layout_name = "cramped_room"

    def _equal_pickle_and_env_state_dict(
        self, pickle_state_dict, env_state_dict
    ):
        return equal_dicts(
            pickle_state_dict,
            env_state_dict,
            ["timestep", "all_orders", "bonus_orders"],
        )

    def setUp(self):
        print(
            "\nIn Class {}, in Method {}".format(
                self.__class__.__name__, self._testMethodName
            )
        )
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

        self.base_mdp = OvercookedGridworld.from_layout_name(self.layout_name)
        self.mlam = MediumLevelActionManager.from_pickle_or_compute(
            self.base_mdp, NO_COUNTERS_PARAMS, force_compute=True, info=False
        )
        self.env = OvercookedEnv.from_mdp(
            self.base_mdp, horizon=self.horizon, info_level=0
        )
        self.starting_state_dict = (
            self.base_mdp.get_standard_start_state().to_dict()
        )

        outfile = process_human_trials_main(
            self.infile,
            self.temp_dir,
            insert_interacts=True,
            verbose=False,
            forward_port=False,
            fix_json=False,
        )
        with open(outfile, "rb") as f:
            self.human_data = pickle.load(f)[self.layout_name]

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_state(self):
        idx = 0
        for state_dict, joint_action in self.human_data[:100]:
            if state_dict.items() == self.starting_state_dict.items():
                self.env.reset()
            else:
                self.assertTrue(
                    self._equal_pickle_and_env_state_dict(
                        state_dict, self.env.state.to_dict()
                    ),
                    "Expected state:\t\n{}\n\nActual state:\t\n{}".format(
                        self.env.state.to_dict(), state_dict
                    ),
                )
            self.env.step(joint_action=joint_action)
            idx += 1


if __name__ == "__main__":
    unittest.main()
