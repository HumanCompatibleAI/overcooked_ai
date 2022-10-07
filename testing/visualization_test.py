import copy
import json
import os
import unittest

import numpy as np
import pygame

from overcooked_ai_py.agents.agent import RandomAgent
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.overcooked_mdp import (
    OvercookedGridworld,
    OvercookedState,
    Recipe,
)
from overcooked_ai_py.static import TESTING_DATA_DIR
from overcooked_ai_py.utils import generate_temporary_file_path, load_from_json
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer


def get_file_count(directory_path):
    path, dirs, files = next(os.walk(directory_path))
    return len(files)


state_visualizer_dir = os.path.join(TESTING_DATA_DIR, "test_state_visualizer")
example_img_path = generate_temporary_file_path(
    prefix="overcooked_visualized_state_", extension=".png"
)


def test_render_state_from_dict(test_dict):
    input_dict = copy.deepcopy(test_dict)
    test_dict = copy.deepcopy(test_dict)
    test_dict["kwargs"]["state"] = OvercookedState.from_dict(
        test_dict["kwargs"]["state"]
    )
    # check only if it raise error or not, for image fidelity render_state check is used
    StateVisualizer(**test_dict["config"]).display_rendered_state(
        img_path=example_img_path, **test_dict["kwargs"]
    )

    actual_result = pygame.surfarray.array3d(
        StateVisualizer(**test_dict["config"]).render_state(
            **test_dict["kwargs"]
        )
    )
    expected_result = np.load(
        os.path.join(state_visualizer_dir, test_dict["result_array_filename"])
    )
    if not actual_result.shape == expected_result.shape:
        print("test with: ", input_dict["result_array_filename"], "is failed")
        print(
            "test not passed, wrong output shape",
            actual_result.shape,
            "!=",
            expected_result.shape,
        )
        print(json.dumps(input_dict, indent=4, sort_keys=True))
        return False

    wrong_rows, wrong_columns, wrong_color_channels = np.where(
        actual_result != expected_result
    )
    wrong_coordinates = set(
        [(row, col) for row, col in zip(wrong_rows, wrong_columns)]
    )
    incorrect_pixels_num = len(wrong_coordinates)
    all_pixels_num = int(expected_result.size / 3)
    if incorrect_pixels_num:
        wrong_coordinate_list = sorted(list(wrong_coordinates))
        print("test with: ", input_dict["result_array_filename"], "is failed")
        print(
            "test not passed, wrong color of",
            incorrect_pixels_num,
            "pixels out of",
            all_pixels_num,
        )
        print(
            "first 100 wrong pixels coordinates", wrong_coordinate_list[:100]
        )
        print("coordinate\texpected\tactual")
        for i in range(10):
            (wrong_x, wrong_y) = wrong_coord = wrong_coordinate_list[i]
            print(
                "%s\t%s\t%s"
                % (
                    str(wrong_coord),
                    str(expected_result[wrong_x, wrong_y]),
                    str(actual_result[wrong_x, wrong_y]),
                )
            )
        print("test_dict", json.dumps(input_dict))
        return False
    print("test with: ", input_dict["result_array_filename"], "is ok")
    return True


class TestStateVisualizer(unittest.TestCase):
    def setUp(self):
        Recipe.configure({})

    def test_setting_up_configs(self):
        default_values = copy.deepcopy(StateVisualizer.DEFAULT_VALUES)

        init_config = {"tile_size": 123}
        configure_config = {"tile_size": 234}
        configure_defaults_config = {"tile_size": 345}
        assert (
            default_values["tile_size"]
            != init_config["tile_size"]
            != configure_config["tile_size"]
            != configure_defaults_config["tile_size"]
        )

        visualizer = StateVisualizer(**init_config)
        self.assertEqual(init_config["tile_size"], visualizer.tile_size)

        visualizer.configure(**configure_config)
        self.assertEqual(configure_config["tile_size"], visualizer.tile_size)

        StateVisualizer.configure_defaults(**configure_defaults_config)
        self.assertEqual(
            configure_defaults_config["tile_size"],
            StateVisualizer.DEFAULT_VALUES["tile_size"],
        )
        self.assertEqual(
            configure_defaults_config["tile_size"], StateVisualizer().tile_size
        )

        invalid_kwargs = {"invalid_argument": 123}
        self.assertRaises(AssertionError, StateVisualizer, **invalid_kwargs)
        self.assertRaises(
            AssertionError,
            StateVisualizer.configure_defaults,
            **invalid_kwargs
        )
        self.assertRaises(
            AssertionError, visualizer.configure, **invalid_kwargs
        )

    def test_properties(self):
        visualizer = StateVisualizer(
            tile_size=30, hud_interline_size=7, hud_font_size=26
        )
        self.assertEqual(visualizer.scale_by_factor, 2)
        self.assertEqual(visualizer.hud_line_height, 26 + 7)

    def test_hud_display(self):
        for d in load_from_json(
            os.path.join(
                state_visualizer_dir, "render_state_data_test_hud.json"
            )
        ):
            test_render_state_from_dict(d)

    def test_differnet_sizes(self):
        for d in load_from_json(
            os.path.join(
                state_visualizer_dir, "render_state_data_test_sizes.json"
            )
        ):
            test_render_state_from_dict(d)

    def test_cooking_timer_display(self):
        for d in load_from_json(
            os.path.join(
                state_visualizer_dir,
                "render_state_data_test_cooking_display.json",
            )
        ):
            test_render_state_from_dict(d)

    def test_various_states(self):
        for d in load_from_json(
            os.path.join(
                state_visualizer_dir, "render_state_data_test_various.json"
            )
        ):
            test_render_state_from_dict(d)

    def test_generated_layout_states(self):
        for d in load_from_json(
            os.path.join(
                state_visualizer_dir,
                "render_state_data_test_generated_layout.json",
            )
        ):
            test_render_state_from_dict(d)

    def test_default_hud_data_from_trajectories(self):
        traj_path = os.path.join(
            TESTING_DATA_DIR, "test_state_visualizer", "test_trajectory.json"
        )  # NOTE: for test purposes reward is added here despite there was no soup delivery in trajectory
        test_trajectory = AgentEvaluator.load_traj_from_json(traj_path)
        hud_data_path = os.path.join(
            TESTING_DATA_DIR,
            "test_state_visualizer",
            "expected_default_hud_data_from_trajectories.json",
        )
        expected_hud_data = load_from_json(hud_data_path)
        result_hud_data = StateVisualizer().default_hud_data_from_trajectories(
            test_trajectory
        )
        self.assertEqual(
            json.dumps(result_hud_data, sort_keys=True),
            json.dumps(expected_hud_data, sort_keys=True),
        )

    def test_action_probs_display(self):
        for d in load_from_json(
            os.path.join(
                state_visualizer_dir,
                "render_state_data_test_action_probs_display.json",
            )
        ):
            test_render_state_from_dict(d)

    def test_trajectory_visualization(self):
        # we don't have good way to check slider automatically so its mostly test for basic stuff like numer of outputed images, if using method raises error etc.
        traj_path = os.path.join(
            TESTING_DATA_DIR, "test_state_visualizer", "test_trajectory.json"
        )
        test_trajectory = AgentEvaluator.load_traj_from_json(traj_path)
        expected_images_num = len(test_trajectory["ep_states"][0])
        assert expected_images_num == 10
        action_probs = [
            [RandomAgent(all_actions=True).action(state)[1]["action_probs"]]
            * 2
            for state in test_trajectory["ep_states"][0]
        ]

        result_img_directory_path = (
            StateVisualizer().display_rendered_trajectory(
                test_trajectory,
                action_probs=action_probs,
                ipython_display=False,
            )
        )
        self.assertEqual(
            get_file_count(result_img_directory_path), expected_images_num
        )

        custom_img_directory_path = generate_temporary_file_path(
            prefix="overcooked_visualized_trajectory", extension=""
        )
        self.assertNotEqual(
            custom_img_directory_path, result_img_directory_path
        )
        result_img_directory_path = (
            StateVisualizer().display_rendered_trajectory(
                test_trajectory,
                img_directory_path=custom_img_directory_path,
                ipython_display=False,
            )
        )
        self.assertEqual(custom_img_directory_path, result_img_directory_path)
        self.assertEqual(
            get_file_count(result_img_directory_path), expected_images_num
        )


if __name__ == "__main__":
    unittest.main()
