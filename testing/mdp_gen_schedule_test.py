import unittest
import numpy as np

from overcooked_ai_py.mdp.actions import Direction, Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, PlayerState, ObjectState
from overcooked_ai_py.agents.benchmarking import AgentEvaluator

np.random.seed(42)

n, s = Direction.NORTH, Direction.SOUTH
e, w = Direction.EAST, Direction.WEST
stay, interact = Action.STAY, Action.INTERACT
P, Obj = PlayerState, ObjectState

force_compute_large = False
force_compute = True
DISPLAY = False

simple_mdp = OvercookedGridworld.from_layout_name('cramped_room')
large_mdp = OvercookedGridworld.from_layout_name('corridor')


def params_schedule_fn_constant_09_01(outside_information):
    """
    In this preliminary version, the outside information is ignored
    """
    mdp_default_gen_params = {
        "inner_shape": (7, 5),
        "prop_empty": 0.9,
        "prop_feats": 0.1,
        "start_all_orders": [
            {"ingredients": ["onion", "onion", "onion"]}
        ],
        "display": False,
        "rew_shaping_params": None
    }
    return mdp_default_gen_params


def params_schedule_fn_constant_07_03(outside_info):
    """
    In this preliminary version, the outside information is ignored
    """
    mdp_default_gen_params = {
        "inner_shape": (7, 5),
        "prop_empty": 0.7,
        "prop_feats": 0.3,
        "start_all_orders": [
            {"ingredients": ["onion", "onion", "onion"]}
        ],
        "display": False,
        "rew_shaping_params": None
    }
    return mdp_default_gen_params


def params_schedule_fn_constant_05_05(outside_info):
    """
    In this preliminary version, the outside information is ignored
    """
    mdp_default_gen_params = {
        "inner_shape": (7, 5),
        "prop_empty": 0.5,
        "prop_feats": 0.5,
        "start_all_orders": [
            {"ingredients": ["onion", "onion", "onion"]}
        ],
        "display": False,
        "rew_shaping_params": None
    }
    return mdp_default_gen_params


def params_schedule_fn_interval(outside_info):
    """
    outside_information (dict):
        progress (float in [0, 1] interval) a number that indicate progress
    """
    assert outside_info != {} and "progress" in outside_info, \
        "if this happens during initialization, please add initial_info to env_params to address the issue"
    progress = outside_info["progress"]
    prop_empty = 0.9 - 0.4 * progress
    prop_feats = 0.1 + 0.4 * progress
    mdp_params_generated = {
        "inner_shape": (7, 5),
        "prop_empty": prop_empty,
        "prop_feats": prop_feats,
        "start_all_orders": [
            {"ingredients": ["onion", "onion", "onion"]}
        ],
        "display": False,
        "rew_shaping_params": None
    }
    return mdp_params_generated


default_env_params_infinite = {"horizon": 400, "num_mdp": np.inf}

default_env_params_infinite_interval = {"horizon": 400, "num_mdp": np.inf, "initial_info": {"progress": 0}}


class TestParamScheduleFnConstant(unittest.TestCase):

    def test_constant_schedule_095_01(self):
        ae = AgentEvaluator.from_mdp_params_infinite(mdp_params=None, env_params=default_env_params_infinite,
                                                     outer_shape= (7, 5),
                                                     mdp_params_schedule_fn=params_schedule_fn_constant_09_01)
        num_empty_grid = []
        for i in range(500):
            ae.env.reset()
            empty_i = len(ae.env.mdp.terrain_pos_dict[' '])
            num_empty_grid.append(empty_i)
        avg_num_empty = sum(num_empty_grid)/len(num_empty_grid)
        print("avg number of empty grid:", avg_num_empty)
        # the number of empty square should be consistant"

        self.assertTrue(13.9 < avg_num_empty < 14.1)

    def test_constant_schedule_07_03(self):
        ae = AgentEvaluator.from_mdp_params_infinite(mdp_params=None, env_params=default_env_params_infinite,
                                                     outer_shape= (7, 5),
                                                     mdp_params_schedule_fn=params_schedule_fn_constant_07_03)
        num_empty_grid = []
        for i in range(500):
            ae.env.reset()
            empty_i = len(ae.env.mdp.terrain_pos_dict[' '])
            num_empty_grid.append(empty_i)
        avg_num_empty = sum(num_empty_grid)/len(num_empty_grid)
        print("avg number of empty grid:", avg_num_empty)
        # the number of empty square should be fairlyconsistant"
        self.assertTrue(11.5 < avg_num_empty < 11.8)

    def test_constant_schedule_05_05(self):
        ae = AgentEvaluator.from_mdp_params_infinite(mdp_params=None, env_params=default_env_params_infinite,
                                                     outer_shape= (7, 5),
                                                     mdp_params_schedule_fn=params_schedule_fn_constant_05_05)
        num_empty_grid = []
        for i in range(500):
            ae.env.reset()
            empty_i = len(ae.env.mdp.terrain_pos_dict[' '])
            num_empty_grid.append(empty_i)
        avg_num_empty = sum(num_empty_grid)/len(num_empty_grid)
        print("avg number of empty grid:", avg_num_empty)
        # the number of empty square should be fairlyconsistant"
        self.assertTrue(10.4 < avg_num_empty < 10.9)


class TestParamScheduleFnInterval(unittest.TestCase):
    def test_interval_schedule(self):
        ae = AgentEvaluator.from_mdp_params_infinite(mdp_params=None, env_params=default_env_params_infinite_interval,
                                                     outer_shape= (7, 5),
                                                     mdp_params_schedule_fn=params_schedule_fn_interval)
        num_empty_grid = []
        for i in range(4000):
            ae.env.reset(outside_info={"progress": i/4000})
            empty_i = len(ae.env.mdp.terrain_pos_dict[' '])
            num_empty_grid.append(empty_i)
        avg_num_empty_09_01 = sum(num_empty_grid[0:50]) / 50
        self.assertTrue(13.9 < avg_num_empty_09_01 < 14.1)
        avg_num_empty_07_03 = sum(num_empty_grid[1975:2025]) / 50
        self.assertTrue(11.5 < avg_num_empty_07_03 < 11.8)
        avg_num_empty_05_05 = sum(num_empty_grid[3950:4000]) / 50
        self.assertTrue(10.4 < avg_num_empty_05_05 < 10.9)
        print("avg number of empty grids:", avg_num_empty_09_01, avg_num_empty_07_03, avg_num_empty_05_05)


if __name__ == '__main__':
    unittest.main()

