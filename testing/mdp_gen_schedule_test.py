import numpy as np

from overcooked_ai_py.agents.benchmarking import AgentEvaluator

np.random.seed(42)


def params_schedule_fn_constant_09_01(outside_information):
    return {
        "inner_shape": (7, 5),
        "prop_empty": 0.9,
        "prop_feats": 0.1,
        "start_all_orders": [{"ingredients": ["onion", "onion", "onion"]}],
        "display": False,
        "rew_shaping_params": None,
    }


def params_schedule_fn_constant_07_03(outside_info):
    return {
        "inner_shape": (7, 5),
        "prop_empty": 0.7,
        "prop_feats": 0.3,
        "start_all_orders": [{"ingredients": ["onion", "onion", "onion"]}],
        "display": False,
        "rew_shaping_params": None,
    }


def params_schedule_fn_constant_05_05(outside_info):
    return {
        "inner_shape": (7, 5),
        "prop_empty": 0.5,
        "prop_feats": 0.5,
        "start_all_orders": [{"ingredients": ["onion", "onion", "onion"]}],
        "display": False,
        "rew_shaping_params": None,
    }


def params_schedule_fn_interval(outside_info):
    assert (
        outside_info != {} and "progress" in outside_info
    ), "if this happens during initialization, please add initial_info to env_params to address the issue"
    progress = outside_info["progress"]
    prop_empty = 0.9 - 0.4 * progress
    prop_feats = 0.1 + 0.4 * progress
    return {
        "inner_shape": (7, 5),
        "prop_empty": prop_empty,
        "prop_feats": prop_feats,
        "start_all_orders": [{"ingredients": ["onion", "onion", "onion"]}],
        "display": False,
        "rew_shaping_params": None,
    }


default_env_params_infinite = {"horizon": 400, "num_mdp": np.inf}

default_env_params_infinite_interval = {
    "horizon": 400,
    "num_mdp": np.inf,
    "initial_info": {"progress": 0},
}


def _avg_empty_grids(ae, num_iters, outside_info_fn=None):
    num_empty_grid = []
    for i in range(num_iters):
        kwargs = {"outside_info": outside_info_fn(i)} if outside_info_fn else {}
        ae.env.reset(**kwargs)
        num_empty_grid.append(len(ae.env.mdp.terrain_pos_dict[" "]))
    return num_empty_grid


class TestParamScheduleFnConstant:
    def test_constant_schedule_095_01(self):
        ae = AgentEvaluator.from_mdp_params_infinite(
            mdp_params=None,
            env_params=default_env_params_infinite,
            outer_shape=(7, 5),
            mdp_params_schedule_fn=params_schedule_fn_constant_09_01,
        )
        num_empty_grid = _avg_empty_grids(ae, 500)
        avg_num_empty = sum(num_empty_grid) / len(num_empty_grid)
        assert 13.9 < avg_num_empty < 14.1

    def test_constant_schedule_07_03(self):
        ae = AgentEvaluator.from_mdp_params_infinite(
            mdp_params=None,
            env_params=default_env_params_infinite,
            outer_shape=(7, 5),
            mdp_params_schedule_fn=params_schedule_fn_constant_07_03,
        )
        num_empty_grid = _avg_empty_grids(ae, 500)
        avg_num_empty = sum(num_empty_grid) / len(num_empty_grid)
        assert 11.5 < avg_num_empty < 11.8

    def test_constant_schedule_05_05(self):
        ae = AgentEvaluator.from_mdp_params_infinite(
            mdp_params=None,
            env_params=default_env_params_infinite,
            outer_shape=(7, 5),
            mdp_params_schedule_fn=params_schedule_fn_constant_05_05,
        )
        num_empty_grid = _avg_empty_grids(ae, 500)
        avg_num_empty = sum(num_empty_grid) / len(num_empty_grid)
        assert 10.3 < avg_num_empty < 11.0


class TestParamScheduleFnInterval:
    def test_interval_schedule(self):
        ae = AgentEvaluator.from_mdp_params_infinite(
            mdp_params=None,
            env_params=default_env_params_infinite_interval,
            outer_shape=(7, 5),
            mdp_params_schedule_fn=params_schedule_fn_interval,
        )
        num_empty_grid = _avg_empty_grids(
            ae, 4000, outside_info_fn=lambda i: {"progress": i / 4000}
        )
        avg_num_empty_09_01 = sum(num_empty_grid[0:50]) / 50
        assert 13.9 < avg_num_empty_09_01 < 14.1
        avg_num_empty_07_03 = sum(num_empty_grid[1975:2025]) / 50
        assert 11.5 < avg_num_empty_07_03 < 11.8
        avg_num_empty_05_05 = sum(num_empty_grid[3950:4000]) / 50
        assert 10.3 < avg_num_empty_05_05 < 11.0
