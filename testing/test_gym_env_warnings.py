import warnings

import gymnasium
import numpy as np
import pytest

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld


@pytest.fixture
def env():
    base_mdp = OvercookedGridworld.from_layout_name("cramped_room")
    base_env = OvercookedEnv.from_mdp(base_mdp, horizon=100)
    env = gymnasium.make(
        "Overcooked-v0",
        base_env=base_env,
        featurize_fn=base_env.featurize_state_mdp,
    )
    yield env
    env.close()


class TestNoGymnasiumWarnings:
    def test_reset_no_warnings(self, env):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            env.reset()
        gym_warnings = [
            w for w in caught if "gymnasium" in (w.filename or "").lower()
        ]
        assert gym_warnings == [], (
            f"Gymnasium warnings during reset(): "
            f"{[str(w.message) for w in gym_warnings]}"
        )

    def test_step_no_warnings(self, env):
        env.reset()
        action = (0, 0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            env.step(action)
        gym_warnings = [
            w for w in caught if "gymnasium" in (w.filename or "").lower()
        ]
        assert gym_warnings == [], (
            f"Gymnasium warnings during step(): "
            f"{[str(w.message) for w in gym_warnings]}"
        )


class TestObservationFormat:
    def test_reset_returns_tuple_of_arrays(self, env):
        obs, info = env.reset()
        assert isinstance(obs, tuple), f"Expected tuple, got {type(obs)}"
        assert len(obs) == 2
        for i, ob in enumerate(obs):
            assert isinstance(ob, np.ndarray), (
                f"obs[{i}] should be np.ndarray, got {type(ob)}"
            )

    def test_step_returns_tuple_of_arrays(self, env):
        env.reset()
        obs, reward, done, truncated, info = env.step((0, 0))
        assert isinstance(obs, tuple), f"Expected tuple, got {type(obs)}"
        assert len(obs) == 2
        for i, ob in enumerate(obs):
            assert isinstance(ob, np.ndarray), (
                f"obs[{i}] should be np.ndarray, got {type(ob)}"
            )

    def test_obs_matches_observation_space(self, env):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs), (
            "reset() obs not in observation_space"
        )

    def test_step_obs_matches_observation_space(self, env):
        env.reset()
        obs, _, _, _, _ = env.step((0, 0))
        assert env.observation_space.contains(obs), (
            "step() obs not in observation_space"
        )

    def test_obs_shape_matches_featurize(self, env):
        obs, _ = env.reset()
        expected_shape = env.unwrapped.mdp.get_featurize_state_shape()
        assert obs[0].shape == expected_shape
        assert obs[1].shape == expected_shape

    def test_obs_dtype(self, env):
        obs, _ = env.reset()
        assert np.issubdtype(obs[0].dtype, np.floating)


class TestInfoDict:
    def test_reset_info_has_overcooked_state(self, env):
        _, info = env.reset()
        assert "overcooked_state" in info

    def test_reset_info_has_other_agent_env_idx(self, env):
        _, info = env.reset()
        assert "other_agent_env_idx" in info
        assert info["other_agent_env_idx"] in (0, 1)

    def test_step_info_has_overcooked_state(self, env):
        env.reset()
        _, _, _, _, info = env.step((0, 0))
        assert "overcooked_state" in info

    def test_step_info_has_other_agent_env_idx(self, env):
        env.reset()
        _, _, _, _, info = env.step((0, 0))
        assert "other_agent_env_idx" in info
        assert info["other_agent_env_idx"] in (0, 1)


class TestObservationSpace:
    def test_is_tuple_space(self, env):
        assert isinstance(env.observation_space, gymnasium.spaces.Tuple)
        assert len(env.observation_space.spaces) == 2

    def test_tuple_contains_box_spaces(self, env):
        for space in env.observation_space.spaces:
            assert isinstance(space, gymnasium.spaces.Box)
