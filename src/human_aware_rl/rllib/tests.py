import copy
import unittest
from math import isclose

import numpy as np

from human_aware_rl.rllib.rllib import OvercookedMultiAgent
from human_aware_rl.rllib.utils import (
    get_required_arguments,
    iterable_equal,
    softmax,
)


class RllibEnvTest(unittest.TestCase):
    def setUp(self):
        print(
            "\nIn Class {}, in Method {}".format(
                self.__class__.__name__, self._testMethodName
            )
        )
        self.params = copy.deepcopy(OvercookedMultiAgent.DEFAULT_CONFIG)
        self.timesteps = [0, 10, 100, 500, 1000, 1500, 2000, 2500]

    def tearDown(self):
        pass

    def _assert_lists_almost_equal(self, first, second, places=7):
        for a, b in zip(first, second):
            self.assertAlmostEqual(a, b, places=places)

    def _test_bc_schedule(self, bc_schedule, expected_bc_factors):
        self.params["multi_agent_params"]["bc_schedule"] = bc_schedule
        env = OvercookedMultiAgent.from_config(self.params)
        actual_bc_factors = []

        for t in self.timesteps:
            env.anneal_bc_factor(t)
            actual_bc_factors.append(env.bc_factor)

        self._assert_lists_almost_equal(expected_bc_factors, actual_bc_factors)

    def _test_bc_creation_proportion(self, env, factor, trials=10000):
        env.bc_factor = factor
        tot_bc = 0
        for _ in range(trials):
            env.reset(regen_mdp=False)
            num_bc = sum(
                map(lambda agent: int(agent.startswith("bc")), env.curr_agents)
            )
            self.assertLessEqual(num_bc, 1)
            tot_bc += num_bc
        actual_factor = tot_bc / trials
        self.assertAlmostEqual(actual_factor, factor, places=1)

    def test_env_creation(self):
        # Valid creation
        env = OvercookedMultiAgent.from_config(self.params)
        for param, expected in self.params["multi_agent_params"].items():
            self.assertEqual(expected, getattr(env, param))

        # Invalid bc_schedules
        invalid_schedules = [
            [(-1, 0.0), (1.0, 1e5)],
            [(0.0, 0.0), (10, 1), (5, 0.5)],
            [(0, 0), (5, 1), (10, 1.5)],
        ]
        for sched in invalid_schedules:
            self.params["multi_agent_params"]["bc_schedule"] = sched
            self.assertRaises(
                AssertionError, OvercookedMultiAgent.from_config, self.params
            )

    def test_reward_shaping_annealing(self):
        self.params["multi_agent_params"]["reward_shaping_factor"] = 1
        self.params["multi_agent_params"]["reward_shaping_horizon"] = 1e3

        expected_rew_factors = [
            1,
            990 / 1e3,
            900 / 1e3,
            500 / 1e3,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        actual_rew_factors = []

        env = OvercookedMultiAgent.from_config(self.params)

        for t in self.timesteps:
            env.anneal_reward_shaping_factor(t)
            actual_rew_factors.append(env.reward_shaping_factor)

        self._assert_lists_almost_equal(
            expected_rew_factors, actual_rew_factors
        )

    def test_bc_annealing(self):
        # Test no annealing
        self._test_bc_schedule(
            OvercookedMultiAgent.self_play_bc_schedule,
            [0.0] * len(self.timesteps),
        )

        # Test annealing
        anneal_bc_schedule = [(0, 0.0), (1e3, 1.0), (2e3, 0.0)]
        expected_bc_factors = [
            0.0,
            10 / 1e3,
            100 / 1e3,
            500 / 1e3,
            1.0,
            500 / 1e3,
            0.0,
            0.0,
        ]
        self._test_bc_schedule(anneal_bc_schedule, expected_bc_factors)

    def test_agent_creation(self):
        env = OvercookedMultiAgent.from_config(self.params)
        obs = env.reset()

        # Check that we have the right number of agents with valid names
        self.assertEqual(len(env.curr_agents), 2)
        self.assertListEqual(list(obs.keys()), env.curr_agents)

        # Ensure that bc agents are created 'factor' percentage of the time
        bc_factors = [0.0, 0.1, 0.5, 0.9, 1.0]
        for factor in bc_factors:
            self._test_bc_creation_proportion(env, factor)


class RllibUtilsTest(unittest.TestCase):
    def setUp(self):
        print(
            "\nIn Class {}, in Method {}".format(
                self.__class__.__name__, self._testMethodName
            )
        )
        pass

    def tearDown(self):
        pass

    def test_softmax(self):
        logits = np.array(
            [
                [0.1, 0.1, 0.1],
                [-0.1, 0.0, 0.1],
                [0.5, -1.2, 3.2],
                [-1.6, -2.0, -1.5],
            ]
        )
        expected = np.array(
            [
                [0.33333333, 0.33333333, 0.33333333],
                [0.30060961, 0.33222499, 0.3671654],
                [0.06225714, 0.01137335, 0.92636951],
                [0.36029662, 0.24151404, 0.39818934],
            ]
        )

        actual = softmax(logits)

        self.assertTrue(np.allclose(expected, actual))

    def test_iterable_equal(self):
        a = [(1,), (1, 2)]
        b = ([1], [1, 2])

        self.assertTrue(iterable_equal(a, b))

        a = [(1, 2), (1)]
        b = [(1,), (1, 2)]

        self.assertFalse(iterable_equal(a, b))

    def test_get_required_arguments(self):
        def foo1(a):
            pass

        def foo2(a, b):
            pass

        def foo3(a, b, c):
            pass

        def foo4(a, b, c="bar"):
            pass

        def foo5(a, b="bar", d="baz", **kwargs):
            pass

        fns = [foo1, foo2, foo3, foo4, foo5]
        expected = [1, 2, 3, 2, 1]

        for fn, expected in zip(fns, expected):
            self.assertEqual(expected, len(get_required_arguments(fn)))


if __name__ == "__main__":
    unittest.main()
