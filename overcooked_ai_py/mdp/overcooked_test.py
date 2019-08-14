import unittest
import numpy as np

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import PlayerState, OvercookedGridworld, OvercookedState, ObjectState
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, DEFAULT_ENV_PARAMS
from overcooked_ai_py.mdp.layout_generator import LayoutGenerator
from overcooked_ai_py.agents.agent import AgentGroup, AgentPair, GreedyHumanModel, FixedPlanAgent
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS
from overcooked_ai_py.utils import save_pickle, load_pickle, iterate_over_files_in_dir


START_ORDER_LIST = ["any"]
n, s = Direction.NORTH, Direction.SOUTH
e, w = Direction.EAST, Direction.WEST
stay, interact = Action.STAY, Action.INTERACT
P, Obj = PlayerState, ObjectState


class TestDirection(unittest.TestCase):

    def test_direction_number_conversion(self):
        all_directions = Direction.ALL_DIRECTIONS
        all_numbers = []

        for direction in Direction.ALL_DIRECTIONS:
            number = Direction.DIRECTION_TO_INDEX[direction]
            direction_again = Direction.INDEX_TO_DIRECTION[number]
            self.assertEqual(direction, direction_again)
            all_numbers.append(number)

        # Check that all directions are distinct
        num_directions = len(all_directions)
        self.assertEqual(len(set(all_directions)), num_directions)
        
        # Check that the numbers are 0, 1, ... num_directions - 1
        self.assertEqual(set(all_numbers), set(range(num_directions)))

class TestGridworld(unittest.TestCase):

    # TODO: write more smaller targeted tests to be loaded from jsons

    def setUp(self):
        self.base_mdp = OvercookedGridworld.from_layout_name(
            "mdp_test",
            **{'cook_time': 5, 'start_order_list': ['onion', 'any']}
        )

    def test_constructor_invalid_inputs(self):
        # Height and width must be at least 3.
        with self.assertRaises(AssertionError):
            mdp = OvercookedGridworld.from_grid(['X', 'X', 'X'])
        with self.assertRaises(AssertionError):
            mdp = OvercookedGridworld.from_grid([['X', 'X', 'X']])
        with self.assertRaises(AssertionError):
            # Borders must be present.
            mdp = OvercookedGridworld.from_grid(['XOSX',
                                                 'P  D',
                                                 ' 21 '])

        with self.assertRaises(AssertionError):
            # The grid can't be ragged.
            mdp = OvercookedGridworld.from_grid(['XXPXX',
                                                 'O  2XX',
                                                 'X1 3 X',
                                                 'XDXSXX'])

        with self.assertRaises(AssertionError):
            # The agents must be numbered 1 and 2.
            mdp = OvercookedGridworld.from_grid(['XXPXX',
                                                 'O  3O',
                                                 'X1  X',
                                                 'XDXSX'])

        with self.assertRaises(AssertionError):
            # The agents must be numbered 1 and 2.
            mdp = OvercookedGridworld.from_grid(['XXPXX',
                                                 'O  1O',
                                                 'X1  X',
                                                 'XDXSX'])

        with self.assertRaises(AssertionError):
            # B is not a valid element.
            mdp = OvercookedGridworld.from_grid(['XBPXX',
                                                 'O  2O',
                                                 'X1  X',
                                                 'XDXSX'])

    def test_start_positions(self):
        expected_start_state = OvercookedState(
            [PlayerState((1, 2), Direction.NORTH), PlayerState((3, 1), Direction.NORTH)], {}, order_list=['onion', 'any'])
        actual_start_state = self.base_mdp.get_standard_start_state()
        self.assertEqual(actual_start_state, expected_start_state, '\n' + str(actual_start_state) + '\n' + str(expected_start_state))

    def test_file_constructor(self):
        mdp = OvercookedGridworld.from_layout_name('corridor')
        expected_start_state = OvercookedState(
            [PlayerState((3, 1), Direction.NORTH), PlayerState((10, 1), Direction.NORTH)], {}, order_list=None)
        actual_start_state = mdp.get_standard_start_state()
        self.assertEqual(actual_start_state, expected_start_state, '\n' + str(actual_start_state) + '\n' + str(expected_start_state))

    def test_actions(self):
        bad_state = OvercookedState(
            [PlayerState((0, 0), Direction.NORTH), PlayerState((3, 1), Direction.NORTH)], {}, order_list=['any'])
        with self.assertRaises(AssertionError):
            self.base_mdp.get_actions(bad_state)

        self.assertEqual(self.base_mdp.get_actions(self.base_mdp.get_standard_start_state()),
                         [Action.ALL_ACTIONS, Action.ALL_ACTIONS])

    def test_transitions_and_environment(self):
        bad_state = OvercookedState(
            [P((0, 0), s), P((3, 1), s)], {}, order_list=[])

        with self.assertRaises(AssertionError):
            self.base_mdp.get_state_transition(bad_state, stay)

        env = OvercookedEnv(self.base_mdp)
        env.state.order_list = ['onion', 'any']

        def check_transition(action, expected_state, expected_reward=0):
            state = env.state
            pred_state, sparse_reward, dense_reward = self.base_mdp.get_state_transition(state, action)
            self.assertEqual(pred_state, expected_state, '\n' + str(pred_state) + '\n' + str(expected_state))
            new_state, sparse_reward, _, _ = env.step(action)
            self.assertEqual(new_state, expected_state)
            self.assertEqual(sparse_reward, expected_reward)

        check_transition([n, e], OvercookedState(
            [P((1, 1), n),
             P((3, 1), e)],
            {}, order_list=['onion', 'any']))

    def test_common_mdp_jsons(self):
        traj_test_json_paths = iterate_over_files_in_dir("../common_tests/trajectory_tests/")
        for test_json_path in traj_test_json_paths:
            test_trajectory = AgentEvaluator.load_traj_from_json(test_json_path)
            try:
                AgentEvaluator.check_trajectories(test_trajectory)
            except AssertionError as e:
                self.fail("File {} failed with error:\n{}".format(test_json_path, e))

    def test_four_player_mdp(self):
        try:
            OvercookedGridworld.from_layout_name("multiplayer_schelling")
        except AssertionError as e:
            print("Loading > 2 player map failed with error:", e)

def random_joint_action():
    num_actions = len(Action.ALL_ACTIONS)
    a_idx0, a_idx1 = np.random.randint(low=0, high=num_actions, size=2)
    return (Action.INDEX_TO_ACTION[a_idx0], Action.INDEX_TO_ACTION[a_idx1])


class TestFeaturizations(unittest.TestCase):

    def setUp(self):
        self.base_mdp = OvercookedGridworld.from_layout_name("simple")
        self.mlp = MediumLevelPlanner.from_pickle_or_compute(self.base_mdp, NO_COUNTERS_PARAMS, force_compute=True)
        self.env = OvercookedEnv(self.base_mdp, **DEFAULT_ENV_PARAMS)
        self.rnd_agent_pair = AgentPair(GreedyHumanModel(self.mlp), GreedyHumanModel(self.mlp))
        np.random.seed(0)

    def test_lossless_state_featurization(self):
        trajs = self.env.get_rollouts(self.rnd_agent_pair, num_games=5)
        featurized_observations = [[self.base_mdp.lossless_state_encoding(state) for state in ep_states] for ep_states in trajs["ep_observations"]]
        expected_featurization = load_pickle("data/testing/lossless_state_featurization")
        self.assertTrue(np.array_equal(expected_featurization, featurized_observations))

    def test_state_featurization(self):
        trajs = self.env.get_rollouts(self.rnd_agent_pair, num_games=5)
        featurized_observations = [[self.base_mdp.featurize_state(state, self.mlp) for state in ep_states] for ep_states in trajs["ep_observations"]]
        expected_featurization = load_pickle("data/testing/state_featurization")
        self.assertTrue(np.array_equal(expected_featurization, featurized_observations))


class TestOvercookedEnvironment(unittest.TestCase):

    def setUp(self):
        self.base_mdp = OvercookedGridworld.from_layout_name("simple")
        self.env = OvercookedEnv(self.base_mdp, **DEFAULT_ENV_PARAMS)
        self.rnd_agent_pair = AgentPair(FixedPlanAgent([stay, w, w]), FixedPlanAgent([stay, e, e]))
        np.random.seed(0)

    def test_constructor(self):
        try:
            OvercookedEnv(self.base_mdp, horizon=10)
        except Exception as e:
            self.fail("Failed to instantiate OvercookedEnv:\n{}".format(e))

        with self.assertRaises(TypeError):
            OvercookedEnv(self.base_mdp, **{"invalid_env_param": None})

    def test_step_fn(self):
        for _ in range(10):
            joint_action = random_joint_action()
            self.env.step(joint_action)

    def test_execute_plan(self):
        action_plan = [random_joint_action() for _ in range(10)]
        self.env.execute_plan(self.base_mdp.get_standard_start_state(), action_plan)

    def test_run_agents(self):
        start_state = self.env.state
        self.env.run_agents(self.rnd_agent_pair)
        self.assertNotEqual(self.env.state, start_state)

    def test_rollouts(self):
        try:
            self.env.get_rollouts(self.rnd_agent_pair, 3)
        except Exception as e:
            self.fail("Failed to get rollouts from environment:\n{}".format(e))

    def test_one_player_env(self):
        mdp = OvercookedGridworld.from_layout_name("simple_single")
        env = OvercookedEnv(mdp, horizon=12)
        a0 = FixedPlanAgent([stay, w, w, e, e, n, e, interact, w, n, interact])
        ag = AgentGroup(a0)
        env.run_agents(ag, display=False)
        self.assertEqual(
            env.state.players_pos_and_or,
            (((2, 1), (0, -1)),)
        )

    def test_four_player_env_fixed(self):
        mdp = OvercookedGridworld.from_layout_name("multiplayer_schelling")
        assert mdp.num_players == 4
        env = OvercookedEnv(mdp, horizon=16)
        a0 = FixedPlanAgent([stay, w, w])
        a1 = FixedPlanAgent([stay, stay, e, e, n, n, n, e, interact, n, n, w, w, w, n, interact, e])
        a2 = FixedPlanAgent([stay, w, interact, n, n, e, e, e, n, e, n, interact, w])
        a3 = FixedPlanAgent([e, interact, n, n, w, w, w, n, interact, e, s])
        ag = AgentGroup(a0, a1, a2, a3)
        env.run_agents(ag, display=False)
        self.assertEqual(
            env.state.players_pos_and_or,
            (((1, 1), (-1, 0)), ((3, 1), (0, -1)), ((2, 1), (-1, 0)), ((4, 2), (0, 1)))
        )

    def test_multiple_mdp_env(self):
        mdp0 = OvercookedGridworld.from_layout_name("simple")
        mdp1 = OvercookedGridworld.from_layout_name("random0")
        mdp_fn = lambda: np.random.choice([mdp0, mdp1])
        
        # Default env
        env = OvercookedEnv(mdp_fn, horizon=100)
        env.get_rollouts(self.rnd_agent_pair, 5)

    def test_starting_position_randomization(self):
        self.base_mdp = OvercookedGridworld.from_layout_name("simple")
        start_state_fn = self.base_mdp.get_random_start_state_fn(random_start_pos=True, rnd_obj_prob_thresh=0.0)
        env = OvercookedEnv(self.base_mdp, start_state_fn)
        start_state = env.state.players_pos_and_or
        for _ in range(3):
            env.reset()
            print(env)
            curr_terrain = env.state.players_pos_and_or
            self.assertFalse(np.array_equal(start_state, curr_terrain))

    def test_starting_obj_randomization(self):
        self.base_mdp = OvercookedGridworld.from_layout_name("simple")
        start_state_fn = self.base_mdp.get_random_start_state_fn(random_start_pos=False, rnd_obj_prob_thresh=0.8)
        env = OvercookedEnv(self.base_mdp, start_state_fn)
        start_state = env.state.all_objects_list
        for _ in range(3):
            env.reset()
            print(env)
            curr_terrain = env.state.all_objects_list
            self.assertFalse(np.array_equal(start_state, curr_terrain))

    def test_failing_rnd_layout(self):
        with self.assertRaises(TypeError):
            mdp_gen_params = {"None": None}
            mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(**mdp_gen_params)
            OvercookedEnv(mdp=mdp_fn, **DEFAULT_ENV_PARAMS)

    def test_random_layout(self):
        mdp_gen_params = {"prop_feats": (1, 1)}
        mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(**mdp_gen_params)
        env = OvercookedEnv(mdp=mdp_fn, **DEFAULT_ENV_PARAMS)
        start_terrain = env.mdp.terrain_mtx

        for _ in range(3):
            env.reset()
            print(env)
            curr_terrain = env.mdp.terrain_mtx
            self.assertFalse(np.array_equal(start_terrain, curr_terrain))

        mdp_gen_params = {"mdp_choices": ['simple', 'unident_s']}
        mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(**mdp_gen_params)
        env = OvercookedEnv(mdp=mdp_fn, **DEFAULT_ENV_PARAMS)
        
        layouts_seen = []
        for _ in range(10):
            layouts_seen.append(env.mdp.terrain_mtx)
            env.reset()
        all_same_layout = all([np.array_equal(env.mdp.terrain_mtx, terrain) for terrain in layouts_seen])
        self.assertFalse(all_same_layout)
        
        
class TestGymEnvironment(unittest.TestCase):

    def setUp(self):
        self.base_mdp = OvercookedGridworld.from_layout_name("simple")
        self.env = OvercookedEnv(self.base_mdp, **DEFAULT_ENV_PARAMS)
        self.rnd_agent_pair = AgentPair(FixedPlanAgent([]), FixedPlanAgent([]))
        np.random.seed(0)

    # TODO: write more tests here

if __name__ == '__main__':
    unittest.main()
