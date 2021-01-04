import unittest
import numpy as np
from overcooked_ai_py.mdp.layout_evaluator import terrain_analysis, path_to_actions, \
    UNDEFIND_ACTION, remove_extra_action, add_action_from_location, calculate_entropy_of_path, \
    ENTROPY_RHO, path_to_actions_with_padding, graph_from_terrain, shortest_walk_path

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
"""
class TestHandoverEvaluation(unittest.TestCase):

    def setUp(self):
        self.divided_mtx = [
            ['X', 'P', 'X', 'X', 'X'],
            ['O', '2', 'X', ' ', 'O'],
            ['X', ' ', 'X', '1', 'X'],
            ['X', 'D', 'X', 'S', 'X'],

        ]
        print(self.divided_mtx)

        self.diagnoal_divided_mtx = [
            ['X', 'X', 'X', 'X', 'S', 'X'],
            ['X', 'X', 'X', ' ', '1', 'O'],
            ['P', '2', ' ', 'X', 'X', 'X'],
            ['X', 'D', 'X', 'X', 'X', 'X'],
        ]
        print(self.diagnoal_divided_mtx)

    def test_simple(self):
        terrain_analysis(self.divided_mtx, False)

    def test_diagonal(self):
        terrain_analysis(self.diagnoal_divided_mtx, False)


class TestMotionExtractor(unittest.TestCase):

    def setUp(self):
        self.divided_mtx = [
            ['X', 'P', 'X', 'X', 'X', 'X'],
            ['O', '2', 'X', ' ', ' ', 'O'],
            ['X', ' ', 'X', '1', ' ', 'X'],
            ['X', 'D', 'X', 'S', 'X', 'X'],
        ]
    def test_remove_extra(self):

        # test removing extra action for
        # ['X', 'X', 'X', 'X']
        # ['X', '1', ' ', 'O']
        # ['X', 'X', 'X', 'X']
        # vs
        # ['X', 'X', 'O', 'X']
        # ['X', '1', ' ', 'X']
        # ['X', 'X', 'X', 'X']

        actions_remove = [(1, 0), (1, 0), 'interact']
        self.assertEqual(remove_extra_action(actions_remove), [(1,0), 'interact'])

        actions_no_remove = [(1, 0), (0, -1), 'interact']
        self.assertEqual(remove_extra_action(actions_no_remove), [(1, 0), (0, -1), 'interact'])

    def test_add_action(self):

        # row 1 of terrain
        # ['O', ' ', 'X', ' ', 'P']
        # test addition of actions for agent picking up from counter (covers all cases)

        # no movement = no action
        actions = []
        curr_loc = 'UND_L'
        next_loc = 'UND_L'
        add_action_from_location(curr_loc, next_loc, actions)
        self.assertEqual(actions, [])

        # first of two actions that signify counter picking up
        curr_loc = next_loc
        next_loc = (1, 3)
        add_action_from_location(curr_loc, next_loc, actions)
        self.assertEqual(actions, ['UND_A'])

        # second of two actions that signify counter picking up
        curr_loc = next_loc
        next_loc = (1, 4)
        add_action_from_location(curr_loc, next_loc, actions)
        self.assertEqual(actions, ['interact'])

        # real movement action
        curr_loc = next_loc
        next_loc = (1, 5)
        add_action_from_location(curr_loc, next_loc, actions)
        self.assertEqual(actions, ['interact', (1, 0)])

        # stop moving because of interaction
        curr_loc = next_loc
        next_loc = 'UND_L'
        add_action_from_location(curr_loc, next_loc, actions)
        self.assertEqual(actions, ['interact', (1, 0), 'interact'])

    def test_onion_pickup_2(self):
        # basic motion test 1
        path_0 = ['UND_L', 'UND_L']
        path_1 = [(1, 1), (1, 0)]
        self.assertEqual(path_to_actions(path_0, path_1, self.divided_mtx), (
            [UNDEFIND_ACTION, UNDEFIND_ACTION],
            [(-1, 0), 'interact']
        ))

    def test_onion_pickup_1(self):
        # basic motion test 2
        path_0 = [(2, 3), (1, 3), (1, 4), (1, 5)]
        path_1 = ['UND_L', 'UND_L', 'UND_L', 'UND_L']
        self.assertEqual(path_to_actions(path_0, path_1, self.divided_mtx), (
            [(0, -1), (1, 0), 'interact'],
            [UNDEFIND_ACTION, UNDEFIND_ACTION, UNDEFIND_ACTION]
        ))


    def test_onion_drop_1(self):
        # this tests counter handover
        path_0 = [(1, 4), (1, 3), (1, 2), 'UND_L', 'UND_L']
        path_1 = ['UND_L', 'UND_L', (1, 2), (1, 1), (0, 1)]
        self.assertEqual(path_to_actions(path_0, path_1, self.divided_mtx), (
            [(-1, 0), 'interact', UNDEFIND_ACTION, UNDEFIND_ACTION, UNDEFIND_ACTION],
            [UNDEFIND_ACTION, UNDEFIND_ACTION, 'interact', (0, -1), 'interact']
        ))

    def test_dish_serving(self):
        # this tests counter handover in the other direction
        # this tests the border cases for remove_extra_action
        path_0 = ['UND_L', (1, 2), (1, 3), (2, 3), (3, 3)]
        path_1 = [(1, 1), (1, 2), 'UND_L', 'UND_L', 'UND_L']
        self.assertEqual(path_to_actions(path_0, path_1, self.divided_mtx), (
            [UNDEFIND_ACTION, UNDEFIND_ACTION, 'interact', (0, 1), 'interact'],
            [(1, 0), 'interact', UNDEFIND_ACTION, UNDEFIND_ACTION, UNDEFIND_ACTION]
        ))

    def test_dish_pickup_1(self):
        # this test abbreviation of additional movement because of natural agent facing
        path_0 = ['UND_L', 'UND_L', 'UND_L']
        path_1 = [(1, 1), (2, 1), (3, 1)]
        self.assertEqual(path_to_actions(path_0, path_1, self.divided_mtx), (
            [UNDEFIND_ACTION, UNDEFIND_ACTION],
            [(0, 1), 'interact']
        ))

class TestMotionExtractorWithPathPadding(unittest.TestCase):
    def setUp(self):
        self.divided_mtx = [
            ['X', 'P', 'X', 'X', 'X', 'X'],
            ['O', ' ', 'X', ' ', ' ', 'O'],
            ['X', ' ', 'X', ' ', ' ', 'X'],
            ['X', 'D', 'X', 'S', 'X', 'X'],
        ]

        self.keyhole_maze = [
            ['X', 'P', 'X', 'X', 'X', 'X', 'S', 'X', 'X', 'X'],
            ['X', ' ', 'X', ' ', ' ', ' ', ' ', 'X', ' ', 'X'],
            ['X', ' ', 'X', ' ', 'X', 'X', ' ', 'X', ' ', 'X'],
            ['X', ' ', 'X', ' ', 'X', 'X', ' ', 'X', ' ', 'X'],
            ['X', ' ', 'X', ' ', 'X', 'X', ' ', 'X', ' ', 'X'],
            ['X', ' ', 'X', ' ', 'X', 'X', ' ', 'X', ' ', 'X'],
            ['X', ' ', 'X', ' ', 'X', 'X', ' ', 'X', ' ', 'X'],
            ['X', ' ', 'X', ' ', 'X', 'X', ' ', 'X', ' ', 'O'],
            ['X', ' ', ' ', ' ', 'X', 'X', ' ', ' ', ' ', 'X'],
            ['X', 'X', 'D', 'X', 'X', 'X', 'X', 'X', 'X', 'X']
        ]

        self.walk_graph, self.handover_graph = graph_from_terrain(self.keyhole_maze)

    def testShortestWalkPath(self):
        action_path = shortest_walk_path(self.walk_graph, (4, 8), (7, 9), self.keyhole_maze)
        self.assertEqual(action_path, [(0, 1), (0, 1), (0, 1), (1, 0)])

        action_path = shortest_walk_path(self.walk_graph, (7, 8), (0, 1), self.keyhole_maze)
        self.assertEqual(action_path, [(0, 1), (-1, 0), (-1, 0), (0, -1), (0, -1), (0, -1),
                                       (0, -1), (0, -1), (0, -1), (0, -1), (-1, 0), (-1, 0),
                                       (-1, 0), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1),
                                       (0, 1), (0, 1), (-1, 0), (-1, 0), (0, -1), (0, -1),
                                       (0, -1), (0, -1), (0, -1), (0, -1), (0, -1), (0, -1)])

    def testBasicMotion(self):
        # this tests the path padding for undefined actions when one agent should stay
        path_0 = ['UND_L', 'UND_L']
        path_1 = [(1, 1), (1, 0)]
        self.assertEqual(path_to_actions_with_padding(path_0, path_1, self.divided_mtx, (2, 3), (1, 1)), (
            [(0, 0), (0, 0)],
            [(-1, 0), 'interact']
        ))

    def testCounterHandoverTurning(self):
        # this tests the path padding for undefined actions when there is a counter handover
        # with turning to reach the counter
        # with the padded path being shorter than the undefined
        path_0 = [(1, 4), (1, 3), (1, 2), 'UND_L', 'UND_L']
        path_1 = ['UND_L', 'UND_L', (1, 2), (1, 1), (0, 1)]
        prev_loc_0 = (1, 4)
        prev_loc_1 = (1, 1)
        self.assertEqual(path_to_actions_with_padding(path_0, path_1, self.divided_mtx, prev_loc_0, prev_loc_1), (
            [(-1, 0), 'interact', (0, 0), (0, 0), (0, 0)],
            [(0, 0), (1, 0), 'interact', (0, -1), 'interact']
        ))

    def testCounterHandoverNoTurning(self):
        # this tests the path padding for undefined actions when there is a counter handover
        # without turning to reach the counter
        # with the padded path being longer than the undefined
        path_0 = ['UND_L', 'UND_L', (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (0, 6)]
        path_1 = [(1, 1), (1, 2), 'UND_L', 'UND_L', 'UND_L', 'UND_L', 'UND_L', 'UND_L']
        prev_loc_0 = (1, 6)
        prev_loc_1 = (1, 1)
        self.assertEqual(path_to_actions_with_padding(path_0, path_1, self.keyhole_maze, prev_loc_0, prev_loc_1), (
            [(-1, 0), (-1, 0), (-1, 0), 'interact', (1, 0), (1, 0), (1, 0), (0, -1), 'interact'],
            [(1, 0), 'interact', (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
        ))

class TestEntropyComparison(unittest.TestCase):
    def setUp(self):
        self.divided_mtx_basic = [
            ['X', 'P', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'],
            ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
            ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
            ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
            ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '1', 'X'],
            ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
            ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
            ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'O'],
            ['X', ' ', '2', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
            ['X', 'X', 'D', 'X', 'X', 'X', 'X', 'S', 'X', 'X']
        ]
        self.divided_mtx_center_counter = [
            ['X', 'P', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'],
            ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
            ['X', ' ', 'X', 'X', 'X', 'X', 'X', 'X', ' ', 'X'],
            ['X', ' ', 'X', 'X', 'X', 'X', 'X', 'X', ' ', 'X'],
            ['X', ' ', 'X', 'X', 'X', 'X', 'X', 'X', '1', 'X'],
            ['X', ' ', 'X', 'X', 'X', 'X', 'X', 'X', ' ', 'X'],
            ['X', ' ', 'X', 'X', 'X', 'X', 'X', 'X', ' ', 'X'],
            ['X', ' ', 'X', 'X', 'X', 'X', 'X', 'X', ' ', 'O'],
            ['X', ' ', '2', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
            ['X', 'X', 'D', 'X', 'X', 'X', 'X', 'S', 'X', 'X']
        ]

        self.divided_mtx_keyhole_maze = [
            ['X', 'P', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'],
            ['X', ' ', 'X', ' ', ' ', ' ', ' ', 'X', ' ', 'X'],
            ['X', ' ', 'X', ' ', 'X', 'X', ' ', 'X', ' ', 'X'],
            ['X', ' ', 'X', ' ', 'X', 'X', ' ', 'X', ' ', 'X'],
            ['X', ' ', 'X', ' ', 'X', 'X', ' ', 'X', '1', 'X'],
            ['X', ' ', 'X', ' ', 'X', 'X', ' ', 'X', ' ', 'X'],
            ['X', ' ', 'X', ' ', 'X', 'X', ' ', 'X', ' ', 'X'],
            ['X', ' ', 'X', ' ', 'X', 'X', ' ', 'X', ' ', 'O'],
            ['X', ' ', '2', ' ', 'X', 'X', ' ', ' ', ' ', 'X'],
            ['X', 'X', 'D', 'X', 'X', 'X', 'X', 'S', 'X', 'X']
        ]

    def test_entropy_calculation(self):
        # This tests the calculate_entropy_of_path helper function
        path = [(0, 1), (0, 1), (1, 0), 'interact']
        path_undefined_at_end = [(0, 1), (0, 1), (1, 0), 'interact', 'UND_A', 'UND_A']
        path_undefined_at_start = ['UND_A', 'UND_A', (0, 1), (0, 1), (1, 0), 'interact']

        entropy = -np.log(2) - np.log(1) - np.log(1) + 3 * np.log(ENTROPY_RHO)

        self.assertAlmostEqual(calculate_entropy_of_path(path, ENTROPY_RHO), entropy)
        self.assertAlmostEqual(calculate_entropy_of_path(path_undefined_at_end, ENTROPY_RHO), entropy)
        self.assertAlmostEqual(calculate_entropy_of_path(path_undefined_at_start, ENTROPY_RHO), entropy)

    def test_entropy_comparison_basic(self):
        # This tests entropy comparison for some basic variation in layouts

        # same number of segments (3) and total path length (12)
        path_0 = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (1, 0),
                  (1, 0), (0, -1), (0, -1), (0, -1), (0, -1), (0, -1)]
        path_1 = [(0, 1), (0, 1), (0, 1), (0, 1), (1, 0), (1, 0),
                  (1, 0), (1, 0), (0, -1), (0, -1), (0, -1), (0, -1)]

        entropy_0 = calculate_entropy_of_path(path_0, ENTROPY_RHO)
        entropy_1 = calculate_entropy_of_path(path_1, ENTROPY_RHO)

        self.assertGreater(entropy_0, entropy_1)

        print('same # segments and path length')
        print('----------------------------------------')
        print('5 2 5:', entropy_0)
        print('4 4 4:', entropy_1)
        print()

        # different number of segments (adding a segment to the end)
        path_0 = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (1, 0),
                  (1, 0), (0, -1), (0, -1), (0, -1), (0, -1), (0, -1), (1, 0), (1, 0)]
        path_1 = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (1, 0),
                  (1, 0), (0, -1), (0, -1), (0, -1), (0, -1), (0, -1)]

        entropy_0 = calculate_entropy_of_path(path_0, ENTROPY_RHO)
        entropy_1 = calculate_entropy_of_path(path_1, ENTROPY_RHO)

        self.assertGreater(entropy_0, entropy_1)

        print('same path but one has an extra segment')
        print('----------------------------------------')
        print('5 2 5 2:', entropy_0)
        print('5 2 5:', entropy_1)
        print()

        # different number of segments with same path length
        path_0 = [(0, 1), (0, 1), (0, 1), (1, 0), (1, 0), (0, -1),
                  (0, -1), (0, -1), (0, 1), (0, 1), (0, 1), (0, 1)]
        path_1 = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (1, 0),
                  (1, 0), (0, -1), (0, -1), (0, -1), (0, -1), (0, -1)]

        entropy_0 = calculate_entropy_of_path(path_0, ENTROPY_RHO)
        entropy_1 = calculate_entropy_of_path(path_1, ENTROPY_RHO)

        self.assertGreater(entropy_0, entropy_1)

        print('diff # segments with same path length')
        print('----------------------------------------')
        print('3 2 3 4:', entropy_0)
        print('5 2 5:', entropy_1)
        print()

        # same number of segments with different path length
        path_0 = [(0, 1), (0, 1), (0, 1), (1, 0), (1, 0), (0, -1), (0, -1), (0, -1)]
        path_1 = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (1, 0),
                  (1, 0), (0, -1), (0, -1), (0, -1), (0, -1), (0, -1)]

        entropy_0 = calculate_entropy_of_path(path_0, ENTROPY_RHO)
        entropy_1 = calculate_entropy_of_path(path_1, ENTROPY_RHO)

        self.assertGreater(entropy_0, entropy_1)

        print('same # segments with diff path length')
        print('----------------------------------------')
        print('3 2 3:', entropy_0)
        print('5 2 5:', entropy_1)
        print()

        # different number of segments (adding a segment to the end with length > ro (5))
        path_0 = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (1, 0),
                  (1, 0), (0, -1), (0, -1), (0, -1), (0, -1), (0, -1),
                  (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0)]
        path_1 = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (1, 0),
                  (1, 0), (0, -1), (0, -1), (0, -1), (0, -1), (0, -1)]

        entropy_0 = calculate_entropy_of_path(path_0, ENTROPY_RHO)
        entropy_1 = calculate_entropy_of_path(path_1, ENTROPY_RHO)

        #self.assertGreater(entropy_0, entropy_1)

        print('same path but one has extra segment > ro (5)')
        print('----------------------------------------')
        print('5 2 5 6:', entropy_0)
        print('5 2 5:', entropy_1)
        print()

    def test_entropy_comparison_onion_dropoff(self):
        # This tests how entropy compares for the onion pickup task for three different layouts

        basic = [(-1, 0), (-1, 0), (-1, 0), (-1, 0), (-1, 0), (-1, 0), (-1, 0),
                 (0, -1), (0, -1), (0, -1), (0, -1), (0, -1), (0, -1), 'interact']
        counter = [(0, -1), (0, -1), (0, -1), (0, -1), (0, -1), (0, -1), (-1, 0),
                   (-1, 0), (-1, 0), (-1, 0), (-1, 0), (-1, 0), (-1, 0), (0, -1), 'interact']
        maze = [(0, 1), (-1, 0), (-1, 0), (0, -1), (0, -1), (0, -1), (0, -1), (0, -1),
                (0, -1), (0, -1), (-1, 0), (-1, 0), (-1, 0), (0, 1), (0, 1), (0, 1),
                (0, 1), (0, 1), (0, 1), (0, 1), (-1, 0), (-1, 0), (0, -1), (0, -1),
                (0, -1), (0, -1), (0, -1),(0, -1), (0, -1), 'interact']

        # checks that the hard coded action paths above are accurate and end in the correct spots
        start_loc = [8, 7]
        end_loc_with_turn = [1, 0]
        end_loc_wo_turn = [1, 1]

        curr_loc = start_loc.copy()
        for action in basic:
            if action != 'interact':
                curr_loc[0] += action[0]
                curr_loc[1] += action[1]
        self.assertEqual(end_loc_wo_turn, curr_loc, 'basic')

        curr_loc = start_loc.copy()
        for action in counter:
            if action != 'interact':
                curr_loc[0] += action[0]
                curr_loc[1] += action[1]
        self.assertEqual(end_loc_with_turn, curr_loc, 'counter')

        curr_loc = start_loc.copy()
        for action in maze:
            if action != 'interact':
                curr_loc[0] += action[0]
                curr_loc[1] += action[1]
        self.assertEqual(end_loc_wo_turn, curr_loc, 'maze')

        # assembles the three paths into a list
        paths_to_compare = {'basic': basic, 'counter': counter, 'maze': maze}

        # calculate and save entropies of each path
        entropies = {}
        for path in paths_to_compare:
            entropies[path] = calculate_entropy_of_path(paths_to_compare[path], ENTROPY_RHO)

        print(entropies)

    def test_entropy_full_path(self):
        # this tests the entropy comparison for the first foundn full path for both agents

        # assembles terrains into a list to reduce repetitive code
        terrains = {'basic': self.divided_mtx_basic, 'counter': self.divided_mtx_center_counter, 'maze': self.divided_mtx_keyhole_maze}
        paths = {}
        entropies = {}

        # analyzes the terrain and retrieves the first possible full action path (first is chosen for simplicity)
        for terrain in terrains:
            analysis = terrain_analysis(terrains[terrain])
            paths[terrain] = (analysis['player 1 action paths'][0], analysis['player 2 action paths'][0])

        # calculates and stores the entropies of each full action path
        for path in paths:
            entropies[path] = calculate_entropy_of_path(paths[path][0], ENTROPY_RHO) + \
                              calculate_entropy_of_path(paths[path][1], ENTROPY_RHO)

        print(entropies)


class TestMotionExtractorRealMDP(unittest.TestCase):

    def mtx_test_helper(self, terrain_mtx, verbose=False):

        mdp = OvercookedGridworld.from_grid(terrain_mtx)
        env = OvercookedEnv.from_mdp(mdp)

        res = terrain_analysis(terrain_mtx)
        pairs_of_action_paths_by_total_length = res['pairs of action paths by total length']

        for path_length in pairs_of_action_paths_by_total_length:
            for pair_of_action_paths_by_total_length in pairs_of_action_paths_by_total_length[path_length]:
                action_path_0 = pair_of_action_paths_by_total_length[0]
                action_path_1 = pair_of_action_paths_by_total_length[1]
                joint_action_path = []
                for action_0, action_1 in zip(action_path_0, action_path_1):
                    joint_action_path.append((action_0, action_1))
                if verbose:
                    print("joint_action_path", joint_action_path)
                env.reset()
                sparse_rew = 0
                for joint_action in joint_action_path:
                    env.display_states(env.state)
                    _, sparse_rew, _, _ = env.step(joint_action)
                    if verbose:
                        print("joint action:", joint_action, "sparse reward: ", sparse_rew)
                if verbose:
                    print("final state")
                    env.display_states(env.state)
                self.assertTrue(sparse_rew == 20, "the action paths did not successfully result in soups in the end")

    def test_divided(self):
        # This should aready pass
        divided_mtx = [
            ['X', 'P', 'X', 'X', 'X', 'X'],
            ['O', ' ', 'X', ' ', ' ', 'O'],
            ['X', '1', 'X', '2', ' ', 'X'],
            ['X', 'D', 'X', 'S', 'X', 'X'],
        ]
        self.mtx_test_helper(divided_mtx, verbose=True)


    def test_counter_pre_movement(self):
        divided_counter_pre_mtx = [
            ['X', 'P', 'X', 'X', 'X', 'X'],
            ['X', ' ', 'X', ' ', ' ', 'O'],
            ['X', ' ', 'X', 'X', ' ', 'X'],
            ['X', '1', 'X', 'X', '2', 'X'],
            ['X', 'D', 'X', 'X', 'S', 'X'],
        ]
        self.mtx_test_helper(divided_counter_pre_mtx, verbose=True)

    def test_connected(self):
        connected_mtx = [
            ['X', 'P', 'X', 'X', 'X', 'X'],
            ['O', ' ', ' ', ' ', ' ', 'O'],
            ['X', '1', ' ', '2', ' ', 'X'],
            ['X', 'D', 'X', 'S', 'X', 'X'],
        ]
        self.mtx_test_helper(connected_mtx, verbose=True)


    def test_small_keyhole(self):
        small_keyhole_mtx = [
            ['X', 'P', 'X', 'X', 'X', 'X'],
            ['O', ' ', ' ', ' ', ' ', 'O'],
            ['X', ' ', 'X', ' ', ' ', 'X'],
            ['X', '1', 'X', '2', ' ', 'X'],
            ['X', 'D', 'X', 'S', 'X', 'X'],
        ]
        self.mtx_test_helper(small_keyhole_mtx, verbose=True)
        """

class TestMLASearchNodeWaypointTracking(unittest.TestCase):
    def setUp(self):
        self.divided_mtx = [
            ['X', 'P', 'X', 'X', 'X', 'X'],
            ['O', ' ', 'X', ' ', ' ', 'O'],
            ['X', '1', 'X', '2', ' ', 'X'],
            ['X', 'D', 'X', 'S', 'X', 'X'],
        ]

        self.connected_mtx = [
            ['X', 'P', 'X', 'X', 'X', 'X'],
            ['O', ' ', ' ', ' ', ' ', 'O'],
            ['X', '1', ' ', '2', ' ', 'X'],
            ['X', 'D', 'X', 'S', 'X', 'X'],
        ]

        self.small_keyhole_mtx = [
            ['X', 'P', 'X', 'X', 'X', 'X'],
            ['O', ' ', ' ', ' ', ' ', 'O'],
            ['X', ' ', 'X', ' ', ' ', 'X'],
            ['X', '1', 'X', '2', ' ', 'X'],
            ['X', 'D', 'X', 'S', 'X', 'X'],
        ]

    def test_divided_waypoints(self):
        print("testing divided waypoints------------")
        analysis = terrain_analysis(self.divided_mtx)
        waypoints = analysis['waypoints']
        path1 = analysis['player 1 action paths']
        path2 = analysis['player 2 action paths']
        for waypoint in waypoints:
            print("possible waypoints\n", waypoint)
        print()

    def test_connected_waypoints(self):
        print("testing connected waypoints------------")
        analysis = terrain_analysis(self.connected_mtx)
        waypoints = analysis['waypoints']
        path1 = analysis['player 1 action paths']
        path2 = analysis['player 2 action paths']
        for waypoint in waypoints:
            print("possible waypoints\n", waypoint)
        print()

    def test_keyhole_waypoints(self):
        print("testing keyhole waypoints------------")
        analysis = terrain_analysis(self.small_keyhole_mtx)
        waypoints = analysis['waypoints']
        path1 = analysis['player 1 action paths']
        path2 = analysis['player 2 action paths']
        for waypoint in waypoints:
            print("possible waypoints\n", waypoint)
        print()


if __name__ == '__main__':
    unittest.main()

