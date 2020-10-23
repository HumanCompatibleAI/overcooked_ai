import unittest
import numpy as np
from overcooked_ai_py.mdp.layout_evaluator import terrain_analysis, path_to_actions, \
    UNDEFIND_ACTION, remove_extra_action, add_action_from_location, calculate_entropy_of_path
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld


# class TestSimpleEvaluation(unittest.TestCase):
#
#     def setUp(self):
#         self.cramped_room_terrain_mtx = OvercookedGridworld.from_layout_name("cramped_room").terrain_mtx
#         print(self.cramped_room_terrain_mtx)
#
#     def test_0(self):
#         terrain_analysis(self.cramped_room_terrain_mtx, False)

class TestHandoverEvaluation(unittest.TestCase):

    def setUp(self):
        self.divided_mtx = [
            ['X', 'P', 'X', 'X', 'X'],
            ['O', '2', 'X', ' ', 'O'],
            ['X', ' ', 'X', '1', 'X'],
            ['X', 'D', 'X', 'S', 'X'],

        ]
        print(self.divided_mtx)

    def test_0(self):
        terrain_analysis(self.divided_mtx, False)


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
        ro = 5

        path = [(0, 1), (0, 1), (1, 0), 'interact']
        path_undefined_at_end = [(0, 1), (0, 1), (1, 0), 'interact', 'UND_A', 'UND_A']
        path_undefined_at_start = ['UND_A', 'UND_A', (0, 1), (0, 1), (1, 0), 'interact']

        entropy = -np.log(2) - np.log(1) - np.log(1) + 3 * np.log(ro)

        self.assertAlmostEqual(calculate_entropy_of_path(path, ro), entropy)
        self.assertAlmostEqual(calculate_entropy_of_path(path_undefined_at_end, ro), entropy)
        self.assertAlmostEqual(calculate_entropy_of_path(path_undefined_at_start, ro), entropy)

    def test_entropy_comparison_onion_dropoff(self):
        # This tests how entropy compares for the onion pickup task for three different layouts

        basic = [(-1, 0), (-1, 0), (-1, 0), (-1, 0), (-1, 0), (-1, 0), (-1, 0), \
                 (0, -1), (0, -1), (0, -1), (0, -1), (0, -1), (0, -1), 'interact']
        counter = [(0, -1), (0, -1), (0, -1), (0, -1), (0, -1), (0, -1), (-1, 0),\
                   (-1, 0), (-1, 0), (-1, 0), (-1, 0), (-1, 0), (-1, 0), (0, -1), 'interact']
        maze = [(0, 1), (-1, 0), (-1, 0), (0, -1), (0, -1), (0, -1), (0, -1), (0, -1), \
                (0, -1), (0, -1), (-1, 0), (-1, 0), (-1, 0), (0, 1), (0, 1), (0, 1), \
                (0, 1), (0, 1), (0, 1), (0, 1), (-1, 0), (-1, 0), (0, -1), (0, -1), \
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

        # assembles the three paths into a list and sets our parameter ro
        paths_to_compare = {'basic': basic, 'counter': counter, 'maze': maze}
        ro = 5

        # calculate and save entropies of each path
        entropies = {}
        for path in paths_to_compare:
            entropies[path] = calculate_entropy_of_path(paths_to_compare[path], ro)

        print(entropies)

    def test_entropy_full_path(self):
        # this tests the entropy comparison for the first foundn full path for both agents
    
        # assembles terrains into a list to reduce repetitive code
        terrains = {'basic': self.divided_mtx_basic, 'counter': self.divided_mtx_center_counter, 'maze': self.divided_mtx_keyhole_maze}
        paths = {}
        entropies = {}
        ro = 5

        # analyzes the terrain and retrieves the first possible full action path (first is chosen for simplicity)
        for terrain in terrains:
            analysis = terrain_analysis(terrains[terrain])
            paths[terrain] = (analysis['player 1 action paths'][0], analysis['player 2 action paths'][0])

        # calculates and stores the entropies of each full action path
        for path in paths:
            entropies[path] = calculate_entropy_of_path(paths[path][0], ro) + \
                              calculate_entropy_of_path(paths[path][1], ro)

        print(entropies)




if __name__ == '__main__':
    unittest.main()

