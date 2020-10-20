import unittest
from overcooked_ai_py.mdp.layout_evaluator import terrain_analysis, path_to_actions, \
    UNDEFIND_ACTION, remove_extra_action, add_action_from_location
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

if __name__ == '__main__':
    unittest.main()

