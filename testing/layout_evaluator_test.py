import unittest
from overcooked_ai_py.mdp.layout_evaluator import terrain_analysis, path_to_actions, UNDEFIND_ACTION
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
            ['X', 'P', 'X', 'X', 'X'],
            ['O', '2', 'X', ' ', 'O'],
            ['X', ' ', 'X', '1', 'X'],
            ['X', 'D', 'X', 'S', 'X'],
        ]

    def test_onion_pickup_2(self):
        # basic motion test 1
        path_0 = ['UND', 'UND']
        path_1 = [(1, 1), (1, 0)]
        self.assertEqual(path_to_actions(path_0, path_1, self.divided_mtx), (
            [UNDEFIND_ACTION, UNDEFIND_ACTION],
            [(-1, 0), 'interact']
        ))

    def test_onion_pickup_1(self):
        # basic motion test 2
        path_0 = [(2, 3), (1, 3), (1, 4)]
        path_1 = ['UND', 'UND', 'UND']
        self.assertEqual(path_to_actions(path_0, path_1, self.divided_mtx), (
            [(0, -1), (1, 0), 'interact'],
            [UNDEFIND_ACTION, UNDEFIND_ACTION, UNDEFIND_ACTION]
        ))


    def test_onion_drop_1(self):
        # this tests counter handover
        path_0 = [(1, 3), (1, 2), 'UND', 'UND']
        path_1 = ['UND', (1, 2), (1, 1), (0, 1)]
        self.assertEqual(path_to_actions(path_0, path_1, self.divided_mtx), (
            [(-1, 0), 'interact', UNDEFIND_ACTION, UNDEFIND_ACTION, UNDEFIND_ACTION],
            [UNDEFIND_ACTION, UNDEFIND_ACTION, 'interact', (0, -1), 'interact']
        ))

    def test_dish_pickup_1(self):
        # this test abbreviation of additional movement because of natural agent facing
        path_0 = ['UND', 'UND', 'UND']
        path_1 = [(1, 1), (2, 1), (3, 1)]
        self.assertEqual(path_to_actions(path_0, path_1, self.divided_mtx), (
            [UNDEFIND_ACTION, UNDEFIND_ACTION],
            [(0, 1), 'interact']
        ))

if __name__ == '__main__':
    unittest.main()

