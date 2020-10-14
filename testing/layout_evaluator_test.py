import unittest
from overcooked_ai_py.mdp.layout_evaluator import terrain_analysis
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


if __name__ == '__main__':
    unittest.main()

