import unittest

class TestHanabi(unittest.TestCase):
    def setUp(self):
        self.mdp1 = OvercookedGridworld.from_file('layouts/mdp_test.layout', START_ORDER_LIST, EXPLOSION_TIME)

    def test_constructor_invalid_inputs(self):
        # Height and width must be at least 3.
        with self.assertRaises(AssertionError):
            mdp = OvercookedGridworld.from_grid(['X', 'X', 'X'], START_ORDER_LIST, EXPLOSION_TIME)
        with self.assertRaises(AssertionError):
            mdp = OvercookedGridworld.from_grid([['X', 'X', 'X']], START_ORDER_LIST, EXPLOSION_TIME)
        with self.assertRaises(AssertionError):
            # Borders must be present.
            mdp = OvercookedGridworld.from_grid(['XOSX',
                                                 'P  D',
                                                 ' 21 '],
                                                 START_ORDER_LIST, 
                                                 EXPLOSION_TIME)

        with self.assertRaises(AssertionError):
            # The grid can't be ragged.
            mdp = OvercookedGridworld.from_grid(['XXPXX',
                                                 'O  2XX',
                                                 'X1 3 X',
                                                 'XDXSXX'],
                                                 START_ORDER_LIST, 
                                                 EXPLOSION_TIME)

        with self.assertRaises(AssertionError):
            # There can't be more than two agents
            mdp = OvercookedGridworld.from_grid(['XXPXX',
                                                 'O  2O',
                                                 'X1 3X',
                                                 'XDXSX'],
                                                 START_ORDER_LIST, 
                                                 EXPLOSION_TIME)

        with self.assertRaises(AssertionError):
            # There can't be fewer than two agents.
            mdp = OvercookedGridworld.from_grid(['XXPXX',
                                                 'O   O',
                                                 'X1  X',
                                                 'XDXSX'],
                                                 START_ORDER_LIST, 
                                                 EXPLOSION_TIME)

        with self.assertRaises(AssertionError):
            # The agents must be numbered 1 and 2.
            mdp = OvercookedGridworld.from_grid(['XXPXX',
                                                 'O  3O',
                                                 'X1  X',
                                                 'XDXSX'],
                                                 START_ORDER_LIST, 
                                                 EXPLOSION_TIME)

        with self.assertRaises(AssertionError):
            # The agents must be numbered 1 and 2.
            mdp = OvercookedGridworld.from_grid(['XXPXX',
                                                 'O  1O',
                                                 'X1  X',
                                                 'XDXSX'],
                                                 START_ORDER_LIST, 
                                                 EXPLOSION_TIME)

        with self.assertRaises(AssertionError):
            # B is not a valid element.
            mdp = OvercookedGridworld.from_grid(['XBPXX',
                                                 'O  2O',
                                                 'X1  X',
                                                 'XDXSX'],
                                                 START_ORDER_LIST, 
                                                 EXPLOSION_TIME)

    def test_start_positions(self):
        expected_state = OvercookedState(
            [PlayerState((1, 2), Direction.NORTH), PlayerState((3, 1), Direction.NORTH)], {}, order_list=['any'])
        self.assertEqual(self.mdp1.get_start_state(), expected_state)