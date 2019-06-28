import unittest

from overcooked_gridworld.mdp.actions import Action, Direction
from overcooked_gridworld.mdp.overcooked_mdp import PlayerState, OvercookedGridworld, OvercookedState, ObjectState
from overcooked_gridworld.mdp.overcooked_env import OvercookedEnv

START_ORDER_LIST = ["any"]

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
    def setUp(self):
        self.base_mdp = OvercookedGridworld.from_layout_name(
            "mdp_test", 
            {'cook_time': 5, 'start_order_list': ['onion', 'any']}
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
            # There can't be more than two agents
            mdp = OvercookedGridworld.from_grid(['XXPXX',
                                                 'O  2O',
                                                 'X1 3X',
                                                 'XDXSX'])

        with self.assertRaises(AssertionError):
            # There can't be fewer than two agents.
            mdp = OvercookedGridworld.from_grid(['XXPXX',
                                                 'O   O',
                                                 'X1  X',
                                                 'XDXSX'])

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
        actual_start_state = self.base_mdp.get_start_state()
        self.assertEqual(actual_start_state, expected_start_state, '\n' + str(actual_start_state) + '\n' + str(expected_start_state))

    def test_file_constructor(self):
        mdp = OvercookedGridworld.from_layout_name('corridor')
        expected_start_state = OvercookedState(
            [PlayerState((3, 1), Direction.NORTH), PlayerState((10, 1), Direction.NORTH)], {}, order_list=None)
        actual_start_state = mdp.get_start_state()
        self.assertEqual(actual_start_state, expected_start_state, '\n' + str(actual_start_state) + '\n' + str(expected_start_state))

    def test_actions(self):
        bad_state = OvercookedState(
            [PlayerState((0, 0), Direction.NORTH), PlayerState((3, 1), Direction.NORTH)], {}, order_list=['any'])
        with self.assertRaises(AssertionError):
            self.base_mdp.get_actions(bad_state)

        self.assertEqual(self.base_mdp.get_actions(self.base_mdp.get_start_state()),
                         [Action.ALL_ACTIONS, Action.ALL_ACTIONS])

    def test_transitions_and_environment(self):
        n, s = Direction.NORTH, Direction.SOUTH
        e, w = Direction.EAST, Direction.WEST
        stay, interact = Action.STAY, Action.INTERACT
        P, Obj = PlayerState, ObjectState

        bad_state = OvercookedState(
            [P((0, 0), s), P((3, 1), s)], {}, order_list=[])

        with self.assertRaises(AssertionError):
            self.base_mdp.get_transition_states_and_probs(bad_state, stay)

        env = OvercookedEnv(self.base_mdp)
        env.state.order_list = ['onion', 'any']

        def check_transition(action, expected_state, expected_reward=0):
            state = env.state
            pred_state, sparse_reward, dense_reward = self.base_mdp.get_transition_states_and_probs(state, action)
            self.assertEqual(pred_state, expected_state, '\n' + str(pred_state) + '\n' + str(expected_state))
            new_state, sparse_reward, _, _ = env.step(action)
            self.assertEqual(new_state, expected_state)
            self.assertEqual(sparse_reward, expected_reward)

        check_transition([n, e], OvercookedState(
            [P((1, 1), n),
             P((3, 1), e)],
            {}, order_list=['onion', 'any']))

        check_transition([w, interact], OvercookedState(
            [P((1, 1), w),
             P((3, 1), e, Obj('onion', (3, 1)))],
            {}, order_list=['onion', 'any']))

        check_transition([interact, w],OvercookedState(
            [P((1, 1), w, Obj('onion', (1, 1))),
             P((2, 1), w, Obj('onion', (2, 1)))],
            {}, order_list=['onion', 'any']))

        check_transition([e, n], OvercookedState(
            [P((1, 1), e, Obj('onion', (1, 1))),
             P((2, 1), n, Obj('onion', (2, 1)))],
            {}, order_list=['onion', 'any']))

        check_transition([stay, interact], OvercookedState(
            [P((1, 1), e, Obj('onion', (1, 1))),
             P((2, 1), n)],
            {(2, 0): Obj('soup', (2, 0), ('onion', 1, 0))},
            order_list=['onion', 'any']))

        check_transition([e, e], OvercookedState(
            [P((2, 1), e, Obj('onion', (2, 1))),
             P((3, 1), e)],
            {(2, 0): Obj('soup', (2, 0), ('onion', 1, 0))},
            order_list=['onion', 'any']))

        check_transition([n, interact], OvercookedState(
            [P((2, 1), n, Obj('onion', (2, 1))),
             P((3, 1), e, Obj('onion', (3, 1)))],
            {(2, 0): Obj('soup', (2, 0), ('onion', 1, 0))},
            order_list=['onion', 'any']))

        check_transition([interact, w], OvercookedState(
            [P((2, 1), n),
             P((3, 1), w, Obj('onion', (3, 1)))],
            {(2, 0): Obj('soup', (2, 0), ('onion', 2, 0))},
            order_list=['onion', 'any']))

        check_transition([w, w], OvercookedState(
            [P((1, 1), w),
             P((2, 1), w, Obj('onion', (2, 1)))],
            {(2, 0): Obj('soup', (2, 0), ('onion', 2, 0))},
            order_list=['onion', 'any']))

        check_transition([s, n], OvercookedState(
            [P((1, 2), s),
             P((2, 1), n, Obj('onion', (2, 1)))],
            {(2, 0): Obj('soup', (2, 0), ('onion', 2, 0))},
            order_list=['onion', 'any']))

        check_transition([interact, interact], OvercookedState(
            [P((1, 2), s, Obj('dish', (1, 2))),
             P((2, 1), n)],
            {(2, 0): Obj('soup', (2, 0), ('onion', 3, 1))},
            order_list=['onion', 'any']))

        check_transition([e, s], OvercookedState(
            [P((1, 2), e, Obj('dish', (1, 2))),
             P((2, 1), s)],
            {(2, 0): Obj('soup', (2, 0), ('onion', 3, 2))},
            order_list=['onion', 'any']))

        check_transition([e, interact], OvercookedState(
            [P((2, 2), e, Obj('dish', (2, 2))),
             P((2, 1), s)],
            {(2, 0): Obj('soup', (2, 0), ('onion', 3, 3))},
            order_list=['onion', 'any']))

        check_transition([n, e], OvercookedState(
            [P((2, 1), n, Obj('dish', (2, 1))),
             P((3, 1), e)],
            {(2, 0): Obj('soup', (2, 0), ('onion', 3, 4))},
            order_list=['onion', 'any']))

        check_transition([interact, interact], OvercookedState(
            [P((2, 1), n, Obj('dish', (2, 1))),
             P((3, 1), e, Obj('onion', (3, 1)))],
            {(2, 0): Obj('soup', (2, 0), ('onion', 3, 5))},
            order_list=['onion', 'any']))
        
        check_transition([stay, stay], OvercookedState(
            [P((2, 1), n, Obj('dish', (2, 1))),
             P((3, 1), e, Obj('onion', (3, 1)))],
            {(2, 0): Obj('soup', (2, 0), ('onion', 3, 5))},
            order_list=['onion', 'any']))

        check_transition([interact, interact], OvercookedState(
            [P((2, 1), n, Obj('soup', (2, 1), ('onion', 3, 5))),
             P((3, 1), e, Obj('onion', (3, 1)))],
            {}, order_list=['onion', 'any']))

        check_transition([e, w], OvercookedState(
            [P((2, 1), e, Obj('soup', (2, 1), ('onion', 3, 5))),
             P((3, 1), w, Obj('onion', (3, 1)))],
            {}, order_list=['onion', 'any']))

        check_transition([e, s], OvercookedState(
            [P((3, 1), e, Obj('soup', (3, 1), ('onion', 3, 5))),
             P((3, 2), s, Obj('onion', (3, 2)))],
            {}, order_list=['onion', 'any']))

        check_transition([s, interact], OvercookedState(
            [P((3, 1), s, Obj('soup', (3, 1), ('onion', 3, 5))),
             P((3, 2), s, Obj('onion', (3, 2)))],
            {}, order_list=['onion', 'any']))

        check_transition([s, w], OvercookedState(
            [P((3, 2), s, Obj('soup', (3, 2), ('onion', 3, 5))),
             P((2, 2), w, Obj('onion', (2, 2)))],
            {}, order_list=['onion', 'any']))

        check_transition([interact, n], OvercookedState(
            [P((3, 2), s),
             P((2, 1), n, Obj('onion', (2, 1)))],
            {}, order_list=['any']), expected_reward=self.base_mdp.delivery_reward)

        check_transition([e, interact], OvercookedState(
            [P((3, 2), e),
             P((2, 1), n)],
            {(2, 0): Obj('soup', (2, 0), ('onion', 1, 0))}, 
            order_list=['any']))

        check_transition([interact, s], OvercookedState(
            [P((3, 2), e, Obj('tomato', (3, 2))),
             P((2, 2), s)],
            {(2, 0): Obj('soup', (2, 0), ('onion', 1, 0))},
            order_list=['any']))

        check_transition([w, w], OvercookedState(
            [P((2, 2), w, Obj('tomato', (2, 2))),
             P((1, 2), w)],
            {(2, 0): Obj('soup', (2, 0), ('onion', 1, 0))},
            order_list=['any']))
        
        check_transition([n, interact], OvercookedState(
            [P((2, 1), n, Obj('tomato', (2, 1))),
             P((1, 2), w, Obj('tomato', (1, 2)))],
            {(2, 0): Obj('soup', (2, 0), ('onion', 1, 0))},
            order_list=['any']))
        
        check_transition([interact, interact], OvercookedState(
            [P((2, 1), n, Obj('tomato', (2, 1))),
             P((1, 2), w, Obj('tomato', (1, 2)))],
            {(2, 0): Obj('soup', (2, 0), ('onion', 1, 0))},
            order_list=['any']))

        check_transition([s, interact], OvercookedState(
            [P((2, 2), s, Obj('tomato', (2, 2))),
             P((1, 2), w, Obj('tomato', (1, 2)))],
            {(2, 0): Obj('soup', (2, 0), ('onion', 1, 0))},
            order_list=['any']))
        
        check_transition([interact, interact], OvercookedState(
            [P((2, 2), s),
             P((1, 2), w, Obj('tomato', (1, 2)))],
            {(2, 0): Obj('soup', (2, 0), ('onion', 1, 0)),
             (2, 3): Obj('soup', (2, 3), ('tomato', 1, 0))}, 
            order_list=['any']))
                
class TestEnvironment(unittest.TestCase):
    # TODO: Here
    pass

if __name__ == '__main__':
    unittest.main()
