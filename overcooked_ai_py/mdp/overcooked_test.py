import unittest
import numpy as np
from math import factorial
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import PlayerState, OvercookedGridworld, OvercookedState, ObjectState, SoupState, Recipe
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, DEFAULT_ENV_PARAMS
from overcooked_ai_py.mdp.layout_generator import LayoutGenerator
from overcooked_ai_py.agents.agent import AgentGroup, AgentPair, GreedyHumanModel, FixedPlanAgent
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS, MotionPlanner
from overcooked_ai_py.utils import save_pickle, load_pickle, iterate_over_json_files_in_dir


START_ORDER_LIST = ["any"]
n, s = Direction.NORTH, Direction.SOUTH
e, w = Direction.EAST, Direction.WEST
stay, interact = Action.STAY, Action.INTERACT
P, Obj = PlayerState, ObjectState

def comb(n, k):
    return factorial(n) / (factorial(n - k) * factorial(k))


class TestRecipe(unittest.TestCase):

    def setUp(self):
        self.r1 = Recipe([Recipe.ONION, Recipe.ONION, Recipe.ONION])
        self.r2 = Recipe([Recipe.ONION, Recipe.ONION, Recipe.ONION])
        self.r3 = Recipe([Recipe.ONION, Recipe.TOMATO])
        self.r4 = Recipe([Recipe.ONION, Recipe.TOMATO])
        self.r5 = Recipe([Recipe.TOMATO, Recipe.ONION])
        self.r6 = Recipe([Recipe.ONION, Recipe.ONION])

        self.recipes = [self.r1, self.r2, self.r3, self.r4, self.r5, self.r6]

    def tearDown(self):
        Recipe.configure({})

    def test_eq(self):

        self.assertEqual(self.r1, self.r2, "Failed basic equality check")
        self.assertNotEqual(self.r1, self.r3, "Failed Basic inequality check")
        self.assertNotEqual(self.r1, self.r6, "Failed inequality check with all one ingredient")
        self.assertEqual(self.r3, self.r4, "Failed basic equality check")
        self.assertEqual(self.r4, self.r5, "Failed ordered equality check")

    def test_caching(self):

        self.assertIs(self.r1, self.r2)
        self.assertIs(self.r3, self.r4)
        self.assertIs(self.r4, self.r5)
        self.assertFalse(self.r6 is self.r1, "different recipes cached to same value")

    def test_value(self):
        # TODO
        for recipe in self.recipes:
            self.assertEqual(recipe.value, 20)

    def test_time(self):
        # TODO
        for recipe in self.recipes:
            self.assertEqual(recipe.time, 20)

    def test_all_recipes(self):
        for recipe in self.recipes:
            self.assertTrue(recipe in Recipe.ALL_RECIPES)

        self.assertEqual(len(Recipe.ALL_RECIPES), self._expected_num_recipes(len(Recipe.ALL_INGREDIENTS), Recipe.MAX_NUM_INGREDIENTS))

        Recipe.configure({ "max_num_ingredients" : 4 })

        self.assertEqual(len(Recipe.ALL_RECIPES), self._expected_num_recipes(len(Recipe.ALL_INGREDIENTS), 4))

    def test_invalid_input(self):

        self.assertRaises(ValueError, Recipe, [Recipe.ONION, Recipe.TOMATO, "carrot"])
        self.assertRaises(ValueError, Recipe, [Recipe.ONION]*4)
        self.assertRaises(ValueError, Recipe, [])
        self.assertRaises(ValueError, Recipe, "invalid argument")


    def _expected_num_recipes(self, num_ingredients, max_len):
        return comb(num_ingredients + max_len, num_ingredients) - 1

class TestSoupState(unittest.TestCase):
    
    def setUp(self):
        self.s1 = SoupState.get_soup((0, 0), num_onions=0, num_tomatoes=0)
        self.s2 = SoupState.get_soup((0, 1), num_onions=2, num_tomatoes=1)
        self.s3 = SoupState.get_soup((1, 1), num_onions=1, num_tomatoes=0, cooking_tick=1)
        self.s4 = SoupState.get_soup((1, 0), num_onions=0, num_tomatoes=2, finished=True)

    def test_position(self):
        new_pos = (2, 0)
        self.s4.position = new_pos

        for ingredient in self.s4._ingredients:
            self.assertEqual(new_pos, ingredient.position)
        self.assertEqual(new_pos, self.s4.position)

    def test_is_cooking(self):
        self.assertFalse(self.s1.is_cooking)
        self.assertFalse(self.s2.is_cooking)
        self.assertTrue(self.s3.is_cooking)
        self.assertFalse(self.s4.is_cooking)

    def test_is_ready(self):
        self.assertFalse(self.s1.is_ready)
        self.assertFalse(self.s2.is_ready)
        self.assertFalse(self.s3.is_ready)
        self.assertTrue(self.s4.is_ready)

    def test_is_idle(self):
        self.assertTrue(self.s1.is_idle)
        self.assertTrue(self.s2.is_idle)
        self.assertFalse(self.s3.is_idle)
        self.assertFalse(self.s4.is_idle)

    def test_is_full(self):
        self.assertFalse(self.s1.is_full)
        self.assertTrue(self.s2.is_full)
        self.assertTrue(self.s3.is_full)
        self.assertTrue(self.s4.is_full)

    def test_cooking(self):
        self.s1.add_ingredient_from_str(Recipe.ONION)
        self.s1.add_ingredient_from_str(Recipe.TOMATO)
        
        self.assertTrue(self.s1.is_idle)
        self.assertFalse(self.s1.is_cooking)
        self.assertFalse(self.s1.is_full)
        
        self.s1.begin_cooking()

        self.assertFalse(self.s1.is_idle)
        self.assertTrue(self.s1.is_full)
        self.assertTrue(self.s1.is_cooking)

        for _ in range(self.s1.cook_time):
            self.s1.cook()

        self.assertFalse(self.s1.is_cooking)
        self.assertFalse(self.s1.is_idle)
        self.assertTrue(self.s1.is_full)
        self.assertTrue(self.s1.is_ready)

    def test_attributes(self):
        self.assertListEqual(self.s1.ingredients, [])
        self.assertListEqual(self.s2.ingredients, [Recipe.ONION, Recipe.ONION, Recipe.TOMATO])
        self.assertListEqual(self.s3.ingredients, [Recipe.ONION])
        self.assertListEqual(self.s4.ingredients, [Recipe.TOMATO, Recipe.TOMATO])

        try:
            self.s1.recipe
            self.fail("Expected ValueError to be raised")
        except ValueError as e: 
            pass
        except Exception as e:
            self.fail("Expected ValueError to be raised, {} raised instead".format(e))

        try:
            self.s2.recipe
            self.fail("Expected ValueError to be raised")
        except ValueError as e: 
            pass
        except Exception as e:
            self.fail("Expected ValueError to be raised, {} raised instead".format(e))
        self.assertEqual(self.s3.recipe, Recipe([Recipe.ONION]))
        self.assertEqual(self.s4.recipe, Recipe([Recipe.TOMATO, Recipe.TOMATO]))

    def test_invalid_ops(self):
        
        # Cannot cook an empty soup
        self.assertRaises(ValueError, self.s1.begin_cooking)

        # Must call 'begin_cooking' before cooking a soup
        self.assertRaises(ValueError, self.s2.cook)

        # Cannot cook a done soup
        self.assertRaises(ValueError, self.s4.cook)

        # Cannot begin cooking a soup that is already cooking
        self.assertRaises(ValueError, self.s3.begin_cooking)

        # Cannot begin cooking a soup that is already done
        self.assertRaises(ValueError, self.s4.begin_cooking)

        # Cannot add ingredients to a soup that is cooking
        self.assertRaises(ValueError, self.s3.add_ingredient_from_str, Recipe.ONION)

        # Cannot add ingredients to a soup that is ready
        self.assertRaises(ValueError, self.s4.add_ingredient_from_str, Recipe.ONION)

        # Cannot remove an ingredient from a soup that is ready
        self.assertRaises(ValueError, self.s4.pop_ingredient)

        # Cannot remove an ingredient from a soup that is cooking
        self.assertRaises(ValueError, self.s3.pop_ingredient)

        # Cannot remove an ingredient from a soup that is empty
        self.assertRaises(ValueError, self.s1.pop_ingredient)



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
            [PlayerState((1, 2), Direction.NORTH), PlayerState((3, 1), Direction.NORTH)], {})
        actual_start_state = self.base_mdp.get_standard_start_state()
        self.assertEqual(actual_start_state, expected_start_state, '\n' + str(actual_start_state) + '\n' + str(expected_start_state))

    def test_file_constructor(self):
        mdp = OvercookedGridworld.from_layout_name('corridor')
        expected_start_state = OvercookedState(
            [PlayerState((3, 1), Direction.NORTH), PlayerState((10, 1), Direction.NORTH)], {})
        actual_start_state = mdp.get_standard_start_state()
        self.assertEqual(actual_start_state, expected_start_state, '\n' + str(actual_start_state) + '\n' + str(expected_start_state))

    def test_actions(self):
        bad_state = OvercookedState(
            [PlayerState((0, 0), Direction.NORTH), PlayerState((3, 1), Direction.NORTH)], {})
        with self.assertRaises(AssertionError):
            self.base_mdp.get_actions(bad_state)

        self.assertEqual(self.base_mdp.get_actions(self.base_mdp.get_standard_start_state()),
                         [Action.ALL_ACTIONS, Action.ALL_ACTIONS])

    def test_from_dict(self):
        state_dict = {"players": [{"position": [2, 1], "orientation": [0, -1], "held_object": None }, {"position": [1, 1], "orientation": [0, -1], "held_object": None }], "objects": [{"name": "onion", "position": [1, 0], "state": None }], "order_list": None }
        state = OvercookedState.from_dict(state_dict)
        print(state)


    def test_transitions_and_environment(self):
        bad_state = OvercookedState(
            [P((0, 0), s), P((3, 1), s)], {})

        with self.assertRaises(AssertionError):
            self.base_mdp.get_state_transition(bad_state, stay)

        env = OvercookedEnv.from_mdp(self.base_mdp)

        def check_transition(action, expected_state, expected_reward=0):
            state = env.state
            pred_state, _ = self.base_mdp.get_state_transition(state, action)
            self.assertEqual(pred_state, expected_state, '\n' + str(pred_state) + '\n' + str(expected_state))
            new_state, sparse_reward, _, _ = env.step(action)
            self.assertEqual(new_state, expected_state)
            self.assertEqual(sparse_reward, expected_reward)

        check_transition([n, e], OvercookedState(
            [P((1, 1), n),
             P((3, 1), e)],
            {}, timestep=1))

    def test_common_mdp_jsons(self):
        traj_test_json_paths = iterate_over_json_files_in_dir("../common_tests/trajectory_tests/")
        for test_json_path in traj_test_json_paths:
            test_trajectory = AgentEvaluator.load_traj_from_json(test_json_path)
            # try:
            #     AgentEvaluator.check_trajectories(test_trajectory, from_json=True)
            # except AssertionError as e:
            #     self.fail("File {} failed with error:\n{}".format(test_json_path, e))
            AgentEvaluator.check_trajectories(test_trajectory, from_json=True)

    def test_four_player_mdp(self):
        try:
            OvercookedGridworld.from_layout_name("multiplayer_schelling")
        except AssertionError as e:
            print("Loading > 2 player map failed with error:", e)

    def test_potential_function(self):
        self.base_mdp_2 = OvercookedGridworld.from_layout_name('mdp_test_tomato')
        mp = MotionPlanner(self.base_mdp_2)
        state = self.base_mdp_2.get_standard_start_state()
        val0 = self.base_mdp_2.potential_function(state, mp)
        self.assertEqual(val0, 0, "Start potential should be zero")

        # Pick up onion
        print("pick up onion")
        print(self.base_mdp_2.state_string(state))
        print("potential: ", self.base_mdp_2.potential_function(state, mp))
        actions = [Direction.EAST, Action.INTERACT]
        for action in actions:
            state, _ = self.base_mdp_2.get_state_transition(state, [Action.STAY, action])
            print(self.base_mdp_2.state_string(state))
            print("potential: ", self.base_mdp_2.potential_function(state, mp))
        val1 = self.base_mdp_2.potential_function(state, mp)
        

        # Pick up tomato
        print("pick up tomtato")
        actions = [Direction.WEST, Action.INTERACT]
        for action in actions:
            state, _ = self.base_mdp_2.get_state_transition(state, [action, Action.STAY])
            print(self.base_mdp_2.state_string(state))
            print("potential: ", self.base_mdp_2.potential_function(state, mp))
        val2 = self.base_mdp_2.potential_function(state, mp)

        self.assertLess(val0, val1, "Picking up onion should increase potential")
        self.assertLess(val1, val2, "Picking up tomato should increase potential")

        # Pot tomato
        print("pot tomato")
        actions = [Direction.EAST, Direction.NORTH, Action.INTERACT, Direction.WEST]
        for action in actions:
            state, _ = self.base_mdp_2.get_state_transition(state, [action, Action.STAY])
            print(self.base_mdp_2.state_string(state))
            print("potential: ", self.base_mdp_2.potential_function(state, mp))
        val3 = self.base_mdp_2.potential_function(state, mp)
        
        # Pot onion
        print("pot onion")
        actions = [Direction.WEST, Direction.NORTH, Action.INTERACT]
        for action in actions:
            state, _ = self.base_mdp_2.get_state_transition(state, [Action.STAY, action])
            print(self.base_mdp_2.state_string(state))
            print("potential: ", self.base_mdp_2.potential_function(state, mp))
        val4 = self.base_mdp_2.potential_function(state, mp)

        self.assertLess(val2, val3, "Potting tomato should increase potential")
        self.assertLess(val3, val4, "Potting onion should increase potential")

        ## Repeat on second pot ##

        # Pick up onion
        print("pick up onion")
        state, _ = self.base_mdp_2.get_state_transition(state, [Action.INTERACT, Action.STAY])
        val5 = self.base_mdp_2.potential_function(state, mp)
        print(self.base_mdp_2.state_string(state))
        print("potential: ", self.base_mdp_2.potential_function(state, mp))

        # Pick up tomato
        print("pick up tomato")
        actions = [Direction.SOUTH, Direction.EAST, Action.INTERACT]
        for action in actions:
            state, _ = self.base_mdp_2.get_state_transition(state, [Action.STAY, action])
            print(self.base_mdp_2.state_string(state))
            print("potential: ", self.base_mdp_2.potential_function(state, mp))
        val6 = self.base_mdp_2.potential_function(state, mp)

        self.assertLess(val4, val5, "Picking up onion should increase potential")
        self.assertLess(val5, val6, "Picking up tomato should increase potential")

        # Pot onion
        print("pot onion")
        actions = [Direction.SOUTH, Direction.EAST, Direction.SOUTH, Action.INTERACT, Direction.WEST]
        for action in actions:
            state, _ = self.base_mdp_2.get_state_transition(state, [action, Action.STAY])
            print(self.base_mdp_2.state_string(state))
            print("potential: ", self.base_mdp_2.potential_function(state, mp))
        val7 = self.base_mdp_2.potential_function(state, mp)

        # Pot tomato
        print("pot tomato")
        actions = [Direction.WEST, Direction.SOUTH, Action.INTERACT]
        for action in actions:
            state, _ = self.base_mdp_2.get_state_transition(state, [Action.STAY, action])
            print(self.base_mdp_2.state_string(state))
            print("potential: ", self.base_mdp_2.potential_function(state, mp))
        val8 = self.base_mdp_2.potential_function(state, mp)

        

        self.assertLess(val6, val7, "Potting onion should increase potential")
        self.assertLess(val7, val8, "Potting tomato should increase potential")

        ## Useless pickups ##
        
        # pickup tomato
        print("pickup tomato")
        actions = [Action.INTERACT, Direction.NORTH]
        for action in actions:
            state, _ = self.base_mdp_2.get_state_transition(state, [action, Action.STAY])
            print(self.base_mdp_2.state_string(state))
            print("potential: ", self.base_mdp_2.potential_function(state, mp))
        val9 = self.base_mdp_2.potential_function(state, mp)

        # pickup tomato
        print("pickup tomato")
        actions = [Direction.EAST, Action.INTERACT]
        for action in actions:
            state, _ = self.base_mdp_2.get_state_transition(state, [Action.STAY, action])
            print(self.base_mdp_2.state_string(state))
            print("potential: ", self.base_mdp_2.potential_function(state, mp))
        val10 = self.base_mdp_2.potential_function(state, mp)

        self.assertLessEqual(val9, val8, "Extraneous pickup should not increase potential")
        self.assertLessEqual(val10, val8, "Extraneous pickup should not increase potential")

        ## Catastrophic soup failure ##
        
        # pot tomato
        print("pot catastrophic tomato")
        actions = [Direction.WEST, Direction.SOUTH, Action.INTERACT]
        for action in actions:
            state, _ = self.base_mdp_2.get_state_transition(state, [Action.STAY, action])
            print(self.base_mdp_2.state_string(state))
            print("potential: ", self.base_mdp_2.potential_function(state, mp))
        val11 = self.base_mdp_2.potential_function(state, mp)

        self.assertLess(val11, val10, "Catastrophic potting should decrease potential")

        ## Bonus soup creation

        # pick up onion
        print("pick up onion")
        actions = [Direction.NORTH, Action.INTERACT, Direction.WEST, Action.INTERACT]
        for action in actions:
            state, _ = self.base_mdp_2.get_state_transition(state, [action, Action.STAY])
            print(self.base_mdp_2.state_string(state))
            print("potential: ", self.base_mdp_2.potential_function(state, mp))
        val12 = self.base_mdp_2.potential_function(state, mp)

        # pot onion
        print("pot onion")
        actions = [Direction.EAST, Direction.NORTH, Action.INTERACT]
        for action in actions:
            state, _ = self.base_mdp_2.get_state_transition(state, [action, Action.STAY])
            print(self.base_mdp_2.state_string(state))
            print("potential: ", self.base_mdp_2.potential_function(state, mp))
        val13 = self.base_mdp_2.potential_function(state, mp)

        # Cook soup
        print("cook soup")
        actions = [Action.INTERACT, Direction.WEST]
        for action in actions:
            state, _ = self.base_mdp_2.get_state_transition(state, [action, Action.STAY])
            print(self.base_mdp_2.state_string(state))
            print("potential: ", self.base_mdp_2.potential_function(state, mp))
        val14 = self.base_mdp_2.potential_function(state, mp)

        self.assertLess(val11, val12, "Useful onion pickup should increase potential")
        self.assertLess(val12, val13, "Potting useful onion should increase potential")
        self.assertLess(val13, val14, "Cooking optimal soup should increase potential")

        ## Soup pickup ##

        # Pick up dish
        print("pick up dish")
        actions = [Direction.WEST, Direction.SOUTH, Action.INTERACT]
        for action in actions:
            state, _ = self.base_mdp_2.get_state_transition(state, [Action.STAY, action])
            print(self.base_mdp_2.state_string(state))
            print("potential: ", self.base_mdp_2.potential_function(state, mp))
        val15 = self.base_mdp_2.potential_function(state, mp)

        # Move towards pot
        print("move towards pot")
        actions = [Direction.EAST, Direction.NORTH]
        for action in actions:
            state, _ = self.base_mdp_2.get_state_transition(state, [Action.STAY, action])
            print(self.base_mdp_2.state_string(state))
            print("potential: ", self.base_mdp_2.potential_function(state, mp))
        val16 = self.base_mdp_2.potential_function(state, mp)

        # Pickup soup
        print("pickup soup")
        state, _ = self.base_mdp_2.get_state_transition(state, [Action.STAY, Action.INTERACT])
        print(self.base_mdp_2.state_string(state))
        print("potential: ", self.base_mdp_2.potential_function(state, mp))
        val17 = self.base_mdp_2.potential_function(state, mp)

        self.assertLess(val14, val15, "Useful dish pickups should increase potential")
        self.assertLess(val15, val16, "Moving towards soup with dish should increase potential")
        self.assertLess(val16, val17, "Picking up soup should increase potential")

        ## Removing failed soup from pot

        # move towards failed soup
        print("move torwards failed soup")
        actions = [Direction.SOUTH, Direction.EAST, Direction.SOUTH]
        for action in actions:
            state, _ = self.base_mdp_2.get_state_transition(state, [action, Action.STAY])
            print(self.base_mdp_2.state_string(state))
            print("potential: ", self.base_mdp_2.potential_function(state, mp))
        val18 = self.base_mdp_2.potential_function(state, mp)

        # Cook failed soup
        actions = [Action.INTERACT]
        for action in actions:
            state, _ = self.base_mdp_2.get_state_transition(state, [action, Action.STAY])
            print(self.base_mdp_2.state_string(state))
            print("potential: ", self.base_mdp_2.potential_function(state, mp))
        val19 = self.base_mdp_2.potential_function(state, mp)

        # Pickup dish
        print("pickup dish")
        actions = [Direction.WEST, Direction.SOUTH, Action.INTERACT]
        for action in actions:
            state, _ = self.base_mdp_2.get_state_transition(state, [action, Action.STAY])
            print(self.base_mdp_2.state_string(state))
            print("potential: ", self.base_mdp_2.potential_function(state, mp))
        val20 = self.base_mdp_2.potential_function(state, mp)

        # Move towards soup
        print("move towards soup")
        actions = [Direction.EAST, Direction.SOUTH]
        for action in actions:
            state, _ = self.base_mdp_2.get_state_transition(state, [action, Action.STAY])
            print(self.base_mdp_2.state_string(state))
            print("potential: ", self.base_mdp_2.potential_function(state, mp))
        val21 = self.base_mdp_2.potential_function(state, mp)

        self.assertLess(val17, val18, "Moving towards failed soup should increase potential")
        self.assertLess(val18, val19, "Cooking failed soup should increase potential")
        self.assertLess(val19, val20, "Dish pickup for failed soup is still useful")
        self.assertLess(val20, val21, "Moving towars pertinant pot with dish should increase potential")

        ## Deliver failed soup ##

        # Pickup failed soup
        actions = [Action.INTERACT]
        for action in actions:
            state, _ = self.base_mdp_2.get_state_transition(state, [action, Action.STAY])
            print(self.base_mdp_2.state_string(state))
            print("potential: ", self.base_mdp_2.potential_function(state, mp))
        val22 = self.base_mdp_2.potential_function(state, mp)

        # Move towards serving area
        print("move towards servering area")
        actions = [Direction.EAST, Direction.SOUTH]
        for action in actions:
            state, _ = self.base_mdp_2.get_state_transition(state, [action, Action.STAY])
            print(self.base_mdp_2.state_string(state))
            print("potential: ", self.base_mdp_2.potential_function(state, mp))
        val23 = self.base_mdp_2.potential_function(state, mp)

        # Move away from serving area
        print("move away from serving area")
        state, _ = self.base_mdp_2.get_state_transition(state, [Direction.NORTH, Action.STAY])
        print(self.base_mdp_2.state_string(state))
        print("potential: ", self.base_mdp_2.potential_function(state, mp))
        val24 = self.base_mdp_2.potential_function(state, mp)

        self.assertLess(val21, val22, "Picking up failed soup should increase potential")
        self.assertAlmostEqual(val23, val22, delta=0.2, msg="Moving to serve failed soup doesn't change potential much")
        self.assertAlmostEqual(val23, val24, delta=0.2, msg="Moving away from serving area with failed soup doesn't change much")

        ## Deliver successful soup ##

        # Move towards serving area
        print("move towards serving area")
        actions = [Direction.SOUTH, Direction.EAST, Direction.SOUTH]
        for action in actions:
            state, _ = self.base_mdp_2.get_state_transition(state, [Action.STAY, action])
            print(self.base_mdp_2.state_string(state))
            print("potential: ", self.base_mdp_2.potential_function(state, mp))
        val25 = self.base_mdp_2.potential_function(state, mp)

        # Deliver soup
        print("deliver successful soup")
        state, rewards = self.base_mdp_2.get_state_transition(state, [Action.STAY, Action.INTERACT])
        print(self.base_mdp_2.state_string(state))
        print("potential: ", self.base_mdp_2.potential_function(state, mp))

        self.assertLess(val24, val25, "Moving towards serving area with valid soup increases potential")
        self.assertEqual(sum(rewards['sparse_reward_by_agent']), 50, "Soup was not properly devivered, probably an error with MDP logic")

        # Restore recipe configuration for other mdp
        OvercookedGridworld.from_layout_name(
            "mdp_test",
            **{'cook_time': 5, 'start_order_list': ['onion', 'any']}
        )





def random_joint_action():
    num_actions = len(Action.ALL_ACTIONS)
    a_idx0, a_idx1 = np.random.randint(low=0, high=num_actions, size=2)
    return (Action.INDEX_TO_ACTION[a_idx0], Action.INDEX_TO_ACTION[a_idx1])


class TestFeaturizations(unittest.TestCase):

    def setUp(self):
        self.base_mdp = OvercookedGridworld.from_layout_name("cramped_room")
        self.mlp = MediumLevelPlanner.from_pickle_or_compute(self.base_mdp, NO_COUNTERS_PARAMS, force_compute=True)
        self.env = OvercookedEnv.from_mdp(self.base_mdp, **DEFAULT_ENV_PARAMS)
        self.rnd_agent_pair = AgentPair(GreedyHumanModel(self.mlp), GreedyHumanModel(self.mlp))
        np.random.seed(0)

    def test_lossless_state_featurization_shape(self):
        s = self.base_mdp.get_standard_start_state()
        obs = self.base_mdp.lossless_state_encoding(s)[0]
        self.assertTrue(np.array_equal(obs.shape, self.base_mdp.lossless_state_encoding_shape), "{} vs {}".format(obs.shape, self.base_mdp.lossless_state_encoding_shape))

    def test_state_featurization_shape(self):
        s = self.base_mdp.get_standard_start_state()
        obs = self.base_mdp.featurize_state(s, self.mlp)[0]
        self.assertTrue(np.array_equal(obs.shape, self.base_mdp.featurize_state_shape), "{} vs {}".format(obs.shape, self.base_mdp.featurize_state_shape))

    def test_lossless_state_featurization(self):
        trajs = self.env.get_rollouts(self.rnd_agent_pair, num_games=5)
        featurized_observations = [[self.base_mdp.lossless_state_encoding(state) for state in ep_states] for ep_states in trajs["ep_states"]]
        # NOTE: If the featurizations are updated intentionally, you can overwrite the expected
        # featurizations by uncommenting the following line:
        # save_pickle(featurized_observations, "data/testing/lossless_state_featurization")
        expected_featurization = load_pickle("./data/testing/lossless_state_featurization")
        self.assertTrue(np.array_equal(expected_featurization, featurized_observations))

    def test_state_featurization(self):
        trajs = self.env.get_rollouts(self.rnd_agent_pair, num_games=5)
        featurized_observations = [[self.base_mdp.featurize_state(state, self.mlp) for state in ep_states] for ep_states in trajs["ep_states"]]
        # NOTE: If the featurizations are updated intentionally, you can overwrite the expected
        # featurizations by uncommenting the following line:
        # save_pickle(featurized_observations, "data/testing/state_featurization")
        expected_featurization = load_pickle("./data/testing/state_featurization")
        self.assertTrue(np.array_equal(expected_featurization, featurized_observations))


class TestOvercookedEnvironment(unittest.TestCase):

    def setUp(self):
        self.base_mdp = OvercookedGridworld.from_layout_name("cramped_room")
        self.env = OvercookedEnv.from_mdp(self.base_mdp, **DEFAULT_ENV_PARAMS)
        self.rnd_agent_pair = AgentPair(FixedPlanAgent([stay, w, w]), FixedPlanAgent([stay, e, e]))
        np.random.seed(0)

    def test_constructor(self):
        try:
            OvercookedEnv.from_mdp(self.base_mdp, horizon=10)
        except Exception as e:
            self.fail("Failed to instantiate OvercookedEnv:\n{}".format(e))

        with self.assertRaises(TypeError):
            OvercookedEnv.from_mdp(self.base_mdp, **{"invalid_env_param": None})

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
            print(e.with_traceback())
            self.fail("Failed to get rollouts from environment:\n{}".format(e))

    def test_one_player_env(self):
        mdp = OvercookedGridworld.from_layout_name("cramped_room_single")
        env = OvercookedEnv.from_mdp(mdp, horizon=12)
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
        env = OvercookedEnv.from_mdp(mdp, horizon=16)
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
        mdp0 = OvercookedGridworld.from_layout_name("cramped_room")
        mdp1 = OvercookedGridworld.from_layout_name("counter_circuit")
        mdp_fn = lambda: np.random.choice([mdp0, mdp1])
        
        # Default env
        env = OvercookedEnv(mdp_fn, horizon=100)
        env.get_rollouts(self.rnd_agent_pair, 5)

    def test_starting_position_randomization(self):
        self.base_mdp = OvercookedGridworld.from_layout_name("cramped_room")
        start_state_fn = self.base_mdp.get_random_start_state_fn(random_start_pos=True, rnd_obj_prob_thresh=0.0)
        env = OvercookedEnv.from_mdp(self.base_mdp, start_state_fn)
        start_state = env.state.players_pos_and_or
        for _ in range(3):
            env.reset()
            print(env)
            curr_terrain = env.state.players_pos_and_or
            self.assertFalse(np.array_equal(start_state, curr_terrain))

    def test_starting_obj_randomization(self):
        self.base_mdp = OvercookedGridworld.from_layout_name("cramped_room")
        start_state_fn = self.base_mdp.get_random_start_state_fn(random_start_pos=False, rnd_obj_prob_thresh=0.8)
        env = OvercookedEnv.from_mdp(self.base_mdp, start_state_fn)
        start_state = env.state.all_objects_list
        for _ in range(3):
            env.reset()
            # print(env)
            curr_terrain = env.state.all_objects_list
            self.assertFalse(np.array_equal(start_state, curr_terrain))

    def test_failing_rnd_layout(self):
        with self.assertRaises(TypeError):
            mdp_gen_params = {"None": None}
            mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(**mdp_gen_params)
            OvercookedEnv(mdp_fn, **DEFAULT_ENV_PARAMS)

    def test_random_layout(self):
        mdp_gen_params = {"prop_feats": (1, 1)}
        mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(**mdp_gen_params)
        env = OvercookedEnv(mdp_fn, **DEFAULT_ENV_PARAMS)
        start_terrain = env.mdp.terrain_mtx

        for _ in range(3):
            env.reset()
            # print(env)
            curr_terrain = env.mdp.terrain_mtx
            self.assertFalse(np.array_equal(start_terrain, curr_terrain))

        mdp_gen_params = {"mdp_choices": ['cramped_room', 'asymmetric_advantages']}
        mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(**mdp_gen_params)
        env = OvercookedEnv(mdp_fn, **DEFAULT_ENV_PARAMS)
        
        layouts_seen = []
        for _ in range(10):
            layouts_seen.append(env.mdp.terrain_mtx)
            env.reset()
        all_same_layout = all([np.array_equal(env.mdp.terrain_mtx, terrain) for terrain in layouts_seen])
        self.assertFalse(all_same_layout)
        
        
class TestGymEnvironment(unittest.TestCase):

    def setUp(self):
        self.base_mdp = OvercookedGridworld.from_layout_name("cramped_room")
        self.env = OvercookedEnv.from_mdp(self.base_mdp, **DEFAULT_ENV_PARAMS)
        self.rnd_agent_pair = AgentPair(FixedPlanAgent([]), FixedPlanAgent([]))
        np.random.seed(0)

    # TODO: write more tests here

if __name__ == '__main__':
    unittest.main()
