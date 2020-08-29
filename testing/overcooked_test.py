import unittest, os
import json
import numpy as np
from math import factorial
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import PlayerState, OvercookedGridworld, OvercookedState, ObjectState, SoupState, Recipe
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, DEFAULT_ENV_PARAMS
from overcooked_ai_py.mdp.layout_generator import LayoutGenerator, ONION_DISPENSER, TOMATO_DISPENSER, POT, DISH_DISPENSER, SERVING_LOC
from overcooked_ai_py.agents.agent import AgentGroup, AgentPair, GreedyHumanModel, FixedPlanAgent, RandomAgent
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS, MotionPlanner
from overcooked_ai_py.utils import save_pickle, load_pickle, iterate_over_json_files_in_dir, load_from_json, save_as_json
from utils import TESTING_DATA_DIR, generate_serialized_trajectory

START_ORDER_LIST = ["any"]
n, s = Direction.NORTH, Direction.SOUTH
e, w = Direction.EAST, Direction.WEST
stay, interact = Action.STAY, Action.INTERACT
P, Obj = PlayerState, ObjectState

def comb(n, k):
    return factorial(n) / (factorial(n - k) * factorial(k))


class TestRecipe(unittest.TestCase):

    def setUp(self):
        Recipe.configure({})
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

    def test_recipes_generation(self):
        self.assertRaises(AssertionError, Recipe.generate_random_recipes, max_size=Recipe.MAX_NUM_INGREDIENTS+1)
        self.assertRaises(AssertionError, Recipe.generate_random_recipes, min_size=0)
        self.assertRaises(AssertionError, Recipe.generate_random_recipes, min_size=3, max_size=2)
        self.assertRaises(AssertionError, Recipe.generate_random_recipes, ingredients=["onion", "tomato", "fake_ingredient"])
        self.assertRaises(AssertionError, Recipe.generate_random_recipes, n=99999)
        self.assertEqual(len(Recipe.generate_random_recipes(n=3)), 3)
        self.assertEqual(len(Recipe.generate_random_recipes(n=99, unique=False)), 99)

        two_sized_recipes = [Recipe(["onion", "onion"]), Recipe(["onion", "tomato"]), Recipe(["tomato", "tomato"])]
        for _ in range(100):
            self.assertCountEqual(two_sized_recipes, Recipe.generate_random_recipes(n=3, min_size=2, max_size=2, ingredients=["onion", "tomato"]))

        only_onions_recipes = [Recipe(["onion", "onion"]), Recipe(["onion", "onion", "onion"])]
        for _ in range(100):
            self.assertCountEqual(only_onions_recipes, Recipe.generate_random_recipes(n=2, min_size=2, max_size=3, ingredients=["onion"]))
        
        self.assertCountEqual(only_onions_recipes, set([Recipe.generate_random_recipes(n=1, recipes=only_onions_recipes)[0] for _ in range(100)])) # false positives rate for this test is 1/10^99 

    def _expected_num_recipes(self, num_ingredients, max_len):
        return comb(num_ingredients + max_len, num_ingredients) - 1

class TestSoupState(unittest.TestCase):
    
    def setUp(self):
        Recipe.configure({})
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
        self.base_mdp = OvercookedGridworld.from_layout_name("mdp_test")


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
        actual_start_state = self.base_mdp.get_standard_start_state()
        expected_state_path = os.path.join(TESTING_DATA_DIR, "test_start_positions", "expected.json")

        # NOTE: Uncomment the following line if expected start state deliberately changed
        # save_as_json(actual_start_state.to_dict(), expected_state_path)

        expected_start_state = OvercookedState.from_dict(load_from_json(expected_state_path))
        self.assertEqual(actual_start_state, expected_start_state, '\n' + str(actual_start_state) + '\n' + str(expected_start_state))

    def test_file_constructor(self):
        mdp = OvercookedGridworld.from_layout_name('corridor')
        expected_start_state = OvercookedState(
            [PlayerState((3, 1), Direction.NORTH), PlayerState((10, 1), Direction.NORTH)], {},
            all_orders=[{ "ingredients" : ["onion", "onion", "onion"]}])
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


    def test_transitions_and_environment(self):
        bad_state = OvercookedState(
            [P((0, 0), s), P((3, 1), s)], {})

        with self.assertRaises(AssertionError):
            self.base_mdp.get_state_transition(bad_state, stay)

        env = OvercookedEnv.from_mdp(self.base_mdp)

        def check_transition(action, expected_path, recompute=False):
            # Compute actual values
            state = env.state
            pred_state, _ = self.base_mdp.get_state_transition(state, action)
            new_state, sparse_reward, _, _ = env.step(action)
            self.assertEqual(pred_state, new_state, '\n' + str(pred_state) + '\n' + str(new_state))

            # Recompute expected values if desired
            if recompute:
                actual = {
                    "state" : pred_state.to_dict(),
                    "reward" : sparse_reward
                }
                save_as_json(actual, expected_path)

            # Compute expected values
            expected = load_from_json(expected_path)
            expected_state = OvercookedState.from_dict(expected['state'])
            expected_reward = expected['reward']
            
            # Make sure everything lines up (note __eq__ is transitive)
            self.assertTrue(pred_state.time_independent_equal(expected_state), '\n' + str(pred_state) + '\n' + str(expected_state))
            self.assertEqual(sparse_reward, expected_reward)

        expected_path = os.path.join(TESTING_DATA_DIR, "test_transitions_and_environments", "expected.json")

        # NOTE: set 'recompute=True' if deliberately updating state dynamics
        check_transition([n, e], expected_path, recompute=False)

    def test_mdp_dynamics(self):
        traj_path = os.path.join(TESTING_DATA_DIR, 'test_mdp_dynamics', 'expected.json')

        # NOTE: uncomment the following line to recompute trajectories if MDP dymamics were deliberately updated
        generate_serialized_trajectory(self.base_mdp, traj_path)

        test_trajectory = AgentEvaluator.load_traj_from_json(traj_path)
        AgentEvaluator.check_trajectories(test_trajectory, from_json=True)

    def test_mdp_serialization(self):
        # Where to store serialized states -- will be overwritten each timestep
        dummy_path = os.path.join(TESTING_DATA_DIR, 'test_mdp_serialization', 'dummy.json')

        # Get starting seed and random agent pair
        seed = 47
        random_pair = AgentPair(RandomAgent(all_actions=True), RandomAgent(all_actions=True))

        # Run rollouts with different seeds until sparse reward is achieved
        sparse_reward = 0
        while sparse_reward <= 0:
            np.random.seed(seed)
            state = self.base_mdp.get_standard_start_state()
            for _ in range(1500):
                # Ensure serialization and deserializations are inverses
                reconstructed_state = OvercookedState.from_dict(load_from_json(save_as_json(state.to_dict(), dummy_path)))
                self.assertEqual(state, reconstructed_state, "\nState: \t\t\t{}\nReconstructed State: \t{}".format(state, reconstructed_state))

                # Advance state
                joint_action, _ = zip(*random_pair.joint_action(state))
                state, infos = self.base_mdp.get_state_transition(state, joint_action)
                sparse_reward += sum(infos['sparse_reward_by_agent'])
            seed += 1

    def test_four_player_mdp(self):
        try:
            OvercookedGridworld.from_layout_name("multiplayer_schelling")
        except AssertionError as e:
            print("Loading > 2 player map failed with error:", e)

    def test_potential_function(self):
        mp = MotionPlanner(self.base_mdp)
        state = self.base_mdp.get_standard_start_state()
        val0 = self.base_mdp.potential_function(state, mp)

        # Pick up onion
        print("pick up onion")
        print(self.base_mdp.state_string(state))
        print("potential: ", self.base_mdp.potential_function(state, mp))
        actions = [Direction.EAST, Action.INTERACT]
        for action in actions:
            state, _ = self.base_mdp.get_state_transition(state, [Action.STAY, action])
            print(self.base_mdp.state_string(state))
            print("potential: ", self.base_mdp.potential_function(state, mp))
        val1 = self.base_mdp.potential_function(state, mp)
        

        # Pick up tomato
        print("pick up tomtato")
        actions = [Direction.WEST, Action.INTERACT]
        for action in actions:
            state, _ = self.base_mdp.get_state_transition(state, [action, Action.STAY])
            print(self.base_mdp.state_string(state))
            print("potential: ", self.base_mdp.potential_function(state, mp))
        val2 = self.base_mdp.potential_function(state, mp)

        self.assertLess(val0, val1, "Picking up onion should increase potential")
        self.assertLess(val1, val2, "Picking up tomato should increase potential")

        # Pot tomato
        print("pot tomato")
        actions = [Direction.EAST, Direction.NORTH, Action.INTERACT, Direction.WEST]
        for action in actions:
            state, _ = self.base_mdp.get_state_transition(state, [action, Action.STAY])
            print(self.base_mdp.state_string(state))
            print("potential: ", self.base_mdp.potential_function(state, mp))
        val3 = self.base_mdp.potential_function(state, mp)
        
        # Pot onion
        print("pot onion")
        actions = [Direction.WEST, Direction.NORTH, Action.INTERACT]
        for action in actions:
            state, _ = self.base_mdp.get_state_transition(state, [Action.STAY, action])
            print(self.base_mdp.state_string(state))
            print("potential: ", self.base_mdp.potential_function(state, mp))
        val4 = self.base_mdp.potential_function(state, mp)

        self.assertLess(val2, val3, "Potting tomato should increase potential")
        self.assertLess(val3, val4, "Potting onion should increase potential")

        ## Repeat on second pot ##

        # Pick up onion
        print("pick up onion")
        state, _ = self.base_mdp.get_state_transition(state, [Action.INTERACT, Action.STAY])
        val5 = self.base_mdp.potential_function(state, mp)
        print(self.base_mdp.state_string(state))
        print("potential: ", self.base_mdp.potential_function(state, mp))

        # Pick up tomato
        print("pick up tomato")
        actions = [Direction.SOUTH, Direction.EAST, Action.INTERACT]
        for action in actions:
            state, _ = self.base_mdp.get_state_transition(state, [Action.STAY, action])
            print(self.base_mdp.state_string(state))
            print("potential: ", self.base_mdp.potential_function(state, mp))
        val6 = self.base_mdp.potential_function(state, mp)

        self.assertLess(val4, val5, "Picking up onion should increase potential")
        self.assertLess(val5, val6, "Picking up tomato should increase potential")

        # Pot onion
        print("pot onion")
        actions = [Direction.SOUTH, Direction.EAST, Direction.SOUTH, Action.INTERACT, Direction.WEST]
        for action in actions:
            state, _ = self.base_mdp.get_state_transition(state, [action, Action.STAY])
            print(self.base_mdp.state_string(state))
            print("potential: ", self.base_mdp.potential_function(state, mp))
        val7 = self.base_mdp.potential_function(state, mp)

        # Pot tomato
        print("pot tomato")
        actions = [Direction.WEST, Direction.SOUTH, Action.INTERACT]
        for action in actions:
            state, _ = self.base_mdp.get_state_transition(state, [Action.STAY, action])
            print(self.base_mdp.state_string(state))
            print("potential: ", self.base_mdp.potential_function(state, mp))
        val8 = self.base_mdp.potential_function(state, mp)

        

        self.assertLess(val6, val7, "Potting onion should increase potential")
        self.assertLess(val7, val8, "Potting tomato should increase potential")

        ## Useless pickups ##
        
        # pickup tomato
        print("pickup tomato")
        actions = [Action.INTERACT, Direction.NORTH]
        for action in actions:
            state, _ = self.base_mdp.get_state_transition(state, [action, Action.STAY])
            print(self.base_mdp.state_string(state))
            print("potential: ", self.base_mdp.potential_function(state, mp))
        val9 = self.base_mdp.potential_function(state, mp)

        # pickup tomato
        print("pickup tomato")
        actions = [Direction.EAST, Action.INTERACT]
        for action in actions:
            state, _ = self.base_mdp.get_state_transition(state, [Action.STAY, action])
            print(self.base_mdp.state_string(state))
            print("potential: ", self.base_mdp.potential_function(state, mp))
        val10 = self.base_mdp.potential_function(state, mp)

        self.assertLessEqual(val9, val8, "Extraneous pickup should not increase potential")
        self.assertLessEqual(val10, val8, "Extraneous pickup should not increase potential")

        ## Catastrophic soup failure ##
        
        # pot tomato
        print("pot catastrophic tomato")
        actions = [Direction.WEST, Direction.SOUTH, Action.INTERACT]
        for action in actions:
            state, _ = self.base_mdp.get_state_transition(state, [Action.STAY, action])
            print(self.base_mdp.state_string(state))
            print("potential: ", self.base_mdp.potential_function(state, mp))
        val11 = self.base_mdp.potential_function(state, mp)

        self.assertLess(val11, val10, "Catastrophic potting should decrease potential")

        ## Bonus soup creation

        # pick up onion
        print("pick up onion")
        actions = [Direction.NORTH, Action.INTERACT, Direction.WEST, Action.INTERACT]
        for action in actions:
            state, _ = self.base_mdp.get_state_transition(state, [action, Action.STAY])
            print(self.base_mdp.state_string(state))
            print("potential: ", self.base_mdp.potential_function(state, mp))
        val12 = self.base_mdp.potential_function(state, mp)

        # pot onion
        print("pot onion")
        actions = [Direction.EAST, Direction.NORTH, Action.INTERACT]
        for action in actions:
            state, _ = self.base_mdp.get_state_transition(state, [action, Action.STAY])
            print(self.base_mdp.state_string(state))
            print("potential: ", self.base_mdp.potential_function(state, mp))
        val13 = self.base_mdp.potential_function(state, mp)

        # Cook soup
        print("cook soup")
        actions = [Action.INTERACT, Direction.WEST]
        for action in actions:
            state, _ = self.base_mdp.get_state_transition(state, [action, Action.STAY])
            print(self.base_mdp.state_string(state))
            print("potential: ", self.base_mdp.potential_function(state, mp))
        val14 = self.base_mdp.potential_function(state, mp)

        self.assertLess(val11, val12, "Useful onion pickup should increase potential")
        self.assertLess(val12, val13, "Potting useful onion should increase potential")
        self.assertLess(val13, val14, "Cooking optimal soup should increase potential")

        ## Soup pickup ##

        # Pick up dish
        print("pick up dish")
        actions = [Direction.WEST, Direction.SOUTH, Action.INTERACT]
        for action in actions:
            state, _ = self.base_mdp.get_state_transition(state, [Action.STAY, action])
            print(self.base_mdp.state_string(state))
            print("potential: ", self.base_mdp.potential_function(state, mp))
        val15 = self.base_mdp.potential_function(state, mp)

        # Move towards pot
        print("move towards pot")
        actions = [Direction.EAST, Direction.NORTH]
        for action in actions:
            state, _ = self.base_mdp.get_state_transition(state, [Action.STAY, action])
            print(self.base_mdp.state_string(state))
            print("potential: ", self.base_mdp.potential_function(state, mp))
        val16 = self.base_mdp.potential_function(state, mp)

        # Pickup soup
        print("pickup soup")
        state, _ = self.base_mdp.get_state_transition(state, [Action.STAY, Action.INTERACT])
        print(self.base_mdp.state_string(state))
        print("potential: ", self.base_mdp.potential_function(state, mp))
        val17 = self.base_mdp.potential_function(state, mp)

        self.assertLess(val14, val15, "Useful dish pickups should increase potential")
        self.assertLess(val15, val16, "Moving towards soup with dish should increase potential")
        self.assertLess(val16, val17, "Picking up soup should increase potential")

        ## Removing failed soup from pot

        # move towards failed soup
        print("move torwards failed soup")
        actions = [Direction.SOUTH, Direction.EAST, Direction.SOUTH]
        for action in actions:
            state, _ = self.base_mdp.get_state_transition(state, [action, Action.STAY])
            print(self.base_mdp.state_string(state))
            print("potential: ", self.base_mdp.potential_function(state, mp))
        val18 = self.base_mdp.potential_function(state, mp)

        # Cook failed soup
        actions = [Action.INTERACT]
        for action in actions:
            state, _ = self.base_mdp.get_state_transition(state, [action, Action.STAY])
            print(self.base_mdp.state_string(state))
            print("potential: ", self.base_mdp.potential_function(state, mp))
        val19 = self.base_mdp.potential_function(state, mp)

        # Pickup dish
        print("pickup dish")
        actions = [Direction.WEST, Direction.SOUTH, Action.INTERACT]
        for action in actions:
            state, _ = self.base_mdp.get_state_transition(state, [action, Action.STAY])
            print(self.base_mdp.state_string(state))
            print("potential: ", self.base_mdp.potential_function(state, mp))
        val20 = self.base_mdp.potential_function(state, mp)

        # Move towards soup
        print("move towards soup")
        actions = [Direction.EAST, Direction.SOUTH]
        for action in actions:
            state, _ = self.base_mdp.get_state_transition(state, [action, Action.STAY])
            print(self.base_mdp.state_string(state))
            print("potential: ", self.base_mdp.potential_function(state, mp))
        val21 = self.base_mdp.potential_function(state, mp)

        self.assertLess(val17, val18, "Moving towards failed soup should increase potential")
        self.assertLess(val18, val19, "Cooking failed soup should increase potential")
        self.assertLess(val19, val20, "Dish pickup for failed soup is still useful")
        self.assertLess(val20, val21, "Moving towars pertinant pot with dish should increase potential")

        ## Deliver failed soup ##

        # Pickup failed soup
        actions = [Action.INTERACT]
        for action in actions:
            state, _ = self.base_mdp.get_state_transition(state, [action, Action.STAY])
            print(self.base_mdp.state_string(state))
            print("potential: ", self.base_mdp.potential_function(state, mp))
        val22 = self.base_mdp.potential_function(state, mp)

        # Move towards serving area
        print("move towards servering area")
        actions = [Direction.EAST, Direction.SOUTH]
        for action in actions:
            state, _ = self.base_mdp.get_state_transition(state, [action, Action.STAY])
            print(self.base_mdp.state_string(state))
            print("potential: ", self.base_mdp.potential_function(state, mp))
        val23 = self.base_mdp.potential_function(state, mp)

        # Move away from serving area
        print("move away from serving area")
        state, _ = self.base_mdp.get_state_transition(state, [Direction.NORTH, Action.STAY])
        print(self.base_mdp.state_string(state))
        print("potential: ", self.base_mdp.potential_function(state, mp))
        val24 = self.base_mdp.potential_function(state, mp)

        self.assertLess(val21, val22, "Picking up failed soup should increase potential")
        self.assertAlmostEqual(val23, val22, delta=0.2, msg="Moving to serve failed soup doesn't change potential much")
        self.assertAlmostEqual(val23, val24, delta=0.2, msg="Moving away from serving area with failed soup doesn't change much")

        ## Deliver successful soup ##

        # Move towards serving area
        print("move towards serving area")
        actions = [Direction.SOUTH, Direction.EAST, Direction.SOUTH]
        for action in actions:
            state, _ = self.base_mdp.get_state_transition(state, [Action.STAY, action])
            print(self.base_mdp.state_string(state))
            print("potential: ", self.base_mdp.potential_function(state, mp))
        val25 = self.base_mdp.potential_function(state, mp)

        # Deliver soup
        print("deliver successful soup")
        state, rewards = self.base_mdp.get_state_transition(state, [Action.STAY, Action.INTERACT])
        print(self.base_mdp.state_string(state))
        print("potential: ", self.base_mdp.potential_function(state, mp))

        self.assertLess(val24, val25, "Moving towards serving area with valid soup increases potential")
        self.assertEqual(sum(rewards['sparse_reward_by_agent']), 50, "Soup was not properly devivered, probably an error with MDP logic")





def random_joint_action():
    num_actions = len(Action.ALL_ACTIONS)
    a_idx0, a_idx1 = np.random.randint(low=0, high=num_actions, size=2)
    return (Action.INDEX_TO_ACTION[a_idx0], Action.INDEX_TO_ACTION[a_idx1])


class TestFeaturizations(unittest.TestCase):

    def setUp(self):
        self.base_mdp = OvercookedGridworld.from_layout_name("cramped_room")
        self.mlam = MediumLevelActionManager.from_pickle_or_compute(self.base_mdp, NO_COUNTERS_PARAMS, force_compute=True)
        self.env = OvercookedEnv.from_mdp(self.base_mdp, **DEFAULT_ENV_PARAMS)
        self.rnd_agent_pair = AgentPair(GreedyHumanModel(self.mlam), GreedyHumanModel(self.mlam))
        np.random.seed(0)

    def test_lossless_state_featurization_shape(self):
        s = self.base_mdp.get_standard_start_state()
        obs = self.base_mdp.lossless_state_encoding(s)[0]
        self.assertTrue(np.array_equal(obs.shape, self.base_mdp.lossless_state_encoding_shape), "{} vs {}".format(obs.shape, self.base_mdp.lossless_state_encoding_shape))

    def test_state_featurization_shape(self):
        s = self.base_mdp.get_standard_start_state()
        obs = self.base_mdp.featurize_state(s, self.mlam)[0]
        self.assertTrue(np.array_equal(obs.shape, self.base_mdp.featurize_state_shape), "{} vs {}".format(obs.shape, self.base_mdp.featurize_state_shape))

    def test_lossless_state_featurization(self):
        trajs = self.env.get_rollouts(self.rnd_agent_pair, num_games=5)
        featurized_observations = [[self.base_mdp.lossless_state_encoding(state) for state in ep_states] for ep_states in trajs["ep_states"]]
        
        pickle_path = os.path.join(TESTING_DATA_DIR, "test_lossless_state_featurization", "expected")

        # NOTE: If the featurizations are updated intentionally, you can overwrite the expected
        # featurizations by uncommenting the following line:
        # save_pickle(featurized_observations, pickle_path)
        expected_featurization = load_pickle(pickle_path)
        self.assertTrue(np.array_equal(expected_featurization, featurized_observations))

    def test_state_featurization(self):
        trajs = self.env.get_rollouts(self.rnd_agent_pair, num_games=5)
        featurized_observations = [[self.base_mdp.featurize_state(state, self.mlam) for state in ep_states] for ep_states in trajs["ep_states"]]
        pickle_path = os.path.join(TESTING_DATA_DIR, "test_state_featurization", 'expected')
        # NOTE: If the featurizations are updated intentionally, you can overwrite the expected
        # featurizations by uncommenting the following line:
        # save_pickle(featurized_observations, pickle_path)
        expected_featurization = load_pickle(pickle_path)
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

    def test_display(self):
        mdp0 = OvercookedGridworld.from_layout_name("cramped_room")
        mdp_fn = lambda _ignored: mdp0
        env = OvercookedEnv(mdp_fn, horizon=20)
        env.get_rollouts(self.rnd_agent_pair, 1, display=True)

    def test_display_phi(self):
        mdp0 = OvercookedGridworld.from_layout_name("cramped_room")
        mdp_fn = lambda _ignored: mdp0
        env = OvercookedEnv(mdp_fn, horizon=20)
        env.get_rollouts(self.rnd_agent_pair, 1, display=True, display_phi=True)

    def test_multiple_mdp_env(self):
        mdp0 = OvercookedGridworld.from_layout_name("cramped_room")
        mdp1 = OvercookedGridworld.from_layout_name("counter_circuit")
        mdp_fn = lambda _ignored: np.random.choice([mdp0, mdp1])
        
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
            curr_terrain = env.state.players_pos_and_or
            self.assertFalse(np.array_equal(start_state, curr_terrain))

    def test_starting_obj_randomization(self):
        self.base_mdp = OvercookedGridworld.from_layout_name("cramped_room")
        start_state_fn = self.base_mdp.get_random_start_state_fn(random_start_pos=False, rnd_obj_prob_thresh=0.8)
        env = OvercookedEnv.from_mdp(self.base_mdp, start_state_fn)
        start_state = env.state.all_objects_list
        for _ in range(3):
            env.reset()
            curr_terrain = env.state.all_objects_list
            self.assertFalse(np.array_equal(start_state, curr_terrain))

    def test_failing_rnd_layout(self):
        with self.assertRaises(TypeError):
            mdp_gen_params = {"None": None}
            mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(**mdp_gen_params)
            OvercookedEnv(mdp_fn, **DEFAULT_ENV_PARAMS)

    def test_random_layout(self):
        mdp_gen_params = {"inner_shape": (5, 4),
                            "prop_empty": 0.8,
                            "prop_feats": 0.2,
                            "start_all_orders" : [
                                { "ingredients" : ["onion", "onion", "onion"]}
                            ],
                            "recipe_values" : [20],
                            "recipe_times" : [20],
                            "display": False
                          }
        mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_gen_params, outer_shape=(5, 4))
        env = OvercookedEnv(mdp_fn, **DEFAULT_ENV_PARAMS)
        start_terrain = env.mdp.terrain_mtx

        for _ in range(3):
            env.reset()
            curr_terrain = env.mdp.terrain_mtx
            self.assertFalse(np.array_equal(start_terrain, curr_terrain))

        mdp_gen_params = {"layout_name": 'cramped_room'}
        mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_gen_params)
        env = OvercookedEnv(mdp_fn, **DEFAULT_ENV_PARAMS)
        
        layouts_seen = []
        for _ in range(5):
            layouts_seen.append(env.mdp.terrain_mtx)
            env.reset()

        all_same_layout = all([np.array_equal(env.mdp.terrain_mtx, terrain) for terrain in layouts_seen])
        self.assertTrue(all_same_layout)

        mdp_gen_params = {"layout_name": 'asymmetric_advantages'}
        mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_gen_params)
        env = OvercookedEnv(mdp_fn, **DEFAULT_ENV_PARAMS)
        for _ in range(5):
            layouts_seen.append(env.mdp.terrain_mtx)
            env.reset()

        all_same_layout = all([np.array_equal(env.mdp.terrain_mtx, terrain) for terrain in layouts_seen])
        self.assertFalse(all_same_layout)
        
    def test_random_layout_feature_types(self):
        mandatory_features = {POT, DISH_DISPENSER, SERVING_LOC}
        optional_features = {ONION_DISPENSER, TOMATO_DISPENSER}
        optional_features_combinations = [{ONION_DISPENSER, TOMATO_DISPENSER}, {ONION_DISPENSER}, {TOMATO_DISPENSER}]

        for optional_features_combo in optional_features_combinations:
            left_out_optional_features = optional_features - optional_features_combo
            used_features = list(optional_features_combo | mandatory_features)
            mdp_gen_params = {"prop_feats": 0.9,
                            "feature_types": used_features,
                            "prop_empty": 0.1,
                            "inner_shape": (6, 5),
                            "display": False,
                            "start_all_orders" : [
                                { "ingredients" : ["onion", "onion", "onion"]}
                            ]}
            mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_gen_params, outer_shape=(6, 5))
            env = OvercookedEnv(mdp_fn, **DEFAULT_ENV_PARAMS)
            for _ in range(10):
                env.reset()
                curr_terrain = env.mdp.terrain_mtx
                terrain_features = set.union(*(set(line) for line in curr_terrain))
                self.assertTrue(all(elem in terrain_features for elem in used_features)) # all used_features are actually used
                if left_out_optional_features:
                    self.assertFalse(any(elem in terrain_features for elem in left_out_optional_features)) # all left_out optional_features are not used

    def test_random_layout_generated_recipes(self):
        only_onions_recipes = [Recipe(["onion", "onion"]), Recipe(["onion", "onion", "onion"])]
        only_onions_dict_recipes = [r.to_dict() for r in only_onions_recipes]

        # checking if recipes are generated from mdp_params
        mdp_gen_params = {"generate_all_orders": {"n":2, "ingredients": ["onion"], "min_size":2, "max_size":3},
                        "prop_feats": 0.9,
                        "prop_empty": 0.1,
                        "inner_shape": (6, 5),
                        "display": False}
        mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_gen_params, outer_shape=(6, 5))
        env = OvercookedEnv(mdp_fn, **DEFAULT_ENV_PARAMS)
        for _ in range(10):
            env.reset()
            self.assertCountEqual(env.mdp.start_all_orders, only_onions_dict_recipes)
            self.assertEqual(len(env.mdp.start_bonus_orders), 0)
        
        # checking if bonus_orders is subset of all_orders even if not specified

        mdp_gen_params = {"generate_all_orders": {"n":2, "ingredients": ["onion"], "min_size":2, "max_size":3},
                        "generate_bonus_orders": {"n":1, "min_size":2, "max_size":3},
                        "prop_feats": 0.9,
                        "prop_empty": 0.1,
                        "inner_shape": (6, 5),
                        "display": False}
        mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_gen_params, outer_shape=(6,5))
        env = OvercookedEnv(mdp_fn, **DEFAULT_ENV_PARAMS)
        for _ in range(10):
            env.reset()
            self.assertCountEqual(env.mdp.start_all_orders, only_onions_dict_recipes)
            self.assertEqual(len(env.mdp.start_bonus_orders), 1)
            self.assertTrue(env.mdp.start_bonus_orders[0] in only_onions_dict_recipes)

        # checking if after reset there are new recipes generated
        mdp_gen_params = {"generate_all_orders": {"n":3, "min_size":2, "max_size":3},
                        "prop_feats": 0.9,
                        "prop_empty": 0.1,
                        "inner_shape": (6, 5),
                        "display": False,
                        "feature_types": [POT, DISH_DISPENSER, SERVING_LOC, ONION_DISPENSER, TOMATO_DISPENSER]
                        }
        mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_gen_params, outer_shape=(6,5))
        env = OvercookedEnv(mdp_fn, **DEFAULT_ENV_PARAMS)
        generated_recipes_strings = set()
        for _ in range(20):
            env.reset()
            generated_recipes_strings |= {json.dumps(o, sort_keys=True) for o in env.mdp.start_all_orders}
        self.assertTrue(len(generated_recipes_strings) > 3)
        
        
class TestGymEnvironment(unittest.TestCase):

    def setUp(self):
        self.base_mdp = OvercookedGridworld.from_layout_name("cramped_room")
        self.env = OvercookedEnv.from_mdp(self.base_mdp, **DEFAULT_ENV_PARAMS)
        self.rnd_agent_pair = AgentPair(FixedPlanAgent([]), FixedPlanAgent([]))
        np.random.seed(0)

    # TODO: write more tests here

if __name__ == '__main__':
    unittest.main()
