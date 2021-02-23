import unittest, os, copy, shutil, gym, json
import numpy as np
from math import factorial
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import PlayerState, OvercookedGridworld, OvercookedState, ObjectState, SoupState, Recipe, Order, OrdersList
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

        self.pickle_temp_dir = os.path.join(TESTING_DATA_DIR, 'recipes')

        if not os.path.exists(self.pickle_temp_dir):
            os.makedirs(self.pickle_temp_dir)

    def tearDown(self):
        Recipe.configure({})

        if os.path.exists(self.pickle_temp_dir):
            shutil.rmtree(self.pickle_temp_dir)
            

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

    def test_serialization(self):
        loaded_recipes = []

        # Save and then load every recipe instance
        for i, recipe in enumerate(self.recipes):
            pickle_path = os.path.join(self.pickle_temp_dir, 'recipe_{}'.format(i))
            save_pickle(recipe, pickle_path)
            loaded = load_pickle(pickle_path)
            loaded_recipes.append(loaded)
        
        # Ensure loaded recipes equal corresponding original recipe
        for original, loaded in zip(self.recipes, loaded_recipes):
            self.assertEqual(original, loaded)


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

    def test_ingredients_list_standarization(self):
        o = Recipe.ONION
        t = Recipe.TOMATO
        self.assertEqual(
            Recipe.standarized_ingredients([o,o,t]),
            Recipe.standarized_ingredients([t,o,o])
            )
        self.assertEqual(
            Recipe.standarized_ingredients([t,o,t]),
            Recipe.standarized_ingredients([t,t,o])
            )
        self.assertNotEqual(
            Recipe.standarized_ingredients([o,o,t]),
            Recipe.standarized_ingredients([t,t,o])
            )
        self.assertNotEqual(
            Recipe.standarized_ingredients([t,o,t]),
            Recipe.standarized_ingredients([t,t,t])
            )
    
    def test_ingredients_diff(self):
        o = Recipe.ONION
        t = Recipe.TOMATO
        self.assertEqual(
            Recipe.ingredients_diff([t,o,t], [o,t]),
            Recipe.standarized_ingredients([t])
            )
        self.assertEqual(
            Recipe.ingredients_diff([t,o], [o,t]),
            Recipe.standarized_ingredients([])
            )
        self.assertEqual(
            Recipe.ingredients_diff([t, t, t], [o,o]),
            Recipe.standarized_ingredients([t, t, t])
            )
    
    def test_neighbours(self):
        o = Recipe.ONION
        t = Recipe.TOMATO
        expected_neighbours_ingredients = set(
            [Recipe.standarized_ingredients([o,o,t]),
            Recipe.standarized_ingredients([o,t,t])])
        expected_neighbours_recipes = set(Recipe(i) for i in expected_neighbours_ingredients)
        self.assertEqual(
            set(Recipe.neighbors_ingredients([o,t])), 
            expected_neighbours_ingredients
        )
        self.assertEqual(
            set(Recipe([o,t]).neighbors()), 
            expected_neighbours_recipes
        )

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


class TestOrder(unittest.TestCase):

    def setUp(self):
        Recipe.configure({})
        self.onion_recipe = Recipe(["onion", "onion", "onion"])
        self.onion_recipe_dict = Recipe(["onion", "onion", "onion"]).to_dict()
        self.order_kwargs = dict(recipe=self.onion_recipe, time_to_expire=1, expire_penalty=2, base_reward=3,
            linear_time_bonus_reward=4, order_id="5", is_bonus=True)


    def test_init(self):
        order_from_recipe_obj = Order(self.onion_recipe, order_id="123")
        order_from_recipe_dict = Order(self.onion_recipe_dict, order_id="123")

        self.assertEqual(order_from_recipe_obj, order_from_recipe_dict,
            "creation of Order from recipe and recipe dict gives different result")
        order_from_kwargs = Order(**self.order_kwargs)
        for k, v in self.order_kwargs.items():
            self.assertEqual(getattr(order_from_kwargs, k), v)


    def test_serialization(self):
        order = Order(**self.order_kwargs)
        order_dict = order.to_dict()
        deserialized_order = Order.from_dict(order_dict)
        self.assertEqual(order, deserialized_order)


    def test_properties(self):
        default_params_order = order_from_recipe_obj = Order(self.onion_recipe)
        self.assertEqual(order_from_recipe_obj.recipe, self.onion_recipe)

        pernament_order = Order(self.onion_recipe, time_to_expire=None)
        self.assertFalse(pernament_order.is_temporary)
        self.assertFalse(pernament_order.is_expired)
        temporary_order = Order(self.onion_recipe, time_to_expire=3)
        self.assertTrue(temporary_order.is_temporary)
        self.assertFalse(temporary_order.is_expired)
        expired_order = Order(self.onion_recipe, time_to_expire=0)
        self.assertTrue(expired_order.is_expired)

        self.assertEqual(default_params_order.base_reward, self.onion_recipe.value)
        custom_base_reward = 5678
        base_reward_order = Order(self.onion_recipe, base_reward=custom_base_reward)
        self.assertEqual(base_reward_order.base_reward, custom_base_reward)
        base_reward_order.base_reward = custom_base_reward2 = 6789
        self.assertEqual(base_reward_order.base_reward, custom_base_reward2)


    def test_equalities(self):
        order = Order(self.onion_recipe)
        order2 = Order(self.onion_recipe)
        self.assertTrue(order.ids_independent_equal(order2))
        self.assertNotEqual(order, order2)
        order3 = Order(self.onion_recipe, time_to_expire=3)
        self.assertFalse(order.ids_independent_equal(order3))
        self.assertNotEqual(order, order3)
        order4 = Order(self.onion_recipe, expire_penalty=4)
        self.assertFalse(order.ids_independent_equal(order4))
        self.assertNotEqual(order, order4)
        order5 = Order(self.onion_recipe, base_reward=5)
        self.assertFalse(order.ids_independent_equal(order5))
        self.assertNotEqual(order, order5)
        order6 = Order(self.onion_recipe, linear_time_bonus_reward=6)
        self.assertFalse(order.ids_independent_equal(order6))
        self.assertNotEqual(order, order6)
        order7 = Order(self.onion_recipe, is_bonus=True)
        self.assertFalse(order.ids_independent_equal(order7))
        self.assertNotEqual(order, order7)


    def test_calculate_reward(self):
        timesteps_in_near_future = 3
        timesteps_in_far_future = 999
        linear_time_bonus_reward = 2.0
        temporary_order_expire_time = 12
        pernament_order = Order(self.onion_recipe, time_to_expire=None)
        temporary_order = Order(self.onion_recipe, time_to_expire=temporary_order_expire_time, linear_time_bonus_reward=linear_time_bonus_reward)

        self.assertEqual(pernament_order.calculate_reward(), self.onion_recipe.value)
        self.assertEqual(pernament_order.calculate_reward(), pernament_order.calculate_future_reward(0))
        self.assertEqual(pernament_order.calculate_reward(), pernament_order.calculate_future_reward(999))

        self.assertEqual(temporary_order.calculate_reward(), self.onion_recipe.value + temporary_order_expire_time * linear_time_bonus_reward )
        self.assertEqual(temporary_order.calculate_reward(), temporary_order.calculate_future_reward(0))
        self.assertEqual(temporary_order.base_reward + (temporary_order_expire_time - timesteps_in_near_future)*linear_time_bonus_reward,
         temporary_order.calculate_future_reward(timesteps_in_near_future))
        self.assertEqual(0,
            temporary_order.calculate_future_reward(timesteps_in_far_future))

        pernament_order.base_reward = temporary_order.base_reward = custom_base_reward = 5678
        self.assertEqual(pernament_order.calculate_reward(), custom_base_reward)
        self.assertEqual(pernament_order.calculate_reward(), pernament_order.calculate_future_reward(0))
        self.assertEqual(pernament_order.calculate_reward(), pernament_order.calculate_future_reward(999))

        self.assertEqual(temporary_order.calculate_reward(), custom_base_reward + temporary_order_expire_time * linear_time_bonus_reward)
        self.assertEqual(temporary_order.calculate_reward(), temporary_order.calculate_future_reward(0))
        self.assertEqual(temporary_order.base_reward + (temporary_order_expire_time - timesteps_in_near_future)*linear_time_bonus_reward,
         temporary_order.calculate_future_reward(timesteps_in_near_future))
        self.assertEqual(0,
            temporary_order.calculate_future_reward(timesteps_in_far_future))


    def test_step(self):
        pernament_order = Order(self.onion_recipe, time_to_expire=None)
        for _ in range(100):
            reward = pernament_order.step()
            self.assertEqual(reward, 0)

        expiring_order = Order(self.onion_recipe, time_to_expire=8)
        for i in range(10):
            if i < 8:
                reward = expiring_order.step()
                if i == 8:
                    self.assertEqual(reward, 0)
                else:
                    self.assertEqual(reward, 0)
            else:
                self.assertRaises(AssertionError, expiring_order.step)


    def test_will_be_expired_in(self):
        pernament_order = Order(self.onion_recipe, time_to_expire=None)
        self.assertFalse(pernament_order.will_be_expired_in(9999))
        expiring_order = Order(self.onion_recipe, time_to_expire=8)
        self.assertTrue(expiring_order.will_be_expired_in(9))
        self.assertFalse(pernament_order.will_be_expired_in(7))


class TestOrdersList(unittest.TestCase):
    def setUp(self):
        Recipe.configure({})
        self.example_orders = [
            Order(recipe=Recipe(["onion", "onion", "onion"])),
            Order(recipe=Recipe(["tomato", "tomato", "tomato"]), is_bonus=True),
            Order(recipe=Recipe(["onion", "onion"]), time_to_expire=5, expire_penalty=4, base_reward=3, linear_time_bonus_reward=2, is_bonus=True)
            ]
        self.example_orders_recipes = [o.recipe for o in self.example_orders]
        self.example_non_bonus_orders = self.example_orders[:1]
        self.example_bonus_orders = self.example_orders[1:]
        self.example_orders_to_add = [
            Order(recipe=Recipe(["onion", "onion", "tomato"]), time_to_expire=6, expire_penalty=5, base_reward=4, linear_time_bonus_reward=3, is_bonus=True),
            Order(recipe=Recipe(["onion", "tomato", "tomato"]), time_to_expire=7, expire_penalty=6, base_reward=5, linear_time_bonus_reward=4),
            Order(recipe=Recipe(["onion", "onion"]), time_to_expire=8, expire_penalty=7, base_reward=6, linear_time_bonus_reward=5, is_bonus=True)
        ]
        self.example_orders_to_add_recipes = [o.recipe for o in self.example_orders_to_add]
        self.example_bonus_recipes = [Recipe(["onion", "onion"]),
            Recipe(["onion", "onion", "tomato"]),
            Recipe(["tomato", "tomato", "tomato"])] # copied directly from example_orders and example_orders_to_add

        self.example_orders_list_kwargs = dict(orders=self.example_orders,
         orders_to_add=self.example_orders_to_add, add_new_order_every=6,
          time_to_next_order=4
        )
        self.example_orders_list = OrdersList(**self.example_orders_list_kwargs)
        self.empty_orders_list = OrdersList([])
        self.recipe_outside_orders_list = Recipe(["onion"])


    def test_init(self):
        self.assertCountEqual(self.example_orders, self.example_orders_list.orders)
        self.assertCountEqual(self.example_orders_to_add, self.example_orders_list.orders_to_add)
        self.assertEqual(42, OrdersList(add_new_order_every=42, orders_to_add=self.example_orders_to_add).add_new_order_every)
        self.assertEqual(34, OrdersList(time_to_next_order=34, orders_to_add=self.example_orders_to_add).time_to_next_order)

        self.assertRaises(AssertionError, lambda kwargs: OrdersList(**kwargs), {"add_new_order_every":-1 })
        self.assertRaises(AssertionError, lambda kwargs: OrdersList(**kwargs), {"add_new_order_every":10, "orders_to_add": []})
        self.assertRaises(AssertionError, lambda kwargs: OrdersList(**kwargs), {"time_to_next_order":10, "orders_to_add": []})


    def test_equalities(self):
        example_orders_list_copy = copy.deepcopy(self.example_orders_list)
        for order in example_orders_list_copy.orders:
            order.order_id = Order.create_order_id()
        self.assertNotEqual(example_orders_list_copy, self.example_orders_list)
        self.assertTrue(example_orders_list_copy.ids_independent_equal(self.example_orders_list))
        example_orders_list_copy.add_new_order_every = 99
        self.assertFalse(example_orders_list_copy.ids_independent_equal(self.example_orders_list))
        example_orders_list_copy.add_new_order_every = self.example_orders_list.add_new_order_every
        example_orders_list_copy.time_to_next_order = 99
        self.assertFalse(example_orders_list_copy.ids_independent_equal(self.example_orders_list))


    def test_properties(self):
        self.assertCountEqual(self.example_orders_list.all_recipes,
            set(self.example_orders_to_add_recipes+self.example_orders_recipes))
        self.assertCountEqual(self.example_bonus_recipes, self.example_orders_list.bonus_recipes)

        self.assertTrue(self.example_orders_list.is_adding_orders)
        not_adding_orders_kwargs = copy.deepcopy(self.example_orders_list_kwargs)
        not_adding_orders_kwargs["add_new_order_every"] = None
        not_adding_orders_list = OrdersList(**not_adding_orders_kwargs)
        self.assertFalse(not_adding_orders_list.is_adding_orders)
        self.assertRaises(AttributeError, lambda x: setattr(not_adding_orders_list, "is_adding_orders", x), True)

        self.assertTrue(self.example_orders_list.contains_temporary_orders)
        not_temporary_orders_list = OrdersList([Order(Recipe(["onion", "onion", "onion"]))])
        self.assertFalse(not_temporary_orders_list.contains_temporary_orders)

        self.assertCountEqual(self.example_orders_list.bonus_orders, self.example_bonus_orders)
        self.assertRaises(AttributeError, lambda x: setattr(self.example_orders_list, "bonus_orders", x), self.example_bonus_orders)

        self.assertCountEqual(self.example_orders_list.non_bonus_orders, self.example_non_bonus_orders)
        self.assertRaises(AttributeError, lambda x: setattr(self.example_orders_list, "non_bonus_orders", x), self.example_non_bonus_orders)


    def test_builtin_functions(self):
        _ = str(self.example_orders_list)
        _ = repr(self.example_orders_list)
        _ = hash(self.example_orders_list)
        self.assertTrue(bool(self.example_orders_list))

        self.assertFalse(bool(self.empty_orders_list))
        #self.assertCountEqual(list(self.example_orders_list), self.example_orders_list.orders) # testing __iter__
        self.assertEqual(len(self.example_orders_list), 3)
        self.assertEqual(self.example_orders_list, copy.deepcopy(self.example_orders_list))


    def test_orders_sorting(self):
        temporary_order = Order(recipe=Recipe(["tomato", "onion", "onion"]), time_to_expire=2, order_id="123", base_reward=100)
        pernament_order1 = Order(recipe=Recipe(["onion", "onion", "onion"]), time_to_expire=None, order_id="234", base_reward=50)
        pernament_order2 = Order(recipe=Recipe(["tomato", "tomato", "onion"]), time_to_expire=None, order_id="345", base_reward=20)
        orders_list = OrdersList([pernament_order1, temporary_order, pernament_order2])
        self.assertEqual(temporary_order, orders_list.orders_sorted_by_urgency()[0])
        self.assertEqual(temporary_order, orders_list.orders_sorted_by_urgency(1)[0])
        self.assertEqual(1, orders_list.enumerated_orders_sorted_by_urgency(1)[0][0])
        self.assertEqual(pernament_order1, orders_list.orders_sorted_by_urgency(2)[0])
        self.assertEqual(0, orders_list.enumerated_orders_sorted_by_urgency(2)[0][0])

        temporary_order1 = Order(recipe=Recipe(["tomato", "onion", "onion"]), time_to_expire=2, order_id="123", base_reward=100)
        temporary_order2 = Order(recipe=Recipe(["tomato", "onion", "onion"]), time_to_expire=1, order_id="234", base_reward=50)
        temporary_order3 = Order(recipe=Recipe(["tomato", "onion", "onion"]), time_to_expire=3, order_id="345", base_reward=20)
        orders_list = OrdersList([temporary_order1, temporary_order2, temporary_order3])
        self.assertEqual(temporary_order2, orders_list.orders_sorted_by_urgency()[0])
        self.assertEqual(1, orders_list.enumerated_orders_sorted_by_urgency()[0][0])

        temporary_order1 = Order(recipe=Recipe(["tomato", "onion", "onion"]), time_to_expire=10, order_id="123", base_reward=30)
        temporary_order2 = Order(recipe=Recipe(["tomato", "onion", "onion"]), time_to_expire=10, order_id="234", base_reward=100)
        temporary_order3 = Order(recipe=Recipe(["tomato", "onion", "onion"]), time_to_expire=10, order_id="345", base_reward=20)
        orders_list = OrdersList([temporary_order1, temporary_order2, temporary_order3])
        self.assertEqual(temporary_order2, orders_list.orders_sorted_by_urgency()[0])
        self.assertEqual(1, orders_list.enumerated_orders_sorted_by_urgency(2)[0][0])
        self.assertEqual(temporary_order2, orders_list.orders_sorted_by_urgency(999)[0])
        self.assertEqual(1, orders_list.enumerated_orders_sorted_by_urgency(999)[0][0])

        default_sorting_f = OrdersList._order_urgency_sort_key
        new_sorting_f = lambda x, n_timesteps_into_future: x.order_id
        OrdersList._order_urgency_sort_key = new_sorting_f
        self.assertEqual(temporary_order3, orders_list.orders_sorted_by_urgency()[0])
        self.assertEqual(2, orders_list.enumerated_orders_sorted_by_urgency()[0][0])
        self.assertEqual(temporary_order3, orders_list.orders_sorted_by_urgency(999)[0])
        self.assertEqual(2, orders_list.enumerated_orders_sorted_by_urgency(999)[0][0])
        OrdersList._order_urgency_sort_key = default_sorting_f


    def test_get_matching_order(self):
        for i, order in enumerate(self.example_orders):
            self.assertEqual(i, self.example_orders_list.matching_order_index(recipe=order.recipe))
            self.assertEqual(order, self.example_orders_list.get_matching_order(recipe=order.recipe))

            self.assertEqual(i, self.example_orders_list.matching_order_index(order_id=order.order_id))
            self.assertEqual(order, self.example_orders_list.get_matching_order(order_id=order.order_id))
            for order_temporary in [order.is_temporary, None]:
                self.assertEqual(i, self.example_orders_list.matching_order_index(recipe=order.recipe, temporary_order=order_temporary))
                self.assertEqual(order, self.example_orders_list.get_matching_order(recipe=order.recipe, temporary_order=order_temporary))

                self.assertEqual(i, self.example_orders_list.matching_order_index(order_id=order.order_id, temporary_order=order_temporary))
                self.assertEqual(order, self.example_orders_list.get_matching_order(order_id=order.order_id, temporary_order=order_temporary))
        order_temporary = not order.is_temporary
        self.assertIsNone(self.example_orders_list.matching_order_index(recipe=order.recipe, temporary_order=order_temporary))
        self.assertIsNone(self.example_orders_list.get_matching_order(recipe=order.recipe, temporary_order=order_temporary))

        self.assertIsNone(self.example_orders_list.matching_order_index(order_id=order.order_id, temporary_order=order_temporary))
        self.assertIsNone(self.example_orders_list.get_matching_order(order_id=order.order_id, temporary_order=order_temporary))


    def test_adding_orders(self):
        orders_list1 = copy.deepcopy(self.empty_orders_list)
        for i, order in enumerate(self.example_orders):
            orders_list1.add_order(order)
            self.assertCountEqual(orders_list1.orders, self.example_orders[:i+1])

        orders_list2 = copy.deepcopy(self.empty_orders_list)
        orders_list2.add_orders(self.example_orders)
        self.assertCountEqual(orders_list1.orders, orders_list2.orders)

        orders_list = copy.deepcopy(self.example_orders_list)
        for i in range(100):
            orders_list.add_random_orders()
        self.assertEqual(len(orders_list.orders), len(set([o.order_id for o in orders_list.orders])),
            "order_ids are not unique when using add_random_orders_method")

        for order_to_add in self.example_orders_to_add:
            self.assertIsNotNone(orders_list.get_matching_order(recipe=order_to_add.recipe, most_urgent=False),
                "not every order_to_add was added despite many addings (something wrong with randomness?)")

        # we will assume orders are different by its time_to_expire values
        orders_to_add_expire_times = set([o.time_to_expire for o in self.example_orders_to_add])
        assert len(orders_to_add_expire_times) == len(self.example_orders_to_add)
        assert not any((o.time_to_expire in orders_to_add_expire_times) for o in self.example_orders_list.orders)
        for i in range(100):
            orders_list = copy.deepcopy(self.example_orders_list)
            orders_list.add_random_orders(2, replace=False)
            self.assertEqual(len(self.example_orders_list)+2, len(orders_list))
            for order_to_add in self.example_orders_to_add:
                self.assertTrue(sum(order_to_add.time_to_expire == order.time_to_expire for order in orders_list.orders) < 2,
                "some order was added more than once despite replace=False")

        added_more_than_once = False
        for i in range(100):
            orders_list = copy.deepcopy(self.example_orders_list)
            orders_list.add_random_orders(2, replace=True)
            for order_to_add in self.example_orders_to_add:
                if sum(order_to_add.time_to_expire == order.time_to_expire for order in orders_list.orders) > 1:
                    added_more_than_once=True
                    break
        self.assertTrue(added_more_than_once, "same orders_to_add were never added more than once despite replace=True")

        orders_list = copy.deepcopy(self.empty_orders_list)
        pernament_order1 = copy.deepcopy(self.example_orders[0])
        pernament_order1.order_id = "123"
        pernament_order2 = copy.deepcopy(self.example_orders[0])
        pernament_order2.order_id = "234"
        orders_list.add_order(pernament_order1)
        self.assertRaises(AssertionError, orders_list.add_order, pernament_order2)

        assert self.example_orders[0].recipe != self.example_orders[1].recipe
        orders_list = copy.deepcopy(self.empty_orders_list)
        duplicated_order_id = "123"
        pernament_order1 = copy.deepcopy(self.example_orders[0])
        pernament_order1.order_id = duplicated_order_id
        pernament_order2 = copy.deepcopy(self.example_orders[1])
        pernament_order2.order_id = duplicated_order_id
        orders_list.add_order(pernament_order1)
        self.assertRaises(AssertionError, orders_list.add_order, pernament_order2)

    def test_remove_order(self):
        orders_list = copy.deepcopy(self.example_orders_list)
        self.assertIsNone(orders_list.remove_order(order_id="unexisting_order_id"))
        self.assertIsNone(orders_list.remove_order(recipe=self.recipe_outside_orders_list))

        pernament_order = self.example_orders[0]
        assert not pernament_order.is_temporary
        orders_list = copy.deepcopy(self.example_orders_list)
        self.assertIsNone(orders_list.remove_order(recipe=pernament_order.recipe, temporary_order=True))
        self.assertEqual(pernament_order, orders_list.remove_order(recipe=pernament_order.recipe, temporary_order=False))
        self.assertEqual(len(orders_list), len(self.example_orders_list)-1)
        orders_list = copy.deepcopy(self.example_orders_list)
        self.assertEqual(pernament_order, orders_list.remove_order(recipe=pernament_order.recipe, temporary_order=None))
        self.assertEqual(len(orders_list), len(self.example_orders_list)-1)
        orders_list = copy.deepcopy(self.example_orders_list)
        self.assertIsNone(orders_list.remove_order(order_id=orders_list.orders[0].order_id, temporary_order=True))
        self.assertEqual(pernament_order, orders_list.remove_order(order_id=orders_list.orders[0].order_id, temporary_order=False))
        self.assertEqual(len(orders_list), len(self.example_orders_list)-1)
        orders_list = copy.deepcopy(self.example_orders_list)
        self.assertEqual(pernament_order, orders_list.remove_order(order_id=orders_list.orders[0].order_id, temporary_order=None))
        self.assertEqual(len(orders_list), len(self.example_orders_list)-1)

        temporary_order = self.example_orders[2]
        assert temporary_order.is_temporary
        orders_list = copy.deepcopy(self.example_orders_list)
        self.assertIsNone(orders_list.remove_order(recipe=temporary_order.recipe, temporary_order=False))
        self.assertEqual(temporary_order, orders_list.remove_order(recipe=temporary_order.recipe, temporary_order=True))
        self.assertEqual(len(orders_list), len(self.example_orders_list)-1)
        orders_list = copy.deepcopy(self.example_orders_list)
        self.assertEqual(temporary_order, orders_list.remove_order(recipe=temporary_order.recipe, temporary_order=None))
        self.assertEqual(len(orders_list), len(self.example_orders_list)-1)
        orders_list = copy.deepcopy(self.example_orders_list)
        self.assertIsNone(orders_list.remove_order(order_id=orders_list.orders[2].order_id, temporary_order=False))
        self.assertEqual(temporary_order, orders_list.remove_order(order_id=orders_list.orders[2].order_id, temporary_order=True))
        self.assertEqual(len(orders_list), len(self.example_orders_list)-1)
        orders_list = copy.deepcopy(self.example_orders_list)
        self.assertEqual(temporary_order, orders_list.remove_order(order_id=orders_list.orders[2].order_id, temporary_order=None))
        self.assertEqual(len(orders_list), len(self.example_orders_list)-1)


    def test_order_fulfilling(self):
        orders_list = copy.deepcopy(self.example_orders_list)
        pernament_order = self.example_orders[0]
        assert not pernament_order.is_temporary
        temporary_order = self.example_orders[2]
        assert temporary_order.is_temporary
        self.assertIsNone(orders_list.fulfill_order(recipe=self.recipe_outside_orders_list))

        self.assertEqual(pernament_order, orders_list.fulfill_order(recipe=pernament_order.recipe))
        self.assertEqual(orders_list.fulfilled_orders, [pernament_order])
        self.assertEqual(len(orders_list), len(self.example_orders_list))

        self.assertEqual(temporary_order, orders_list.fulfill_order(recipe=temporary_order.recipe))
        self.assertEqual(orders_list.fulfilled_orders, [pernament_order, temporary_order])

        self.assertEqual(len(orders_list), len(self.example_orders_list)-1)
        list_with_removed_order = copy.deepcopy(self.example_orders_list)
        list_with_removed_order.remove_order(recipe=temporary_order.recipe)

              
        # orders list would differ by fulfilled_orders only
        self.assertFalse(list_with_removed_order.ids_independent_equal(orders_list))

        list_with_removed_order.fulfilled_orders = orders_list.fulfilled_orders = []
        self.assertTrue(list_with_removed_order.ids_independent_equal(orders_list))
        self.assertIsNone(orders_list.fulfill_order(recipe=temporary_order.recipe))


    def test_step(self):
        # testing if time_to_next_order resets properly
        orders_list = copy.deepcopy(self.example_orders_list)
        first_order_in = orders_list.time_to_next_order
        for i in range(100):
            self.assertEqual(orders_list.time_to_next_order%orders_list.add_new_order_every,
             (first_order_in-i)%orders_list.add_new_order_every)
            self.assertLessEqual(orders_list.time_to_next_order, orders_list.add_new_order_every)
            self.assertGreater(orders_list.time_to_next_order, 0)
            orders_list.step()

        # testing if adding new orders works ok
        orders_list = OrdersList(orders=[], orders_to_add=[(Order(Recipe(["onion"]), time_to_expire=9999))],
            add_new_order_every=5)
        for i in range(100):
            self.assertEqual(len(orders_list), int(i/orders_list.add_new_order_every))
            orders_list.step()

        # testing if expiring orders works ok
        orders_list = copy.deepcopy(self.empty_orders_list)
        for i in range(1, 51):
            orders_list.add_order(Order(Recipe(["onion"]), time_to_expire=2*i))

        for i in range(100):
            self.assertEqual(len(orders_list), 50-int(i/2))
            orders_list.step()


    def test_serialization(self):
        orders_list_dict = self.example_orders_list.to_dict()
        deserialized_orders_list = OrdersList.from_dict(orders_list_dict)
        self.assertEqual(self.example_orders_list, deserialized_orders_list)


    def test_from_recipes_list(self):
        non_bonus_orders_recipes = [Recipe(["onion"]).to_dict(), Recipe(["tomato"]).to_dict()]
        bonus_orders_recipes = [Recipe(["onion", "onion"]).to_dict(), Recipe(["tomato", "tomato"]).to_dict()]
        all_orders_recipes = bonus_orders_recipes + non_bonus_orders_recipes
        orders_list = OrdersList.from_recipes_lists(all_orders_recipes, bonus_orders_recipes)
        self.assertCountEqual([r.to_dict() for r in orders_list.all_recipes], all_orders_recipes)
        self.assertCountEqual([r.to_dict() for r in orders_list.bonus_recipes], bonus_orders_recipes)


    def test_dict_to_all_recipes_dicts(self):
        self.assertCountEqual(OrdersList.dict_to_all_recipes_dicts(self.example_orders_list.to_dict()), [r.to_dict() for r in self.example_orders_list.all_recipes])


    def test_max_orders_num(self):
        orders_list = OrdersList(orders=[], orders_to_add=[(Order(Recipe(["onion"]), time_to_expire=9999))],
            add_new_order_every=1, max_orders_num=5)
        for i in range(100):
            orders_list.step()
        self.assertEqual(len(orders_list.orders), 5)
        self.assertEqual(orders_list.num_orders_in_queue, 95)
        orders_list.orders = []
        orders_list.step()
        self.assertEqual(len(orders_list.orders), 5)
        self.assertEqual(orders_list.num_orders_in_queue, 91)
        

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
        self.assertTrue(actual_start_state.ids_independent_equal(expected_start_state), '\n' + str(actual_start_state) + '\n' + str(expected_start_state))

    def test_file_constructor(self):
        mdp = OvercookedGridworld.from_layout_name('corridor')
        expected_start_state = OvercookedState(
            [PlayerState((3, 1), Direction.NORTH), PlayerState((10, 1), Direction.NORTH)], {},
            all_orders=[{ "ingredients" : ["onion", "onion", "onion"]}])
        actual_start_state = mdp.get_standard_start_state()
        self.assertTrue(actual_start_state.ids_independent_equal(expected_start_state), '\n' + str(actual_start_state) + '\n' + str(expected_start_state))

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
            self.assertTrue(pred_state.custom_equal(expected_state, time_independent=True, ids_independent=True), '\n' + str(pred_state) + '\n' + str(expected_state))
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
                sparse_reward += infos['sparse_reward_sum']


    def test_four_player_mdp(self):
        try:
            OvercookedGridworld.from_layout_name("multiplayer_schelling")
        except AssertionError as e:
            print("Loading > 2 player map failed with error:", e)


    def test_state_orders_list(self):
        all_orders_dicts = [Recipe(["onion", "onion", "onion"]).to_dict(), Recipe(["tomato", "tomato", "tomato"]).to_dict()]
        bonus_orders_dicts = [ Recipe(["tomato", "tomato", "tomato"]).to_dict()]

        all_orders = [Recipe.from_dict(d) for d in all_orders_dicts]
        bonus_orders = [Recipe.from_dict(d) for d in bonus_orders_dicts]

        players = [PlayerState((3, 1), Direction.NORTH), PlayerState((10, 1), Direction.NORTH)]
        objects = {}
        state = OvercookedState(players, objects, all_orders=all_orders_dicts)
        self.assertCountEqual(state.all_orders, all_orders)
        state = OvercookedState(players, objects, all_orders=all_orders_dicts, bonus_orders=bonus_orders_dicts)
        self.assertCountEqual(state.bonus_orders, bonus_orders)
        self.assertCountEqual(state.all_orders, all_orders)

        state2 = OvercookedState(players, objects,
            orders_list=OrdersList.from_recipes_lists(all_orders_dicts, bonus_orders_dicts))
        self.assertTrue(state.ids_independent_equal(state2))
        state3 = OvercookedState(players, objects,
            orders_list=OrdersList.from_recipes_lists(all_orders_dicts, bonus_orders_dicts).to_dict())
        self.assertTrue(state.ids_independent_equal(state3))

        state_error_kwargs = {"players": players, "objects": objects, "all_orders": all_orders_dicts, "orders_list": state.orders_list}
        self.assertRaises(AssertionError, lambda kwargs: OvercookedState(**kwargs), state_error_kwargs)
        state_error_kwargs = {"players": players, "objects": objects, "bonus_orders": bonus_orders_dicts, "orders_list": state.orders_list}
        self.assertRaises(AssertionError, lambda kwargs: OvercookedState(**kwargs), state_error_kwargs)


    def test_gridworld_orders_list(self):
        def standardize_recipes_dicts_list_format(dicts_list):
            return [Recipe.from_dict(r).to_dict() for r in dicts_list]

        terrain = [['X', 'X', 'P', 'X', 'X'],
                   ['O', ' ', ' ', '2', 'O'],
                   ['T', '1', ' ', ' ', 'T'],
                   ['X', 'D', 'P', 'S', 'X']]

        start_all_orders = [
            { "ingredients" : ["onion"]},
            { "ingredients" : ["onion", "onion", "onion"]},
            { "ingredients" : ["onion", "tomato", "onion"]}]
        start_all_orders = standardize_recipes_dicts_list_format(start_all_orders)
        start_bonus_orders = [
            { "ingredients" : ["onion", "tomato", "onion"]}]
        start_bonus_orders = standardize_recipes_dicts_list_format(start_bonus_orders)
        orders_list_no_bonus = OrdersList.from_recipes_lists(start_all_orders, [])
        orders_list = OrdersList.from_recipes_lists(start_all_orders, start_bonus_orders)

        mdp = OvercookedGridworld.from_grid(terrain,
            base_layout_params={"start_all_orders": start_all_orders})
        self.assertCountEqual(mdp.start_all_orders, start_all_orders)
        self.assertCountEqual(mdp.start_bonus_orders, [])
        self.assertTrue(mdp.start_orders_list.ids_independent_equal(orders_list_no_bonus))

        mdp = OvercookedGridworld.from_grid(terrain,
            base_layout_params={"start_all_orders": start_all_orders,
            "start_bonus_orders":start_bonus_orders})
        self.assertCountEqual(mdp.start_all_orders, start_all_orders)
        self.assertCountEqual(mdp.start_bonus_orders, start_bonus_orders)
        self.assertTrue(mdp.start_orders_list.ids_independent_equal(orders_list))

        mdp = OvercookedGridworld.from_grid(terrain,
            base_layout_params={"start_orders_list": copy.deepcopy(orders_list)})
        self.assertCountEqual(mdp.start_all_orders, start_all_orders)
        self.assertCountEqual(mdp.start_bonus_orders, start_bonus_orders)
        self.assertTrue(mdp.start_orders_list.ids_independent_equal(orders_list))

        self.assertRaises(AssertionError, lambda args: OvercookedGridworld.from_grid(*args),
        [terrain, {"start_orders_list": copy.deepcopy(orders_list),
                    "start_bonus_orders": start_all_orders}])


    def test_get_recipe_value(self):
        mdp = OvercookedGridworld.from_layout_name("mdp_test")
        recipe_outside_orders_list = Recipe(["tomato", "tomato", "tomato"])
        non_bonus_recipe = [o.recipe for o in mdp.start_orders_list.non_bonus_orders if len(o.recipe.ingredients) > 1][0]
        almost_non_bonus_recipe = Recipe(non_bonus_recipe.ingredients[:-1])
        bonus_recipe = mdp.start_orders_list.bonus_orders[0].recipe
        almost_bonus_recipe = Recipe(bonus_recipe.ingredients[:-1])
        state = mdp.get_standard_start_state()
        assert state.orders_list.get_matching_order(recipe=recipe_outside_orders_list) is None

        self.assertEqual(non_bonus_recipe.value, mdp.get_recipe_value(state, non_bonus_recipe))
        self.assertEqual(non_bonus_recipe.value, mdp.get_recipe_value(state, non_bonus_recipe, steps_into_future=999))
        self.assertEqual(bonus_recipe.value*mdp.order_bonus, mdp.get_recipe_value(state, bonus_recipe))
        self.assertEqual(bonus_recipe.value*mdp.order_bonus, mdp.get_recipe_value(state, bonus_recipe, steps_into_future=999))
        self.assertEqual(0, mdp.get_recipe_value(state, recipe_outside_orders_list))
        gamma, pot_onion_steps, pot_tomato_steps = 0.99, 5, 10 # these may not represent true values for mdp_test layout
        potential_params = {"gamma": gamma, "pot_onion_steps": pot_onion_steps, "pot_tomato_steps": pot_tomato_steps}
        steps_into_future = 20
        self.assertEqual(gamma**(pot_onion_steps+non_bonus_recipe.time) * non_bonus_recipe.value,
            mdp.get_recipe_value(state, non_bonus_recipe, discounted=True, base_recipe=almost_non_bonus_recipe,
            potential_params=potential_params
        ))
        self.assertEqual(gamma**steps_into_future * non_bonus_recipe.value,
            mdp.get_recipe_value(state, non_bonus_recipe, discounted=True, base_recipe=almost_non_bonus_recipe,
            potential_params=potential_params, steps_into_future=steps_into_future
        ))
        self.assertEqual(gamma**(pot_tomato_steps+bonus_recipe.time) * bonus_recipe.value * mdp.order_bonus,
            mdp.get_recipe_value(state, bonus_recipe, discounted=True, base_recipe=almost_bonus_recipe,
            potential_params=potential_params
        ))
        self.assertEqual(gamma**steps_into_future * bonus_recipe.value * mdp.order_bonus,
            mdp.get_recipe_value(state, bonus_recipe, discounted=True, base_recipe=almost_bonus_recipe,
            potential_params=potential_params, steps_into_future=steps_into_future
        ))

        bonus_order = Order(recipe=bonus_recipe, time_to_expire=30, expire_penalty=120, base_reward=20,
            linear_time_bonus_reward=3, order_id="123", is_bonus=True)
        non_bonus_order = Order(recipe=non_bonus_recipe, time_to_expire=30, expire_penalty=100, base_reward=30,
            linear_time_bonus_reward=4, order_id="234", is_bonus=False)

        orders_list = OrdersList([copy.deepcopy(bonus_order), copy.deepcopy(non_bonus_order)])
        mdp = OvercookedGridworld.from_layout_name("mdp_test", start_orders_list=orders_list, start_bonus_orders=[], start_all_orders=[])
        state = mdp.get_standard_start_state()

        for include_expire_penalty in [True, False]:
            error_msg = "include_expire_penalty was "+str(include_expire_penalty)
            self.assertEqual(non_bonus_order.calculate_reward()+non_bonus_order.expire_penalty*include_expire_penalty,
                mdp.get_recipe_value(state, non_bonus_recipe, include_expire_penalty=include_expire_penalty), error_msg)
            self.assertEqual(non_bonus_order.calculate_reward()+non_bonus_order.expire_penalty*include_expire_penalty,
                mdp.get_recipe_value(state, non_bonus_recipe, include_expire_penalty=include_expire_penalty), error_msg)
            self.assertEqual(0, mdp.get_recipe_value(state, non_bonus_recipe, include_expire_penalty=include_expire_penalty, steps_into_future=999), error_msg)
            self.assertEqual(0, mdp.get_recipe_value(state, non_bonus_recipe, include_expire_penalty=include_expire_penalty, steps_into_future=999, discounted=True, potential_params=potential_params), error_msg)
            self.assertEqual(bonus_order.calculate_reward()*mdp.order_bonus+bonus_order.expire_penalty*include_expire_penalty,
                mdp.get_recipe_value(state, bonus_recipe, include_expire_penalty=include_expire_penalty), error_msg)
            self.assertEqual(0, mdp.get_recipe_value(state, non_bonus_recipe, include_expire_penalty=include_expire_penalty, steps_into_future=999), error_msg)
            self.assertEqual(0, mdp.get_recipe_value(state, non_bonus_recipe, include_expire_penalty=include_expire_penalty, steps_into_future=999, discounted=True, potential_params=potential_params), error_msg)


            self.assertEqual(gamma**(pot_onion_steps+non_bonus_recipe.time) *
                    (non_bonus_order.calculate_future_reward(pot_onion_steps+non_bonus_recipe.time) + non_bonus_order.expire_penalty*include_expire_penalty),
                mdp.get_recipe_value(state, non_bonus_recipe, discounted=True, base_recipe=almost_non_bonus_recipe,
                potential_params=potential_params, include_expire_penalty=include_expire_penalty), error_msg)
            self.assertEqual(gamma**steps_into_future * (non_bonus_order.calculate_future_reward(steps_into_future) + non_bonus_order.expire_penalty*include_expire_penalty),
                mdp.get_recipe_value(state, non_bonus_recipe, discounted=True, base_recipe=almost_non_bonus_recipe,
                potential_params=potential_params, include_expire_penalty=include_expire_penalty, steps_into_future=steps_into_future), error_msg)
            self.assertEqual(gamma**(pot_tomato_steps+bonus_recipe.time) * (bonus_order.calculate_future_reward(pot_tomato_steps+bonus_recipe.time) * mdp.order_bonus + bonus_order.expire_penalty*include_expire_penalty),
                mdp.get_recipe_value(state, bonus_recipe, discounted=True, base_recipe=almost_bonus_recipe,
                potential_params=potential_params, include_expire_penalty=include_expire_penalty), error_msg)
            self.assertEqual(gamma**steps_into_future * (bonus_order.calculate_future_reward(steps_into_future)  * mdp.order_bonus + bonus_order.expire_penalty*include_expire_penalty),
                mdp.get_recipe_value(state, bonus_recipe, discounted=True, base_recipe=almost_bonus_recipe,
                potential_params=potential_params, include_expire_penalty=include_expire_penalty, steps_into_future=steps_into_future), error_msg)


    def test_get_optimal_possible_recipe_caching(self):
        # testing only if caching works properly
        potential_params = {"gamma": 0.99, "pot_onion_steps": 5, "pot_tomato_steps": 10} # these may not represent true values for mdp_test layout
        potential_params2 = {"gamma": 0.98, "pot_onion_steps": 5, "pot_tomato_steps": 10}
        temporary_order = Order(Recipe(["tomato", "onion", "onion"]), time_to_expire=6)
        pernament_order = Order(Recipe(["tomato", "onion", "tomato"]), time_to_expire=None)
        temporary_orders_list = OrdersList([copy.deepcopy(temporary_order)])
        assert temporary_orders_list.contains_temporary_orders
        assert not temporary_orders_list.is_adding_orders
        adding_orders_list =  OrdersList([copy.deepcopy(pernament_order)], orders_to_add=[copy.deepcopy(temporary_order)], add_new_order_every=10)
        assert adding_orders_list.is_adding_orders
        assert not adding_orders_list.contains_temporary_orders
        def get_cache_field(mdp, discounted):
            if discounted:
                return mdp._opt_recipe_discount_cache
            else:
                return mdp._opt_recipe_cache
        def set_cache_field(mdp, discounted, value):
            if discounted:
                mdp._opt_recipe_discount_cache = value
            else:
                mdp._opt_recipe_cache = value
        for discounted in [True, False]:
            error_msg = "discounted "+str(discounted)
            mdp = OvercookedGridworld.from_layout_name("mdp_test")
            state = mdp.get_standard_start_state()
            start_recipe = Recipe(["tomato"])
            self.assertCountEqual({}, get_cache_field(mdp, discounted), error_msg)
            _ = mdp.get_optimal_possible_recipe(state, start_recipe, discounted=discounted, potential_params=potential_params)
            self.assertTrue(get_cache_field(mdp, discounted), error_msg)
            set_val = {start_recipe: ("some_recipe", "some value")}
            set_cache_field(mdp, discounted, set_val)
            self.assertEqual(set_val[start_recipe][0], mdp.get_optimal_possible_recipe(state, start_recipe, discounted=discounted, potential_params=potential_params), "cache was not used, "+error_msg)
    
            if discounted:
                mdp._prev_potential_params = potential_params2
                self.assertNotEqual(set_val[start_recipe][0], mdp.get_optimal_possible_recipe(state, start_recipe, discounted=discounted, potential_params=potential_params), "cache was used when it should not, "+error_msg)

            mdp = OvercookedGridworld.from_layout_name("mdp_test", start_orders_list=temporary_orders_list, start_bonus_orders=[], start_all_orders=[])
            state = mdp.get_standard_start_state()
            _ = mdp.get_optimal_possible_recipe(state, start_recipe, discounted=discounted, potential_params=potential_params)
            self.assertCountEqual({}, get_cache_field(mdp, discounted), error_msg)

            mdp = OvercookedGridworld.from_layout_name("mdp_test", start_orders_list=adding_orders_list, start_bonus_orders=[], start_all_orders=[])
            state = mdp.get_standard_start_state()
            _ = mdp.get_optimal_possible_recipe(state, start_recipe, discounted=discounted, potential_params=potential_params)
            self.assertCountEqual({}, get_cache_field(mdp, discounted), error_msg)


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
    
    def test_is_terminal(self):
        mdp = self.base_mdp
        mdp.orders_to_end_episode = 3
        state = mdp.get_standard_start_state()
        recipe = state.orders_list.orders[0].recipe
        
        state.orders_list.fulfill_order(recipe)
        self.assertFalse(mdp.is_terminal(state)) # 1 >= 3
        state.orders_list.fulfill_order(recipe)
        self.assertFalse(mdp.is_terminal(state)) # 2 >= 3
        state.orders_list.fulfill_order(recipe)
        self.assertTrue(mdp.is_terminal(state)) # 3 >= 3
        state.orders_list.fulfill_order(recipe)
        self.assertTrue(mdp.is_terminal(state)) # 4 >= 3
        # 4 orders fulfilled now
        mdp.orders_to_end_episode = 100
        self.assertFalse(mdp.is_terminal(state))
        mdp.orders_to_end_episode = 0
        self.assertFalse(mdp.is_terminal(state))
        mdp.orders_to_end_episode = None
        self.assertFalse(mdp.is_terminal(state))
        mdp.orders_to_end_episode = 4
        self.assertTrue(mdp.is_terminal(state))


def random_joint_action():
    num_actions = len(Action.ALL_ACTIONS)
    a_idx0, a_idx1 = np.random.randint(low=0, high=num_actions, size=2)
    return (Action.INDEX_TO_ACTION[a_idx0], Action.INDEX_TO_ACTION[a_idx1])


class TestFeaturizations(unittest.TestCase):

    def setUp(self):
        self.base_mdp = OvercookedGridworld.from_layout_name("cramped_room")
        self.mlam = MediumLevelActionManager.from_pickle_or_compute(self.base_mdp, NO_COUNTERS_PARAMS, force_compute=True)
        self.env = OvercookedEnv.from_mdp(self.base_mdp, **DEFAULT_ENV_PARAMS)
        self.greedy_human_model_pair = AgentPair(GreedyHumanModel(self.mlam), GreedyHumanModel(self.mlam))
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
        trajs = self.env.get_rollouts(self.greedy_human_model_pair, num_games=5)
        featurized_observations = [[self.base_mdp.lossless_state_encoding(state) for state in ep_states] for ep_states in trajs["ep_states"]]
        
        pickle_path = os.path.join(TESTING_DATA_DIR, "test_lossless_state_featurization", "expected")

        # NOTE: If the featurizations are updated intentionally, you can overwrite the expected
        # featurizations by uncommenting the following line:
        # save_pickle(featurized_observations, pickle_path)
        expected_featurization = load_pickle(pickle_path)
        self.assertTrue(np.array_equal(expected_featurization, featurized_observations))

    def test_state_featurization(self):
        trajs = self.env.get_rollouts(self.greedy_human_model_pair, num_games=5)
        featurized_observations = [[self.base_mdp.featurize_state(state, self.mlam) for state in ep_states] for ep_states in trajs["ep_states"]]
        pickle_path = os.path.join(TESTING_DATA_DIR, "test_state_featurization", 'expected')
        # NOTE: If the featurizations are updated intentionally, you can overwrite the expected
        # featurizations by uncommenting the following line:
        # save_pickle(featurized_observations, pickle_path)
        expected_featurization = load_pickle(pickle_path)
        self.assertTrue(np.array_equal(expected_featurization, featurized_observations))

    def test_multi_hot_orders_encoding(self):
        state = self.base_mdp.get_standard_start_state()
        self.assertEqual(self.base_mdp.multi_hot_orders_encoding_single_agent(state), np.array([1.0]))
        self.assertEqual(self.base_mdp.multi_hot_orders_encoding(state), [np.array([1.0])]*2)
        self.assertEqual(self.base_mdp.multi_hot_orders_encoding_shape, (1,))

    def test_sparse_categorical_joint_action_encoding(self):
        joint_action = [Action.INTERACT, Action.STAY]
        print(self.base_mdp.sparse_categorical_joint_action_encoding_shape)
        self.assertEqual(self.base_mdp.sparse_categorical_joint_action_encoding(joint_action).tolist(), [[5], [4]])
        self.assertEqual(self.base_mdp.sparse_categorical_joint_action_encoding_shape.tolist(), [2, 1])

    def test_one_hot_joint_action_encoding(self):
        joint_action = [Action.INTERACT, Action.STAY]
        one_hot_action = [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]
        self.assertEqual(self.base_mdp.one_hot_joint_action_encoding(joint_action).tolist(), one_hot_action)
        self.assertEqual(self.base_mdp.one_hot_joint_action_encoding_shape.tolist(), [2, 6])

    def test_gym_spaces(self):
        space1 = self.base_mdp.multi_hot_orders_encoding_gym_space
        self.assertIsInstance(space1, gym.spaces.MultiBinary)

        space2 = self.base_mdp.lossless_state_encoding_gym_space
        self.assertIsInstance(space2, gym.spaces.Box)

        space3 = self.base_mdp.featurize_state_gym_space
        self.assertIsInstance(space3, gym.spaces.Box)


class TestOvercookedEnvironment(unittest.TestCase):

    def setUp(self):
        self.base_mdp = OvercookedGridworld.from_layout_name("cramped_room")
        self.env = OvercookedEnv.from_mdp(self.base_mdp, **DEFAULT_ENV_PARAMS)
        self.rnd_agent_pair = AgentPair(FixedPlanAgent([stay, w, w]), FixedPlanAgent([stay, e, e]))
        np.random.seed(0)

    def ids_independent_equal_envs(self, env1, env2):
        return env1.env_params == env2.env_params and env1.mdp.ids_independent_equal(env2.mdp)

    def test_constructor(self):
        try:
            OvercookedEnv.from_mdp(self.base_mdp, horizon=10)
        except Exception as e:
            self.fail("Failed to instantiate OvercookedEnv:\n{}".format(e))

        with self.assertRaises(TypeError):
            OvercookedEnv.from_mdp(self.base_mdp, **{"invalid_env_param": None})

    def test_init_from_trajectories_json(self):

        agent_eval = AgentEvaluator.from_mdp(self.base_mdp, DEFAULT_ENV_PARAMS)
        self.assertTrue(self.ids_independent_equal_envs(self.env, agent_eval.env))
        trajectory = agent_eval.evaluate_random_pair()
        env_from_traj = OvercookedEnv.from_trajectories_json(trajectory)
        self.assertTrue(self.ids_independent_equal_envs(self.env, env_from_traj))

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
