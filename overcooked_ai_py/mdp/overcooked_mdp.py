import itertools, copy
import numpy as np
from functools import reduce
from collections import defaultdict
from overcooked_ai_py.utils import pos_distance, load_from_json
from overcooked_ai_py import read_layout_dict
from overcooked_ai_py.mdp.actions import Action, Direction


class classproperty(property):
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()

class Recipe:

    TOMATO = 'tomato'
    ONION = 'onion'
    ALL_INGREDIENTS = [ONION, TOMATO]
    MAX_NUM_INGREDIENTS = 3

    ALL_RECIPES_CACHE = {}

    _computed = False
    
    def __new__(cls, ingredients):
        # Some basic argument verification
        if not ingredients or not hasattr(ingredients, '__iter__') or len(ingredients) == 0:
            raise ValueError("Invalid input recipe. Must be ingredients iterable with non-zero length")
        for elem in ingredients:
            if not elem in cls.ALL_INGREDIENTS:
                raise ValueError("Invalid ingredient: {0}. Recipe can only contain ingredients {1}".format(elem, cls.ALL_INGREDIENTS))
        if not len(ingredients) <= cls.MAX_NUM_INGREDIENTS:
            raise ValueError("Recipe of length {0} is invalid. Recipe can contain at most {1} ingredients".format(len(ingredients), cls.MAX_NUM_INGREDIENTS))
        key = hash(tuple(sorted(ingredients)))
        if key in cls.ALL_RECIPES_CACHE:
            return cls.ALL_RECIPES_CACHE[key]
        cls.ALL_RECIPES_CACHE[key] = super(Recipe, cls).__new__(cls)
        return cls.ALL_RECIPES_CACHE[key]

    def __init__(self, ingredients):
        self._ingredients = ingredients

    def __hash__(self):
        return hash(self.ingredients)

    def __eq__(self, other):
        return self.ingredients == other.ingredients

    def __repr__(self):
        return self.ingredients.__repr__()

    def __iter__(self):
        return iter(self.ingredients)

    def __copy__(self):
        return Recipe(self.ingredients)

    def __deepcopy__(self, memo):
        ingredients_cpy = copy.deepcopy(self.ingredients)
        return Recipe(ingredients_cpy)

    @classmethod
    def _compute_all_recipes(cls):
        for i in range(cls.MAX_NUM_INGREDIENTS):
            for ingredient_list in itertools.combinations_with_replacement(cls.ALL_INGREDIENTS, i + 1):
                cls(ingredient_list)

    @property
    def ingredients(self):
        return tuple(sorted(self._ingredients))

    @ingredients.setter
    def ingredients(self, _):
        raise AttributeError("Recpes are read-only. Do not modify instance attributes after creation")

    @property
    def value(self):
        # TODO
        if hasattr(self, '_delivery_reward'):
            return self._delivery_reward
        num_onions = len([ingredient for ingredient in self.ingredients if ingredient == self.ONION])
        num_tomatoes = len([ingredient for ingredient in self.ingredients if ingredient == self.TOMATO])
        return 3 * num_tomatoes + 2 * num_onions

    @property
    def time(self):
        # TODO
        if hasattr(self, '_cook_time'):
            return self._cook_time
        if hasattr(self, '_time_mapping') and self in self._time_mapping:
            return self._time_mapping[self]
        if hasattr(self, '_onion_time') and hasattr(self, '_tomato_time'):
            num_onions = len([ingredient for ingredient in self.ingredients if ingredient == self.ONION])
            num_tomatoes = len([ingredient for ingredient in self.ingredients if ingredient == self.TOMATO])
            return self._onion_time * num_onions + self._tomato_time * num_tomatoes
        return 20

    def to_dict(self):
        return { 'ingredients' : self.ingredients }

    @classproperty
    def ALL_RECIPES(cls):
        if not cls._computed:
            cls._compute_all_recipes()
            cls._computed = True
        return set(cls.ALL_RECIPES_CACHE.values())

    @classmethod
    def configure(cls, conf):
        cls._computed = False
        cls.MAX_NUM_INGREDIENTS = conf.get('max_num_ingredients', 3)

        # Backwards compatibility
        if 'cook_time' in conf:
            cls._cook_time = conf['cook_time']

        if 'delivery_reward' in conf:
            cls._delivery_reward = conf['delivery_reward']

        if 'recipe_values' in conf:
            if not 'all_orders' in conf:
                raise ValueError("Invalid recipe configuration")
            if not len(conf['all_orders']) == len(conf['recipe_values']):
                raise ValueError("Invalid recipe configuration")
            cls._value_mapping = {
                cls.from_dict(recipe) : value for (recipe, value) in zip(conf['all_orders'], conf['recipe_values'])
            }

        if 'recipe_times' in conf:
            if not 'all_orders' in conf:
                raise ValueError("Invalid recipe configuration")
            if not len(conf['all_orders']) == len(conf['recipe_times']):
                raise ValueError("Invalid recipe configuration")
            cls._time_mapping = {
                cls.from_dict(recipe) : time for (recipe, time) in zip(conf['all_orders'], conf['recipe_times'])
            }

        if 'tomato_time' in conf:
            cls._tomato_time = conf['tomato_time']

        if 'onion_time' in conf:
            cls._onion_time = conf['onion_time']

    @classmethod
    def from_dict(cls, obj_dict):
        return cls(**obj_dict)
        



class ObjectState(object):
    """
    State of an object in OvercookedGridworld.
    """

    def __init__(self, name, position, **kwargs):
        """
        name (str): The name of the object
        position (int, int): Tuple for the current location of the object.
        """
        self.name = name
        self._position = tuple(position)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, new_pos):
        self._position = new_pos

    def is_valid(self):
        return self.name in ['onion', 'tomato', 'dish']

    def deepcopy(self):
        return ObjectState(self.name, self.position)

    def __eq__(self, other):
        return isinstance(other, ObjectState) and \
            self.name == other.name and \
            self.position == other.position

    def __hash__(self):
        return hash((self.name, self.position))

    def __repr__(self):
        return '{}@{}'.format(
            self.name, self.position)

    def to_dict(self):
        return {
            "name": self.name,
            "position": self.position
        }

    @classmethod
    def from_dict(cls, obj_dict):
        obj_dict = copy.deepcopy(obj_dict)
        return ObjectState(**obj_dict)


class SoupState(ObjectState):


    def __init__(self, position, ingredients=[], cooking_tick=-1, **kwargs):
        """
        Represents a soup object. An object becomes a soup the instant it is placed in a pot. The
        soup's recipe is a list of ingredient names used to create it. A soup's recipe is undetermined
        until it has begun cooking. 

        name (str): always "soup" for a soup instance
        position (tupe): (x, y) coordinates in the grid
        ingrdients (list(ObjectState)): Objects that have been used to cook this soup. Determiens @property recipe
        cooking (int): How long the soup has been cooking for. -1 means cooking hasn't started yet
        """
        super(SoupState, self).__init__("soup", position)
        self._ingredients = ingredients
        self._cooking_tick = cooking_tick
        self._recipe = None

    def __eq__(self, other):
        return isinstance(other, SoupState) and self.name == other.name and self.position == other.position and self._cooking_tick == other._cooking_tick and \
            all([this_i == other_i for this_i, other_i in zip(self._ingredients, other._ingredients)])

    def __hash__(self):
        ingredient_hash = hash(tuple([hash(i) for i in self._ingredients]))
        supercls_hash = super(SoupState, self).__hash__()
        return hash((supercls_hash, self._cooking_tick, ingredient_hash))

    def __repr__(self):
        supercls_str = super(SoupState, self).__repr__()
        ingredients_str = self._ingredients.__repr__()
        return "{}\nIngredients:\t{}\nCooking Time:\t{}".format(supercls_str, ingredients_str, self._cooking_tick)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, new_pos):
        self._position = new_pos
        for ingredient in self._ingredients:
            ingredient.position = new_pos

    @property
    def ingredients(self):
        return [ingredient.name for ingredient in self._ingredients]

    @property
    def is_cooking(self):
        return not self.is_idle and not self.is_ready

    @property
    def recipe(self):
        if self.is_idle:
            raise ValueError("Recipe is not determined until soup begins cooking")
        if not self._recipe:
            self._recipe = Recipe(self.ingredients)
        return self._recipe

    @property
    def value(self):
        return self.recipe.value

    @property
    def cook_time(self):
        return self.recipe.time

    @property
    def is_ready(self):
        if self.is_idle:
            return False
        return self._cooking_tick >= self.cook_time

    @property
    def is_idle(self):
        return self._cooking_tick < 0

    @property
    def is_full(self):
        return not self.is_idle or len(self.ingredients) == Recipe.MAX_NUM_INGREDIENTS

    def is_valid(self):
        if not all([ingredient.position == self.position for ingredient in self._ingredients]):
            return False
        if len(self.ingredients) > Recipe.MAX_NUM_INGREDIENTS:
            return False
        return True

    def auto_finish(self):
        if len(self.ingredients) == 0:
            raise ValueError("Cannot finish soup with no ingredients")
        self._cooking_tick = 0
        self._cooking_tick = self.cook_time

    def add_ingredient(self, ingredient):
        if not ingredient.name in Recipe.ALL_INGREDIENTS:
            raise ValueError("Invalid ingredient")
        if self.is_full:
            raise ValueError("Reached maximum number of ingredients in recipe")
        ingredient.position = self.position
        self._ingredients.append(ingredient)

    def add_ingredient_from_str(self, ingredient_str):
        ingredient_obj = ObjectState(ingredient_str, self.position)
        self.add_ingredient(ingredient_obj)

    def pop_ingredient(self):
        if not self.is_idle:
            raise ValueError("Cannot remove an ingredient from this soup at this time")
        if len(self._ingredients) == 0:
            raise ValueError("No ingredient to remove")
        return self._ingredients.pop()

    def begin_cooking(self):
        if not self.is_idle:
            raise ValueError("Cannot begin cooking this soup at this time")
        if len(self.ingredients) == 0:
            raise ValueError("Must add at least one ingredient to soup before you can begin cooking")
        self._cooking_tick = 0

    def cook(self):
        if self.is_idle:
            raise ValueError("Must begin cooking before advancing cook tick")
        if self.is_ready:
            raise ValueError("Cannot cook a soup that is already done")
        self._cooking_tick += 1


    def deepcopy(self):
        return SoupState(self.position, [ingredient.deepcopy() for ingredient in self._ingredients], self._cooking_tick)
    
    def to_dict(self):
        info_dict = super(SoupState, self).to_dict()
        ingrdients_dict = [ingredient.to_dict() for ingredient in self._ingredients]
        info_dict['_ingredients'] = ingrdients_dict
        info_dict['_cooking_tick'] = self._cooking_tick
        info_dict['is_cooking'] = self.is_cooking
        info_dict['is_ready'] = self.is_ready
        info_dict['is_idle'] = self.is_ready
        info_dict['cook_time'] = -1 if self.is_idle else self.cook_time
        return info_dict

    @classmethod
    def from_dict(cls, obj_dict):
        obj_dict = copy.deepcopy(obj_dict)
        if obj_dict['name'] != 'soup':
            return super(SoupState, cls).from_dict(obj_dict)

        if 'state' in obj_dict:
            # Legacy soup representation
            ingredient, num_ingredient, time = obj_dict['state']
            cooking_tick = -1 if time == 0 else time
            finished = time >= 20
            if ingredient == Recipe.TOMATO:
                return SoupState.get_soup(obj_dict['position'], num_tomatoes=num_ingredient, cooking_tick=cooking_tick, finished=finished)
            else:
                return SoupState.get_soup(obj_dict['position'], num_onions=num_ingredient, cooking_tick=cooking_tick, finished=finished)

        ingredients_objs = [ObjectState.from_dict(ing_dict) for ing_dict in obj_dict['_ingredients']]
        obj_dict['_ingredients'] = ingredients_objs
        return cls(**obj_dict)

    @classmethod
    def get_soup(cls, position, num_onions=1, num_tomatoes=0, cooking_tick=-1, finished=False, **kwargs):
        if num_onions < 0 or num_tomatoes < 0:
            raise ValueError("Number of active ingredients must be positive")
        if num_onions + num_tomatoes > Recipe.MAX_NUM_INGREDIENTS:
            raise ValueError("Too many ingredients specified for this soup")
        if cooking_tick >= 0 and num_tomatoes + num_onions == 0:
            raise ValueError("_cooking_tick must be -1 for empty soup")
        if finished and num_tomatoes + num_onions == 0:
            raise ValueError("Empty soup cannot be finished")
        onions = [ObjectState(Recipe.ONION, position) for _ in range(num_onions)]
        tomatoes = [ObjectState(Recipe.TOMATO, position) for _ in range(num_tomatoes)]
        ingredients = onions + tomatoes
        soup = cls(position, ingredients, cooking_tick)
        if finished:
            soup.auto_finish()
        return soup
        


class PlayerState(object):
    """
    State of a player in OvercookedGridworld.

    position: (x, y) tuple representing the player's location.
    orientation: Direction.NORTH/SOUTH/EAST/WEST representing orientation.
    held_object: ObjectState representing the object held by the player, or
                 None if there is no such object.
    """
    def __init__(self, position, orientation, held_object=None):
        self.position = tuple(position)
        self.orientation = tuple(orientation)
        self.held_object = held_object

        assert self.orientation in Direction.ALL_DIRECTIONS
        if self.held_object is not None:
            assert isinstance(self.held_object, ObjectState)
            assert self.held_object.position == self.position

    @property
    def pos_and_or(self):
        return (self.position, self.orientation)

    def has_object(self):
        return self.held_object is not None

    def get_object(self):
        assert self.has_object()
        return self.held_object

    def set_object(self, obj):
        assert not self.has_object()
        obj.position = self.position
        self.held_object = obj
 
    def remove_object(self):
        assert self.has_object()
        obj = self.held_object
        self.held_object = None
        return obj
    
    def update_pos_and_or(self, new_position, new_orientation):
        self.position = new_position
        self.orientation = new_orientation
        if self.has_object():
            self.get_object().position = new_position

    def deepcopy(self):
        new_obj = None if self.held_object is None else self.held_object.deepcopy()
        return PlayerState(self.position, self.orientation, new_obj)

    def __eq__(self, other):
        return isinstance(other, PlayerState) and \
            self.position == other.position and \
            self.orientation == other.orientation and \
            self.held_object == other.held_object

    def __hash__(self):
        return hash((self.position, self.orientation, self.held_object))

    def __repr__(self):
        return '{} facing {} holding {}'.format(
            self.position, self.orientation, str(self.held_object))
    
    def to_dict(self):
        return {
            "position": self.position,
            "orientation": self.orientation,
            "held_object": self.held_object.to_dict() if self.held_object is not None else None
        }

    @staticmethod
    def from_dict(player_dict):
        player_dict = copy.deepcopy(player_dict)
        held_obj = player_dict["held_object"]
        if held_obj is not None:
            player_dict["held_object"] = ObjectState.from_dict(held_obj)
        return PlayerState(**player_dict)


class OvercookedState(object):
    """A state in OvercookedGridworld."""
    def __init__(self, players, objects, bonus_orders=[], all_orders=[], timestep=0, **kwargs):
        """
        players (list(PlayerState)): Currently active PlayerStates (index corresponds to number)
        objects (dict({tuple:list(ObjectState)})):  Dictionary mapping positions (x, y) to ObjectStates. 
            NOTE: Does NOT include objects held by players (they are in 
            the PlayerState objects).
        bonus_orders (list(dict)):   Current orders worth a bonus
        all_orders (list(dict)):     Current orders allowed at all
        timestep (int):  The current timestep of the state

        """
        bonus_orders = set([Recipe.from_dict(order) for order in bonus_orders])
        all_orders = set([Recipe.from_dict(order) for order in all_orders])
        for pos, obj in objects.items():
            assert obj.position == pos
        self.players = tuple(players)
        self.objects = objects
        self.bonus_orders = bonus_orders
        self._all_orders = all_orders
        self.timestep = timestep

    @property
    def player_positions(self):
        return tuple([player.position for player in self.players])

    @property
    def player_orientations(self):
        return tuple([player.orientation for player in self.players])

    @property
    def players_pos_and_or(self):
        """Returns a ((pos1, or1), (pos2, or2)) tuple"""
        return tuple(zip(*[self.player_positions, self.player_orientations]))

    @property
    def unowned_objects_by_type(self):
        """
        Returns dictionary of (obj_name: ObjState)
        for all objects in the environment, NOT including
        ones held by players.
        """
        objects_by_type = defaultdict(list)
        for _pos, obj in self.objects.items():
            objects_by_type[obj.name].append(obj)
        return objects_by_type

    @property
    def player_objects_by_type(self):
        """
        Returns dictionary of (obj_name: ObjState)
        for all objects held by players.
        """
        player_objects = defaultdict(list)
        for player in self.players:
            if player.has_object():
                player_obj = player.get_object()
                player_objects[player_obj.name].append(player_obj)
        return player_objects

    @property
    def all_objects_by_type(self):
        """
        Returns dictionary of (obj_name: ObjState)
        for all objects in the environment, including
        ones held by players.
        """
        all_objs_by_type = self.unowned_objects_by_type.copy()
        for obj_type, player_objs in self.player_objects_by_type.items():
            all_objs_by_type[obj_type].extend(player_objs)
        return all_objs_by_type

    @property
    def all_objects_list(self):
        all_objects_lists = list(self.all_objects_by_type.values()) + [[], []]
        return reduce(lambda x, y: x + y, all_objects_lists)

    @property
    def all_orders(self):
        return self._all_orders if self._all_orders else Recipe.ALL_RECIPES

    def has_object(self, pos):
        return pos in self.objects

    def get_object(self, pos):
        assert self.has_object(pos)
        return self.objects[pos]

    def add_object(self, obj, pos=None):
        if pos is None:
            pos = obj.position

        assert not self.has_object(pos)
        obj.position = pos
        self.objects[pos] = obj

    def remove_object(self, pos):
        assert self.has_object(pos)
        obj = self.objects[pos]
        del self.objects[pos]
        return obj

    @classmethod
    def from_players_pos_and_or(cls, players_pos_and_or, bonus_orders=set(), all_orders=set()):
        """
        Make a dummy OvercookedState with no objects based on the passed in player
        positions and orientations and order list
        """
        return cls(
            [PlayerState(*player_pos_and_or) for player_pos_and_or in players_pos_and_or], 
            objects={}, bonus_orders=bonus_orders, all_orders=all_orders)

    @classmethod
    def from_player_positions(cls, player_positions, bonus_orders=set(), all_orders=set()):
        """
        Make a dummy OvercookedState with no objects and with players facing
        North based on the passed in player positions and order list
        """
        dummy_pos_and_or = [(pos, Direction.NORTH) for pos in player_positions]
        return cls.from_players_pos_and_or(dummy_pos_and_or, bonus_orders, all_orders)

    def deepcopy(self):
        return OvercookedState(
            players=[player.deepcopy() for player in self.players],
            objects={pos:obj.deepcopy() for pos, obj in self.objects.items()}, 
            bonus_orders=[order.to_dict() for order in self.bonus_orders],
            all_orders=[order.to_dict() for order in self.all_orders],
            timestep=self.timestep)

    def time_independent_equal(self, other):
        order_lists_equal = self.all_orders == other.all_orders and self.bonus_orders == other.bonus_orders

        return isinstance(other, OvercookedState) and \
            self.players == other.players and \
            set(self.objects.items()) == set(other.objects.items()) and \
            order_lists_equal

    def __eq__(self, other):
        return self.time_independent_equal(other) and self.timestep == other.timestep

    def __hash__(self):
        order_list_hash = hash(tuple(sorted(self.bonus_orders))) + hash(tuple(sorted(self.all_orders)))
        return hash(
            (self.players, tuple(self.objects.values()), order_list_hash)
        )

    def __str__(self):
        return 'Players: {}, Objects: {}, Bonus orders: {} All orders: {} Timestep: {}'.format( 
            str(self.players), str(list(self.objects.values())), str(self.bonus_orders), str(self.all_orders), str(self.timestep))

    def to_dict(self):
        return {
            "players": [p.to_dict() for p in self.players],
            "objects": [obj.to_dict() for obj in self.objects.values()],
            "bonus_orders": [order.to_dict() for order in self.bonus_orders],
            "all_orders" : [order.to_dict() for order in self.all_orders]
        }

    @staticmethod
    def from_dict(state_dict):
        state_dict = copy.deepcopy(state_dict)
        state_dict["players"] = [PlayerState.from_dict(p) for p in state_dict["players"]]
        object_list = [SoupState.from_dict(o) for o in state_dict["objects"]]
        state_dict["objects"] = { ob.position : ob for ob in object_list }
        state_dict["bonus_orders"] = [Recipe.from_dict(order) for order in state_dict.get('bonus_orders', [])]
        state_dict['all_orders'] = [Recipe.from_dict(order) for order in state_dict.get('all_orders', [])]
        return OvercookedState(**state_dict)
    
    @staticmethod
    def from_json(filename):
        return load_from_json(filename)


BASE_REW_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0
}

EVENT_TYPES = [
    # Onion events
    'onion_pickup',
    'useful_onion_pickup',
    'onion_drop',
    'useful_onion_drop',
    'potting_onion',

    # Dish events
    'dish_pickup',
    'useful_dish_pickup',
    'dish_drop',
    'useful_dish_drop',

    # Soup events
    'soup_pickup',
    'soup_delivery',
    'soup_drop'
]

class OvercookedGridworld(object):
    """
    An MDP grid world based off of the Overcooked game.
    TODO: clean the organization of this class further.
    """


    #########################
    # INSTANTIATION METHODS #
    #########################

    def __init__(self, terrain, start_player_positions, start_bonus_orders=[], rew_shaping_params=None, layout_name="unnamed_layout", start_all_orders=[r.to_dict() for r in Recipe.ALL_RECIPES], num_items_for_soup=3, order_bonus=2, **kwargs):
        """
        terrain: a matrix of strings that encode the MDP layout
        layout_name: string identifier of the layout
        start_player_positions: tuple of positions for both players' starting positions
        start_bonus_orders: List of recipes dicts that are worth a bonus 
        rew_shaping_params: reward given for completion of specific subgoals
        all_orders: List of all available order dicts the players can make
        num_items_for_soup: Maximum number of ingredients that can be placed in a soup
        order_bonus: Multiplicative factor for serving a bonus recipe
        """
        self.height = len(terrain)
        self.width = len(terrain[0])
        self.shape = (self.width, self.height)
        self.terrain_mtx = terrain
        self.terrain_pos_dict = self._get_terrain_type_pos_dict()
        self.start_player_positions = start_player_positions
        self.num_players = len(start_player_positions)
        self.start_bonus_orders = start_bonus_orders
        self.start_all_orders = start_all_orders
        self.reward_shaping_params = BASE_REW_SHAPING_PARAMS if rew_shaping_params is None else rew_shaping_params
        self.layout_name = layout_name
        self.order_bonus = order_bonus
        self._configure_recipes(start_all_orders, num_items_for_soup, **kwargs)

    @staticmethod
    def from_layout_name(layout_name, **params_to_overwrite):
        """
        Generates a OvercookedGridworld instance from a layout file.

        One can overwrite the default mdp configuration using partial_mdp_config.
        """
        params_to_overwrite = params_to_overwrite.copy()
        base_layout_params = read_layout_dict(layout_name)

        grid = base_layout_params['grid']
        del base_layout_params['grid']
        base_layout_params['layout_name'] = layout_name

        # Clean grid
        grid = [layout_row.strip() for layout_row in grid.split("\n")]
        return OvercookedGridworld.from_grid(grid, base_layout_params, params_to_overwrite)

    @staticmethod
    def from_grid(layout_grid, base_layout_params={}, params_to_overwrite={}, debug=False):
        """
        Returns instance of OvercookedGridworld with terrain and starting 
        positions derived from layout_grid.
        One can override default configuration parameters of the mdp in
        partial_mdp_config.
        """
        mdp_config = base_layout_params.copy()

        layout_grid = [[c for c in row] for row in layout_grid]
        OvercookedGridworld._assert_valid_grid(layout_grid)

        player_positions = [None] * 9
        for y, row in enumerate(layout_grid):
            for x, c in enumerate(row):
                if c in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    layout_grid[y][x] = ' '

                    # -1 is to account for fact that player indexing starts from 1 rather than 0
                    assert player_positions[int(c) - 1] is None, 'Duplicate player in grid'
                    player_positions[int(c) - 1] = (x, y)

        num_players = len([x for x in player_positions if x is not None])
        player_positions = player_positions[:num_players]

        # After removing player positions from grid we have a terrain mtx
        mdp_config["terrain"] = layout_grid
        mdp_config["start_player_positions"] = player_positions

        for k, v in params_to_overwrite.items():
            curr_val = mdp_config.get(k, None)
            if debug:
                print("Overwriting mdp layout standard config value {}:{} -> {}".format(k, curr_val, v))
            mdp_config[k] = v

        return OvercookedGridworld(**mdp_config)


    def _configure_recipes(self, start_all_orders, num_items_for_soup, **kwargs):
        Recipe.configure({'num_items_for_soup' : num_items_for_soup, "all_orders" : start_all_orders, **kwargs })

    #####################
    # BASIC CLASS UTILS #
    #####################

    def __eq__(self, other):
        return np.array_equal(self.terrain_mtx, other.terrain_mtx) and \
                self.start_player_positions == other.start_player_positions and \
                self.start_bonus_orders == other.start_bonus_orders and \
                self.start_all_orders == other.start_all_orders and \
                self.reward_shaping_params == other.reward_shaping_params and \
                self.layout_name == other.layout_name
    
    def copy(self):
        return OvercookedGridworld(
            terrain=self.terrain_mtx.copy(),
            start_player_positions=self.start_player_positions,
            start_bonus_orders=self.start_bonus_orders,
            rew_shaping_params=copy.deepcopy(self.reward_shaping_params),
            layout_name=self.layout_name,
            start_all_orders=self.start_all_orders
        )

    @property
    def mdp_params(self):
        return {
            "layout_name": self.layout_name,
            "terrain": self.terrain_mtx,
            "start_player_positions": self.start_player_positions,
            "start_bonus_orders": self.start_bonus_orders,
            "rew_shaping_params": copy.deepcopy(self.reward_shaping_params),
            "start_all_orders" : self.start_all_orders
        }


    ##############
    # GAME LOGIC #
    ##############

    def get_actions(self, state):
        """
        Returns the list of lists of valid actions for 'state'.

        The ith element of the list is the list of valid actions that player i
        can take.
        """
        self._check_valid_state(state)
        return [self._get_player_actions(state, i) for i in range(len(state.players))]

    def _get_player_actions(self, state, player_num):
        """All actions are allowed to all players in all states."""
        return Action.ALL_ACTIONS

    def _check_action(self, state, joint_action):
        for p_action, p_legal_actions in zip(joint_action, self.get_actions(state)):
            if p_action not in p_legal_actions:
                raise ValueError('Invalid action')

    def get_standard_start_state(self):
        start_state = OvercookedState.from_player_positions(
            self.start_player_positions, bonus_orders=self.start_bonus_orders, all_orders=self.start_all_orders
        )
        return start_state

    def get_random_start_state_fn(self, random_start_pos=False, rnd_obj_prob_thresh=0.0):
        def start_state_fn():
            if random_start_pos:
                valid_positions = self.get_valid_joint_player_positions()
                start_pos = valid_positions[np.random.choice(len(valid_positions))]
            else:
                start_pos = self.start_player_positions

            start_state = OvercookedState.from_player_positions(start_pos, bonus_orders=self.start_bonus_orders, all_orders=self.start_all_orders)

            if rnd_obj_prob_thresh == 0:
                return start_state

            # Arbitrary hard-coding for randomization of objects
            # For each pot, add a random amount of onions and tomatoes with prob rnd_obj_prob_thresh
            # Begin the soup cooking with probability rnd_obj_prob_thresh
            pots = self.get_pot_states(start_state)["empty"]
            for pot_loc in pots:
                p = np.random.rand()
                if p < rnd_obj_prob_thresh:
                    n = int(np.random.randint(low=1, high=4))
                    m = int(np.random.randint(low=0, high=4-n))
                    q = np.random.rand()
                    cooking_tick = 0 if q < rnd_obj_prob_thresh else -1
                    start_state.objects[pot_loc] = SoupState.get_soup(pot_loc, num_onions=n, num_tomatoes=m, cooking_tick=cooking_tick)

            # For each player, add a random object with prob rnd_obj_prob_thresh
            for player in start_state.players:
                p = np.random.rand()
                if p < rnd_obj_prob_thresh:
                    # Different objects have different probabilities
                    obj = np.random.choice(["dish", "onion", "soup"], p=[0.2, 0.6, 0.2])
                    n = int(np.random.randint(low=1, high=4))
                    m = int(np.random.randint(low=0, high=4-n))
                    if obj == "soup":
                        player.set_object(
                            SoupState.get_soup(player.position, num_onions=n, num_tomatoes=m, finished=True)
                        )
                    else:
                        player.set_object(ObjectState(obj, player.position))
            return start_state
        return start_state_fn

    def is_terminal(self, state):
        # There is a finite horizon, handled by the environment.
        return False

    def get_state_transition(self, state, joint_action):
        """Gets information about possible transitions for the action.

        Returns the next state, sparse reward and reward shaping.
        Assumes all actions are deterministic.

        NOTE: Sparse reward is given only when soups are delivered, 
        shaped reward is given only for completion of subgoals 
        (not soup deliveries).
        """
        events_infos = { event : [False] * self.num_players for event in EVENT_TYPES }
        phi_s = self.potential_function(state)

        assert not self.is_terminal(state), "Trying to find successor of a terminal state: {}".format(state)
        for action, action_set in zip(joint_action, self.get_actions(state)):
            if action not in action_set:
                raise ValueError("Illegal action %s in state %s" % (action, state))
        
        new_state = state.deepcopy()

        # Resolve interacts first
        sparse_reward_by_agent, shaped_reward_by_agent = self.resolve_interacts(new_state, joint_action, events_infos)

        assert new_state.player_positions == state.player_positions
        assert new_state.player_orientations == state.player_orientations
        
        # Resolve player movements
        self.resolve_movement(new_state, joint_action)

        # Finally, environment effects
        self.step_environment_effects(new_state)

        # Additional dense reward logic
        # shaped_reward += self.calculate_distance_based_shaped_reward(state, new_state)

        phi_s_prime = self.potential_function(new_state)
        infos = {
            "event_infos": events_infos,
            "sparse_reward_by_agent": sparse_reward_by_agent,
            "shaped_reward_by_agent": shaped_reward_by_agent,
            "phi_s": phi_s,
            "phi_s_prime": phi_s_prime
        }
        return new_state, infos

    def resolve_interacts(self, new_state, joint_action, events_infos):
        """
        Resolve any INTERACT actions, if present.

        Currently if two players both interact with a terrain, we resolve player 1's interact 
        first and then player 2's, without doing anything like collision checking.
        """
        pot_states = self.get_pot_states(new_state)
        # We divide reward by agent to keep track of who contributed
        sparse_reward, shaped_reward = [0] * self.num_players, [0] * self.num_players 

        for player_idx, (player, action) in enumerate(zip(new_state.players, joint_action)):

            if action != Action.INTERACT:
                continue

            pos, o = player.position, player.orientation
            i_pos = Action.move_in_direction(pos, o)
            terrain_type = self.get_terrain_type_at_pos(i_pos)

            # NOTE: we always log pickup/drop before performing it, as that's
            # what the logic of determining whether the pickup/drop is useful assumes
            if terrain_type == 'X':

                if player.has_object() and not new_state.has_object(i_pos):
                    obj_name = player.get_object().name
                    self.log_object_drop(events_infos, new_state, obj_name, pot_states, player_idx)

                    # Drop object on counter
                    obj = player.remove_object()
                    new_state.add_object(obj, i_pos)
                    
                elif not player.has_object() and new_state.has_object(i_pos):
                    obj_name = new_state.get_object(i_pos).name
                    self.log_object_pickup(events_infos, new_state, obj_name, pot_states, player_idx)

                    # Pick up object from counter
                    obj = new_state.remove_object(i_pos)
                    player.set_object(obj)
                    

            elif terrain_type == 'O' and player.held_object is None:
                self.log_object_pickup(events_infos, new_state, "onion", pot_states, player_idx)

                # Onion pickup from dispenser
                obj = ObjectState('onion', pos)
                player.set_object(obj)

            elif terrain_type == 'T' and player.held_object is None:
                # Tomato pickup from dispenser
                player.set_object(ObjectState('tomato', pos))

            elif terrain_type == 'D' and player.held_object is None:
                self.log_object_pickup(events_infos, new_state, "dish", pot_states, player_idx)

                # Give shaped reward if pickup is useful
                if self.is_dish_pickup_useful(new_state, pot_states):
                    shaped_reward[player_idx] += self.reward_shaping_params["DISH_PICKUP_REWARD"]

                # Perform dish pickup from dispenser
                obj = ObjectState('dish', pos)
                player.set_object(obj)

            elif terrain_type == 'P' and not player.has_object():
                # Cooking soup
                if self.soup_to_be_cooked_at_location(new_state, i_pos):
                    soup = new_state.get_object(i_pos)
                    soup.begin_cooking()
            
            elif terrain_type == 'P' and player.has_object():

                if player.get_object().name == 'dish' and self.soup_ready_at_location(new_state, i_pos):
                    self.log_object_pickup(events_infos, new_state, "soup", pot_states, player_idx)

                    # Pick up soup
                    player.remove_object() # Remove the dish
                    obj = new_state.remove_object(i_pos) # Get soup
                    player.set_object(obj)
                    shaped_reward[player_idx] += self.reward_shaping_params["SOUP_PICKUP_REWARD"]

                elif player.get_object().name in Recipe.ALL_INGREDIENTS:
                    # Adding ingredient to soup

                    if not new_state.has_object(i_pos):
                        # Pot was empty, add soup to it
                        new_state.add_object(SoupState(i_pos, ingredients=[]))

                    # Add ingredient if possible
                    soup = new_state.get_object(i_pos)
                    if not soup.is_full:
                        obj = player.remove_object()
                        soup.add_ingredient(obj)
                        shaped_reward[player_idx] += self.reward_shaping_params["PLACEMENT_IN_POT_REW"]

                        # Log onion potting
                        if obj.name == Recipe.ONION:
                            events_infos['potting_onion'][player_idx] = True

            elif terrain_type == 'S' and player.has_object():
                obj = player.get_object()
                if obj.name == 'soup':

                    delivery_rew = self.deliver_soup(new_state, player, obj)
                    sparse_reward[player_idx] += delivery_rew

                    # Log soup delivery
                    events_infos['soup_delivery'][player_idx] = True                        

        return sparse_reward, shaped_reward

    def deliver_soup(self, state, player, soup):
        """
        Deliver the soup, and get reward if there is no order list
        or if the type of the delivered soup matches the next order.
        """
        assert soup.name == 'soup', "Tried to deliver something that wasn't soup"
        assert soup.is_ready, "Tried to deliever soup that isn't ready"
        player.remove_object()

        if not soup.recipe in state.all_orders:
            return 0

        if not soup.recipe in state.bonus_orders:
            return soup.value

        return self.order_bonus * soup.value


    def resolve_movement(self, state, joint_action):
        """Resolve player movement and deal with possible collisions"""
        new_positions, new_orientations = self.compute_new_positions_and_orientations(state.players, joint_action)
        for player_state, new_pos, new_o in zip(state.players, new_positions, new_orientations):
            player_state.update_pos_and_or(new_pos, new_o)

    def compute_new_positions_and_orientations(self, old_player_states, joint_action):
        """Compute new positions and orientations ignoring collisions"""
        new_positions, new_orientations = list(zip(*[
            self._move_if_direction(p.position, p.orientation, a) \
            for p, a in zip(old_player_states, joint_action)]))
        old_positions = tuple(p.position for p in old_player_states)
        new_positions = self._handle_collisions(old_positions, new_positions)
        return new_positions, new_orientations

    def is_transition_collision(self, old_positions, new_positions):
        # Checking for any players ending in same square
        if self.is_joint_position_collision(new_positions):
            return True
        # Check if any two players crossed paths
        for idx0, idx1 in itertools.combinations(range(self.num_players), 2):
            p1_old, p2_old = old_positions[idx0], old_positions[idx1]
            p1_new, p2_new = new_positions[idx0], new_positions[idx1]
            if p1_new == p2_old and p1_old == p2_new:
                return True
        return False

    def is_joint_position_collision(self, joint_position):
        return any(pos0 == pos1 for pos0, pos1 in itertools.combinations(joint_position, 2))
            
    def step_environment_effects(self, state):
        state.timestep += 1
        for obj in state.objects.values():
            if obj.name == 'soup' and obj.is_cooking:
                obj.cook()
                

    def _handle_collisions(self, old_positions, new_positions):
        """If agents collide, they stay at their old locations"""
        if self.is_transition_collision(old_positions, new_positions):
            return old_positions
        return new_positions

    def _get_terrain_type_pos_dict(self):
        pos_dict = defaultdict(list)
        for y, terrain_row in enumerate(self.terrain_mtx):
            for x, terrain_type in enumerate(terrain_row):
                pos_dict[terrain_type].append((x, y))
        return pos_dict

    def _move_if_direction(self, position, orientation, action):
        """Returns position and orientation that would 
        be obtained after executing action"""
        if action not in Action.MOTION_ACTIONS:
            return position, orientation
        new_pos = Action.move_in_direction(position, action)
        new_orientation = orientation if action == Action.STAY else action
        if new_pos not in self.get_valid_player_positions():
            return position, new_orientation
        return new_pos, new_orientation


    #######################
    # LAYOUT / STATE INFO #
    #######################

    def get_valid_player_positions(self):
        return self.terrain_pos_dict[' ']

    def get_valid_joint_player_positions(self):
        """Returns all valid tuples of the form (p0_pos, p1_pos, p2_pos, ...)"""
        valid_positions = self.get_valid_player_positions() 
        all_joint_positions = list(itertools.product(valid_positions, repeat=self.num_players))
        valid_joint_positions = [j_pos for j_pos in all_joint_positions if not self.is_joint_position_collision(j_pos)]
        return valid_joint_positions

    def get_valid_player_positions_and_orientations(self):
        valid_states = []
        for pos in self.get_valid_player_positions():
            valid_states.extend([(pos, d) for d in Direction.ALL_DIRECTIONS])
        return valid_states

    def get_valid_joint_player_positions_and_orientations(self):
        """All joint player position and orientation pairs that are not
        overlapping and on empty terrain."""
        valid_player_states = self.get_valid_player_positions_and_orientations()

        valid_joint_player_states = []
        for players_pos_and_orientations in itertools.product(valid_player_states, repeat=self.num_players):
            joint_position = [plyer_pos_and_or[0] for plyer_pos_and_or in players_pos_and_orientations]
            if not self.is_joint_position_collision(joint_position):
                valid_joint_player_states.append(players_pos_and_orientations)

        return valid_joint_player_states

    def get_adjacent_features(self, player):
        adj_feats = []
        pos = player.position
        for d in Direction.ALL_DIRECTIONS:
            adj_pos = Action.move_in_direction(pos, d)
            adj_feats.append((adj_pos, self.get_terrain_type_at_pos(adj_pos)))
        return adj_feats

    def get_terrain_type_at_pos(self, pos):
        x, y = pos
        return self.terrain_mtx[y][x]

    def get_dish_dispenser_locations(self):
        return list(self.terrain_pos_dict['D'])

    def get_onion_dispenser_locations(self):
        return list(self.terrain_pos_dict['O'])

    def get_tomato_dispenser_locations(self):
        return list(self.terrain_pos_dict['T'])

    def get_serving_locations(self):
        return list(self.terrain_pos_dict['S'])

    def get_pot_locations(self):
        return list(self.terrain_pos_dict['P'])

    def get_counter_locations(self):
        return list(self.terrain_pos_dict['X'])

    @property
    def num_pots(self):
        return len(self.get_pot_locations())

    def get_pot_states(self, state):
        """Returns dict with structure:
        {
         empty: [positions of empty pots]
        'x_items': [soup objects with x items that have yet to start cooking],
        'cooking': [soup objs that are cooking but not ready]
        'ready': [ready soup objs],
        }
        NOTE: all returned pots are just pot positions
        """
        pots_states_dict = defaultdict(list)
        for pot_pos in self.get_pot_locations():
            if not state.has_object(pot_pos):
                pots_states_dict['empty'].append(pot_pos)
            else:
                soup = state.get_object(pot_pos)
                if soup.is_ready:
                    pots_states_dict['ready'].append(pot_pos)
                elif soup.is_cooking:
                    pots_states_dict['cooking'].append(pot_pos)
                else:
                    num_ingredients = len(soup.ingredients)
                    pots_states_dict['{}_items'.format(num_ingredients)].append(pot_pos)

        return pots_states_dict

    def get_counter_objects_dict(self, state, counter_subset=None):
        """Returns a dictionary of pos:objects on counters by type"""
        counters_considered = self.terrain_pos_dict['X'] if counter_subset is None else counter_subset
        counter_objects_dict = defaultdict(list)
        for obj in state.objects.values():
            if obj.position in counters_considered:
                counter_objects_dict[obj.name].append(obj.position)
        return counter_objects_dict

    def get_empty_counter_locations(self, state):
        counter_locations = self.get_counter_locations()
        return [pos for pos in counter_locations if not state.has_object(pos)]

    def get_empty_pots(self, pot_states):
        """Returns pots that have 0 items in them"""
        return pot_states["empty"]

    def get_ready_pots(self, pot_states):
        return pot_states['ready']

    def get_cooking_pots(self, pot_states):
        return pot_states['cooking']

    def get_full_but_not_cooking_pots(self, pot_states):
        return pot_states['{}_items'.format(Recipe.MAX_NUM_INGREDIENTS)]


    def get_full_pots(self, pot_states):
        return self.get_cooking_pots(pot_states) + self.get_ready_pots(pot_states) + self.get_full_but_not_cooking_pots(pot_states)

    def get_partially_full_pots(self, pot_states):
        return list(set().union(*[pot_states['{}_items'.format(i)] for i in range(1, Recipe.MAX_NUM_INGREDIENTS)]))

    def soup_ready_at_location(self, state, pos):
        if not state.has_object(pos):
            return False
        obj = state.get_object(pos)
        assert obj.name == 'soup', 'Object in pot was not soup'
        return obj.is_ready

    def soup_to_be_cooked_at_location(self, state, pos):
        if not state.has_object(pos):
            return False
        obj = state.get_object(pos)
        return obj.name == 'soup' and not obj.is_cooking and not obj.is_ready and len(obj.ingredients) > 0

    def _check_valid_state(self, state):
        """Checks that the state is valid.

        Conditions checked:
        - Players are on free spaces, not terrain
        - Held objects have the same position as the player holding them
        - Non-held objects are on terrain
        - No two players or non-held objects occupy the same position
        - Objects have a valid state (eg. no pot with 4 onions)
        """
        all_objects = list(state.objects.values())
        for player_state in state.players:
            # Check that players are not on terrain
            pos = player_state.position
            assert pos in self.get_valid_player_positions()

            # Check that held objects have the same position
            if player_state.held_object is not None:
                all_objects.append(player_state.held_object)
                assert player_state.held_object.position == player_state.position

        for obj_pos, obj_state in state.objects.items():
            # Check that the hash key position agrees with the position stored
            # in the object state
            assert obj_state.position == obj_pos
            # Check that non-held objects are on terrain
            assert self.get_terrain_type_at_pos(obj_pos) != ' '

        # Check that players and non-held objects don't overlap
        all_pos = [player_state.position for player_state in state.players]
        all_pos += [obj_state.position for obj_state in state.objects.values()]
        assert len(all_pos) == len(set(all_pos)), "Overlapping players or objects"

        # Check that objects have a valid state
        for obj_state in all_objects:
            assert obj_state.is_valid()

    def find_free_counters_valid_for_both_players(self, state, mlp):
        """Finds all empty counter locations that are accessible to both players"""
        one_player, other_player = state.players
        free_counters = self.get_empty_counter_locations(state)
        free_counters_valid_for_both = []
        for free_counter in free_counters:
            goals = mlp.mp.motion_goals_for_pos[free_counter]
            if any([mlp.mp.is_valid_motion_start_goal_pair(one_player.pos_and_or, goal) for goal in goals]) and \
            any([mlp.mp.is_valid_motion_start_goal_pair(other_player.pos_and_or, goal) for goal in goals]):
                free_counters_valid_for_both.append(free_counter)
        return free_counters_valid_for_both

    @staticmethod
    def _assert_valid_grid(grid):
        """Raises an AssertionError if the grid is invalid.

        grid:  A sequence of sequences of spaces, representing a grid of a
        certain height and width. grid[y][x] is the space at row y and column
        x. A space must be either 'X' (representing a counter), ' ' (an empty
        space), 'O' (onion supply), 'P' (pot), 'D' (dish supply), 'S' (serving
        location), '1' (player 1) and '2' (player 2).
        """
        height = len(grid)
        width = len(grid[0])

        # Make sure the grid is not ragged
        assert all(len(row) == width for row in grid), 'Ragged grid'

        # Borders must not be free spaces
        def is_not_free(c):
            return c in 'XOPDST'

        for y in range(height):
            assert is_not_free(grid[y][0]), 'Left border must not be free'
            assert is_not_free(grid[y][-1]), 'Right border must not be free'
        for x in range(width):
            assert is_not_free(grid[0][x]), 'Top border must not be free'
            assert is_not_free(grid[-1][x]), 'Bottom border must not be free'

        all_elements = [element for row in grid for element in row]
        digits = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        layout_digits = [e for e in all_elements if e in digits]
        num_players = len(layout_digits)
        assert num_players > 0, "No players (digits) in grid"
        layout_digits = list(sorted(map(int, layout_digits)))
        assert layout_digits == list(range(1, num_players + 1)), "Some players were missing"

        assert all(c in 'XOPDST123456789 ' for c in all_elements), 'Invalid character in grid'
        assert all_elements.count('1') == 1, "'1' must be present exactly once"
        assert all_elements.count('D') >= 1, "'D' must be present at least once"
        assert all_elements.count('S') >= 1, "'S' must be present at least once"
        assert all_elements.count('P') >= 1, "'P' must be present at least once"
        assert all_elements.count('O') >= 1 or all_elements.count('T') >= 1, "'O' or 'T' must be present at least once"


    ################################
    # EVENT LOGGING HELPER METHODS #
    ################################

    def log_object_pickup(self, events_infos, state, obj_name, pot_states, player_index):
        """Player picked an object up from a counter or a dispenser"""
        obj_pickup_key = obj_name + "_pickup"
        if obj_pickup_key not in events_infos:
            # TODO: add support for tomato event logging
            if obj_name == "tomato":
                return
            raise ValueError("Unknown event {}".format(obj_pickup_key))
        events_infos[obj_pickup_key][player_index] = True
        
        USEFUL_PICKUP_FNS = {
            "onion": self.is_onion_pickup_useful,
            "dish": self.is_dish_pickup_useful
        }
        if obj_name in USEFUL_PICKUP_FNS:
            if USEFUL_PICKUP_FNS[obj_name](state, pot_states, player_index):
                obj_useful_key = "useful_" + obj_name + "_pickup"
                events_infos[obj_useful_key][player_index] = True

    def log_object_drop(self, events_infos, state, obj_name, pot_states, player_index):
        """Player dropped the object on a counter"""
        obj_drop_key = obj_name + "_drop"
        if obj_drop_key not in events_infos:
            # TODO: add support for tomato event logging
            if obj_name == "tomato":
                return
            raise ValueError("Unknown event {}".format(obj_drop_key))
        events_infos[obj_drop_key][player_index] = True
        
        USEFUL_DROP_FNS = {
            "onion": self.is_onion_drop_useful,
            "dish": self.is_dish_drop_useful
        }
        if obj_name in USEFUL_DROP_FNS:
            if USEFUL_DROP_FNS[obj_name](state, pot_states, player_index):
                obj_useful_key = "useful_" + obj_name + "_drop"
                events_infos[obj_useful_key][player_index] = True

    def is_dish_pickup_useful(self, state, pot_states, player_index=None):
        """
        NOTE: this only works if self.num_players == 2
        Useful if:
        - Pot is ready/cooking and there is no player with a dish               \ 
        - 2 pots are ready/cooking and there is one player with a dish          | -> number of dishes in players hands < number of ready/cooking/partially full soups 
        - Partially full pot is ok if the other player is on course to fill it  /

        We also want to prevent picking up and dropping dishes, so add the condition
        that there must be no dishes on counters
        """
        if self.num_players != 2: return False

        # This next line is to prevent reward hacking (this logic is also used by reward shaping)
        dishes_on_counters = self.get_counter_objects_dict(state)["dish"]
        no_dishes_on_counters = len(dishes_on_counters) == 0

        num_player_dishes = len(state.player_objects_by_type['dish'])
        non_empty_pots = len(self.get_ready_pots(pot_states) + self.get_cooking_pots(pot_states) + self.get_partially_full_pots(pot_states))
        return no_dishes_on_counters and num_player_dishes < non_empty_pots

    def is_dish_drop_useful(self, state, pot_states, player_index):
        """
        NOTE: this only works if self.num_players == 2
        Useful if:
        - Onion is needed (all pots are non-full)
        - Nobody is holding onions
        """
        if self.num_players != 2: return False
        all_non_full = len(self.get_full_pots(pot_states)) == 0
        other_player = state.players[1 - player_index]
        other_player_holding_onion = other_player.has_object() and other_player.get_object().name == "onion"
        return all_non_full and not other_player_holding_onion

    def is_onion_pickup_useful(self, state, pot_states, player_index):
        """
        NOTE: this only works if self.num_players == 2
        Always useful unless:
        - All pots are full & other agent is not holding a dish
        """
        if self.num_players != 2: return False
        all_pots_full = self.num_pots == len(self.get_full_pots(pot_states))
        other_player = state.players[1 - player_index]
        other_player_has_dish = other_player.has_object() and other_player.get_object().name == "dish"
        return not (all_pots_full and not other_player_has_dish)
        
    def is_onion_drop_useful(self, state, pot_states, player_index):
        """
        NOTE: this only works if self.num_players == 2
        Useful if:
        - Dish is needed (all pots are full)
        - Nobody is holding a dish
        """
        if self.num_players != 2: return False
        all_pots_full = len(self.get_full_pots(pot_states)) == self.num_pots
        other_player = state.players[1 - player_index]
        other_player_holding_dish = other_player.has_object() and other_player.get_object().name == "dish"
        return all_pots_full and not other_player_holding_dish


    #####################
    # TERMINAL GRAPHICS #
    #####################

    def state_string(self, state):
        """String representation of the current state"""
        players_dict = {player.position: player for player in state.players}

        grid_string = ""
        for y, terrain_row in enumerate(self.terrain_mtx):
            for x, element in enumerate(terrain_row):
                if (x, y) in players_dict.keys():
                    player = players_dict[(x, y)]
                    orientation = player.orientation
                    assert orientation in Direction.ALL_DIRECTIONS

                    player_idx_lst = [i for i, p in enumerate(state.players) if p.position == player.position]
                    assert len(player_idx_lst) == 1

                    grid_string += Action.ACTION_TO_CHAR[orientation]
                    player_object = player.held_object
                    if player_object:
                        grid_string += player_object.name[:1]
                        grid_string += str(player_idx_lst[0])
                    else:
                        grid_string += str(player_idx_lst[0])
                else:
                    if element == "X" and state.has_object((x, y)):
                        state_obj = state.get_object((x, y))
                        grid_string = grid_string + element + state_obj.name[:1]

                    elif element == "P" and state.has_object((x, y)):
                        soup = state.get_object((x, y))
                        # TODO: Figure out a way to represent more rich soup space
                        if Recipe.ONION in soup.ingredients:
                            grid_string += "ø"
                        elif Recipe.TOMATO in soup.ingredients:
                            grid_string += "†"
                        else:
                            grid_string += " "

                        if soup.is_cooking:
                            grid_string += str(soup._cooking_tick)
                        
                        # NOTE: do not currently have terminal graphics 
                        # support for cooking times greater than 3.
                        elif len(soup.ingredients) == 2:
                            grid_string += "="
                        elif len(soup.ingredients) == 1:
                            grid_string += "-"
                        else:
                            grid_string += " "
                    else:
                        grid_string += element + " "

            grid_string += "\n"
        
        if state.bonus_orders:
            grid_string += "Bonus orders: {}\n".format(
                state.bonus_orders
            )
        grid_string += "State potential value: {}\n".format(self.potential_function(state))
        return grid_string

    ###################
    # STATE ENCODINGS #
    ###################

    @property
    def lossless_state_encoding_shape(self):
        return np.array(list(self.shape) + [20])

    def lossless_state_encoding(self, overcooked_state, debug=False):
        """Featurizes a OvercookedState object into a stack of boolean masks that are easily readable by a CNN"""
        assert self.num_players == 2, "Functionality has to be added to support encondings for > 2 players"
        assert type(debug) is bool
        base_map_features = ["pot_loc", "counter_loc", "onion_disp_loc", "dish_disp_loc", "serve_loc"]
        variable_map_features = ["onions_in_pot", "onions_cook_time", "onion_soup_loc", "dishes", "onions"]

        all_objects = overcooked_state.all_objects_list

        def make_layer(position, value):
                layer = np.zeros(self.shape)
                layer[position] = value
                return layer

        def process_for_player(primary_agent_idx):
            # Ensure that primary_agent_idx layers are ordered before other_agent_idx layers
            other_agent_idx = 1 - primary_agent_idx
            ordered_player_features = ["player_{}_loc".format(primary_agent_idx), "player_{}_loc".format(other_agent_idx)] + \
                        ["player_{}_orientation_{}".format(i, Direction.DIRECTION_TO_INDEX[d])
                        for i, d in itertools.product([primary_agent_idx, other_agent_idx], Direction.ALL_DIRECTIONS)]

            LAYERS = ordered_player_features + base_map_features + variable_map_features
            state_mask_dict = {k:np.zeros(self.shape) for k in LAYERS}

            # MAP LAYERS
            for loc in self.get_counter_locations():
                state_mask_dict["counter_loc"][loc] = 1

            for loc in self.get_pot_locations():
                state_mask_dict["pot_loc"][loc] = 1

            for loc in self.get_onion_dispenser_locations():
                state_mask_dict["onion_disp_loc"][loc] = 1

            for loc in self.get_dish_dispenser_locations():
                state_mask_dict["dish_disp_loc"][loc] = 1

            for loc in self.get_serving_locations():
                state_mask_dict["serve_loc"][loc] = 1

            # PLAYER LAYERS
            for i, player in enumerate(overcooked_state.players):
                player_orientation_idx = Direction.DIRECTION_TO_INDEX[player.orientation]
                state_mask_dict["player_{}_loc".format(i)] = make_layer(player.position, 1)
                state_mask_dict["player_{}_orientation_{}".format(i, player_orientation_idx)] = make_layer(player.position, 1)

            # OBJECT & STATE LAYERS
            for obj in all_objects:
                if obj.name == "soup":
                    if Recipe.ONION in obj.ingredients:
                        if obj.position in self.get_pot_locations():
                            state_mask_dict["onions_in_pot"] += make_layer(obj.position, len(obj.ingredients))
                            state_mask_dict["onions_cook_time"] += make_layer(obj.position, max(0, obj._cooking_tick))
                        else:
                            # If player soup is not in a pot, put it in separate mask
                            state_mask_dict["onion_soup_loc"] += make_layer(obj.position, 1)
                    else:
                        raise ValueError("Unrecognized soup")

                elif obj.name == "dish":
                    state_mask_dict["dishes"] += make_layer(obj.position, 1)
                elif obj.name == "onion":
                    state_mask_dict["onions"] += make_layer(obj.position, 1)
                else:
                    raise ValueError("Unrecognized object")

            if debug:
                print(len(LAYERS))
                print(len(state_mask_dict))
                for k, v in state_mask_dict.items():
                    print(k)
                    print(np.transpose(v, (1, 0)))

            # Stack of all the state masks, order decided by order of LAYERS
            state_mask_stack = np.array([state_mask_dict[layer_id] for layer_id in LAYERS])
            state_mask_stack = np.transpose(state_mask_stack, (1, 2, 0))
            assert state_mask_stack.shape[:2] == self.shape
            assert state_mask_stack.shape[2] == len(LAYERS)
            # NOTE: currently not including time left or order_list in featurization
            return np.array(state_mask_stack).astype(int)

        # NOTE: Currently not very efficient, a decent amount of computation repeated here
        num_players = len(overcooked_state.players)
        final_obs_for_players = tuple(process_for_player(i) for i in range(num_players))
        return final_obs_for_players

    @property
    def featurize_state_shape(self):
        return np.array([62])

    def featurize_state(self, overcooked_state, mlp):
        """
        Encode state with some manually designed features.
        NOTE: currently works for just two players.
        """

        all_features = {}

        def make_closest_feature(idx, name, locations):
            "Compute (x, y) deltas to closest feature of type `name`, and save it in the features dict"
            all_features["p{}_closest_{}".format(idx, name)] = self.get_deltas_to_closest_location(player, locations,
                                                                                                   mlp)

        IDX_TO_OBJ = ["onion", "soup", "dish"]
        OBJ_TO_IDX = {o_name: idx for idx, o_name in enumerate(IDX_TO_OBJ)}

        counter_objects = self.get_counter_objects_dict(overcooked_state)
        pot_state = self.get_pot_states(overcooked_state)

        # Player Info
        for i, player in enumerate(overcooked_state.players):
            orientation_idx = Direction.DIRECTION_TO_INDEX[player.orientation]
            all_features["p{}_orientation".format(i)] = np.eye(4)[orientation_idx]
            obj = player.held_object

            if obj is None:
                held_obj_name = "none"
                all_features["p{}_objs".format(i)] = np.zeros(len(IDX_TO_OBJ))
            else:
                held_obj_name = obj.name
                obj_idx = OBJ_TO_IDX[held_obj_name]
                all_features["p{}_objs".format(i)] = np.eye(len(IDX_TO_OBJ))[obj_idx]

            # Closest feature of each type
            if held_obj_name == "onion":
                all_features["p{}_closest_onion".format(i)] = (0, 0)
            else:
                make_closest_feature(i, "onion", self.get_onion_dispenser_locations() + counter_objects["onion"])

            make_closest_feature(i, "empty_pot", pot_state["empty"])
            make_closest_feature(i, "one_onion_pot", pot_state["1_items"])
            make_closest_feature(i, "two_onion_pot", pot_state["2_items"])
            make_closest_feature(i, "cooking_pot", pot_state["cooking"])
            make_closest_feature(i, "ready_pot", pot_state["ready"])

            if held_obj_name == "dish":
                all_features["p{}_closest_dish".format(i)] = (0, 0)
            else:
                make_closest_feature(i, "dish", self.get_dish_dispenser_locations() + counter_objects["dish"])

            if held_obj_name == "soup":
                all_features["p{}_closest_soup".format(i)] = (0, 0)
            else:
                make_closest_feature(i, "soup", counter_objects["soup"])

            make_closest_feature(i, "serving", self.get_serving_locations())

            for direction, pos_and_feat in enumerate(self.get_adjacent_features(player)):
                adj_pos, feat = pos_and_feat

                if direction == player.orientation:
                    # Check if counter we are facing is empty
                    facing_counter = (feat == 'X' and adj_pos not in overcooked_state.objects.keys())
                    facing_counter_feature = [1] if facing_counter else [0]
                    # NOTE: Really, this feature should have been "closest empty counter"
                    all_features["p{}_facing_empty_counter".format(i)] = facing_counter_feature

                all_features["p{}_wall_{}".format(i, direction)] = [0] if feat == ' ' else [1]

        features_np = {k: np.array(v) for k, v in all_features.items()}

        p0, p1 = overcooked_state.players
        p0_dict = {k: v for k, v in features_np.items() if k[:2] == "p0"}
        p1_dict = {k: v for k, v in features_np.items() if k[:2] == "p1"}
        p0_features = np.concatenate(list(p0_dict.values()))
        p1_features = np.concatenate(list(p1_dict.values()))

        p1_rel_to_p0 = np.array(pos_distance(p1.position, p0.position))
        abs_pos_p0 = np.array(p0.position)
        ordered_features_p0 = np.squeeze(np.concatenate([p0_features, p1_features, p1_rel_to_p0, abs_pos_p0]))

        p0_rel_to_p1 = np.array(pos_distance(p0.position, p1.position))
        abs_pos_p1 = np.array(p1.position)
        ordered_features_p1 = np.squeeze(np.concatenate([p1_features, p0_features, p0_rel_to_p1, abs_pos_p1]))
        return ordered_features_p0, ordered_features_p1


    def get_deltas_to_closest_location(self, player, locations, mlp):
        _, closest_loc = mlp.mp.min_cost_to_feature(player.pos_and_or, locations, with_argmin=True)
        if closest_loc is None:
            # "any object that does not exist or I am carrying is going to show up as a (0,0)
            # but I can disambiguate the two possibilities by looking at the features 
            # for what kind of object I'm carrying"
            return (0, 0)
        dy_loc, dx_loc = pos_distance(closest_loc, player.position)
        return dy_loc, dx_loc


    ###############################
    # POTENTIAL REWARD SHAPING FN #
    ###############################

    def potential_function(self, state):
        """
        A potential function used for more principled reward shaping
        For details see "Policy invariance under reward transformations:
        Theory and application to reward shaping"

        Essentially, this is the ɸ(s) function.
        """
        pot_states = self.get_pot_states(state)
        potential_values = {
            'holding_onion': 1,
            'onion_in_pot': 3,
            'holding_dish': 1, # For each dish when there is a full pot, corresponding to it
            'holding_soup': 14
        }

        potential = 0
        potential += len(state.player_objects_by_type['onion']) * potential_values['holding_onion']

        all_pots_with_items = self.get_partially_full_pots(pot_states) + self.get_full_pots(pot_states)
        soup_objects_in_pots_with_items = [state.get_object(pot_loc) for pot_loc in all_pots_with_items]
        total_num_onions_in_pots = sum([len(soup.ingredients) for soup in soup_objects_in_pots_with_items])
        potential += total_num_onions_in_pots * potential_values['onion_in_pot']

        num_ready_pots = len(self.get_full_pots(pot_states))
        num_useful_dishes = min(len(state.player_objects_by_type['dish']), num_ready_pots)
        potential += num_useful_dishes * potential_values['holding_dish']

        potential += len(state.player_objects_by_type['soup']) * potential_values['holding_soup']
        return potential


    ##############
    # DEPRECATED #
    ##############

    def calculate_distance_based_shaped_reward(self, state, new_state):
        """
        Adding reward shaping based on distance to certain features.
        """
        distance_based_shaped_reward = 0
        
        pot_states = self.get_pot_states(new_state)
        ready_pots = pot_states["tomato"]["ready"] + pot_states["onion"]["ready"]
        cooking_pots = ready_pots + pot_states["tomato"]["cooking"] + pot_states["onion"]["cooking"]
        nearly_ready_pots = cooking_pots + pot_states["tomato"]["partially_full"] + pot_states["onion"]["partially_full"]
        dishes_in_play = len(new_state.player_objects_by_type['dish'])
        for player_old, player_new in zip(state.players, new_state.players):
            # Linearly increase reward depending on vicinity to certain features, where distance of 10 achieves 0 reward
            max_dist = 8

            if player_new.held_object is not None and player_new.held_object.name == 'dish' and len(nearly_ready_pots) >= dishes_in_play:
                min_dist_to_pot_new = np.inf
                min_dist_to_pot_old = np.inf
                for pot in nearly_ready_pots:
                    new_dist = np.linalg.norm(np.array(pot) - np.array(player_new.position))
                    old_dist = np.linalg.norm(np.array(pot) - np.array(player_old.position))
                    if new_dist < min_dist_to_pot_new:
                        min_dist_to_pot_new = new_dist
                    if old_dist < min_dist_to_pot_old:
                        min_dist_to_pot_old = old_dist
                if min_dist_to_pot_old > min_dist_to_pot_new:
                    distance_based_shaped_reward += self.reward_shaping_params["POT_DISTANCE_REW"] * (1 - min(min_dist_to_pot_new / max_dist, 1))

            if player_new.held_object is None and len(cooking_pots) > 0 and dishes_in_play == 0:
                min_dist_to_d_new = np.inf
                min_dist_to_d_old = np.inf
                for serving_loc in self.terrain_pos_dict['D']:
                    new_dist = np.linalg.norm(np.array(serving_loc) - np.array(player_new.position))
                    old_dist = np.linalg.norm(np.array(serving_loc) - np.array(player_old.position))
                    if new_dist < min_dist_to_d_new:
                        min_dist_to_d_new = new_dist
                    if old_dist < min_dist_to_d_old:
                        min_dist_to_d_old = old_dist

                if min_dist_to_d_old > min_dist_to_d_new:
                    distance_based_shaped_reward += self.reward_shaping_params["DISH_DISP_DISTANCE_REW"] * (1 - min(min_dist_to_d_new / max_dist, 1))

            if player_new.held_object is not None and player_new.held_object.name == 'soup':
                min_dist_to_s_new = np.inf
                min_dist_to_s_old = np.inf
                for serving_loc in self.terrain_pos_dict['S']:
                    new_dist = np.linalg.norm(np.array(serving_loc) - np.array(player_new.position))
                    old_dist = np.linalg.norm(np.array(serving_loc) - np.array(player_old.position))
                    if new_dist < min_dist_to_s_new:
                        min_dist_to_s_new = new_dist

                    if old_dist < min_dist_to_s_old:
                        min_dist_to_s_old = old_dist
                
                if min_dist_to_s_old > min_dist_to_s_new:
                    distance_based_shaped_reward += self.reward_shaping_params["SOUP_DISTANCE_REW"] * (1 - min(min_dist_to_s_new / max_dist, 1))

        return distance_based_shaped_reward
