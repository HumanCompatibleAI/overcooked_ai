import itertools, copy, uuid, json, gym
import numpy as np
from functools import reduce
from collections import defaultdict, Counter
from overcooked_ai_py.utils import pos_distance, read_layout_dict, delete_duplicates, classproperty
from overcooked_ai_py.mdp.actions import Action, Direction

def custom_method_equal(obj1, obj2, method_name):
    if hasattr(obj1, method_name):
        return getattr(obj1, method_name)(obj2)
    elif hasattr(obj2, method_name):
        return getattr(obj2, method_name)(obj1)
    else:
        return obj1 == obj2

def ids_independent_equal(obj1, obj2):
    return custom_method_equal(obj1, obj2, "ids_independent_equal")

class Recipe:
    MAX_NUM_INGREDIENTS = 3

    TOMATO = 'tomato'
    ONION = 'onion'
    ALL_INGREDIENTS = [ONION, TOMATO]

    ALL_RECIPES_CACHE = {}
    STR_REP = {'tomato': "†", 'onion': "ø"}

    _computed = False
    _configured = False
    _conf = {}
    
    def __new__(cls, ingredients):
        if not cls._configured:
            raise ValueError("Recipe class must be configured before recipes can be created")
        # Some basic argument verification
        if not ingredients or not hasattr(ingredients, '__iter__') or len(ingredients) == 0:
            raise ValueError("Invalid input recipe. Must be ingredients iterable with non-zero length")
        for elem in ingredients:
            if not elem in cls.ALL_INGREDIENTS:
                raise ValueError("Invalid ingredient: {0}. Recipe can only contain ingredients {1}".format(elem, cls.ALL_INGREDIENTS))
        if not len(ingredients) <= cls.MAX_NUM_INGREDIENTS:
            raise ValueError("Recipe of length {0} is invalid: {1}. Recipe can contain at most {2} ingredients".format(len(ingredients), ingredients, cls.MAX_NUM_INGREDIENTS))
        key = hash(tuple(sorted(ingredients)))
        if key in cls.ALL_RECIPES_CACHE:
            return cls.ALL_RECIPES_CACHE[key]
        cls.ALL_RECIPES_CACHE[key] = super(Recipe, cls).__new__(cls)
        return cls.ALL_RECIPES_CACHE[key]

    def __init__(self, ingredients):
        self._ingredients = ingredients

    def __getnewargs__(self):
        return (self._ingredients,)

    def __int__(self):
        num_tomatoes = len([_ for _ in self.ingredients if _ == Recipe.TOMATO])
        num_onions = len([_ for _ in self.ingredients if _ == Recipe.ONION])

        mixed_mask = int(bool(num_tomatoes * num_onions))
        mixed_shift = (Recipe.MAX_NUM_INGREDIENTS + 1)**len(Recipe.ALL_INGREDIENTS)
        encoding = num_onions + (Recipe.MAX_NUM_INGREDIENTS + 1) * num_tomatoes

        return mixed_mask * encoding * mixed_shift + encoding

    def __hash__(self):
        return hash(self.ingredients)

    def __eq__(self, other):
        # The ingredients property already returns sorted items, so equivalence check is sufficient
        return isinstance(other, Recipe) and self.ingredients == other.ingredients

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        return int(self) < int(other)

    def __le__(self, other):
        return int(self) <= int(other)

    def __gt__(self, other):
        return int(self) > int(other)

    def __ge__(self, other):
        return int(self) >= int(other)

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

    @staticmethod
    def standardized_ingredients(ingredients):
        return tuple(sorted(ingredients))

    @property
    def ingredients(self):
        return Recipe.standardized_ingredients(self._ingredients)

    @ingredients.setter
    def ingredients(self, _):
        raise AttributeError("Recpes are read-only. Do not modify instance attributes after creation")

    @property
    def value(self):
        if self._delivery_reward:
            return self._delivery_reward
        if self._value_mapping and self in self._value_mapping:
            return self._value_mapping[self]
        if self._onion_value and self._tomato_value:
            num_onions = len([ingredient for ingredient in self.ingredients if ingredient == self.ONION])
            num_tomatoes = len([ingredient for ingredient in self.ingredients if ingredient == self.TOMATO])
            return self._tomato_value * num_tomatoes + self._onion_value * num_onions
        return 20

    @property
    def time(self):
        if self._cook_time:
            return self._cook_time
        if self._time_mapping and self in self._time_mapping:
            return self._time_mapping[self]
        if self._onion_time and self._tomato_time:
            num_onions = len([ingredient for ingredient in self.ingredients if ingredient == self.ONION])
            num_tomatoes = len([ingredient for ingredient in self.ingredients if ingredient == self.TOMATO])
            return self._onion_time * num_onions + self._tomato_time * num_tomatoes
        return 20

    def to_dict(self):
        return { 'ingredients' : self.ingredients }
    
    @classmethod
    def neighbors_ingredients(cls, ingredients):
        neighbors = []
        if len(ingredients) == cls.MAX_NUM_INGREDIENTS:
            return neighbors
        for ingredient in cls.ALL_INGREDIENTS:
            new_ingredients = Recipe.standardized_ingredients([*ingredients, ingredient])
            neighbors.append(new_ingredients)
        return neighbors


    def neighbors(self):
        """
        Return all "neighbor" recipes to this recipe. A neighbor recipe is one that can be obtained
        by adding exactly one ingredient to the current recipe
        """
        return [Recipe(ingredients) for ingredients in Recipe.neighbors_ingredients(self.ingredients)]

    @classmethod
    def ingredients_diff(cls, ingredients1, ingredients2):
        # returns tuple of ingredients that are missing in ingredients2 to have same elements as ingredients1 
        # (if there are no extra ingredients in ingredients 1 that are not in ingredients2),
        # can be viewed as ingredients1 - ingredients2 (substraction operation)
        ingredients_diffs = {ingredient: len([i for i in ingredients1 if i == ingredient]) - len([i for i in ingredients2 if i == ingredient])
            for ingredient in cls.ALL_INGREDIENTS}
        return Recipe.standardized_ingredients(itertools.chain.from_iterable([[ingredient]*diff for (ingredient, diff) in ingredients_diffs.items() if diff > 0]))

    @classproperty
    def ALL_RECIPES(cls):
        if not cls._computed:
            cls._compute_all_recipes()
            cls._computed = True
        return set(cls.ALL_RECIPES_CACHE.values())

    @classproperty
    def configuration(cls):
        if not cls._configured:
            raise ValueError("Recipe class not yet configured")
        return cls._conf

    @classmethod
    def configure(cls, conf):
        cls._conf = conf
        cls._configured = True
        cls._computed = False
        cls.MAX_NUM_INGREDIENTS = conf.get('max_num_ingredients', 3)

        cls._cook_time = None
        cls._delivery_reward = None
        cls._value_mapping = None
        cls._time_mapping = None
        cls._onion_value = None
        cls._onion_time = None
        cls._tomato_value = None
        cls._tomato_time = None

        ## Basic checks for validity ##

        # Mutual Exclusion
        if 'tomato_time' in conf and not 'onion_time' in conf or 'onion_time' in conf and not 'tomato_time' in conf:
            raise ValueError("Must specify both 'onion_time' and 'tomato_time'")
        if 'tomato_value' in conf and not 'onion_value' in conf or 'onion_value' in conf and not 'tomato_value' in conf:
            raise ValueError("Must specify both 'onion_value' and 'tomato_value'")
        if 'tomato_value' in conf and 'delivery_reward' in conf:
            raise ValueError("'delivery_reward' incompatible with '<ingredient>_value'")
        if 'tomato_value' in conf and 'recipe_values' in conf:
            raise ValueError("'recipe_values' incompatible with '<ingredient>_value'")
        if 'recipe_values' in conf and 'delivery_reward' in conf:
            raise ValueError("'delivery_reward' incompatible with 'recipe_values'")
        if 'tomato_time' in conf and 'cook_time' in conf:
            raise ValueError("'cook_time' incompatible with '<ingredient>_time")
        if 'tomato_time' in conf and 'recipe_times' in conf:
            raise ValueError("'recipe_times' incompatible with '<ingredient>_time'")
        if 'recipe_times' in conf and 'cook_time' in conf:
            raise ValueError("'delivery_reward' incompatible with 'recipe_times'")

        # recipe_ lists and orders compatibility
        if 'recipe_values' in conf:
            if not 'all_orders' in conf or not conf['all_orders']:
                raise ValueError("Must specify 'all_orders' if 'recipe_values' specified")
            if not len(conf['all_orders']) == len(conf['recipe_values']):
                raise ValueError("Number of recipes in 'all_orders' must be the same as number in 'recipe_values")
        if 'recipe_times' in conf:
            if not 'all_orders' in conf or not conf['all_orders']:
                raise ValueError("Must specify 'all_orders' if 'recipe_times' specified")
            if not len(conf['all_orders']) == len(conf['recipe_times']):
                raise ValueError("Number of recipes in 'all_orders' must be the same as number in 'recipe_times")

        
        ## Conifgure ##

        if 'cook_time' in conf:
            cls._cook_time = conf['cook_time']

        if 'delivery_reward' in conf:
            cls._delivery_reward = conf['delivery_reward']

        if 'recipe_values' in conf:
            cls._value_mapping = {
                cls.from_dict(recipe) : value for (recipe, value) in zip(conf['all_orders'], conf['recipe_values'])
            }

        if 'recipe_times' in conf:
            cls._time_mapping = {
                cls.from_dict(recipe) : time for (recipe, time) in zip(conf['all_orders'], conf['recipe_times'])
            }

        if 'tomato_time' in conf:
            cls._tomato_time = conf['tomato_time']

        if 'onion_time' in conf:
            cls._onion_time = conf['onion_time']

        if 'tomato_value' in conf:
            cls._tomato_value = conf['tomato_value']

        if 'onion_value' in conf:
            cls._onion_value = conf['onion_value']
    
    @classmethod
    def generate_random_recipes(cls, n=1, min_size=2, max_size=3, ingredients=None, recipes=None, unique=True):
        """
        n (int): how many recipes generate
        min_size (int): min generated recipe size
        max_size (int): max generated recipe size
        ingredients (list(str)): list of ingredients used for generating recipes (default is cls.ALL_INGREDIENTS)
        recipes (list(Recipe)): list of recipes to choose from (default is cls.ALL_RECIPES)
        unique (bool): if all recipes are unique (without repeats)
        """
        if recipes is None: recipes = cls.ALL_RECIPES

        ingredients = set(ingredients or cls.ALL_INGREDIENTS)
        choice_replace = not(unique)

        assert 1 <= min_size <= max_size <= cls.MAX_NUM_INGREDIENTS
        assert all(ingredient in cls.ALL_INGREDIENTS for ingredient in ingredients)

        def valid_size(r):
            return min_size <= len(r.ingredients) <= max_size

        def valid_ingredients(r):
            return all(i in ingredients for i in r.ingredients)
        
        relevant_recipes = [r for r in recipes if valid_size(r) and valid_ingredients(r)]
        assert choice_replace or (n <= len(relevant_recipes))
        return np.random.choice(relevant_recipes, n, replace=choice_replace)

    @classmethod
    def from_dict(cls, obj_dict):
        return cls(**obj_dict)
        
class Order:
    def __init__(self, recipe, time_to_expire=None, expire_penalty=0, base_reward=None, linear_time_bonus_reward=0.0, order_id=None, is_bonus=False):
        """
        recipe (Recipe or dict): recipe for order
        time_to_expire (int): time for order to disappear and occur expire_penalty
        expire_penalty (int): negative reward when Order expires before fulfilling it
        base_reward (int): reward for fulfilling order independent of time
        linear_time_bonus_reward (float): reward per every remaining timestep for fulfilling order before expiring
        order_id (str): unique id for the order
        is_bonus(bool): indication if order should be shown in bonus_orders lists
        """
        if not isinstance(recipe, Recipe):
            recipe = Recipe.from_dict(recipe)
        self.recipe = recipe
        self.time_to_expire = time_to_expire
        assert expire_penalty >= 0, "expire penalty needs to be 0 or more (to not give reward for missing the order)"
        self.expire_penalty = expire_penalty

        self.order_id = Order.create_order_id()  if order_id is None else order_id
        self._base_reward = base_reward
        self.linear_time_bonus_reward = linear_time_bonus_reward or 0
        self.is_bonus = is_bonus

    @property
    def is_temporary(self):
        return self.time_to_expire is not None

    @is_temporary.setter
    def is_temporary(self, _):
        raise AttributeError("is_temporary param is read-only, edit time_to_expire instead (None is for pernament order)")

    @property
    def base_reward(self):
        return self.recipe.value if self._base_reward is None else self._base_reward

    @base_reward.setter
    def base_reward(self, v):
        self._base_reward = v

    @property
    def is_expired(self):
        return self.time_to_expire == 0

    def __str__(self):
        return str(self.to_dict())[1:-1]

    def __repr__(self):
        return repr(self.to_dict())[1:-1]

    def __hash__(self):
        return hash((self.recipe, self.time_to_expire, self.expire_penalty, self.order_id, self._base_reward, self.linear_time_bonus_reward, self.is_bonus))

    def __deepcopy__(self, memo):
        return Order(recipe=copy.deepcopy(self.recipe),
                    time_to_expire=self.time_to_expire,
                    expire_penalty=self.expire_penalty,
                    base_reward=self._base_reward,
                    linear_time_bonus_reward=self.linear_time_bonus_reward,
                    order_id=self.order_id,
                    is_bonus=self.is_bonus)

    def __eq__(self, other):
        return self.ids_independent_equal(other) and self.order_id == other.order_id

    def ids_independent_equal(self, other):
        return isinstance(other, Order) \
            and self.recipe == other.recipe \
            and self.time_to_expire == other.time_to_expire \
            and self.expire_penalty == other.expire_penalty \
            and self._base_reward == other._base_reward \
            and self.linear_time_bonus_reward == other.linear_time_bonus_reward \
            and self.is_bonus == other.is_bonus

    @staticmethod
    def create_order_id():
        return str(uuid.uuid1())

    @classmethod
    def from_dict(cls, obj_dict):
        kwargs = copy.deepcopy(obj_dict)
        kwargs["recipe"] = Recipe.from_dict(obj_dict["recipe"])
        return cls(**obj_dict)

    def to_dict(self):
        return {"recipe": self.recipe.to_dict(),
                "time_to_expire": self.time_to_expire,
                "expire_penalty":self.expire_penalty,
                "base_reward":self._base_reward,
                "linear_time_bonus_reward":self.linear_time_bonus_reward,
                "order_id": self.order_id,
                "is_bonus": self.is_bonus}

    def step(self):
        if self.is_temporary:
            assert self.time_to_expire > 0
            self.time_to_expire -= 1
            if self.time_to_expire == 0:
                return -self.expire_penalty
        return 0

    def calculate_reward(self):
        return self.calculate_future_reward(0)

    def calculate_future_reward(self, timesteps_into_future=0):
        if not self.is_temporary:
            return self.base_reward
        elif timesteps_into_future > self.time_to_expire:
            return 0
        else:
            return self.base_reward + int(self.linear_time_bonus_reward * abs(self.time_to_expire-timesteps_into_future))

    def will_be_expired_in(self, time):
        return self.is_temporary and self.time_to_expire <= time


class OrdersList:
    def __init__(self, orders=[], orders_to_add=[], add_new_order_every=None, time_to_next_order=None, 
        max_orders_num=None, num_orders_in_queue=0, fulfilled_orders=[]):
        """
        list of orders with methods to interact with them
        orders (list(Order) or list(dict)): initial list of orders
        orders_to_add list (list(Order) or list(dict)): list of orders that will be randomly added
             when time_to_next_order would reach 0
        add_new_order_every (int): number of timesteps between adding consecutive orders
        time_to_next_order (int): remaining timesteps number before next order will be added to this OrderList
        max_orders_num (int): maximum number of orders that can be inside orders ready to be fullfilled at same time, 
            None is for infinity
        num_orders_in_queue (int): number of orders waiting to go into orders list becaue of max_orders_num limit
        fulfilled_orders(list(Order)): list of fullfilled orders; used by mdp to check if episode can be terminated after 
        fulliling selected number of orders (currently mdp uses only size of this list)
        """
        assert add_new_order_every is None or add_new_order_every > 0
        assert (add_new_order_every is None and time_to_next_order is None) or orders_to_add
        self.orders = []
        self.add_orders(orders)
        self.orders_to_add = []
        self.add_orders(orders_to_add, list_to_add=self.orders_to_add)
        self.add_new_order_every = add_new_order_every
        self.time_to_next_order = time_to_next_order or add_new_order_every
        self.num_orders_in_queue = num_orders_in_queue
        self.max_orders_num = max_orders_num
        self.fulfilled_orders = []
        self.add_orders(fulfilled_orders, list_to_add=self.fulfilled_orders)
    
    @property
    def all_recipes(self):
        """
        used to create Recipes config
        """
        all_orders = self.orders + self.orders_to_add
        return list(set([o.recipe for o in all_orders]))

    @property
    def bonus_recipes(self):
        all_orders = self.orders + self.orders_to_add
        return list(set([o.recipe for o in all_orders if o.is_bonus]))

    @property
    def is_adding_orders(self):
        return self.add_new_order_every is not None

    @is_adding_orders.setter
    def is_adding_orders(self, _):
        raise AttributeError("is_adding_orders param is read-only, edit add_new_order_every instead (None is for not adding orders)")

    @property
    def contains_temporary_orders(self):
        return any(o.is_temporary for o in self.orders)

    @property
    def bonus_orders(self):
        return [order for order in self.orders if order.is_bonus]

    @bonus_orders.setter
    def bonus_orders(self, _):
        raise AttributeError("bonus_orders param is read-only, edit orders instead")

    @property
    def non_bonus_orders(self):
        return [order for order in self.orders if not order.is_bonus]

    @non_bonus_orders.setter
    def non_bonus_orders(self, _):
        raise AttributeError("non_bonus_orders param is read-only, edit orders instead")

    def __str__(self):
        return str(self.to_dict())[1:-1]

    def __repr__(self):
        return repr(self.to_dict())[1:-1]

    def __hash__(self):
        return hash((tuple(self.orders), tuple(self.orders_to_add), self.add_new_order_every, self.time_to_next_order))

    def __bool__(self):
        return bool(self.orders)

    def __eq__(self, other):
        return isinstance(other, OrdersList) \
            and OrdersList.orders_iterables_equal(self.orders, other.orders) \
            and self.add_new_order_every == other.add_new_order_every \
            and self.time_to_next_order == other.time_to_next_order \
            and OrdersList.orders_iterables_equal(self.orders_to_add, other.orders_to_add) \
            and self.fulfilled_orders == other.fulfilled_orders
    
    def ids_independent_equal(self, other):
        return isinstance(other, OrdersList) \
            and OrdersList.orders_iterables_equal(self.orders, other.orders, ids_independent_equal) \
            and self.add_new_order_every == other.add_new_order_every \
            and self.time_to_next_order == other.time_to_next_order \
            and OrdersList.orders_iterables_equal(self.orders_to_add, other.orders_to_add, ids_independent_equal) \
            and self.fulfilled_orders == other.fulfilled_orders

    def __len__(self):
        return len(self.orders)

    # when use code below decide if orders order should be based on orders_sorted_by_urgency or not
    #  def __iter__(self):
    #     return iter(self.orders)
    # def __getitem__(self, i):
    #     return self.orders[i]
    # def __setitem__(self, i, v):
    #     self.orders[i] = v
    # def __delitem__(self, i):
    #     del self.orders[i]

    def __deepcopy__(self, memo):
        return OrdersList(orders=[copy.deepcopy(o) for o in self.orders],
                         add_new_order_every=self.add_new_order_every,
                         time_to_next_order=self.time_to_next_order,
                         orders_to_add=[copy.deepcopy(o) for o in self.orders_to_add],
                         num_orders_in_queue=self.num_orders_in_queue,
                         max_orders_num=self.max_orders_num,
                         fulfilled_orders=self.fulfilled_orders
                         )

    @staticmethod
    def orders_iterables_equal(orders1, orders2, f_equal=lambda x, y: x == y):
        orders1 = set(orders1)
        orders2 = set(orders2)
        if len(orders1) != len(orders2):
            return False
        for order1 in orders1:
            if not any(f_equal(order1, order2) for order2 in orders2):
                return False
        return True

    @staticmethod
    def _order_urgency_sort_key(order, n_timesteps_into_future=0):
        # higher -> more urgent
        return (-order.will_be_expired_in(n_timesteps_into_future), order.is_temporary, -(order.time_to_expire or 0), order.is_bonus, order.calculate_reward()+order.expire_penalty)

    def enumerated_orders_sorted_by_urgency(self, n_timesteps_into_future=0):
        # closer to beginning -> more urgent
        return list(reversed(sorted(enumerate(self.orders), key=lambda x: OrdersList._order_urgency_sort_key(x[1], n_timesteps_into_future))))

    def orders_sorted_by_urgency(self, n_timesteps_into_future=0):
        # closer to beginning -> more urgent
        return [x[1] for x in self.enumerated_orders_sorted_by_urgency(n_timesteps_into_future=n_timesteps_into_future)]

    def get_matching_order(self, order_id=None, recipe=None, temporary_order=None, available_after_n_timesteps=-1, most_urgent=True):
        order_idx = self.matching_order_index(order_id=order_id, recipe=recipe, temporary_order=temporary_order, available_after_n_timesteps=available_after_n_timesteps, most_urgent=most_urgent)
        if order_idx is None:
            return None
        else:
            return self.orders[order_idx]

    def matching_order_index(self, order_id=None, recipe=None, temporary_order=None, available_after_n_timesteps=-1, most_urgent=True):
        """
        find order index by order_id or by recipe
        if teporary_order=None find any order, if temporary_order is bool it indicates if we look for only temporary orders or pernament ones
        available_after_n_timesteps allows to search existing orders that will won't expire for n timesteps
        """
        assert recipe is not None or order_id is not None
        if most_urgent:
            enumerated_orders = self.enumerated_orders_sorted_by_urgency(available_after_n_timesteps)
        else:
            enumerated_orders = enumerate(self.orders)

        for i, order in enumerated_orders:
            if (temporary_order is None or temporary_order == order.is_temporary) \
                    and ((not order.is_temporary) or available_after_n_timesteps < order.time_to_expire):
                if order_id is not None and order.order_id == order_id:
                    return i
                if recipe is not None and order.recipe == recipe:
                    return i
        else:
            return None

    def add_order(self, order, list_to_add=None):
        """ Adds order to selected list (self.orders as default); contains additional checks for self.orders
        """
        if list_to_add is None:
            list_to_add = self.orders
        if not isinstance(order, Order):
            order = Order.from_dict(order)
        if list_to_add is self.orders:
            assert order.is_temporary or self.get_matching_order(recipe=order.recipe, temporary_order=False) is None, "More than one non-temporary order with the same recipe is not allowed"
            assert self.get_matching_order(order_id=order.order_id) is None, "More than one order in with the same order_id is not allowed"
        list_to_add.append(order)
    
    def add_orders(self, orders, list_to_add=None):
        for order in orders:
            self.add_order(order, list_to_add)

    def add_random_orders(self, n=1, replace=False):
        """
        add random n orders from orders_to_add
        """
        for order in np.random.choice(list(self.orders_to_add), n, replace=replace):
            order = copy.deepcopy(order)
            order.order_id = Order.create_order_id()
            self.add_order(order)

    def remove_order(self, order_id=None, recipe=None, order_idx=None, temporary_order=True, most_urgent=True):
        """
        remove order by order_id or by recipe
        if teporary_order=None remove also pernament orders, if teporary_order=False remove only pernament orders
        """
        assert recipe is not None or order_id is not None or order_idx is not None
        if order_idx is None:
            order_idx = self.matching_order_index(recipe=recipe, order_id=order_id, temporary_order=temporary_order, most_urgent=most_urgent)
        if order_idx is None:
            order = None
        else:
            order = self.orders.pop(order_idx)
        return order

    def fulfill_order(self, recipe):
        """
        invoke this method on soup delivery
        """
        order_idx = self.matching_order_index(recipe=recipe)
        if order_idx is None:
            return None
        order = self.orders[order_idx]
        self.add_order(order, list_to_add=self.fulfilled_orders)
        if order.is_temporary:
            self.remove_order(order_idx=order_idx)
        return order

    def step(self):
        """
        use this method every timestep
        """
        reward = 0
        order_ids_to_remove = []
        for order in self.orders:
            reward += order.step()
            if order.is_expired:
                order_ids_to_remove.append(order.order_id)
        
        for order_id in order_ids_to_remove:
            self.remove_order(order_id=order_id)
        
        if self.is_adding_orders:
            self.time_to_next_order -= 1
            if self.time_to_next_order < 1:
                self.time_to_next_order = self.add_new_order_every
                self.num_orders_in_queue += 1

            while self.can_add_more_orders(n=1) and self.num_orders_in_queue > 0:
                self.add_random_orders(n=1)
                self.num_orders_in_queue -= 1
        return reward

    def can_add_more_orders(self, n=1):
        return not self.max_orders_num or len(self.orders) + n <= self.max_orders_num
    
    @classmethod
    def from_dict(cls, obj_dict):
        return cls(**copy.deepcopy(obj_dict))

    def to_dict(self):
        return {"orders": [o.to_dict() for o in self.orders],
                "add_new_order_every": self.add_new_order_every,
                "time_to_next_order": self.time_to_next_order,
                "orders_to_add": [o.to_dict() for o in self.orders_to_add],
                "max_orders_num": self.max_orders_num,
                "num_orders_in_queue": self.num_orders_in_queue,
                "fulfilled_orders": [o.to_dict() for o in self.fulfilled_orders]
                }

    @classmethod
    def from_recipes_lists(cls, all_orders_recipes, bonus_orders_recipes):
        bonus_orders_recipes_serialized = set([json.dumps(d, sort_keys=True) for d in bonus_orders_recipes])
        def is_in_bonus_recipes(d):
            return json.dumps(d, sort_keys=True) in bonus_orders_recipes_serialized

        bonus_orders = [Order(Recipe.from_dict(r), is_bonus=True) for r in bonus_orders_recipes]
        non_bonus_orders = [Order(Recipe.from_dict(r)) for r in all_orders_recipes if not is_in_bonus_recipes(r)]

        return cls(orders=bonus_orders+non_bonus_orders)

    @staticmethod
    def dict_to_all_recipes_dicts(d):
        """
        used to create Recipes config
        """
        all_orders = d.get("orders", []) + d.get("orders_to_add", [])
        return delete_duplicates([o["recipe"] for o in all_orders])


class ObjectState(object):
    """
    State of an object in OvercookedGridworld.
    """

    def __init__(self, name, position, object_id=None, **kwargs):
        """
        name (str): The name of the object
        position (int, int): Tuple for the current location of the object.
        object_id(str): Id of the object
        """
        self.name = name
        self._position = tuple(position)
        self.object_id = str(uuid.uuid1()) if object_id is None else object_id

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, new_pos):
        self._position = new_pos

    def is_valid(self):
        return self.name in ['onion', 'tomato', 'dish']

    def deepcopy(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, memo):
        return ObjectState(self.name, self.position, self.object_id)


    def __eq__(self, other):
        return self.ids_independent_equal(other) and \
            self.object_id == other.object_id

    def ids_independent_equal(self, other):
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
            "position": self.position,
            "object_id": self.object_id
        }

    @classmethod
    def from_dict(cls, obj_dict):
        obj_dict = copy.deepcopy(obj_dict)
        return ObjectState(**obj_dict)


class SoupState(ObjectState):
    def __init__(self, position, ingredients=[], cooking_tick=-1, cook_time=None, object_id=None, **kwargs):

        """
        Represents a soup object. An object becomes a soup the instant it is placed in a pot. The
        soup's recipe is a list of ingredient names used to create it. A soup's recipe is undetermined
        until it has begun cooking. 

        position (tuple): (x, y) coordinates in the grid
        ingredients (list(ObjectState)): Objects that have been used to cook this soup. Determiens @property recipe
        cooking (int): How long the soup has been cooking for. -1 means cooking hasn't started yet
        cook_time(int): How long soup needs to be cooked, used only mostly for getting soup from dict with supplied cook_time, if None self.recipe.time is used
        """
        super(SoupState, self).__init__("soup", position, object_id)
        self._ingredients = ingredients
        self._cooking_tick = cooking_tick
        self._recipe = None
        self._cook_time = cook_time

    def __eq__(self, other):
        return isinstance(other, SoupState) and self.name == other.name and self.position == other.position and \
            self._cooking_tick == other._cooking_tick and self._cook_time == other._cook_time and \
            self.object_id == other.object_id and \
            all([this_i == other_i for this_i, other_i in zip(self._ingredients, other._ingredients)]) 

    def ids_independent_equal(self, other):
        return isinstance(other, SoupState) and self.name == other.name and self.position == other.position and \
            self._cooking_tick == other._cooking_tick and self._cook_time == other._cook_time and \
            all([ids_independent_equal(this_i, other_i) for this_i, other_i in zip(self._ingredients, other._ingredients)])

    def __hash__(self):
        ingredient_hash = hash(tuple([hash(i) for i in self._ingredients]))
        supercls_hash = super(SoupState, self).__hash__()
        return hash((supercls_hash, self._cooking_tick, ingredient_hash))

    def __repr__(self):
        supercls_str = super(SoupState, self).__repr__()
        ingredients_str = self._ingredients.__repr__()
        return "{}\nIngredients:\t{}\nCooking Tick:\t{}".format(supercls_str, ingredients_str, self._cooking_tick)

    def __str__(self):
        res = "{"
        for ingredient in sorted(self.ingredients):
            res += Recipe.STR_REP[ingredient]
        if self.is_cooking:
            res += str(self._cooking_tick)
        elif self.is_ready:
            res += str("✓")
        return res

    @ObjectState.position.setter
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
        # used mostly when cook time is supplied by state dict
        if self._cook_time is not None:
            return self._cook_time
        else:
            return self.recipe.time

    @property
    def cook_time_remaining(self):
        return max(0, self.cook_time - self._cooking_tick)

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


    def __deepcopy__(self, memo):
        return SoupState(self.position, [ingredient.deepcopy() for ingredient in self._ingredients],
            cooking_tick=self._cooking_tick, object_id=self.object_id, cook_time=self._cook_time)

    def deepcopy(self):
        return copy.deepcopy(self)

    def to_dict(self):
        info_dict = super(SoupState, self).to_dict()
        ingredients_dict = [ingredient.to_dict() for ingredient in self._ingredients]
        info_dict['_ingredients'] = ingredients_dict
        info_dict['cooking_tick'] = self._cooking_tick
        info_dict['is_cooking'] = self.is_cooking
        info_dict['is_ready'] = self.is_ready
        info_dict['is_idle'] = self.is_idle
        info_dict['cook_time'] = self._cook_time
        info_dict['object_id'] = self.object_id
        # This is for backwards compatibility w/ overcooked-demo
        # Should be removed once overcooked-demo is updated to use 'cooking_tick' instead of '_cooking_tick'
        info_dict['_cooking_tick'] = self._cooking_tick
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
        obj_dict['ingredients'] = ingredients_objs
        return cls(**obj_dict)

    @classmethod
    def get_soup(cls, position, num_onions=1, num_tomatoes=0, cooking_tick=-1, finished=False, object_id=None, **kwargs):
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
        soup = cls(position, ingredients, cooking_tick, object_id)
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
        return copy.deepcopy(self)
    
    def ids_independent_equal(self, other):
        return isinstance(other, PlayerState) and \
            self.position == other.position and \
            self.orientation == other.orientation and \
            ids_independent_equal(self.held_object, other.held_object)

    def __eq__(self, other):
        return self.ids_independent_equal(other) and self.held_object == other.held_object

    def __hash__(self):
        return hash((self.position, self.orientation, self.held_object))

    def __repr__(self):
        return '{} facing {} holding {}'.format(
            self.position, self.orientation, str(self.held_object))
    
    def __deepcopy__(self, memo):
        new_obj = None if self.held_object is None else self.held_object.deepcopy()
        return PlayerState(self.position, self.orientation, new_obj)

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
            player_dict["held_object"] = SoupState.from_dict(held_obj)
        return PlayerState(**player_dict)

class OvercookedState(object):
    """A state in OvercookedGridworld."""
    def __init__(self, players, objects, all_orders=[], bonus_orders=[], orders_list=None, timestep=0, **kwargs):
        """
        players (list(PlayerState)): Currently active PlayerStates (index corresponds to number)
        objects (dict({tuple:list(ObjectState)})):  Dictionary mapping positions (x, y) to ObjectStates. 
            NOTE: Does NOT include objects held by players (they are in 
            the PlayerState objects).
        timestep (int):  The current timestep of the state
        orders_list (OrdersList or dict):    Current orders list
        bonus_orders (list(dict)):   Current orders worth a bonus (user) - legacy param, use orders_list instead
        all_orders (list(dict)):     Current orders allowed at all - legacy param, use orders_list instead
        """

        assert not ((all_orders or bonus_orders) and orders_list), "Use either legacy params 'all_orders' and 'bonus_orders' or new one 'orders_list', but not the both"

        if isinstance(orders_list, OrdersList):
            pass
        elif isinstance(orders_list, dict):
            orders_list = OrdersList(**orders_list)
        else:
            orders_list = OrdersList.from_recipes_lists(all_orders, bonus_orders)

        self.orders_list = orders_list

        for pos, obj in objects.items():
            assert obj.position == pos
        self.players = tuple(players)
        self.objects = objects
        self.timestep = timestep

        assert len(set(self.bonus_orders)) == len(self.bonus_orders), "Bonus orders must not have duplicates"
        assert len(set(self.all_orders)) == len(self.all_orders), "All orders must not have duplicates"
        assert set(self.bonus_orders).issubset(set(self.all_orders)), "Bonus orders must be a subset of all orders"

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
        """
        returns all recipes from orders
        """
        return sorted(self.orders_list.all_recipes)

    @property
    def bonus_orders(self):
        """
        returns all recipes from orders marked as bonus
        """
        return sorted(self.orders_list.bonus_recipes)

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
    def from_players_pos_and_or(cls, players_pos_and_or, orders_list=[], bonus_orders=[], all_orders=[]):
        """
        Make a dummy OvercookedState with no objects based on the passed in player
        positions and orientations and order list
        """
        return cls(
            [PlayerState(*player_pos_and_or) for player_pos_and_or in players_pos_and_or], 
            objects={}, orders_list=orders_list, all_orders=all_orders, bonus_orders=bonus_orders)

    @classmethod
    def from_player_positions(cls, player_positions, orders_list=[], all_orders=[], bonus_orders=[]):
        """
        Make a dummy OvercookedState with no objects and with players facing
        North based on the passed in player positions and order list
        """
        dummy_pos_and_or = [(pos, Direction.NORTH) for pos in player_positions]
        return cls.from_players_pos_and_or(dummy_pos_and_or, orders_list=orders_list, all_orders=all_orders, bonus_orders=bonus_orders)

    def deepcopy(self):
        return copy.deepcopy(self)

    def time_equal(self, other):
        return self.timestep == other.timestep

    def order_lists_equal(self, other, ids_independent=False):
        if ids_independent:
            return ids_independent_equal(self.orders_list, other.orders_list)
        else:
            return self.orders_list == other.orders_list

    def players_equal(self, other, ids_independent=False):
        if ids_independent:
            return all(ids_independent_equal(player1, player2) for player1, player2 in zip(self.players, other.players))
        else:
            return self.players == other.players

    def objects_equal(self, other, ids_independent=False):
        if ids_independent:
            def key_position_and_name(item):
                (key, obj) = item
                return (key, obj.position, obj.name)
            self_items = sorted(list(self.objects.items()), key=key_position_and_name)
            other_items = sorted(list(other.objects.items()), key=key_position_and_name)
            return len(self_items) == len(other_items) and \
                all(key1 == key2 and ids_independent_equal(item1, item2)
                for (key1, item1), (key2, item2) in zip(self_items, other_items))
        else:
            return set(self.objects.items()) == set(other.objects.items())

    def custom_equal(self, other, time_independent=False, ids_independent=False):
        return isinstance(other, OvercookedState) and \
            self.players_equal(other, ids_independent) and \
            self.objects_equal(other, ids_independent) and \
            self.order_lists_equal(other, ids_independent) and \
            (time_independent or self.time_equal(other))
    
    def ids_independent_equal(self, other):
        return self.custom_equal(other, ids_independent=True)

    def time_independent_equal(self, other):
        return self.custom_equal(other, time_independent=True)

    def __eq__(self, other):
        return self.custom_equal(other)

    def __hash__(self):
        orders_list_hash = hash(self.orders_list)
        return hash(
            (self.players, tuple(self.objects.values()), orders_list_hash)
        )

    def __deepcopy__(self, memo):
        return OvercookedState(
            players=[player.deepcopy() for player in self.players],
            objects={pos:obj.deepcopy() for pos, obj in self.objects.items()},
            orders_list=copy.deepcopy(self.orders_list),
            timestep=self.timestep)

    def __str__(self):
        return 'Players: {}, Objects: {}, Orders: {} Timestep: {}'.format(
            str(self.players), str(list(self.objects.values())), str(self.orders_list), str(self.timestep))

    def to_dict(self):
        return {
            "players": [p.to_dict() for p in self.players],
            "objects": [obj.to_dict() for obj in self.objects.values()],
            "orders_list" : self.orders_list.to_dict(),
            "timestep" : self.timestep
        }

    @staticmethod
    def from_dict(state_dict):
        state_dict = copy.deepcopy(state_dict)
        state_dict["players"] = [PlayerState.from_dict(p) for p in state_dict["players"]]
        object_list = [SoupState.from_dict(o) for o in state_dict["objects"]]
        state_dict["objects"] = { ob.position : ob for ob in object_list }
        return OvercookedState(**state_dict)


BASE_REW_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0,
    "TOMATO_COUNTER_PICKUP_REWARD": 0,
    "ONION_COUNTER_PICKUP_REWARD": 0,
    "TOMATO_DISPENSER_PICKUP_REWARD": 1,
    "ONION_DISPENSER_PICKUP_REWARD": 1
}

EVENT_TYPES = [
    # Tomato events
    'tomato_pickup',
    'useful_tomato_pickup',
    'tomato_drop',
    'useful_tomato_drop',
    'potting_tomato',

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
    'soup_drop',

    # Potting events
    'optimal_onion_potting',
    'optimal_tomato_potting',
    'viable_onion_potting',
    'viable_tomato_potting',
    'catastrophic_onion_potting',
    'catastrophic_tomato_potting',
    'useless_onion_potting',
    'useless_tomato_potting'
]

POTENTIAL_CONSTANTS = {
    'default' : {
        'max_delivery_steps' : 10,
        'max_pickup_steps' : 10,
        'pot_onion_steps' : 10,
        'pot_tomato_steps' : 10
    },
    'mdp_test_tomato' : {
        'max_delivery_steps' : 4,
        'max_pickup_steps' : 4,
        'pot_onion_steps' : 5,
        'pot_tomato_steps' : 6
    }
}

class OvercookedGridworld(object):
    """
    An MDP grid world based off of the Overcooked game.
    TODO: clean the organization of this class further.
    """
    DEFAULT_POTENTIAL_PARAMS = {
        "gamma": 0.99,
        "tomato_value": 13,
        "onion_value": 21
        }
    #########################
    # INSTANTIATION METHODS #
    #########################

    def __init__(self, terrain, start_player_positions, rew_shaping_params=None, layout_name="unnamed_layout", start_orders_list=None, start_all_orders=[],
        start_bonus_orders=[], num_items_for_soup=3, order_bonus=2, start_state=None, orders_to_end_episode=None, **kwargs):
        """
        terrain: a matrix of strings that encode the MDP layout
        layout_name: string identifier of the layout
        start_player_positions: tuple of positions for both players' starting positions
        rew_shaping_params: reward given for completion of specific subgoals
        num_items_for_soup: Maximum number of ingredients that can be placed in a soup
        order_bonus: Multiplicative factor for serving a bonus recipe
        start_state: Default start state returned by get_standard_start_state
        start_orders_list: OrdersList dict or object
        start_bonus_orders: Legacy param, use orders instead, list of recipes dicts that are worth a bonus
        start_all_orders: Legacy param, use orders instead, list of all available order dicts the players can make, defaults to all possible recipes if empy list provided
        orders_to_end_episode: number of fulfilled orders that ends the episode
        """
        assert not (start_bonus_orders and start_orders_list), "Use either legacy param 'start_bonus_orders' or new one 'start_orders_list', but not the both"
        if not start_orders_list:
            self._configure_recipes(start_all_orders, num_items_for_soup, **kwargs)
            start_all_orders = [r.to_dict() for r in Recipe.ALL_RECIPES] if not start_all_orders else start_all_orders
            start_orders_list = OrdersList.from_recipes_lists(start_all_orders, start_bonus_orders)
        elif isinstance(start_orders_list, OrdersList):
            if not start_all_orders:
                start_all_orders = [r.to_dict() for r in start_orders_list.all_recipes]
            self._configure_recipes(start_all_orders, num_items_for_soup, **kwargs)
        else:
            if not start_all_orders:
                start_all_orders = OrdersList.dict_to_all_recipes_dicts(start_orders_list)
            self._configure_recipes(start_all_orders, num_items_for_soup, **kwargs)
            start_orders_list = OrdersList(**start_orders_list)

        self.start_orders_list = start_orders_list
        self.height = len(terrain)
        self.width = len(terrain[0])
        self.shape = (self.width, self.height)
        self.terrain_mtx = terrain
        self.terrain_pos_dict = self._get_terrain_type_pos_dict()
        self.start_player_positions = start_player_positions
        self.num_players = len(start_player_positions)
        self.reward_shaping_params = BASE_REW_SHAPING_PARAMS if rew_shaping_params is None else rew_shaping_params
        self.layout_name = layout_name
        self.order_bonus = order_bonus
        self.start_state = start_state
        self.orders_to_end_episode = orders_to_end_episode
        self._opt_recipe_discount_cache = {}
        self._opt_recipe_cache = {}
        self._prev_potential_params = {}
        self._recipe_to_one_hot_encoding_index = {Recipe.from_dict(r): i for i, r in enumerate(self.start_all_orders)}

    @property
    def start_all_orders(self):
        return [r.to_dict() for r in self.start_orders_list.all_recipes]

    @property
    def start_bonus_orders(self):
        return [r.to_dict() for r in self.start_orders_list.bonus_recipes]

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
        if 'start_state' in base_layout_params:
            base_layout_params['start_state'] = OvercookedState.from_dict(base_layout_params['start_state'])

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
        mdp_config = copy.deepcopy(base_layout_params)

        layout_grid = [[c for c in row] for row in layout_grid]
        OvercookedGridworld._assert_valid_grid(layout_grid)

        if "layout_name" not in mdp_config:
            layout_name = "|".join(["".join(line) for line in layout_grid])
            mdp_config["layout_name"] = layout_name

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
        self.recipe_config = {
            "num_items_for_soup" : num_items_for_soup,
            "all_orders" : start_all_orders,
            **kwargs
        }
        Recipe.configure(self.recipe_config)

    #####################
    # BASIC CLASS UTILS #
    #####################

    def __eq__(self, other):
        return self.ids_independent_equal(other) and self.start_orders_list == other.start_orders_list

    def ids_and_reward_shaping_independent_equal(self, other):
        return np.array_equal(self.terrain_mtx, other.terrain_mtx) and \
                self.start_player_positions == other.start_player_positions and \
                ids_independent_equal(self.start_orders_list, other.start_orders_list) and \
                self.layout_name == other.layout_name
    
    def ids_independent_equal(self, other):
        return np.array_equal(self.terrain_mtx, other.terrain_mtx) and \
                self.start_player_positions == other.start_player_positions and \
                ids_independent_equal(self.start_orders_list, other.start_orders_list) and \
                self.reward_shaping_params == other.reward_shaping_params and \
                self.layout_name == other.layout_name

    def copy(self):
        return OvercookedGridworld(
            terrain=self.terrain_mtx.copy(),
            start_player_positions=self.start_player_positions,
            start_orders_list=copy.deepcopy(self.start_orders_list),
            rew_shaping_params=copy.deepcopy(self.reward_shaping_params),
            layout_name=self.layout_name,
        )

    @property
    def mdp_params(self):
        return {
            "layout_name": self.layout_name,
            "terrain": self.terrain_mtx,
            "start_player_positions": self.start_player_positions,
            "start_orders_list": self.start_orders_list.to_dict(),
            "rew_shaping_params": copy.deepcopy(self.reward_shaping_params),
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
        if self.start_state:
            return self.start_state
        start_state = OvercookedState.from_player_positions(
            self.start_player_positions, orders_list=copy.deepcopy(self.start_orders_list),
        )
        return start_state

    def get_random_start_state_fn(self, random_start_pos=False, rnd_obj_prob_thresh=0.0):
        def start_state_fn():
            if random_start_pos:
                valid_positions = self.get_valid_joint_player_positions()
                start_pos = valid_positions[np.random.choice(len(valid_positions))]
            else:
                start_pos = self.start_player_positions

            start_state = OvercookedState.from_player_positions(start_pos, orders_list=copy.deepcopy(self.start_orders_list))

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
        # There is a finite horizon, handled by the environment unless self.orders_to_end_episode is bigger than 0
        if self.orders_to_end_episode and self.orders_to_end_episode <= len(state.orders_list.fulfilled_orders):
            return True
        else:
            return False

    def get_state_transition(self, state, joint_action, display_phi=False, motion_planner=None):
        """Gets information about possible transitions for the action.

        Returns the next state, sparse reward and reward shaping.
        Assumes all actions are deterministic.

        NOTE: Sparse reward is given only when soups are delivered, 
        shaped reward is given only for completion of subgoals 
        (not soup deliveries).
        """
        events_infos = { event : [False] * self.num_players for event in EVENT_TYPES }
        events_list = []

        assert not self.is_terminal(state), "Trying to find successor of a terminal state: {}".format(state)
        for action, action_set in zip(joint_action, self.get_actions(state)):
            if action not in action_set:
                raise ValueError("Illegal action %s in state %s" % (action, state))
        
        new_state = state.deepcopy()

        # Resolve interacts first
        sparse_reward_by_agent, shaped_reward_by_agent = self.resolve_interacts(new_state, joint_action, events_infos, events_list)

        assert new_state.player_positions == state.player_positions
        assert new_state.player_orientations == state.player_orientations
        
        # Resolve player movements
        self.resolve_movement(new_state, joint_action)

        # Finally, environment effects
        self.step_environment_effects(new_state)
        sparse_expired_orders_reward = new_state.orders_list.step()
        sparse_env_reward = [sparse_expired_orders_reward]

        # Additional dense reward logic
        # shaped_reward += self.calculate_distance_based_shaped_reward(state, new_state)
        infos = {
            "event_infos": events_infos,
            "events_list": events_list,
            "sparse_reward_by_agent": sparse_reward_by_agent,
            "shaped_reward_by_agent": shaped_reward_by_agent,
            "sparse_env_reward": sparse_env_reward,
            "sparse_reward_sum": sum(sparse_reward_by_agent) + sum(sparse_env_reward)
        }
        if display_phi:
            assert motion_planner is not None, "motion planner must be defined if display_phi is true"
            infos["phi_s"] = self.potential_function(state, motion_planner)
            infos["phi_s_prime"] = self.potential_function(new_state, motion_planner)
        return new_state, infos

    def resolve_interacts(self, new_state, joint_action, events_infos, events_list):
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
                    # Drop object on counter
                    obj = player.remove_object()
                    new_state.add_object(obj, i_pos)
                    self.log_object_drop(events_infos, events_list, new_state, obj, pot_states, player_idx)
                    
                elif not player.has_object() and new_state.has_object(i_pos):
                    # Pick up object from counter
                    obj = new_state.remove_object(i_pos)
                    player.set_object(obj)
                    self.log_object_pickup(events_infos, events_list, new_state, obj, pot_states, player_idx)
                    if obj.name == "onion":
                        shaped_reward[player_idx] += self.reward_shaping_params["ONION_COUNTER_PICKUP_REWARD"]
                    elif obj.name == "tomato":
                        shaped_reward[player_idx] += self.reward_shaping_params["TOMATO_COUNTER_PICKUP_REWARD"]
            elif terrain_type == 'O' and player.held_object is None:
                # Onion pickup from dispenser
                obj = ObjectState('onion', pos)
                player.set_object(obj)
                self.log_object_pickup(events_infos, events_list, new_state, obj, pot_states, player_idx)
                shaped_reward[player_idx] += self.reward_shaping_params["ONION_DISPENSER_PICKUP_REWARD"]
            elif terrain_type == 'T' and player.held_object is None:
                # Tomato pickup from dispenser
                obj = ObjectState('tomato', pos)
                player.set_object(obj)
                self.log_object_pickup(events_infos, events_list, new_state, obj, pot_states, player_idx)
                shaped_reward[player_idx] += self.reward_shaping_params["TOMATO_DISPENSER_PICKUP_REWARD"]
            elif terrain_type == 'D' and player.held_object is None:
                # Give shaped reward if pickup is useful
                if self.is_dish_pickup_useful(new_state, pot_states):
                    shaped_reward[player_idx] += self.reward_shaping_params["DISH_PICKUP_REWARD"]
                # Perform dish pickup from dispenser
                obj = ObjectState('dish', pos)
                player.set_object(obj)
                self.log_object_pickup(events_infos, events_list, new_state, obj, pot_states, player_idx)

            elif terrain_type == 'P' and not player.has_object():
                # Cooking soup
                if self.soup_to_be_cooked_at_location(new_state, i_pos):
                    soup = new_state.get_object(i_pos)
                    soup.begin_cooking()
            
            elif terrain_type == 'P' and player.has_object():

                if player.get_object().name == 'dish' and self.soup_ready_at_location(new_state, i_pos):
                    # Pick up soup
                    player.remove_object() # Remove the dish
                    obj = new_state.remove_object(i_pos) # Get soup
                    player.set_object(obj)
                    self.log_object_pickup(events_infos, events_list, new_state, obj, pot_states, player_idx)
                    shaped_reward[player_idx] += self.reward_shaping_params["SOUP_PICKUP_REWARD"]

                elif player.get_object().name in Recipe.ALL_INGREDIENTS:
                    # Adding ingredient to soup

                    if not new_state.has_object(i_pos):
                        # Pot was empty, add soup to it
                        new_state.add_object(SoupState(i_pos, ingredients=[]))

                    # Add ingredient if possible
                    soup = new_state.get_object(i_pos)
                    if not soup.is_full:
                        old_soup = soup.deepcopy()
                        obj = player.remove_object()
                        soup.add_ingredient(obj)
                        self.log_object_potting(events_infos, events_list, new_state, old_soup, soup, obj, player_idx)
                        shaped_reward[player_idx] += self.reward_shaping_params["PLACEMENT_IN_POT_REW"]

            elif terrain_type == 'S' and player.has_object():
                obj = player.get_object()
                if obj.name == 'soup':
                    delivery_rew = self.deliver_soup(new_state, player, obj)
                    sparse_reward[player_idx] += delivery_rew
                    # Log soup delivery
                    self.log_soup_delivery(events_infos, events_list, obj, player_idx)

        return sparse_reward, shaped_reward

    def get_recipe_value(self, state, recipe, discounted=False, base_recipe=None, potential_params={}, steps_into_future=None, include_expire_penalty=True):
        """
        Return the reward the player should receive for delivering this recipe after 'steps_into_future' timesteps (default None is for enabling calculation of timesteps),
         by default includes expire penalty as it calculates the comparative value of the order from recipe with the value of letting the order to expire (if it's temporary order)
        """
        if not discounted:
            if not recipe:
                return 0
            matching_order = state.orders_list.get_matching_order(recipe=recipe, available_after_n_timesteps=steps_into_future or 0)
            if not matching_order:
                return 0
            reward = matching_order.calculate_future_reward(steps_into_future or 0)
            if matching_order.is_bonus:
                reward *= self.order_bonus

            if include_expire_penalty:
                reward += matching_order.expire_penalty
            return reward
        else:
            # Calculate missing ingredients needed to complete recipe
            missing_ingredients = list(recipe.ingredients)
            prev_ingredients = list(base_recipe.ingredients) if base_recipe else []
            for ingredient in prev_ingredients:
                missing_ingredients.remove(ingredient)
            n_tomatoes = len([i for i in missing_ingredients if i == Recipe.TOMATO])
            n_onions = len([i for i in missing_ingredients if i == Recipe.ONION])
            if steps_into_future is None:
                gamma, pot_onion_steps, pot_tomato_steps = potential_params['gamma'], potential_params['pot_onion_steps'], potential_params['pot_tomato_steps']
                steps_into_future = pot_onion_steps * n_onions + pot_tomato_steps * n_tomatoes + recipe.time
            else:
                gamma = potential_params['gamma']
            return gamma**steps_into_future * self.get_recipe_value(state, recipe, discounted=False, 
                steps_into_future=steps_into_future, include_expire_penalty=include_expire_penalty)

    def deliver_soup(self, state, player, soup):
        """
        Deliver the soup, and get reward if there is no order list
        or if the type of the delivered soup matches the next order.
        """
        assert soup.name == 'soup', "Tried to deliver something that wasn't soup"
        assert soup.is_ready, "Tried to deliever soup that isn't ready"
        player.remove_object()
        order = state.orders_list.fulfill_order(soup.recipe)
        if not order:
            return 0
        reward = order.calculate_reward()
        if order.is_bonus:
            reward *= self.order_bonus
        return reward


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
                assert soup.name == 'soup', "soup at " + pot_pos + " is not a soup but a " + soup.name
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

    def get_non_empty_pots(self, pot_states):
        return self.get_full_pots(pot_states) + self.get_partially_full_pots(pot_states)

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

    def find_free_counters_valid_for_both_players(self, state, mlam):
        """Finds all empty counter locations that are accessible to both players"""
        one_player, other_player = state.players
        free_counters = self.get_empty_counter_locations(state)
        free_counters_valid_for_both = []
        for free_counter in free_counters:
            goals = mlam.motion_planner.motion_goals_for_pos[free_counter]
            if any([mlam.motion_planner.is_valid_motion_start_goal_pair(one_player.pos_and_or, goal) for goal in goals]) and \
            any([mlam.motion_planner.is_valid_motion_start_goal_pair(other_player.pos_and_or, goal) for goal in goals]):
                free_counters_valid_for_both.append(free_counter)
        return free_counters_valid_for_both

    def _get_optimal_possible_recipe(self, state, recipe, discounted, potential_params, return_value):
        """
        Traverse the recipe-space graph using DFS to find the best possible recipe that can be made
        from the current recipe

        Because we can't have empty recipes, we handle the case by letting recipe==None be a stand-in for empty recipe
        """
        start_recipe = recipe
        visited = set()
        stack = []
        best_recipe = recipe
        best_value = 0
        if not recipe:
            for ingredient in Recipe.ALL_INGREDIENTS:
                stack.append(Recipe([ingredient]))
        else:
            stack.append(recipe)

        while stack:
            curr_recipe = stack.pop()
            if curr_recipe not in visited:
                visited.add(curr_recipe)
                curr_value = self.get_recipe_value(state, curr_recipe, base_recipe=start_recipe, discounted=discounted, potential_params=potential_params)
                if curr_value > best_value:
                    best_value, best_recipe = curr_value, curr_recipe
                
                for neighbor in curr_recipe.neighbors():
                    if not neighbor in visited:
                        stack.append(neighbor)
        
        if return_value:
            return best_recipe, best_value
        return best_recipe

    def get_optimal_possible_recipe(self, state, recipe, discounted=False, potential_params={}, return_value=False):
        """
        Return the best possible recipe that can be made starting with ingredients in `recipe`
        Uses self._optimal_possible_recipe as a cache to avoid re-computing. This only works because
        the recipe values are currently static (i.e. bonus_orders doesn't change). Would need to have cache
        flushed if order dynamics are introduced
        """
        save_cache = (not state.orders_list.contains_temporary_orders) and (not state.orders_list.is_adding_orders)
        clear_cache = (discounted and self._prev_potential_params != potential_params) or not save_cache

        if clear_cache:
            if discounted:
                self._opt_recipe_discount_cache = {}
            else:
                self._opt_recipe_cache = {}
        
        if save_cache:
            if discounted:
                cache = self._opt_recipe_discount_cache
            else:
                cache = self._opt_recipe_cache
        else:
            cache = {}

        if discounted and save_cache:
            self._prev_potential_params = potential_params

        if recipe not in cache:
            # Compute best recipe now and store in cache for later use
            opt_recipe, value = self._get_optimal_possible_recipe(state, recipe, discounted=discounted, potential_params=potential_params, return_value=True)
            cache[recipe] = (opt_recipe, value)

        # Return best recipe (and value) from cache
        if return_value:
            return cache[recipe]
        return cache[recipe][0]
        



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
    def log_object_potting(self, events_infos, events_list, state, old_soup, new_soup, obj, player_index):
        """Player added an ingredient to a pot"""
        obj_pickup_key = "potting_" + obj.name
        if obj_pickup_key not in events_infos:
            raise ValueError("Unknown event {}".format(obj_pickup_key))
        events_infos[obj_pickup_key][player_index] = True

        POTTING_FNS = {
            "optimal" : self.is_potting_optimal,
            "catastrophic" : self.is_potting_catastrophic,
            "viable" : self.is_potting_viable,
            "useless" : self.is_potting_useless
        }
        outcomes = [outcome for outcome, outcome_fn in POTTING_FNS.items() if outcome_fn(state, old_soup, new_soup)]
        for outcome in outcomes:
            potting_key = "{}_{}_potting".format(outcome, obj.name)
            events_infos[potting_key][player_index] = True

        events_list.append(dict(action="potting", player=player_index, object=obj.to_dict(), adjectives=outcomes))
    
    def log_object_pickup(self, events_infos, events_list, state, obj, pot_states, player_index):
        """Player picked an object up from a counter or a dispenser"""
        obj_pickup_key = obj.name + "_pickup"
        if obj_pickup_key not in events_infos:
            raise ValueError("Unknown event {}".format(obj_pickup_key))
        events_infos[obj_pickup_key][player_index] = True
        
        USEFUL_PICKUP_FNS = {
            "tomato" : self.is_ingredient_pickup_useful,
            "onion": self.is_ingredient_pickup_useful,
            "dish": self.is_dish_pickup_useful
        }

        if obj.name in USEFUL_PICKUP_FNS.keys() and USEFUL_PICKUP_FNS[obj.name](state, pot_states, player_index):
            obj_useful_key = "useful_" + obj.name + "_pickup"
            events_infos[obj_useful_key][player_index] = True
            event_adjectives = ["useful"]
        else:
            event_adjectives = []

        events_list.append(dict(action="pickup", player=player_index, object=obj.to_dict(), adjectives=event_adjectives))
    
    def log_object_drop(self, events_infos, events_list, state, obj, pot_states, player_index):
        """Player dropped the object on a counter"""
        obj_drop_key = obj.name + "_drop"
        if obj_drop_key not in events_infos:
            raise ValueError("Unknown event {}".format(obj_drop_key))
        events_infos[obj_drop_key][player_index] = True
        
        USEFUL_DROP_FNS = {
            "tomato" : self.is_ingredient_drop_useful,
            "onion": self.is_ingredient_drop_useful,
            "dish": self.is_dish_drop_useful
        }

        if obj.name in USEFUL_DROP_FNS.keys() and USEFUL_DROP_FNS[obj.name](state, pot_states, player_index):
            obj_useful_key = "useful_" + obj.name + "_drop"
            events_infos[obj_useful_key][player_index] = True
            event_adjectives = ["useful"]
        else:
            event_adjectives = []

        events_list.append(dict(action="drop", player=player_index, object=obj.to_dict(), adjectives=event_adjectives))
    
    def log_soup_delivery(self, events_infos, events_list, obj, player_index):
        events_infos['soup_delivery'][player_index] = True
        events_list.append(dict(action="delivery", player=player_index, object=obj.to_dict(), adjectives=[]))

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
        - Ingredient is needed (all pots are non-full)
        - Nobody is holding ingredient
        """
        if self.num_players != 2: return False
        all_non_full = len(self.get_full_pots(pot_states)) == 0
        other_player = state.players[1 - player_index]
        other_player_holding_ingredient = other_player.has_object() and other_player.get_object().name in Recipe.ALL_INGREDIENTS
        return all_non_full and not other_player_holding_ingredient

    def is_ingredient_pickup_useful(self, state, pot_states, player_index):
        """
        NOTE: this only works if self.num_players == 2
        Always useful unless:
        - All pots are full & other agent is not holding a dish
        Can make it more accurate by checking if ingredient is actually used in possible recipes
        """
        if self.num_players != 2: return False
        all_pots_full = self.num_pots == len(self.get_full_pots(pot_states))
        other_player = state.players[1 - player_index]
        other_player_has_dish = other_player.has_object() and other_player.get_object().name == "dish"
        return not (all_pots_full and not other_player_has_dish)
        
    def is_ingredient_drop_useful(self, state, pot_states, player_index):
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

    def is_potting_optimal(self, state, old_soup, new_soup):
        """
        True if the highest valued soup possible is the same before and after the potting
        NOTE: It is not work perfectly for dynamic (temporary) orders - it does not take into account:
            - optimal order of fulfilling orders (always takes most high value in the current moment)
            - future orders that can be added to order list
        """
        old_recipe = Recipe(old_soup.ingredients) if old_soup.ingredients else None
        new_recipe = Recipe(new_soup.ingredients)
        old_val = self.get_recipe_value(state, self.get_optimal_possible_recipe(state, old_recipe))
        new_val = self.get_recipe_value(state, self.get_optimal_possible_recipe(state, new_recipe))
        return new_val >= old_val

    def is_potting_viable(self, state, old_soup, new_soup):
        """
        True if there exists a non-zero reward soup possible from new ingredients
        NOTE: does not keep care about orders that did not appeared yet
        """
        new_recipe = Recipe(new_soup.ingredients)
        new_val = self.get_recipe_value(state, self.get_optimal_possible_recipe(state, new_recipe))
        return new_val > 0

    def is_potting_catastrophic(self, state, old_soup, new_soup):
        """
        True if no non-zero reward soup is possible from new ingredients
        NOTE: does not keep care about orders that did not appeared yet
        """
        old_recipe = Recipe(old_soup.ingredients) if old_soup.ingredients else None
        new_recipe = Recipe(new_soup.ingredients)
        old_val = self.get_recipe_value(state, self.get_optimal_possible_recipe(state, old_recipe))
        new_val = self.get_recipe_value(state, self.get_optimal_possible_recipe(state, new_recipe))
        return old_val > 0 and new_val == 0

    def is_potting_useless(self, state, old_soup, new_soup):
        """
        True if ingredient added to a soup that was already gauranteed to be worth at most 0 points
        NOTE: does not keep care about orders that did not appeared yet
        """
        old_recipe = Recipe(old_soup.ingredients) if old_soup.ingredients else None
        old_val = self.get_recipe_value(state, self.get_optimal_possible_recipe(state, old_recipe))
        return old_val == 0



    #####################
    # TERMINAL GRAPHICS #
    #####################

    def state_string(self, state):
        """String representation of the current state"""
        players_dict = {player.position: player for player in state.players}

        grid_string = ""
        for y, terrain_row in enumerate(self.terrain_mtx):
            for x, element in enumerate(terrain_row):
                grid_string_add = ""
                if (x, y) in players_dict.keys():
                    player = players_dict[(x, y)]
                    orientation = player.orientation
                    assert orientation in Direction.ALL_DIRECTIONS

                    player_idx_lst = [i for i, p in enumerate(state.players) if p.position == player.position]
                    assert len(player_idx_lst) == 1

                    grid_string_add += Action.ACTION_TO_CHAR[orientation]
                    player_object = player.held_object
                    if player_object:
                        grid_string_add += str(player_idx_lst[0])
                        if player_object.name[0] == "s":
                            # this is a soup
                            grid_string_add += str(player_object)
                        else:
                            grid_string_add += player_object.name[:1]
                    else:
                        grid_string_add += str(player_idx_lst[0])
                else:
                    grid_string_add += element
                    if element == "X" and state.has_object((x, y)):
                        state_obj = state.get_object((x, y))
                        if state_obj.name[0] == "s":
                            grid_string_add += str(state_obj)
                        else:
                            grid_string_add += state_obj.name[:1]

                    elif element == "P" and state.has_object((x, y)):
                        soup = state.get_object((x, y))
                        # display soup
                        grid_string_add += str(soup)

                grid_string += grid_string_add
                grid_string += "".join([" "] * (7 - len(grid_string_add)))
                grid_string += " "

            grid_string += "\n\n"
        
        if state.bonus_orders:
            grid_string += "Bonus orders: {}\n".format(
                state.bonus_orders
            )
        # grid_string += "State potential value: {}\n".format(self.potential_function(state))
        return grid_string

    ###################
    # STATE ENCODINGS #
    ###################
    def multi_hot_orders_encoding_single_agent(self, overcooked_state, skip_not_found_recipes=True):
        result = np.zeros(self.multi_hot_orders_encoding_shape)
        for order in overcooked_state.orders_list.orders:
                idx = self.recipe_one_hot_encoding_index(order.recipe, allow_not_found=skip_not_found_recipes)
                if idx is None:
                    assert skip_not_found_recipes
                    # do that to be backward compatible with pickled states i.e. in human behavioral cloning data in harl repo 
                    # that have all recipes by default
                    pass
                else:
                    result[idx] = 1
        return result
    
    def multi_hot_orders_encoding(self, overcooked_state):
        return [self.multi_hot_orders_encoding_single_agent(overcooked_state)]*self.num_players

    def recipe_one_hot_encoding_index(self, recipe, allow_not_found=False, not_found_value=None):
        if allow_not_found:
            return self._recipe_to_one_hot_encoding_index.get(recipe, not_found_value)
        else:
            return self._recipe_to_one_hot_encoding_index[recipe]

    @property
    def multi_hot_orders_encoding_shape(self):
        return np.array([len(self._recipe_to_one_hot_encoding_index)])

    @property
    def multi_hot_orders_encoding_gym_space(self):
        return gym.spaces.MultiBinary(self.multi_hot_orders_encoding_shape)
    
    @property
    def lossless_state_encoding_shape(self):
        return np.array(list(self.shape) + [26])

    @property
    def lossless_state_encoding_gym_space(self):
        return gym.spaces.Box(
            low=np.ones(self.lossless_state_encoding_shape) * 0,
            high=np.ones(self.lossless_state_encoding_shape) * float("inf"), 
            dtype=np.float32)

    @property
    def sparse_categorical_joint_action_encoding_shape(self):
        return np.array([self.num_players, 1])

    def sparse_categorical_joint_action_encoding(self, joint_action):
        return np.array([[Action.ACTION_TO_INDEX[a]] for a in joint_action]).astype(int)
    
    @property
    def one_hot_joint_action_encoding_shape(self):
        return np.array([self.num_players, len(Action.ACTION_TO_INDEX)])

    def one_hot_joint_action_encoding(self, joint_action):
        # can be improved if too slow https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
        result = np.zeros(self.one_hot_joint_action_encoding_shape)
        for player_idx, action in enumerate(joint_action):
            result[player_idx][Action.ACTION_TO_INDEX[action]] = 1
        return result
        
    def lossless_state_encoding(self, overcooked_state, horizon=400, debug=False):
        """Featurizes a OvercookedState object into a stack of boolean masks that are easily readable by a CNN"""
        assert self.num_players == 2, "Functionality has to be added to support encondings for > 2 players"
        assert type(debug) is bool
        base_map_features = ["pot_loc", "counter_loc", "onion_disp_loc", "tomato_disp_loc",
                             "dish_disp_loc", "serve_loc"]
        variable_map_features = ["onions_in_pot", "tomatoes_in_pot", "onions_in_soup", "tomatoes_in_soup",
                                 "soup_cook_time_remaining", "soup_done", "dishes", "onions", "tomatoes"]
        urgency_features = ["urgency"]
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

            # LAYERS = ordered_player_features + base_map_features + variable_map_features
            LAYERS = ordered_player_features + base_map_features + variable_map_features + urgency_features
            state_mask_dict = {k:np.zeros(self.shape) for k in LAYERS}

            # MAP LAYERS
            if horizon - overcooked_state.timestep < 40:
                state_mask_dict["urgency"] = np.ones(self.shape)

            for loc in self.get_counter_locations():
                state_mask_dict["counter_loc"][loc] = 1

            for loc in self.get_pot_locations():
                state_mask_dict["pot_loc"][loc] = 1

            for loc in self.get_onion_dispenser_locations():
                state_mask_dict["onion_disp_loc"][loc] = 1

            for loc in self.get_tomato_dispenser_locations():
                state_mask_dict["tomato_disp_loc"][loc] = 1

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
                    # removed the next line because onion doesn't have to be in all the soups?
                    # if Recipe.ONION in obj.ingredients:
                    # get the ingredients into a {object: number} dictionary
                    ingredients_dict = Counter(obj.ingredients)
                    # assert "onion" in ingredients_dict.keys()
                    if obj.position in self.get_pot_locations():
                        if obj.is_idle:
                            # onions_in_pot and tomatoes_in_pot are used when the soup is idling, and ingredients could still be added
                            state_mask_dict["onions_in_pot"] += make_layer(obj.position, ingredients_dict["onion"])
                            state_mask_dict["tomatoes_in_pot"] += make_layer(obj.position, ingredients_dict["tomato"])
                        else:
                            state_mask_dict["onions_in_soup"] += make_layer(obj.position, ingredients_dict["onion"])
                            state_mask_dict["tomatoes_in_soup"] += make_layer(obj.position, ingredients_dict["tomato"])
                            state_mask_dict["soup_cook_time_remaining"] += make_layer(obj.position, obj.cook_time - obj._cooking_tick)
                            if obj.is_ready:
                                state_mask_dict["soup_done"] += make_layer(obj.position, 1)

                    else:
                        # If player soup is not in a pot, treat it like a soup that is cooked with remaining time 0
                        state_mask_dict["onions_in_soup"] += make_layer(obj.position, ingredients_dict["onion"])
                        state_mask_dict["tomatoes_in_soup"] += make_layer(obj.position, ingredients_dict["tomato"])
                        state_mask_dict["soup_done"] += make_layer(obj.position, 1)

                elif obj.name == "dish":
                    state_mask_dict["dishes"] += make_layer(obj.position, 1)
                elif obj.name == "onion":
                    state_mask_dict["onions"] += make_layer(obj.position, 1)
                elif obj.name == "tomato":
                    state_mask_dict["tomatoes"] += make_layer(obj.position, 1)
                else:
                    raise ValueError("Unrecognized object")

            if debug:
                print("terrain----")
                print(np.array(self.terrain_mtx))
                print("-----------")
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

    @property
    def featurize_state_gym_space(self):
        return gym.spaces.Box(
            low=np.ones(self.featurize_state_shape) * -10,
            high=np.ones(self.featurize_state_shape) * 10, 
            dtype=np.float32)

    def featurize_state(self, overcooked_state, mlam, horizon=400):
        """
        Encode state with some manually designed features.
        NOTE: currently works for just two players.
        """

        all_features = {}

        def make_closest_feature(idx, name, locations):
            "Compute (x, y) deltas to closest feature of type `name`, and save it in the features dict"
            all_features["p{}_closest_{}".format(idx, name)] = self.get_deltas_to_closest_location(player, locations,
                                                                                                   mlam)
        ingredients = self.possibble_ingredients_from_dispensers
        assert ingredients == ["onion"], "featurization of the state works for layouts containing no ingredient dispensers other than onion"
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

    @property
    def possibble_ingredients_from_dispensers(self):
        terrain_types = set(np.array(self.terrain_mtx).flatten())
        ingredients = []
        if "O" in terrain_types:
            ingredients.append("onion")
        if "T" in terrain_types:
            ingredients.append("tomato")
        return ingredients

    def get_deltas_to_closest_location(self, player, locations, mlam):
        _, closest_loc = mlam.motion_planner.min_cost_to_feature(player.pos_and_or, locations, with_argmin=True)
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

    def potential_function(self, state, mp, gamma=None):
        """
        Essentially, this is the ɸ(s) function.

        The main goal here to to approximately infer the actions of an optimal agent, and derive an estimate for the value
        function of the optimal policy. The perfect potential function is indeed the value function

        At a high level, we assume each agent acts independetly, and greedily optimally, and then, using the decay factor "gamma", 
        we calculate the expected discounted reward under this policy
        
        Some implementation details:
            * the process of delivering a soup is broken into 4 steps
                * Step 1: placing the first ingredient into an empty pot
                * Step 2: placing the remaining ingredients in the pot
                * Step 3: cooking the soup/retreiving a dish with which to serve the soup
                * Step 4: delivering the soup once it is in a dish
            * Here is an exhaustive list of the greedy assumptions made at each step
                * step 1:
                    * If an agent is holding an ingredient that could be used to cook an optimal soup, it will use it in that soup
                    * If no such optimal soup exists, but there is an empty pot, the agent will place the ingredient there
                    * If neither of the above cases holds, no potential is awarded for possessing the ingredient
                * step 2:
                    * The agent will always try to cook the highest valued soup possible based on the current ingredients in a pot
                    * Any agent possessing a missing ingredient for an optimal soup will travel directly to the closest such pot
                    * If the optimal soup has all ingredients, the closest agent not holding anything will go to cook it
                * step 3:
                    * Any player holding a dish attempts to serve the highest valued soup based on recipe values and cook time remaining
                * step 4:
                    * Any agent holding a soup will go directly to the nearest serving area
            * At every step, the expected reward is discounted by multiplying the optimal reward by gamma ^ (estimated #steps to complete greedy action)
            * In the case that certain actions are infeasible (i.e. an agent is holding a soup in step 4, but no path exists to a serving
              area), estimated number of steps in order to complete the action defaults to `max_steps`
            * Cooperative behavior between the two agents is not considered for complexity reasons
            * Soups that are worth <1 points are rounded to be worth 1 point. This is to incentivize the agent to cook a worthless soup
              that happens to be in a pot in order to free up the pot

        Parameters:
            state: OvercookedState instance representing the state to evaluate potential for
            mp: MotionPlanner instance used to calculate gridworld distances to objects
            gamma: float, discount factor
            max_steps: int, number of steps a high level action is assumed to take in worst case
        
        Returns
            phi(state), the potential of the state
        """
        if not hasattr(Recipe, '_tomato_value') or not hasattr(Recipe, '_onion_value'):
            raise ValueError("Potential function requires Recipe onion and tomato values to work properly")

        gamma = gamma if gamma is not None else self.DEFAULT_POTENTIAL_PARAMS["gamma"]
        # Constants needed for potential function
        potential_params = {
            'gamma' : gamma,
            'tomato_value' : Recipe._tomato_value if Recipe._tomato_value else self.DEFAULT_POTENTIAL_PARAMS["tomato_value"],
            'onion_value' : Recipe._onion_value if Recipe._tomato_value else self.DEFAULT_POTENTIAL_PARAMS["onion_value"],
            **POTENTIAL_CONSTANTS.get(self.layout_name, POTENTIAL_CONSTANTS['default'])
        }

        pot_states = self.get_pot_states(state)

        # Base potential value is the geometric sum of making optimal soups infinitely
        opt_recipe, discounted_opt_recipe_value = self.get_optimal_possible_recipe(state, None, discounted=True, potential_params=potential_params, return_value=True)
        opt_recipe_value = self.get_recipe_value(state, opt_recipe)
        discount = discounted_opt_recipe_value / opt_recipe_value
        steady_state_value = (discount / (1 - discount)) * opt_recipe_value
        potential = steady_state_value

        # Get list of all soups that have >0 ingredients, sorted based on value of best possible recipe 
        idle_soups = [state.get_object(pos) for pos in self.get_full_but_not_cooking_pots(pot_states)]
        idle_soups.extend([state.get_object(pos) for pos in self.get_partially_full_pots(pot_states)])
        idle_soups = sorted(idle_soups, key=lambda soup : self.get_optimal_possible_recipe(state, Recipe(soup.ingredients), discounted=True, potential_params=potential_params, return_value=True)[1], reverse=True)

        # Build mapping of non_idle soups to the potential value each one will contribue
        # Default potential value is maximimal discount for last two steps applied to optimal recipe value
        cooking_soups = [state.get_object(pos) for pos in self.get_cooking_pots(pot_states)]
        done_soups = [state.get_object(pos) for pos in self.get_ready_pots(pot_states)]
        non_idle_soup_vals = { soup : gamma**(potential_params['max_delivery_steps'] + max(potential_params['max_pickup_steps'], soup.cook_time - soup._cooking_tick)) * max(self.get_recipe_value(state, soup.recipe), 1) for soup in cooking_soups + done_soups }

        # Get descriptive list of players based on different attributes
        # Note that these lists are mutually exclusive
        players_holding_soups = [player for player in state.players if player.has_object() and player.get_object().name == 'soup']
        players_holding_dishes = [player for player in state.players if player.has_object() and player.get_object().name == 'dish']
        players_holding_tomatoes = [player for player in state.players if player.has_object() and player.get_object().name == Recipe.TOMATO]
        players_holding_onions = [player for player in state.players if player.has_object() and player.get_object().name == Recipe.ONION]
        players_holding_nothing = [player for player in state.players if not player.has_object()]

        ### Step 4 potential ###

        # Add potential for each player with a soup
        for player in players_holding_soups:
            # Even if delivery_dist is infinite, we still award potential (as an agent might need to pass the soup to other player first)
            delivery_dist = mp.min_cost_to_feature(player.pos_and_or, self.terrain_pos_dict['S'])
            potential += gamma**min(delivery_dist, potential_params['max_delivery_steps']) * max(self.get_recipe_value(state, player.get_object().recipe), 1)



        ### Step 3 potential ###

        # Reweight each non-idle soup value based on agents with dishes performing greedily-optimally as outlined in docstring
        for player in players_holding_dishes:
            best_pickup_soup = None
            best_pickup_value = 0

            # find best soup to pick up with dish agent currently has
            for soup in non_idle_soup_vals:
                # How far away the soup is (inf if not-reachable)
                pickup_dist = mp.min_cost_to_feature(player.pos_and_or, [soup.position])

                # mask to award zero score if not reachable
                # Note: this means that potentially "useful" dish pickups (where agent passes dish to other agent
                # that can reach the soup) do not recive a potential bump
                is_useful = int(pickup_dist < np.inf)

                # Always assume worst-case discounting for step 4, and bump zero-valued soups to 1 as mentioned in docstring
                pickup_soup_value = gamma**potential_params['max_delivery_steps'] * max(self.get_recipe_value(state, soup.recipe), 1)
                cook_time_remaining = soup.cook_time - soup._cooking_tick
                discount = gamma**max(cook_time_remaining, min(pickup_dist, potential_params['max_pickup_steps']))

                # Final discount-adjusted value for this player pursuing this soup
                pickup_value = discount * pickup_soup_value * is_useful

                # Update best soup found for this player
                if pickup_dist < np.inf and pickup_value > best_pickup_value:
                    best_pickup_soup = soup
                    best_pickup_value = pickup_value
            
            # Set best-case score for this soup. Can only improve upon previous players policies
            # Note cooperative policies between players not considered
            if best_pickup_soup:
                non_idle_soup_vals[best_pickup_soup] = max(non_idle_soup_vals[best_pickup_soup], best_pickup_value)

        # Apply potential for each idle soup as calculated above
        for soup in non_idle_soup_vals:
            potential += non_idle_soup_vals[soup]



        ### Step 2 potential ###

        # Iterate over idle soups in decreasing order of value so we greedily prioritize higher valued soups
        for soup in idle_soups:
            # Calculate optimal recipe
            curr_recipe = Recipe(soup.ingredients)
            opt_recipe = self.get_optimal_possible_recipe(state, curr_recipe, discounted=True, potential_params=potential_params)

            # Calculate missing ingredients needed to complete optimal recipe
            missing_ingredients = list(opt_recipe.ingredients)
            for ingredient in soup.ingredients:
                missing_ingredients.remove(ingredient)

            # Base discount for steps 3-4
            discount = gamma**(max(potential_params['max_pickup_steps'], opt_recipe.time) + potential_params['max_delivery_steps'])

            # Add a multiplicative discount for each needed ingredient (this has the effect of giving more award to soups
            # that are closer to being completed)
            for ingredient in missing_ingredients:
                # Players who might have an ingredient we need
                pertinent_players = players_holding_tomatoes if ingredient == Recipe.TOMATO else players_holding_onions
                dist = np.inf
                closest_player = None

                # Find closest player with ingredient we need
                for player in pertinent_players:
                    curr_dist = mp.min_cost_to_feature(player.pos_and_or, [soup.position])
                    if curr_dist < dist:
                        dist = curr_dist
                        closest_player = player

                # Update discount to account for adding this missing ingredient (defaults to min_coeff if no pertinent players exist)
                discount *= gamma**min(dist, potential_params['pot_{}_steps'.format(ingredient)])

                # Cross off this player's ingreident contribution so it can't be double-counted
                if closest_player:
                    pertinent_players.remove(closest_player)

            # Update discount to account for time it takes to start the soup cooking once last ingredient is added
            if missing_ingredients:
                # We assume it only takes one timestep if there are missing ingredients since the agent delivering the last ingredient
                # will be at the pot already
                discount *= gamma
            else:
                # Otherwise, we assume that every player holding nothing will make a beeline to this soup since it's already optimal
                cook_dist = min([mp.min_cost_to_feature(player.pos_and_or, [soup.position]) for player in players_holding_nothing], default=np.inf)
                discount *= gamma**min(cook_dist, potential_params['max_pickup_steps'])

            potential += discount * max(self.get_recipe_value(state, opt_recipe), 1)


        ### Step 1 Potential ###

        # Add potential for each tomato that is left over after using all others to complete optimal recipes
        for player in players_holding_tomatoes:
            # will be inf if there exists no empty pot that is reachable
            dist = mp.min_cost_to_feature(player.pos_and_or, self.get_empty_pots(pot_states))
            is_useful = int(dist < np.inf)
            discount = gamma**(min(potential_params['pot_tomato_steps'], dist) + potential_params['max_pickup_steps'] + potential_params['max_delivery_steps']) * is_useful
            potential += discount * potential_params['tomato_value']

        # Add potential for each onion that is remaining after using others to complete optimal recipes if possible
        for player in players_holding_onions:
            dist = mp.min_cost_to_feature(player.pos_and_or, self.get_empty_pots(pot_states))
            is_useful = int(dist < np.inf)
            discount = gamma**(min(potential_params['pot_onion_steps'], dist) + potential_params['max_pickup_steps'] + potential_params['max_delivery_steps']) * is_useful
            potential += discount * potential_params['onion_value']

        # At last
        return potential

    ##############
    # DEPRECATED #
    ##############

    # def calculate_distance_based_shaped_reward(self, state, new_state):
    #     """
    #     Adding reward shaping based on distance to certain features.
    #     """
    #     distance_based_shaped_reward = 0
    #
    #     pot_states = self.get_pot_states(new_state)
    #     ready_pots = pot_states["tomato"]["ready"] + pot_states["onion"]["ready"]
    #     cooking_pots = ready_pots + pot_states["tomato"]["cooking"] + pot_states["onion"]["cooking"]
    #     nearly_ready_pots = cooking_pots + pot_states["tomato"]["partially_full"] + pot_states["onion"]["partially_full"]
    #     dishes_in_play = len(new_state.player_objects_by_type['dish'])
    #     for player_old, player_new in zip(state.players, new_state.players):
    #         # Linearly increase reward depending on vicinity to certain features, where distance of 10 achieves 0 reward
    #         max_dist = 8
    #
    #         if player_new.held_object is not None and player_new.held_object.name == 'dish' and len(nearly_ready_pots) >= dishes_in_play:
    #             min_dist_to_pot_new = np.inf
    #             min_dist_to_pot_old = np.inf
    #             for pot in nearly_ready_pots:
    #                 new_dist = np.linalg.norm(np.array(pot) - np.array(player_new.position))
    #                 old_dist = np.linalg.norm(np.array(pot) - np.array(player_old.position))
    #                 if new_dist < min_dist_to_pot_new:
    #                     min_dist_to_pot_new = new_dist
    #                 if old_dist < min_dist_to_pot_old:
    #                     min_dist_to_pot_old = old_dist
    #             if min_dist_to_pot_old > min_dist_to_pot_new:
    #                 distance_based_shaped_reward += self.reward_shaping_params["POT_DISTANCE_REW"] * (1 - min(min_dist_to_pot_new / max_dist, 1))
    #
    #         if player_new.held_object is None and len(cooking_pots) > 0 and dishes_in_play == 0:
    #             min_dist_to_d_new = np.inf
    #             min_dist_to_d_old = np.inf
    #             for serving_loc in self.terrain_pos_dict['D']:
    #                 new_dist = np.linalg.norm(np.array(serving_loc) - np.array(player_new.position))
    #                 old_dist = np.linalg.norm(np.array(serving_loc) - np.array(player_old.position))
    #                 if new_dist < min_dist_to_d_new:
    #                     min_dist_to_d_new = new_dist
    #                 if old_dist < min_dist_to_d_old:
    #                     min_dist_to_d_old = old_dist
    #
    #             if min_dist_to_d_old > min_dist_to_d_new:
    #                 distance_based_shaped_reward += self.reward_shaping_params["DISH_DISP_DISTANCE_REW"] * (1 - min(min_dist_to_d_new / max_dist, 1))
    #
    #         if player_new.held_object is not None and player_new.held_object.name == 'soup':
    #             min_dist_to_s_new = np.inf
    #             min_dist_to_s_old = np.inf
    #             for serving_loc in self.terrain_pos_dict['S']:
    #                 new_dist = np.linalg.norm(np.array(serving_loc) - np.array(player_new.position))
    #                 old_dist = np.linalg.norm(np.array(serving_loc) - np.array(player_old.position))
    #                 if new_dist < min_dist_to_s_new:
    #                     min_dist_to_s_new = new_dist
    #
    #                 if old_dist < min_dist_to_s_old:
    #                     min_dist_to_s_old = old_dist
    #
    #             if min_dist_to_s_old > min_dist_to_s_new:
    #                 distance_based_shaped_reward += self.reward_shaping_params["SOUP_DISTANCE_REW"] * (1 - min(min_dist_to_s_new / max_dist, 1))
    #
    #     return distance_based_shaped_reward
