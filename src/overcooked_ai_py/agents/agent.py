import itertools, math
import numpy as np
from collections import defaultdict
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import Recipe

class Agent(object):

    def __init__(self):
        self.reset()

    def action(self, state):
        """
        Should return an action, and an action info dictionary.
        If collecting trajectories of the agent with OvercookedEnv, the action
        info data will be included in the trajectory data under `ep_infos`.

        This allows agents to optionally store useful information about them
        in the trajectory for further analysis.
        """
        return NotImplementedError()

    def actions(self, states, agent_indices):
        """
        A multi-state version of the action method. This enables for parallized
        implementations that can potentially give speedups in action prediction. 

        Args:
            states (list): list of OvercookedStates for which we want actions for
            agent_indices (list): list to inform which agent we are requesting the action for in each state

        Returns:
            [(action, action_info), (action, action_info), ...]: the actions and action infos for each state-agent_index pair
        """
        return NotImplementedError()

    @staticmethod
    def a_probs_from_action(action):
        action_idx = Action.ACTION_TO_INDEX[action]
        return np.eye(Action.NUM_ACTIONS)[action_idx]

    @staticmethod
    def check_action_probs(action_probs, tolerance=1e-4):
        """Check that action probabilities sum to ≈ 1.0"""
        probs_sum = sum(action_probs)
        assert math.isclose(probs_sum, 1.0, rel_tol=tolerance), "Action probabilities {} should sum up to approximately 1 but sum up to {}".format(list(action_probs), probs_sum)

    def set_agent_index(self, agent_index):
        self.agent_index = agent_index

    def set_mdp(self, mdp):
        self.mdp = mdp

    def reset(self):
        """
        One should always reset agents in between trajectory rollouts, as resetting
        usually clears history or other trajectory-specific attributes.
        """
        self.agent_index = None
        self.mdp = None


class AgentGroup(object):
    """
    AgentGroup is a group of N agents used to sample 
    joint actions in the context of an OvercookedEnv instance.
    """

    def __init__(self, *agents, allow_duplicate_agents=False):
        self.agents = agents
        self.n = len(self.agents)
        self.reset()

        if not all(a0 is not a1 for a0, a1 in itertools.combinations(agents, 2)):
            assert allow_duplicate_agents, "All agents should be separate instances, unless allow_duplicate_agents is set to true"

    def joint_action(self, state):
        actions_and_probs_n = tuple(a.action(state) for a in self.agents)
        return actions_and_probs_n

    def set_mdp(self, mdp):
        for a in self.agents:
            a.set_mdp(mdp)

    def reset(self):
        """
        When resetting an agent group, we know that the agent indices will remain the same,
        but we have no guarantee about the mdp, that must be set again separately.
        """
        for i, agent in enumerate(self.agents):
            agent.reset()
            agent.set_agent_index(i)


class AgentPair(AgentGroup):
    """
    AgentPair is the N=2 case of AgentGroup. Unlike AgentGroup,
    it supports having both agents being the same instance of Agent.

    NOTE: Allowing duplicate agents (using the same instance of an agent
    for both fields can lead to problems if the agents have state / history)
    """

    def __init__(self, *agents, allow_duplicate_agents=False): 
        super().__init__(*agents, allow_duplicate_agents=allow_duplicate_agents)
        assert self.n == 2
        self.a0, self.a1 = self.agents

    def joint_action(self, state):
        if self.a0 is self.a1:
            # When using the same instance of an agent for self-play,
            # reset agent index at each turn to prevent overwriting it
            self.a0.set_agent_index(0)
            action_and_infos_0 = self.a0.action(state)
            self.a1.set_agent_index(1)
            action_and_infos_1 = self.a1.action(state)
            joint_action_and_infos = (action_and_infos_0, action_and_infos_1)
            return joint_action_and_infos
        else:
            return super().joint_action(state)

class NNPolicy(object):
    """
    This is a common format for NN-based policies. Once one has wrangled the intended trained neural net
    to this format, one can then easily create an Agent with the AgentFromPolicy class.
    """

    def __init__(self):
        pass

    def multi_state_policy(self, states, agent_indices):
        """
        A function that takes in multiple OvercookedState instances and their respective agent indices and returns action probabilities.
        """
        raise NotImplementedError()

    def multi_obs_policy(self, states):
        """
        A function that takes in multiple preprocessed OvercookedState instatences and returns action probabilities.
        """
        raise NotImplementedError()


class AgentFromPolicy(Agent):
    """
    This is a useful Agent class backbone from which to subclass from NN-based agents.
    """
    
    def __init__(self, policy):
        """
        Takes as input an NN Policy instance
        """
        self.policy = policy
        self.reset()
        
    def action(self, state):
        return self.actions([state], [self.agent_index])[0]

    def actions(self, states, agent_indices):
        action_probs_n = self.policy.multi_state_policy(states, agent_indices)
        actions_and_infos_n = []
        for action_probs in action_probs_n:
            action = Action.sample(action_probs)
            actions_and_infos_n.append((action, {"action_probs": action_probs}))
        return actions_and_infos_n

    def set_mdp(self, mdp):
        super().set_mdp(mdp)
        self.policy.mdp = mdp
    
    def reset(self):
        super(AgentFromPolicy, self).reset()
        self.policy.mdp = None

class RandomAgent(Agent):
    """
    An agent that randomly picks motion actions.
    NOTE: Does not perform interact actions, unless specified
    """

    def __init__(self, sim_threads=None, all_actions=False, custom_wait_prob=None):
        self.sim_threads = sim_threads
        self.all_actions = all_actions
        self.custom_wait_prob = custom_wait_prob
    
    def action(self, state):
        action_probs = np.zeros(Action.NUM_ACTIONS)
        legal_actions = list(Action.MOTION_ACTIONS)
        if self.all_actions:
            legal_actions = Action.ALL_ACTIONS
        legal_actions_indices = np.array([Action.ACTION_TO_INDEX[motion_a] for motion_a in legal_actions])
        action_probs[legal_actions_indices] = 1 / len(legal_actions_indices)

        if self.custom_wait_prob is not None:
            stay = Action.STAY
            if np.random.random() < self.custom_wait_prob:
                return stay, {"action_probs": Agent.a_probs_from_action(stay)}
            else:
                action_probs = Action.remove_indices_and_renormalize(action_probs, [Action.ACTION_TO_INDEX[stay]])

        return Action.sample(action_probs), {"action_probs": action_probs}

    def actions(self, states, agent_indices):
        return [self.action(state) for state in states]

    def direct_action(self, obs):
        return [np.random.randint(4) for _ in range(self.sim_threads)]


class StayAgent(Agent):

    def __init__(self, sim_threads=None):
        self.sim_threads = sim_threads
    
    def action(self, state):
        a = Action.STAY
        return a, {}

    def direct_action(self, obs):
        return [Action.ACTION_TO_INDEX[Action.STAY]] * self.sim_threads


class FixedPlanAgent(Agent):
    """
    An Agent with a fixed plan. Returns Stay actions once pre-defined plan has terminated.
    # NOTE: Assumes that calls to action are sequential (agent has history)
    """

    def __init__(self, plan):
        self.plan = plan
        self.i = 0
    
    def action(self, state):
        if self.i >= len(self.plan):
            return Action.STAY, {}
        curr_action = self.plan[self.i]
        self.i += 1
        return curr_action, {}

    def reset(self):
        super().reset()
        self.i = 0


class GreedyHumanModel(Agent):
    """
    Agent that at each step selects a medium level action corresponding
    to the most intuitively high-priority thing to do. Do not seek for cooperation, acts like he is only agent in the environment.
    NOTE: MIGHT NOT WORK IN ALL ENVIRONMENTS, for example forced_coordination.layout,
    in which an individual agent cannot complete the task on their own.
    Also agent can act suboptimaly (even when excluding coordination failures) in some environemnts with multiple (especially temporary) orders.
    for best play counter_goals, counter_drop and counter_pickup in mlam params are assumed to be equal mdp.get_counter_locations()
    """

    def __init__(self, mlam, hl_boltzmann_rational=False, ll_boltzmann_rational=False, hl_temp=1, ll_temp=1,
                 auto_unstuck=True, debug=False,  prioritize_less_ingredients_left_pots=False, 
                 choose_ingredients_pedagogically=False, most_pedagogical_next_ingredients=None):
        self.mlam = mlam
        self.mdp = self.mlam.mdp
        self.debug = debug
        # Bool for perfect rationality vs Boltzmann rationality for high level and low level action selection
        self.hl_boltzmann_rational = hl_boltzmann_rational  # For choices among high level goals of same type
        self.ll_boltzmann_rational = ll_boltzmann_rational  # For choices about low level motion

        # Coefficient for Boltzmann rationality for high level action selection
        self.hl_temperature = hl_temp
        self.ll_temperature = ll_temp

        # Whether to automatically take an action to get the agent unstuck if it's in the same
        # state as the previous turn. If false, the agent is history-less, while if true it has history.
        self.auto_unstuck = auto_unstuck

        # Try to finish soups that are close to finising over bringing ingredient to the closest pot
        self.prioritize_less_ingredients_left_pots = prioritize_less_ingredients_left_pots

        # Try to choose ingredients in the way that lets other agent recognize easier what recipe agent is aiming at
        self.choose_ingredients_pedagogically = choose_ingredients_pedagogically
        if self.choose_ingredients_pedagogically:
            if not most_pedagogical_next_ingredients:
                most_pedagogical_next_ingredients = self._preprocess_most_pedagogical_next_ingredients()
            # see _preprocess_most_pedagogical_next_ingredients for more info on this attribute
            self.most_pedagogical_next_ingredients = most_pedagogical_next_ingredients
        else:
            self.most_pedagogical_next_ingredients = None

        self.reset()

    def next_ingredients(self, current_ingredients, target_ingredients):
        # chooses next ingredient to pickup and pot
        if self.choose_ingredients_pedagogically:
            return self.most_pedagogical_next_ingredients[Recipe.standarized_ingredients(current_ingredients)][Recipe.standarized_ingredients(target_ingredients)]
        else:
            return set(Recipe.ingredients_diff(target_ingredients, current_ingredients))
        
    def _preprocess_most_pedagogical_next_ingredients(self):
        # dict accesed in form of dict[current_ingredients][target_ingredients] and returns set of ingredients that are most pedagogical to pick up;
        #   this method uses number of possible recipes (lower is more pedagogical) available after adding ingredient 
        #   (currently looking only 1 ingredient forward, it works fine for recipes of size 4 or smaller)
        most_pedagogical_next_ingredients = defaultdict(lambda: defaultdict(set))

        target_recipes_ingredients = [r.ingredients for r in self.mdp.start_orders_list.all_recipes]
        if any (len(i)> 4 for i in target_recipes_ingredients):
            print("WARNING, when recipes have a size 5 or bigger calculated pedagogical ingredients \
                to pickup will be most likely suboptimal but still better than random")
        # dict accesed in form of dict[current_ingredients][possible_target_ingredients]
        reachable_target_ingredients = defaultdict(set)
        # dicurrentct accesed in form of dict[target_ingredients][possible_current_ingredients]
        reachable_from_ingredients = defaultdict(set)
        for ingredients in target_recipes_ingredients:
            for i in range(0, len(ingredients)):
                for subset in itertools.combinations(ingredients, i):
                    reachable_target_ingredients[subset].add(ingredients)
                    reachable_from_ingredients[ingredients].add(subset)
        
        for current_ingredients, possible_target_ingredients in sorted(reachable_target_ingredients.items(), key = lambda x: -len(x[0])):
            target_recipes_num = len(possible_target_ingredients)
            for target_ingredients in possible_target_ingredients:
                missing_ingredients = Recipe.ingredients_diff(target_ingredients, current_ingredients)
                if len(set(missing_ingredients)) == 1 or target_recipes_num == 1:
                    best_ingredients = set(missing_ingredients)
                else: # at least 2 ingredients of different type are waiting for adding and there are more than 1 target recipe possible from them
                    # prioritize next ingredients that narrows down possible recipes
                    best_ingredient_min_recipes_num = target_recipes_num
                    best_ingredients = set()
                
                    for possible_next_ingredient in set(missing_ingredients): 
                        new_ingredients = Recipe.standarized_ingredients((possible_next_ingredient,) + current_ingredients)
                        new_ingredients_recipes_num = len(reachable_target_ingredients[new_ingredients])
                        if best_ingredient_min_recipes_num > new_ingredients_recipes_num:
                            best_ingredient_min_recipes_num = new_ingredients_recipes_num
                            best_ingredients = set([possible_next_ingredient])
                        elif best_ingredient_min_recipes_num == new_ingredients_recipes_num:
                            best_ingredients.add(possible_next_ingredient)
            
                most_pedagogical_next_ingredients[current_ingredients][target_ingredients] = best_ingredients
        return most_pedagogical_next_ingredients

    def reset(self):
        super().reset()
        self.prev_state = None

    def actions(self, states, agent_indices):
        actions_and_infos_n = []
        for state, agent_idx in zip(states, agent_indices):
            self.set_agent_index(agent_idx)
            self.reset()
            actions_and_infos_n.append(self.action(state))
        return actions_and_infos_n

    def action(self, state):
        if self.debug: print("timestep", state.timestep, "agent index", self.agent_index)
        possible_motion_goals = self.ml_action(state)

        # Once we have identified the motion goals for the medium
        # level action we want to perform, select the one with lowest cost
        start_pos_and_or = state.players_pos_and_or[self.agent_index]

        chosen_goal, chosen_action, action_probs = self.choose_motion_goal(start_pos_and_or, possible_motion_goals)
        if self.debug: print("chosen_goal:", chosen_goal, "chosen_action:", chosen_action, "action_probs:", action_probs)
        if self.ll_boltzmann_rational and chosen_goal[0] == start_pos_and_or[0]:
            chosen_action, action_probs = self.boltzmann_rational_ll_action(start_pos_and_or, chosen_goal)

        if self.auto_unstuck:
            # HACK: if two agents get stuck, select an action at random that would
            # change the player positions if the other player were not to move
            if self.prev_state is not None and state.players_pos_and_or == self.prev_state.players_pos_and_or:
                if self.agent_index == 0:
                    joint_actions = list(itertools.product(Action.ALL_ACTIONS, [Action.STAY]))
                elif self.agent_index == 1:
                    joint_actions = list(itertools.product([Action.STAY], Action.ALL_ACTIONS))
                else:
                    raise ValueError("Player index not recognized")

                unblocking_joint_actions = []
                for j_a in joint_actions:
                    new_state, _ = self.mlam.mdp.get_state_transition(state, j_a)
                    if new_state.player_positions != self.prev_state.player_positions:
                        unblocking_joint_actions.append(j_a)
                # Getting stuck became a possiblity simply because the nature of a layout (having a dip in the middle)
                if len(unblocking_joint_actions) == 0:
                    unblocking_joint_actions.append([Action.STAY, Action.STAY])
                chosen_action = unblocking_joint_actions[np.random.choice(len(unblocking_joint_actions))][
                    self.agent_index]
                action_probs = self.a_probs_from_action(chosen_action)

            # NOTE: Assumes that calls to the action method are sequential
            self.prev_state = state
        return chosen_action, {"action_probs": action_probs}

    def choose_motion_goal(self, start_pos_and_or, motion_goals):
        """
        For each motion goal, consider the optimal motion plan that reaches the desired location.
        Based on the plan's cost, the method chooses a motion goal (either boltzmann rationally
        or rationally), and returns the plan and the corresponding first action on that plan.
        """
        if self.hl_boltzmann_rational:
            possible_plans = [self.mlam.motion_planner.get_plan(start_pos_and_or, goal) for goal in motion_goals]
            plan_costs = [plan[2] for plan in possible_plans]
            goal_idx, action_probs = self.get_boltzmann_rational_action_idx(plan_costs, self.hl_temperature)
            chosen_goal = motion_goals[goal_idx]
            chosen_goal_action = possible_plans[goal_idx][0][0]
        else:
            chosen_goal, chosen_goal_action = self.get_lowest_cost_action_and_goal(start_pos_and_or, motion_goals)
            action_probs = self.a_probs_from_action(chosen_goal_action)
        return chosen_goal, chosen_goal_action, action_probs

    def get_boltzmann_rational_action_idx(self, costs, temperature):
        """Chooses index based on softmax probabilities obtained from cost array"""
        costs = np.array(costs)
        softmax_probs = np.exp(-costs * temperature) / np.sum(np.exp(-costs * temperature))
        action_idx = np.random.choice(len(costs), p=softmax_probs)
        return action_idx, softmax_probs

    def get_lowest_cost_action_and_goal(self, start_pos_and_or, motion_goals):
        """
        Chooses motion goal that has the lowest cost action plan.
        Returns the motion goal itself and the first action on the plan.
        """
        min_cost = np.Inf
        best_action, best_goal = None, None
        for goal in motion_goals:
            action_plan, _, plan_cost = self.mlam.motion_planner.get_plan(start_pos_and_or, goal)
            if plan_cost < min_cost:
                best_action = action_plan[0]
                min_cost = plan_cost
                best_goal = goal
        return best_goal, best_action

    def boltzmann_rational_ll_action(self, start_pos_and_or, goal, inverted_costs=False):
        """
        Computes the plan cost to reach the goal after taking each possible low level action.
        Selects a low level action boltzmann rationally based on the one-step-ahead plan costs.

        If `inverted_costs` is True, it will make a boltzmann "irrational" choice, exponentially
        favouring high cost plans rather than low cost ones.
        """
        future_costs = []
        for action in Action.ALL_ACTIONS:
            pos, orient = start_pos_and_or
            new_pos_and_or = self.mlam.mdp._move_if_direction(pos, orient, action)
            _, _, plan_cost = self.mlam.motion_planner.get_plan(new_pos_and_or, goal)
            sign = (-1) ** int(inverted_costs)
            future_costs.append(sign * plan_cost)

        action_idx, action_probs = self.get_boltzmann_rational_action_idx(future_costs, self.ll_temperature)
        return Action.ALL_ACTIONS[action_idx], action_probs
    
    def _all_soups_objs(self, state, pot_states_dict):
        possible_soup_keys = ['{}_items'.format(i+1) for i in range(Recipe.MAX_NUM_INGREDIENTS)]
        return [state.get_object(pos) for pos in itertools.chain(*[pot_states_dict[key] for key in possible_soup_keys])]

    def _empty_pot_best_missing_ingredients(self, state):
        # finds ingredients that are part of best possible soup for empty pot case
        target_ingredients = self.mlam.mdp.get_optimal_possible_recipe(state, recipe=None) or tuple()
        return self.next_ingredients(tuple(), target_ingredients)

    def _best_missing_ingredients_for_soups(self, state, pot_states_dict, include_empty_pots=True):
        # return list of tuples (soup_position, missing_ingredients for best recipe from current ingredients in pots)
        soup_objs = self._all_soups_objs(state, pot_states_dict)
        recipes = [Recipe(soup_obj.ingredients) for soup_obj in soup_objs]
        optimal_recipes_dict = {recipe: self.mlam.mdp.get_optimal_possible_recipe(state, recipe, discounted=False, return_value=False) for recipe in set(recipes)}
        
        best_missing_ingredients = [self.next_ingredients(recipe.ingredients, optimal_recipes_dict[recipe].ingredients) 
            for recipe in recipes]
        result = [(soup_obj.position, ingredients) for soup_obj, ingredients in zip(soup_objs, best_missing_ingredients)]
        if include_empty_pots and pot_states_dict["empty"]:
            empty_pot_ingredients = self._empty_pot_best_missing_ingredients(state)
            result += [(pos, empty_pot_ingredients) for pos in pot_states_dict["empty"]]
    
        if self.prioritize_less_ingredients_left_pots and result:
            min_ingredients_left = min([len(ingedients) for (pos, ingredients) in result])
            result = [(pos, ingredients) for (pos, ingredients) in result if len(ingedients) == min_ingredients_left]
        
        return result            
        
    def _motion_goals_with_ingredient(self, state, ingredient_obj, pot_states_dict, counter_objects):
        if ingredient_obj.name in Recipe.ALL_INGREDIENTS:
            best_missing_ingredients_for_soups = self._best_missing_ingredients_for_soups(state, pot_states_dict)
            
            all_ingredients_to_pot = set(itertools.chain(*[ingredients 
                for (pos, ingredients) in best_missing_ingredients_for_soups]))
            
            if ingredient_obj.name in all_ingredients_to_pot:
                goal_positions = [pos for (pos, ingredients) in best_missing_ingredients_for_soups
                    if ingredient_obj.name in ingredients]
                
                if ingredient_obj.name == Recipe.ONION:
                    motion_goals = self.mlam._get_ml_actions_for_positions(goal_positions)
                    if self.debug: print("motion goal: custom put_onion_in_pot_actions", motion_goals)
                elif ingredient_obj.name ==  Recipe.TOMATO:
                    motion_goals = self.mlam._get_ml_actions_for_positions(goal_positions)
                    if self.debug: print("motion goal: custom put_tomato_in_pot_actions", motion_goals)
                else:
                    raise ValueError
            else:
                # nothing to do with ingredient; check there is something better to do without it
                if self.debug: print("checking alternative motion goals if item is dropped")
                alternative_motion_goals = self._motion_goals_pickup_ordered_soups(state, counter_objects) or \
                    self._motion_goals_pickup_soon_needed_dishes(state, pot_states_dict, counter_objects) or \
                    self._motion_goals_pickup_ingredients_or_start_soups(state, pot_states_dict, counter_objects)
                if alternative_motion_goals:
                    motion_goals = self.mlam.place_obj_on_counter_actions(state)
                    if self.debug: print("motion goal: place_obj_on_counter_actions", motion_goals)
                else:
                    motion_goals = []
        return motion_goals


    def _motion_goals_with_dish(self, state, pot_states_dict):
        motion_goals = self.mlam.pickup_soup_with_dish_actions(pot_states_dict, only_nearly_ready=True)
        if self.debug: print("motion goal: pickup_soup_with_dish_actions", motion_goals)
        if not motion_goals: # nothing to do with dish
            motion_goals = self.mlam.place_obj_on_counter_actions(state)
            if self.debug: print("motion goal: place_obj_on_counter_actions", motion_goals)
        return motion_goals
    
    def _motion_goals_with_soup(self, state, soup_obj):
        recipe = soup_obj.recipe
        # detect soups worthless now, but maybe valuable in future (in orders_list.orders_to_add, but not in orders_list.orders)
        if recipe in state.orders_list.all_recipes and state.orders_list.get_matching_order(recipe=recipe) is None:
            motion_goals = self.mlam.place_obj_on_counter_actions(state)
            if self.debug: print("motion goal: place_obj_on_counter_actions", motion_goals)
        else:
            motion_goals = []
        
        if not motion_goals: # placing soup on counter is either bad or impossible
            # deliver soup either for reward or to trash it
            motion_goals = self.mlam.deliver_soup_actions()
            if self.debug: print("motion goal: deliver_soup_actions", motion_goals)
        return motion_goals

    def _motion_goals_pickup_ordered_soups(self, state, counter_objects):
        counter_soups_objs = [state.get_object(pos) for pos in counter_objects['soup']]
        ordered_counter_soups_objs = [soup_obj for soup_obj in counter_soups_objs if state.orders_list.get_matching_order(recipe=soup_obj.recipe)]
        if ordered_counter_soups_objs:
            motion_goals = self.mlam._get_ml_actions_for_positions([soup.position for soup in ordered_counter_soups_objs])
            if self.debug: print("motion goal: custom pickup_counter_soup_actions", motion_goals)
        else:
            motion_goals = []
        return motion_goals
    
    def _motion_goals_pickup_soon_needed_dishes(self, state, pot_states_dict, counter_objects):
        player = state.players[self.agent_index]
        other_player = state.players[1 - self.agent_index]
        other_player_has_dish = other_player.has_object() and other_player.get_object().name == 'dish'
        soups_nearly_ready = len(pot_states_dict['ready']) + len(pot_states_dict['cooking'])
        if soups_nearly_ready - int(other_player_has_dish):
            motion_goals = self.mlam.pickup_dish_actions(counter_objects)
            if self.debug: print("motion goal: pickup_dish_actions", motion_goals)
        else:
            motion_goals = []
        return motion_goals 

    def _motion_goals_pickup_ingredients_or_start_soups(self, state, pot_states_dict, counter_objects):
        # motion goals from empty handed agent
        best_missing_ingredients_for_soups = self._best_missing_ingredients_for_soups(state, pot_states_dict)

        soups_pos_ready_to_cook = [pos for (pos, ingredients) in best_missing_ingredients_for_soups 
            if not ingredients and state.has_object(pos) and not state.get_object(pos).is_ready]            
        if soups_pos_ready_to_cook:
            motion_goals = self.mlam._get_ml_actions_for_positions(soups_pos_ready_to_cook)
            if self.debug: print("motion goal: custom start_cooking_actions", motion_goals) 
        else:
            all_ingredients_to_pot = set(itertools.chain(*[ingredients 
                for (pos, ingredients) in best_missing_ingredients_for_soups]))
                
            motion_goals = []
                
            if Recipe.ONION in all_ingredients_to_pot:
                new_goals = self.mlam.pickup_onion_actions(counter_objects)
                if self.debug: print("motion goal: pickup_onion_actions", new_goals) 
                motion_goals += new_goals

            if Recipe.TOMATO in all_ingredients_to_pot:
                new_goals = self.mlam.pickup_tomato_actions(counter_objects)
                if self.debug: print("motion goal: pickup_tomato_actions", new_goals) 
                motion_goals += new_goals
        return motion_goals
    
    def _motion_goals_backup_pickup(self, state, counter_objects):
        # picking up anything is probably better than nothing, it is last resort before choosing as goal any closest feature
        # pickup ingredient for moment when pot will get empty
        all_ingredients_to_pot = set(self._empty_pot_best_missing_ingredients(state)) or Recipe.ALL_INGREDIENTS
        motion_goals = []
        if Recipe.ONION in all_ingredients_to_pot:
            new_goals = self.mlam.pickup_onion_actions(counter_objects)
            if self.debug: print("motion goal: pickup_onion_actions", new_goals) 
            motion_goals += new_goals
        if Recipe.TOMATO in all_ingredients_to_pot:
            new_goals = self.mlam.pickup_tomato_actions(counter_objects)
            if self.debug: print("motion goal: pickup_tomato_actions", new_goals) 
            motion_goals += new_goals
        return motion_goals
    
    def ml_action(self, state):
        """
        Selects a medium level action for the current state.
        Motion goals can be thought of instructions of the form:
            [do X] at location [Y]

        In this method, X (e.g. deliver the soup, pick up an onion, etc) is chosen based on
        a simple set of greedy heuristics based on the current state, e.g. it tries to use
        every ingredient already put into pot and tries to free up pots by cooking best soup
        in the closest pot even if its suboptimal.

        Effectively, will return a list of all possible locations Y in which the selected
        medium level action X can be performed.
        """
        player = state.players[self.agent_index]
        pot_states_dict = self.mlam.mdp.get_pot_states(state)
        counter_objects = self.mlam.mdp.get_counter_objects_dict(state, list(self.mlam.mdp.terrain_pos_dict['X']))

        if player.has_object():
            player_obj = player.get_object()
            if player_obj.name in Recipe.ALL_INGREDIENTS:
                motion_goals = self._motion_goals_with_ingredient(state, player_obj, pot_states_dict, counter_objects)
            elif player_obj.name == 'dish':
                motion_goals = self._motion_goals_with_dish(state, pot_states_dict)
            elif player_obj.name == 'soup':
                motion_goals = self._motion_goals_with_soup(state, player_obj)
            else:
                raise ValueError()
        else:
            motion_goals = self._motion_goals_pickup_ordered_soups(state, counter_objects)
            if not motion_goals:
                motion_goals = self._motion_goals_pickup_soon_needed_dishes(state, pot_states_dict, counter_objects)
            if not motion_goals:
                motion_goals = self._motion_goals_pickup_ingredients_or_start_soups(state, pot_states_dict, counter_objects)
            if not motion_goals:
                motion_goals = self._motion_goals_backup_pickup(state, counter_objects)
        
        motion_goals = [mg for mg in motion_goals if self.mlam.motion_planner.is_valid_motion_start_goal_pair(player.pos_and_or, mg)]

        if len(motion_goals) == 0:
            if self.debug: print("no valid motion goals, pick closest feature actions")
            motion_goals = self.mlam.go_to_closest_feature_actions(player)
            motion_goals = [mg for mg in motion_goals if self.mlam.motion_planner.is_valid_motion_start_goal_pair(player.pos_and_or, mg)]
            assert len(motion_goals) != 0

        return motion_goals


class SimpleGreedyHumanModel(Agent):
    """
    Agent that at each step selects a medium level action corresponding
    to the most intuitively high-priority thing to do

    NOTE: MIGHT NOT WORK IN ALL ENVIRONMENTS, for example forced_coordination.layout,
    in which an individual agent cannot complete the task on their own.
    Will work only in environments where the only order is 3 onion soup.
    """

    def __init__(self, mlam, hl_boltzmann_rational=False, ll_boltzmann_rational=False, hl_temp=1, ll_temp=1,
                 auto_unstuck=True):
        self.mlam = mlam
        self.mdp = self.mlam.mdp

        # Bool for perfect rationality vs Boltzmann rationality for high level and low level action selection
        self.hl_boltzmann_rational = hl_boltzmann_rational  # For choices among high level goals of same type
        self.ll_boltzmann_rational = ll_boltzmann_rational  # For choices about low level motion

        # Coefficient for Boltzmann rationality for high level action selection
        self.hl_temperature = hl_temp
        self.ll_temperature = ll_temp

        # Whether to automatically take an action to get the agent unstuck if it's in the same
        # state as the previous turn. If false, the agent is history-less, while if true it has history.
        self.auto_unstuck = auto_unstuck
        self.reset()

    def reset(self):
        super().reset()
        self.prev_state = None

    def actions(self, states, agent_indices):
        actions_and_infos_n = []
        for state, agent_idx in zip(states, agent_indices):
            self.set_agent_index(agent_idx)
            self.reset()
            actions_and_infos_n.append(self.action(state))
        return actions_and_infos_n

    def action(self, state):
        possible_motion_goals = self.ml_action(state)

        # Once we have identified the motion goals for the medium
        # level action we want to perform, select the one with lowest cost
        start_pos_and_or = state.players_pos_and_or[self.agent_index]

        chosen_goal, chosen_action, action_probs = self.choose_motion_goal(start_pos_and_or, possible_motion_goals)

        if self.ll_boltzmann_rational and chosen_goal[0] == start_pos_and_or[0]:
            chosen_action, action_probs = self.boltzmann_rational_ll_action(start_pos_and_or, chosen_goal)

        if self.auto_unstuck:
            # HACK: if two agents get stuck, select an action at random that would
            # change the player positions if the other player were not to move
            if self.prev_state is not None and state.players_pos_and_or == self.prev_state.players_pos_and_or:
                if self.agent_index == 0:
                    joint_actions = list(itertools.product(Action.ALL_ACTIONS, [Action.STAY]))
                elif self.agent_index == 1:
                    joint_actions = list(itertools.product([Action.STAY], Action.ALL_ACTIONS))
                else:
                    raise ValueError("Player index not recognized")

                unblocking_joint_actions = []
                for j_a in joint_actions:
                    new_state, _ = self.mlam.mdp.get_state_transition(state, j_a)
                    if new_state.player_positions != self.prev_state.player_positions:
                        unblocking_joint_actions.append(j_a)
                # Getting stuck became a possiblity simply because the nature of a layout (having a dip in the middle)
                if len(unblocking_joint_actions) == 0:
                    unblocking_joint_actions.append([Action.STAY, Action.STAY])
                chosen_action = unblocking_joint_actions[np.random.choice(len(unblocking_joint_actions))][
                    self.agent_index]
                action_probs = self.a_probs_from_action(chosen_action)

            # NOTE: Assumes that calls to the action method are sequential
            self.prev_state = state
        return chosen_action, {"action_probs": action_probs}

    def choose_motion_goal(self, start_pos_and_or, motion_goals):
        """
        For each motion goal, consider the optimal motion plan that reaches the desired location.
        Based on the plan's cost, the method chooses a motion goal (either boltzmann rationally
        or rationally), and returns the plan and the corresponding first action on that plan.
        """
        if self.hl_boltzmann_rational:
            possible_plans = [self.mlam.motion_planner.get_plan(start_pos_and_or, goal) for goal in motion_goals]
            plan_costs = [plan[2] for plan in possible_plans]
            goal_idx, action_probs = self.get_boltzmann_rational_action_idx(plan_costs, self.hl_temperature)
            chosen_goal = motion_goals[goal_idx]
            chosen_goal_action = possible_plans[goal_idx][0][0]
        else:
            chosen_goal, chosen_goal_action = self.get_lowest_cost_action_and_goal(start_pos_and_or, motion_goals)
            action_probs = self.a_probs_from_action(chosen_goal_action)
        return chosen_goal, chosen_goal_action, action_probs

    def get_boltzmann_rational_action_idx(self, costs, temperature):
        """Chooses index based on softmax probabilities obtained from cost array"""
        costs = np.array(costs)
        softmax_probs = np.exp(-costs * temperature) / np.sum(np.exp(-costs * temperature))
        action_idx = np.random.choice(len(costs), p=softmax_probs)
        return action_idx, softmax_probs

    def get_lowest_cost_action_and_goal(self, start_pos_and_or, motion_goals):
        """
        Chooses motion goal that has the lowest cost action plan.
        Returns the motion goal itself and the first action on the plan.
        """
        min_cost = np.Inf
        best_action, best_goal = None, None
        for goal in motion_goals:
            action_plan, _, plan_cost = self.mlam.motion_planner.get_plan(start_pos_and_or, goal)
            if plan_cost < min_cost:
                best_action = action_plan[0]
                min_cost = plan_cost
                best_goal = goal
        return best_goal, best_action

    def boltzmann_rational_ll_action(self, start_pos_and_or, goal, inverted_costs=False):
        """
        Computes the plan cost to reach the goal after taking each possible low level action.
        Selects a low level action boltzmann rationally based on the one-step-ahead plan costs.

        If `inverted_costs` is True, it will make a boltzmann "irrational" choice, exponentially
        favouring high cost plans rather than low cost ones.
        """
        future_costs = []
        for action in Action.ALL_ACTIONS:
            pos, orient = start_pos_and_or
            new_pos_and_or = self.mdp._move_if_direction(pos, orient, action)
            _, _, plan_cost = self.mlam.motion_planner.get_plan(new_pos_and_or, goal)
            sign = (-1) ** int(inverted_costs)
            future_costs.append(sign * plan_cost)

        action_idx, action_probs = self.get_boltzmann_rational_action_idx(future_costs, self.ll_temperature)
        return Action.ALL_ACTIONS[action_idx], action_probs

    def ml_action(self, state):
        """
        Selects a medium level action for the current state.
        Motion goals can be thought of instructions of the form:
            [do X] at location [Y]

        In this method, X (e.g. deliver the soup, pick up an onion, etc) is chosen based on
        a simple set of greedy heuristics based on the current state.

        Effectively, will return a list of all possible locations Y in which the selected
        medium level action X can be performed.
        """
        player = state.players[self.agent_index]
        other_player = state.players[1 - self.agent_index]
        am = self.mlam

        counter_objects = self.mlam.mdp.get_counter_objects_dict(state, list(self.mlam.mdp.terrain_pos_dict['X']))
        pot_states_dict = self.mlam.mdp.get_pot_states(state)


        if not player.has_object():
            ready_soups = pot_states_dict['ready']
            cooking_soups = pot_states_dict['cooking']

            soup_nearly_ready = len(ready_soups) > 0 or len(cooking_soups) > 0
            other_has_dish = other_player.has_object() and other_player.get_object().name == 'dish'

            if soup_nearly_ready and not other_has_dish:
                motion_goals = am.pickup_dish_actions(counter_objects)
            else:
                assert len(state.all_orders) == 1 and list(state.all_orders[0].ingredients) == ["onion", "onion", "onion"], \
                    "The current mid level action manager only support 3-onion-soup order, but got orders" \
                    + str(state.all_orders)
                next_order = list(state.all_orders)[0]
                soups_ready_to_cook_key = '{}_items'.format(len(next_order.ingredients))
                soups_ready_to_cook = pot_states_dict[soups_ready_to_cook_key]
                if soups_ready_to_cook:
                    only_pot_states_ready_to_cook = defaultdict(list)
                    only_pot_states_ready_to_cook[soups_ready_to_cook_key] = soups_ready_to_cook
                    # we want to cook only soups that has same len as order
                    motion_goals = am.start_cooking_actions(only_pot_states_ready_to_cook)
                else:
                    motion_goals = am.pickup_onion_actions(counter_objects)
                # it does not make sense to have tomato logic when the only possible order is 3 onion soup (see assertion above)
                # elif 'onion' in next_order:
                #     motion_goals = am.pickup_onion_actions(counter_objects)
                # elif 'tomato' in next_order:
                #     motion_goals = am.pickup_tomato_actions(counter_objects)
                # else:
                #     motion_goals = am.pickup_onion_actions(counter_objects) + am.pickup_tomato_actions(counter_objects)


        else:
            player_obj = player.get_object()

            if player_obj.name == 'onion':
                motion_goals = am.put_onion_in_pot_actions(pot_states_dict)

            elif player_obj.name == 'tomato':
                motion_goals = am.put_tomato_in_pot_actions(pot_states_dict)

            elif player_obj.name == 'dish':
                motion_goals = am.pickup_soup_with_dish_actions(pot_states_dict, only_nearly_ready=True)

            elif player_obj.name == 'soup':
                motion_goals = am.deliver_soup_actions()

            else:
                raise ValueError()

        motion_goals = [mg for mg in motion_goals if self.mlam.motion_planner.is_valid_motion_start_goal_pair(player.pos_and_or, mg)]

        if len(motion_goals) == 0:
            motion_goals = am.go_to_closest_feature_actions(player)
            motion_goals = [mg for mg in motion_goals if self.mlam.motion_planner.is_valid_motion_start_goal_pair(player.pos_and_or, mg)]
            assert len(motion_goals) != 0

        return motion_goals


class SampleAgent(Agent):
    """ Agent that samples action using action_probs of multiple agents
    """
    def __init__(self, agents):
        self.agents = agents
        self.reset()

    @property
    def agent_index(self):
        return self._agent_index
    
    @agent_index.setter
    def agent_index(self, v):
        for agent in self.agents:
            agent.agent_index = v
        self._agent_index = v
       
    @property
    def mdp(self):
        return self._mdp
    
    @mdp.setter
    def mdp(self, v):
        for agent in self.agents:
            agent.mdp = v
        self._mdp = v
    
    def action(self, state):
        action_probs = np.zeros(Action.NUM_ACTIONS)
        for agent in self.agents:
            action_probs += agent.action(state)[1]["action_probs"]
        action_probs = action_probs/len(self.agents)
        return Action.sample(action_probs), {"action_probs": action_probs}

    def reset(self):
        self._agent_index = None
        self._mdp = None
        for agent in self.agents:
            agent.reset()


class SlowedDownAgent(Agent):
    """
    Agent that is slowed down version of supplied agent.
    Can be used to produce assymetry of power between agents.
    """
    def __init__(self, agent, slowdown_rate=0.5):
        self.agent = agent
        self.slowdown_rate = slowdown_rate
        self.reset()
    
    @property
    def agent_index(self):
        return self.agent.agent_index
    
    @agent_index.setter
    def agent_index(self, v):
        self.agent.agent_index = v
       
    @property
    def mdp(self):
        return self.agent.mdp
    
    @mdp.setter
    def mdp(self, v):
        self.agent.mdp = v

    @property
    def skip_action(self):
        return Action.STAY, {"action_probs": Agent.a_probs_from_action(Action.STAY)}

    def action(self, state):
        last_actions_increment = self.actions_increment
        self.actions_increment += self.slowdown_rate
        # if increment crosses full number then it is time to act
        if int(last_actions_increment) < int(self.actions_increment):
            return self.agent.action(state)
        else:
            return self.skip_action

    def reset(self):
        self.agent_index = None
        self.mdp = None
        self.agent.reset()
        # every agent starts an episode with some non-waiting action and then does waiting action
        self.actions_increment = 1 - self.slowdown_rate
    """
    """
# Deprecated. Need to fix Heuristic to work with the new MDP to reactivate Planning
# class CoupledPlanningAgent(Agent):
#     """
#     An agent that uses a joint planner (mlp, a MediumLevelPlanner) to find near-optimal
#     plans. At each timestep the agent re-plans under the assumption that the other agent
#     is also a CoupledPlanningAgent, and then takes the first action in the plan.
#     """
#
#     def __init__(self, mlp, delivery_horizon=2, heuristic=None):
#         self.mlp = mlp
#         self.mlp.failures = 0
#         self.heuristic = heuristic if heuristic is not None else Heuristic(mlp.mp).simple_heuristic
#         self.delivery_horizon = delivery_horizon
#
#     def action(self, state):
#         try:
#             joint_action_plan = self.mlp.get_low_level_action_plan(state, self.heuristic, delivery_horizon=self.delivery_horizon, goal_info=True)
#         except TimeoutError:
#             print("COUPLED PLANNING FAILURE")
#             self.mlp.failures += 1
#             return Direction.ALL_DIRECTIONS[np.random.randint(4)]
#         return (joint_action_plan[0][self.agent_index], {}) if len(joint_action_plan) > 0 else (Action.STAY, {})
#
#
# class EmbeddedPlanningAgent(Agent):
#     """
#     An agent that uses A* search to find an optimal action based on a model of the other agent,
#     `other_agent`. This class approximates the other agent as being deterministic even though it
#     might be stochastic in order to perform the search.
#     """
#
#     def __init__(self, other_agent, mlp, env, delivery_horizon=2, logging_level=0):
#         """mlp is a MediumLevelPlanner"""
#         self.other_agent = other_agent
#         self.delivery_horizon = delivery_horizon
#         self.mlp = mlp
#         self.env = env
#         self.h_fn = Heuristic(mlp.mp).simple_heuristic
#         self.logging_level = logging_level
#
#     def action(self, state):
#         start_state = state.deepcopy()
#         order_list = start_state.order_list if start_state.order_list is not None else ["any", "any"]
#         start_state.order_list = order_list[:self.delivery_horizon]
#         other_agent_index = 1 - self.agent_index
#         initial_env_state = self.env.state
#         self.other_agent.env = self.env
#
#         expand_fn = lambda state: self.mlp.get_successor_states_fixed_other(state, self.other_agent, other_agent_index)
#         goal_fn = lambda state: len(state.order_list) == 0
#         heuristic_fn = lambda state: self.h_fn(state)
#
#         search_problem = SearchTree(start_state, goal_fn, expand_fn, heuristic_fn, max_iter_count=50000)
#
#         try:
#             ml_s_a_plan, cost = search_problem.A_star_graph_search(info=True)
#         except TimeoutError:
#             print("A* failed, taking random action")
#             idx = np.random.randint(5)
#             return Action.ALL_ACTIONS[idx]
#
#         # Check estimated cost of the plan equals
#         # the sum of the costs of each medium-level action
#         assert sum([len(item[0]) for item in ml_s_a_plan[1:]]) == cost
#
#         # In this case medium level actions are tuples of low level actions
#         # We just care about the first low level action of the first med level action
#         first_s_a = ml_s_a_plan[1]
#
#         # Print what the agent is expecting to happen
#         if self.logging_level >= 2:
#             self.env.state = start_state
#             for joint_a in first_s_a[0]:
#                 print(self.env)
#                 print(joint_a)
#                 self.env.step(joint_a)
#             print(self.env)
#             print("======The End======")
#
#         self.env.state = initial_env_state
#
#         first_joint_action = first_s_a[0][0]
#         if self.logging_level >= 1:
#             print("expected joint action", first_joint_action)
#         action = first_joint_action[self.agent_index]
#         return action, {}
#

# Deprecated. Due to Heuristic and MLP
# class CoupledPlanningPair(AgentPair):
#     """
#     Pair of identical coupled planning agents. Enables to search for optimal
#     action once rather than repeating computation to find action of second agent
#     """
#
#     def __init__(self, agent):
#         super().__init__(agent, agent, allow_duplicate_agents=True)
#
#     def joint_action(self, state):
#         # Reduce computation by half if both agents are coupled planning agents
#         joint_action_plan = self.a0.mlp.get_low_level_action_plan(state, self.a0.heuristic, delivery_horizon=self.a0.delivery_horizon, goal_info=True)
#
#         if len(joint_action_plan) == 0:
#             return ((Action.STAY, {}), (Action.STAY, {}))
#
#         joint_action_and_infos = [(a, {}) for a in joint_action_plan[0]]
#         return joint_action_and_infos
