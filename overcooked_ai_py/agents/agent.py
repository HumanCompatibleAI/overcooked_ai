import scipy
import itertools
import numpy as np
from collections import defaultdict
import random
import logging

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelPlanner, Heuristic


class Agent(object):

    def action(self, state):
        return NotImplementedError()

    def set_agent_index(self, agent_index):
        self.agent_index = agent_index

    def set_mdp(self, mdp):
        self.mdp = mdp

    def reset(self):
        pass


class AgentFromPolicy(Agent):
    
    def __init__(self, state_policy, direct_policy, stochastic=True, action_probs=False):
        self.state_policy = state_policy
        self.direct_policy = direct_policy
        self.history = []
        self.stochastic = stochastic
        self.action_probs = action_probs

    def action(self, state):
        """
        The standard action function call, that takes in a Overcooked state
        and returns the corresponding action.

        Requires having set self.agent_index and self.mdp
        """
        self.history.append(state)
        try:
            return self.state_policy(state, self.mdp, self.agent_index, self.stochastic, self.action_probs)
        except AttributeError as e:
            raise AttributeError("{}. Most likely, need to set the agent_index or mdp of the Agent before calling the action method.".format(e))

    def direct_action(self, obs):
        """
        A action called optimized for multi-threaded environment simulations
        involving the agent. Takes in SIM_THREADS (as defined when defining the agent)
        number of observations in post-processed form, and returns as many actions.
        """
        return self.direct_policy(obs)

    def reset(self):
        self.history = []

class RandomAgent(Agent):
    """
    An agent that randomly picks actions.
    NOTE: Does not perform interact actions
    """

    def __init__(self, sim_threads=None):
        self.sim_threads = sim_threads
    
    def action(self, state):
        idx = np.random.randint(4)
        return Action.ALL_ACTIONS[idx]

    def direct_action(self, obs):
        return [np.random.randint(4) for _ in range(self.sim_threads)]


class StayAgent(Agent):

    def __init__(self, sim_threads=None):
        self.sim_threads = sim_threads
    
    def action(self, state):
        #TODO RS: This used to be Direction.STAY
        return Action.STAY

    def direct_action(self, obs):
        return [Action.ACTION_TO_INDEX[Action.STAY]] * self.sim_threads


class FixedPlanAgent(Agent):
    
    def __init__(self, plan):
        self.plan = plan
        self.i = 0
    
    def action(self, state):
        if self.i >= len(self.plan):
            return None
        curr_action = self.plan[self.i]

        # NOTE: Assumes that calls to action are sequential
        self.i += 1
        return curr_action


class CoupledPlanningAgent(Agent):

    def __init__(self, mlp, delivery_horizon=2, heuristic=None):
        self.mlp = mlp
        self.mlp.failures = 0
        self.heuristic = heuristic if heuristic is not None else Heuristic(mlp.mp).simple_heuristic
        self.delivery_horizon = delivery_horizon

    def action(self, state):
        try:
            joint_action_plan = self.mlp.get_low_level_action_plan(state, self.heuristic, delivery_horizon=self.delivery_horizon, goal_info=True)
        except TimeoutError:
            print("COUPLED PLANNING FAILURE")
            self.mlp.failures += 1
            return Direction.ALL_DIRECTIONS[np.random.randint(4)]
        return joint_action_plan[0][self.agent_index] if len(joint_action_plan) > 0 else None


class EmbeddedPlanningAgent(Agent):

    def __init__(self, other_agent, mlp, env, delivery_horizon=2, debug=False):
        """other_agent_policy returns highest prob action"""
        self.other_agent = other_agent
        self.delivery_horizon = delivery_horizon
        self.mlp = mlp
        self.env = env
        self.h_fn = Heuristic(mlp.mp).simple_heuristic
        self.debug = debug

    def action(self, state):
        from overcooked_ai_py.planning.search import SearchTree
        start_state = state.deepcopy()
        start_state.order_list = start_state.order_list[:self.delivery_horizon]
        other_agent_index = 1 - self.agent_index

        initial_other_agent_type = self.other_agent.stochastic
        self.other_agent.stochastic = False
        initial_env_state = self.env.state
        self.other_agent.env = self.env

        expand_fn = lambda state: self.mlp.get_successor_states_fixed_other(state, self.other_agent, other_agent_index)
        goal_fn = lambda state: len(state.order_list) == 0
        heuristic_fn = lambda state: self.h_fn(state)

        search_problem = SearchTree(start_state, goal_fn, expand_fn, heuristic_fn, max_iter_count=50000)

        try:
            ml_s_a_plan, cost = search_problem.A_star_graph_search(info=True)
        except TimeoutError:
            print("A* failed, taking random action")
            idx = np.random.randint(5)
            return Action.ALL_ACTIONS[idx]

        # Check plan
        actions = [item[0] for item in ml_s_a_plan]
        actions = actions[1:]
        tru_actions = ()
        for a in actions:
            tru_actions = tru_actions + a
        assert len(tru_actions) == cost

        # In this case medium level actions are tuples of low level actions
        # We just care about the first low level action of the first med level action
        first_s_a = ml_s_a_plan[1]

        # if self.debug:
        #     print("WHAT'S ON MY MIND:")
        #     self.env.state = start_state
        #     for joint_a in first_s_a[0]:
        #         print(self.env)
        #         print(joint_a)
        #         self.env.step(joint_a)
        #     print(self.env)
        #     print("======The End======")

        self.env.state = initial_env_state
        self.other_agent.stochastic = initial_other_agent_type

        first_joint_action = first_s_a[0][0]
        if self.debug: 
            print("expected joint action", first_joint_action)
        action = first_joint_action[self.agent_index]
        return action

class GreedyHumanModel_mc(Agent):
    """
    Agent that at each step selects a medium level action corresponding
    to the most intuitively high-priority thing to do
    
    NOTE: MIGHT NOT WORK IN ALL ENVIRONMENTS
    """

    def __init__(self, mlp, boltzmann_rational=False, temp=1):
        self.mlp = mlp
        self.mdp = self.mlp.mdp
        self.prev_state = None
        self.boltzmann_rational = boltzmann_rational
        self.temperature = temp # Some measure of rationality

    def action(self, state):
        motion_goals = self.ml_action(state)

        # Once we have identified the motion goals for the medium
        # level action we want to perform, select the one with lowest cost
        start_pos_and_or = state.players_pos_and_or[self.agent_index]

        if self.boltzmann_rational:
            plans = [self.mlp.mp.get_plan(start_pos_and_or, goal) for goal in motion_goals]
            action_plan = [plan[0] for plan in plans]
            plan_cost = np.array([plan[2] for plan in plans])
            softmax_probs = np.exp(plan_cost * self.temperature) / np.sum(np.exp(plan_cost * self.temperature))
            goal_idx = np.random.choice(len(motion_goals), p=softmax_probs)
            chosen_action = action_plan[goal_idx][0]

        else:  
            min_cost = np.Inf
            best_action = None
            for goal in motion_goals:
                action_plan, _, plan_cost = self.mlp.mp.get_plan(start_pos_and_or, goal)
                if plan_cost < min_cost:
                    best_action = action_plan[0]
                    min_cost = plan_cost
            chosen_action = best_action
        
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
                new_state, _, _ = self.mlp.mdp.get_state_transition(state, j_a)
                if new_state.player_positions != self.prev_state.player_positions:
                    unblocking_joint_actions.append(j_a)

            chosen_action = unblocking_joint_actions[np.random.choice(len(unblocking_joint_actions))][self.agent_index]

        # NOTE: Assumes that calls to action are sequential
        self.prev_state = state
        return chosen_action

    def ml_action(self, state):
        """Selects a medium level action for the current state"""
        player = state.players[self.agent_index]
        other_player = state.players[1 - self.agent_index]
        am = self.mlp.ml_action_manager
        
        counter_objects = self.mlp.mdp.get_counter_objects_dict(state, list(self.mlp.mdp.terrain_pos_dict['X']))
        pot_states_dict = self.mlp.mdp.get_pot_states(state)

        # TODO: this most likely will fail in some tomato scenarios
        next_order = state.order_list[0]

        if not player.has_object():

            if next_order == 'any':
                ready_soups = pot_states_dict['onion']['ready'] + pot_states_dict['tomato']['ready']
                cooking_soups = pot_states_dict['onion']['cooking'] + pot_states_dict['tomato']['cooking']
            else:
                ready_soups = pot_states_dict[next_order]['ready']
                cooking_soups = pot_states_dict[next_order]['cooking']
            
            soup_nearly_ready = len(ready_soups) > 0 or len(cooking_soups) > 0
            other_has_dish = other_player.has_object() and other_player.get_object().name == 'dish'
            
            if soup_nearly_ready and not other_has_dish:
                motion_goals = am.pickup_dish_actions(state, counter_objects)
            else:
                next_order = None
                if len(state.order_list) > 1:
                    next_order = state.order_list[1]
                
                if next_order == 'onion':
                    motion_goals = am.pickup_onion_actions(state, counter_objects)
                elif next_order == 'tomato':
                    motion_goals = am.pickup_tomato_actions(state, counter_objects)
                elif next_order is None or next_order == 'any':
                    motion_goals = am.pickup_onion_actions(state, counter_objects) + am.pickup_tomato_actions(state, counter_objects)

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
        
        motion_goals = list(filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, goal), motion_goals))

        if len(motion_goals) == 0:
            motion_goals = am.go_to_closest_feature_actions(player)
            motion_goals = list(filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, goal), motion_goals))
            assert len(motion_goals) != 0

        return motion_goals


class AgentPair(object):
    """
    The AgentPair object is to be used in the context of a OvercookedEnv, in which
    it can be queried to obtain actions for both the agents.
    """

    def __init__(self, *agents):
        """
        If the pair of agents is in fact a single joint agent, set the agent
        index (used to order the processed observations) to 0, that is consistent
        with training.

        Otherwise, set the agent indices in the same order as the agents have been passed in.
        """
        self.agents = agents

        if len(agents) == 1:
            self.is_joint_agent = True
            self.joint_agent = agents[0]
            self.joint_agent.set_agent_index(0)
        else:
            self.is_joint_agent = False
            self.a0, self.a1 = agents
            self.a0.set_agent_index(0)
            self.a1.set_agent_index(1)

    def set_mdp(self, mdp):
        for a in self.agents:
            a.set_mdp(mdp)

    def joint_action(self, state):
        if self.is_joint_agent:
            joint_action = self.joint_agent.action(state)
            return joint_action
        elif type(self.a0) is CoupledPlanningAgent and type(self.a1) is CoupledPlanningAgent:
            # Reduce computation by half if both agents are coupled planning agents
            joint_action_plan = self.a0.mlp.get_low_level_action_plan(state, self.a0.heuristic, delivery_horizon=self.a0.delivery_horizon, goal_info=True)
            return joint_action_plan[0] if len(joint_action_plan) > 0 else (None, None)
        elif self.a0 is self.a1:
            # When using the same instance of an agent for self-play, 
            # reset agent index at each turn to prevent overwriting it
            self.a0.set_agent_index(0)
            action_0 = self.a0.action(state)
            self.a1.set_agent_index(1)
            action_1 = self.a1.action(state)
            return (action_0, action_1)
        else:
            return (self.a0.action(state), self.a1.action(state))
        
    def reset(self):
        for a in self.agents:
            a.reset()


# ============================ Agents make by pk ===================================#

class GreedyHumanModelv2(Agent):
    """
    Agent that at each step selects a medium level action corresponding to the most intuitively high-priority thing to do

    Enhancements added to v2:
    1) Drop onion if a dish is needed instead
    2) Drop dish if an onion is needed instead
    3) If soup on the counter then pick it up
    3) Added parameter "perseverance", then added mechanism for resolving crashes: each agent steps out of the way with
    a certain probability, which depends on their perseverance and how many timesteps they've been stuck.

    """

    # TODO: Remove any mention of tomato!

    def __init__(self, mlp, player_index, perseverance=0.5):
        self.mlp = mlp
        self.agent_index = player_index
        self.mdp = self.mlp.mdp
        self.prev_state = None
        self.timesteps_stuck = 0  # Count how many times there's a clash with the other player
        self.perseverance = perseverance

    def action(self, state):
        motion_goals = self.ml_action(state)

        # Once we have identified the motion goals for the medium
        # level action we want to perform, select the one with lowest cost
        start_pos_and_or = state.players_pos_and_or[self.agent_index]

        min_cost = np.Inf
        best_action = None
        for goal in motion_goals:
            action_plan, _, plan_cost = self.mlp.mp.get_plan(start_pos_and_or, goal)
            if plan_cost < min_cost:
                best_action = action_plan[0]
                min_cost = plan_cost

        """If the agent is stuck, then take an alternative action with a probability based on the time stuck and the
        agent's "perseverance". Note: We consider an agent stuck if their whole state is unchanged (but this misses the
        case when they try to move and do change direction but can't move <-- here the state changes & they're stuck).
        Also, there exist cases when the state doesn't change but they're not stuck, they just can't complete their action
        (e.g. on unident they could try to use an onion but the other player has already filled the pot)"""
        if self.prev_state is not None and state.players[self.agent_index] == self.prev_state.players[self.agent_index]:
            self.timesteps_stuck += 1
            take_alternative = self.take_alternative_action()
            # logging.info('Player {} timesteps stuck: {}'.format(self.agent_index, self.timesteps_stuck))
            if take_alternative:
                 # logging.info('Taking alternative action!')
                 # Select an action at random that would change the player positions if the other player were not to move
                 if self.agent_index == 0:
                     joint_actions = list(itertools.product(Action.ALL_ACTIONS, [Action.STAY]))
                 elif self.agent_index == 1:
                     joint_actions = list(itertools.product([Action.STAY], Action.ALL_ACTIONS))
                 else:
                     raise ValueError("Player index not recognized")

                 unblocking_joint_actions = []
                 for j_a in joint_actions:
                     new_state, _, _ = self.mlp.mdp.get_state_transition(state, j_a)
                     if new_state.player_positions != self.prev_state.player_positions:
                         unblocking_joint_actions.append(j_a)

                 """Prefer adjacent actions if available:"""
                 # Adjacent actions only exist if best_action is N, S, E, W
                 if best_action in Direction.ALL_DIRECTIONS:

                     # Find the adjacent actions:
                     if self.agent_index == 0:
                         joint_adjacent_actions = list(itertools.product(Direction.get_adjacent_directions(best_action),
                                                                         [Action.STAY]))
                     elif self.agent_index == 1:
                         joint_adjacent_actions = list(itertools.product([Action.STAY],
                                                                         Direction.get_adjacent_directions(best_action)))
                     else:
                         raise ValueError("Player index not recognized")

                     # If at least one of the adjacent actions is in the set of unblocking_joint_actions, then select these:
                     if (joint_adjacent_actions[0] in unblocking_joint_actions
                             or joint_adjacent_actions[1] in unblocking_joint_actions):
                         preferred_unblocking_joint_actions = []
                         # There are only ever two adjacent actions:
                         if joint_adjacent_actions[0] in unblocking_joint_actions:
                             preferred_unblocking_joint_actions.append(joint_adjacent_actions[0])
                         if joint_adjacent_actions[1] in unblocking_joint_actions:
                             preferred_unblocking_joint_actions.append(joint_adjacent_actions[1])
                     elif (joint_adjacent_actions[0] not in unblocking_joint_actions
                           and joint_adjacent_actions[1] not in unblocking_joint_actions):
                         # No adjacent actions in the set of unblocking_joint_actions, so keep these actions
                         preferred_unblocking_joint_actions = unblocking_joint_actions
                     else:
                         raise ValueError("Binary truth value is neither true nor false")

                 # If adjacent actions don't exist then keep unblocking_joint_actions as it is
                 else:
                     preferred_unblocking_joint_actions = unblocking_joint_actions

                 best_action = preferred_unblocking_joint_actions[
                     np.random.choice(len(preferred_unblocking_joint_actions))][self.agent_index]
                 # Note: np.random isn't actually random!

        else:
            self.timesteps_stuck = 0  # Reset to zero if prev & current player states aren't the same (they're not stuck)

        # NOTE: Assumes that calls to action are sequential
        self.prev_state = state
        return best_action

    def ml_action(self, state):
        """Selects a medium level action for the current state"""
        player = state.players[self.agent_index]
        #other_player = state.players[1 - self.agent_index]
        am = self.mlp.ml_action_manager

        counter_objects = self.mlp.mdp.get_counter_objects_dict(state, list(self.mlp.mdp.terrain_pos_dict['X']))
        pot_states_dict = self.mlp.mdp.get_pot_states(state)

        next_order = state.order_list[0]

        # Determine if any soups are "nearly ready", meaning that they are cooking or ready
        if next_order == 'any':
            ready_soups = pot_states_dict['onion']['ready'] + pot_states_dict['tomato']['ready']
            cooking_soups = pot_states_dict['onion']['cooking'] + pot_states_dict['tomato']['cooking']
        else:
            ready_soups = pot_states_dict[next_order]['ready']
            cooking_soups = pot_states_dict[next_order]['cooking']

        soup_nearly_ready = len(ready_soups) > 0 or len(cooking_soups) > 0
        #other_has_dish = other_player.has_object() and other_player.get_object().name == 'dish'  <-- no longer used

        # Determine if any soups need onions
        pots_empty = len(pot_states_dict['empty'])
        pots_partially_full = len(pot_states_dict['onion']['partially_full'])
        soup_needs_onions = pots_empty > 0 or pots_partially_full > 0

        if not player.has_object():

            if soup_nearly_ready:  # PK removed "and not other_has_dish"
                motion_goals = am.pickup_dish_actions(state, counter_objects)
            else:
                next_order = None
                #TODO: This seems to look at the next-but-one order? Should it be order_list[0]? Check this, and modify if needed
                if len(state.order_list) > 1:
                    next_order = state.order_list[1]

                if next_order == 'onion':
                    motion_goals = am.pickup_onion_actions(state, counter_objects)
                elif next_order == 'tomato':
                    motion_goals = am.pickup_tomato_actions(state, counter_objects)
                elif next_order is None or next_order == 'any':
                    motion_goals = am.pickup_onion_actions(state, counter_objects) + am.pickup_tomato_actions(state, counter_objects)

            # If there's a soup on the counter, then override other goals and get the soup
            #TODO: This can cause issues in unident <-- fix it
            if 'soup' in counter_objects:
                motion_goals = am.pickup_counter_soup_actions(state, counter_objects)

        else:
            player_obj = player.get_object()

            if player_obj.name == 'onion':
                if not soup_needs_onions:
                    # If player has an onion but there are no soups to put it in, then drop the onion!
                    motion_goals = am.place_obj_on_counter_actions(state)
                else:
                    motion_goals = am.put_onion_in_pot_actions(pot_states_dict)

            elif player_obj.name == 'tomato':
                motion_goals = am.put_tomato_in_pot_actions(pot_states_dict)

            elif player_obj.name == 'dish':
                # If player has a dish but there are no longer any "nearly ready" soups, then drop the dish!
                if not soup_nearly_ready:
                    motion_goals = am.place_obj_on_counter_actions(state)
                elif soup_nearly_ready:
                    motion_goals = am.pickup_soup_with_dish_actions(pot_states_dict, only_nearly_ready=True)

            elif player_obj.name == 'soup':
                motion_goals = am.deliver_soup_actions()

            else:
                raise ValueError()

        # Remove invalid goals:
        motion_goals = list(filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, goal), motion_goals))

        # If no goals, then just go to nearest feature
        if len(motion_goals) == 0:
            motion_goals = am.go_to_closest_feature_actions(player)
            motion_goals = list(filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, goal), motion_goals))
            assert len(motion_goals) != 0

        return motion_goals

    def take_alternative_action(self):
        """This first gives Prob(taking alternative action)=1 if perseverance=0 and Prob=0 if perseverance=1. Otherwise,
        e.g. perseverance=_, num_items, _ = state.get_object(self.mlp.mdp.get_pot_locations()[0]).state
            #current_list = full_list[num_items:number_of_tasks]

        # if number_of_pots == 2:
            #0.5, for the first timestep Prob~0.5, then Prob-->1 quickly as time increases. Then using this prob
        it randomly determines if the player should take an alternative action."""

        Prob_taking_alternative = 1 - (1 / (np.exp(self.timesteps_stuck*10))) ** (1 - self.perseverance ** (1 / 10))
        rand = random.random()
        take_alternative = rand < Prob_taking_alternative

        # logging.info('time stuck: {}, perserv: {}, Prob_taking_alt: {}, Taking alt: {}'.format(self.timesteps_stuck, self.perseverance, Prob_taking_alternative, take_alternative))

        return take_alternative


class SimpleComplementaryModel(Agent):
    """
    Bulit on greedy human model
    - added a simple heuristic that's used to factor in other player's expected moves
    - motion goals can be retained rather than re-finding goals every timestep

    """

    def __init__(self, mlp, player_index, perseverance=0.5, teamwork=1, retain_goals=0.5):
        self.mlp = mlp
        self.agent_index = player_index
        self.mdp = self.mlp.mdp
        self.prev_state = None
        self.timesteps_stuck = 0  # Count how many times there's a clash with the other player
        self.dont_drop = False
        self.current_motion_goal = None

        # "Personality" parameters (within 0 to 1)
        self.perseverance = perseverance  # perseverance = 1 means the agent always tries to go where it want to do, even when it's stuck
        self.teamwork = teamwork  # teamwork = 0 should make this agent v similar to GreedyHuman
        self.retain_goals = retain_goals  # Prob of keeping the previous goal each timestep (rather than re-calculating)


    def action(self, state):

        logging.info('Player: {}'.format(self.agent_index))
        # Get motion_goals if i) self.current_motion_goal == None; ii) with prob = 1 - self.retain_goals;
        # iii) if stuck for 2 steps (??); iv) if reached goal
        # TODO: modify the getting stuck condition, e.g. include the goal to move out the way as a (different type of)
        #  goal to be saved (see notes)
        rand = random.random()
        if self.current_motion_goal == None or rand > self.retain_goals \
                or self.timesteps_stuck > 0 or self.current_motion_goal == state.players_pos_and_or[self.agent_index]:

            logging.info('Getting new goal:')
            # logging.info('rand: {}, retain_goals: {}, stuck: {}, current goal: {}, current pos or: {}'
            #       .format(rand, self.retain_goals, self.timesteps_stuck,
            #               self.current_motion_goal, state.players_pos_and_or[self.agent_index]))

            motion_goals = self.ml_action(state)

            # Once we have identified the motion goals for the medium
            # level action we want to perform, select the one with lowest cost
            start_pos_and_or = state.players_pos_and_or[self.agent_index]
            min_cost = np.Inf
            best_action = None
            for goal in motion_goals:
                action_plan, _, plan_cost = self.mlp.mp.get_plan(start_pos_and_or, goal)
                if plan_cost < min_cost:
                    best_action = action_plan[0]
                    min_cost = plan_cost
                    best_goal = goal
            # Save motion goal:
            self.current_motion_goal = best_goal
            #TODO: Doesn't this need to be [best_goal]?? Seems to work though?

        else:

            logging.info('Keeping old goal:')
            # logging.info('rand: {}, retain_goals: {}, stuck: {}, current goal: {}, current pos or: {}'
            #       .format(rand, self.retain_goals, self.timesteps_stuck,
            #               self.current_motion_goal, state.players_pos_and_or[self.agent_index]))

            # Use previous goal
            motion_goals = self.current_motion_goal
            # Find action for this goal:
            start_pos_and_or = state.players_pos_and_or[self.agent_index]
            action_plan, _, _ = self.mlp.mp.get_plan(start_pos_and_or, motion_goals)
            best_action = action_plan[0]


        """If the agent is stuck, then take an alternative action with a probability based on the time stuck and the
        agent's "perseverance". Note: We consider an agent stuck if their whole state is unchanged (but this misses the
        case when they try to move and do change direction but can't move <-- here the state changes & they're stuck).
        Also, there exist cases when the state doesn't change but they're not stuck, they just can't complete their action
        (e.g. on unident they could try to use an onion but the other player has already filled the pot)"""
        if self.prev_state is not None and state.players[self.agent_index] == self.prev_state.players[self.agent_index]:
            self.timesteps_stuck += 1
            take_alternative = self.take_alternative_action()
            # logging.info('Player {} timesteps stuck: {}'.format(self.agent_index, self.timesteps_stuck))
            if take_alternative:
                 # logging.info('Taking alternative action!')
                 # Select an action at random that would change the player positions if the other player were not to move
                 if self.agent_index == 0:
                     joint_actions = list(itertools.product(Action.ALL_ACTIONS, [Action.STAY]))
                 elif self.agent_index == 1:
                     joint_actions = list(itertools.product([Action.STAY], Action.ALL_ACTIONS))
                 else:
                     raise ValueError("Player index not recognized")

                 unblocking_joint_actions = []
                 for j_a in joint_actions:
                     new_state, _, _ = self.mlp.mdp.get_state_transition(state, j_a)
                     if new_state.player_positions != self.prev_state.player_positions:
                         unblocking_joint_actions.append(j_a)

                 """Prefer adjacent actions if available:"""
                 # Adjacent actions only exist if best_action is N, S, E, W
                 if best_action in Direction.ALL_DIRECTIONS:

                     # Find the adjacent actions:
                     if self.agent_index == 0:
                         joint_adjacent_actions = list(itertools.product(Direction.get_adjacent_directions(best_action),
                                                                         [Action.STAY]))
                     elif self.agent_index == 1:
                         joint_adjacent_actions = list(itertools.product([Action.STAY],
                                                                         Direction.get_adjacent_directions(best_action)))
                     else:
                         raise ValueError("Player index not recognized")

                     # If at least one of the adjacent actions is in the set of unblocking_joint_actions, then select these:
                     if (joint_adjacent_actions[0] in unblocking_joint_actions
                             or joint_adjacent_actions[1] in unblocking_joint_actions):
                         preferred_unblocking_joint_actions = []
                         # There are only ever two adjacent actions:
                         if joint_adjacent_actions[0] in unblocking_joint_actions:
                             preferred_unblocking_joint_actions.append(joint_adjacent_actions[0])
                         if joint_adjacent_actions[1] in unblocking_joint_actions:
                             preferred_unblocking_joint_actions.append(joint_adjacent_actions[1])
                     elif (joint_adjacent_actions[0] not in unblocking_joint_actions
                           and joint_adjacent_actions[1] not in unblocking_joint_actions):
                         # No adjacent actions in the set of unblocking_joint_actions, so keep these actions
                         preferred_unblocking_joint_actions = unblocking_joint_actions
                     else:
                         raise ValueError("Binary truth value is neither true nor false")

                 # If adjacent actions don't exist then keep unblocking_joint_actions as it is
                 else:
                     preferred_unblocking_joint_actions = unblocking_joint_actions

                 best_action = preferred_unblocking_joint_actions[
                     np.random.choice(len(preferred_unblocking_joint_actions))][self.agent_index]
                 # Note: np.random isn't actually random!

        else:
            self.timesteps_stuck = 0  # Reset to zero if prev & current player states aren't the same (they're not stuck)

        # NOTE: Assumes that calls to action are sequential
        self.prev_state = state
        return best_action


    def ml_action(self, state):
        """Selects a medium level action for the current state"""
        player = state.players[self.agent_index]
        other_player = state.players[1 - self.agent_index]
        am = self.mlp.ml_action_manager

        counter_objects = self.mlp.mdp.get_counter_objects_dict(state, list(self.mlp.mdp.terrain_pos_dict['X']))
        pot_states_dict = self.mlp.mdp.get_pot_states(state)

        #next_order = state.order_list[0]  # Assuming soup only for now

        # (removed tomato)
        ready_soups = pot_states_dict['onion']['ready']
        cooking_soups = pot_states_dict['onion']['cooking']

        soup_nearly_ready = len(ready_soups) > 0 or len(cooking_soups) > 0
        count_soups_nearly_ready = len(ready_soups) + len(cooking_soups)
        other_has_dish = other_player.has_object() and other_player.get_object().name == 'dish'
        other_has_onion = other_player.has_object() and other_player.get_object().name == 'onion'

        number_of_pots = self.mlp.mdp.get_pot_locations().__len__()

        temp_dont_drop = False

        if not player.has_object():

            # Only get the dish if the soup is nearly ready
            if soup_nearly_ready:

                """Get the dish... unless other player is better placed"""

                default_motion_goals = am.pickup_dish_actions(state, counter_objects)

                # Remove invalid goals:
                own_motion_goals = list(filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, goal), default_motion_goals))
                others_motion_goals = list(filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(other_player.pos_and_or, goal), default_motion_goals))

                own_min_cost = self.find_min_plan_cost(own_motion_goals, state, self.agent_index)
                others_min_cost = self.find_min_plan_cost(others_motion_goals, state, 1 - self.agent_index)

                logging.info('Player {}'.format(self.agent_index))
                logging.info('own min cost: {}'.format(own_min_cost))
                logging.info('other\'s min cost: {}'.format(others_min_cost))

                # Get the dish if the other player doesn't have a dish or they do have a dish but are further away
                if (not other_has_dish) and (others_min_cost > own_min_cost):  #TODO: consider having >=

                    logging.info('Getting the dish: assume in better position that other player')
                    motion_goals = default_motion_goals

                elif other_has_dish or ((not other_has_dish) and (others_min_cost <= own_min_cost)):
                    # Now find the *next* task that needs to be done, and do that
                    # Simple way of doing this: IF only one nearly_ready pot at the top of the list THEN get onion
                    # IF both lists start with pot, THEN get a dish (worry about where to put it later!)

                    logging.info('Assume other player is in better position, so do the next task on the list')
                    tasks = self.tasks_to_do(state)

                    # ASSUMING TWO POTS MAX
                    number_of_pots = self.mlp.mdp.get_pot_locations().__len__()
                    if number_of_pots == 1:
                        # Next task will always be onion
                        logging.info('Next task: onion')
                        motion_goals = am.pickup_onion_actions(state, counter_objects)

                        logging.info('Onion shouldnt be dropped, so set temp_dont_drop = True')
                        temp_dont_drop = True

                    elif number_of_pots == 2:

                        if (tasks[0][0] == 'fetch_onion') or (tasks[1][0] == 'fetch_onion'):

                            logging.info('Next task: onion')
                            # Next task will be onion
                            motion_goals = am.pickup_onion_actions(state, counter_objects)

                        elif (tasks[0][0] == 'fetch_dish') and (tasks[1][0] == 'fetch_dish'):

                            logging.info('Next task: dish')
                            # Next task will be dish
                            motion_goals = am.pickup_dish_actions(state, counter_objects)

                        else:
                            raise ValueError('Failed logic')
                            #TODO: is there a special type of logical error??

                else:
                    raise ValueError('Failed logic')

                if random.random() > self.teamwork:
                    # Overwrite consideration of other player
                    logging.info('Overwriting motion_goal with the greed option')
                    motion_goals = default_motion_goals


            elif not soup_nearly_ready:

                """If there are two pots, then we can only get inside this 'if' when both pots have onion first on the list
                of tasks, so we can always get an onion! For 1 pot then just look what's next on the list"""

                default_motion_goals = am.pickup_onion_actions(state, counter_objects)

                # ASSUMING TWO POTS MAX THROUGHOUT
                if number_of_pots == 1:

                    # Remove invalid goals of getting an onion:
                    own_motion_goals = list(
                        filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, goal),
                               default_motion_goals))
                    others_motion_goals = list(
                        filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(other_player.pos_and_or, goal),
                               default_motion_goals))

                    own_min_cost = self.find_min_plan_cost(own_motion_goals, state, self.agent_index)
                    others_min_cost = self.find_min_plan_cost(others_motion_goals, state, 1 - self.agent_index)

                    logging.info('Player {}'.format(self.agent_index))
                    logging.info('own min cost: {}'.format(own_min_cost))
                    logging.info('other\'s min cost: {}'.format(others_min_cost))

                    # Get the onion if the other player doesn't have an onion or they do have an onion but are further away
                    if (not other_has_onion) and (others_min_cost >= own_min_cost):  # TODO: make it random if others_min_cost = own_min_cost ?

                        logging.info('Getting the onion: assume in better position that other player')
                        motion_goals = default_motion_goals

                    elif other_has_onion or ((not other_has_onion) and (others_min_cost <= own_min_cost)):
                        # Now find the *next* task that needs to be done, and do that

                        logging.info('Assume other player is in better position, so do the next task on the list')
                        tasks = self.tasks_to_do(state)

                        if tasks[0][1] == 'fetch_onion':  # tasks[0][1] is the next task (when there's one pot)
                            logging.info('Next task: onion')
                            motion_goals = am.pickup_onion_actions(state, counter_objects)

                        elif tasks[0][1] == 'fetch_dish':
                            logging.info('Next task: dish')
                            motion_goals = am.pickup_onion_actions(state, counter_objects)

                            logging.info('Dish shouldnt be dropped, so set temp_dont_drop = True')
                            temp_dont_drop = True

                elif number_of_pots == 2:
                    # Get an onion regardless: there's always a pot needing an onion (because 'not soup_nearly_ready')
                    motion_goals = default_motion_goals

                else:
                    raise ValueError('Assuming 1 or 2 pots, but there are more??')

                if random.random() > self.teamwork:
                    # Overwrite consideration of other player
                    logging.info('Overwriting motion_goal with the greed option')
                    motion_goals = default_motion_goals

            else:
                raise ValueError('Failed logic')

            # If there's a soup on the counter, then override other goals and get the soup
            # UNLESS other player (strictly) is closer
            if 'soup' in counter_objects:

                default_motion_goals = am.pickup_counter_soup_actions(state, counter_objects)

                # Remove invalid goals of getting an onion:
                own_motion_goals = list(
                    filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, goal),
                           default_motion_goals))
                others_motion_goals = list(
                    filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(other_player.pos_and_or, goal),
                           default_motion_goals))

                own_min_cost = self.find_min_plan_cost(own_motion_goals, state, self.agent_index)
                others_min_cost = self.find_min_plan_cost(others_motion_goals, state, 1 - self.agent_index)

                if others_min_cost > own_min_cost:

                    logging.info('Soup on counter and other player is futher away: get soup')
                    motion_goals = default_motion_goals

                elif others_min_cost <= own_min_cost:

                    logging.info('Soup on counter but other closer')
                    logging.info('Own cost = {}'.format(own_min_cost))



        elif player.has_object():

            # Determine if any soups need onions
            pots_empty = len(pot_states_dict['empty'])
            pots_partially_full = len(pot_states_dict['onion']['partially_full'])
            soups_need_onions = pots_empty + pots_partially_full

            player_obj = player.get_object()

            if player_obj.name == 'onion':

                default_motion_goals = am.put_onion_in_pot_actions(pot_states_dict)

                if soups_need_onions == 0:

                    if self.dont_drop == True:
                        # Don't drop the onion
                        logging.info('Got onion because its the next task: just do nothing')
                        #TODO: unrealistic to do nothing, but the alternatives are hard to code
                        motion_goals = state.players[self.agent_index].pos_and_or
                        temp_dont_drop = True
                    else:
                        # Drop onion
                        logging.info('Onion not needed: drop it')
                        motion_goals = am.place_obj_on_counter_actions(state)

                elif soups_need_onions == 1:
                    # If closer than the other player, then deliver the onion

                    # Remove invalid goals of delivering an onion:
                    own_motion_goals = list(
                        filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, goal),
                               default_motion_goals))
                    others_motion_goals = list(
                        filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(other_player.pos_and_or, goal),
                               default_motion_goals))

                    own_min_cost = self.find_min_plan_cost(own_motion_goals, state, self.agent_index)
                    others_min_cost = self.find_min_plan_cost(others_motion_goals, state, 1 - self.agent_index)

                    if other_player.has_object() and (other_player.get_object().name == 'onion') and (others_min_cost < own_min_cost):

                        if number_of_pots == 2:
                            # Two pots but only 1 needs onion: this means a dish is needed!
                            logging.info('Two pots but only 1 needs onion: this means a dish is needed!')
                            motion_goals = am.place_obj_on_counter_actions(state)

                        elif number_of_pots == 1:
                            logging.info('Only one pot, which needs an onion, but other is closer. So do the next task')
                            # Do the next task
                            tasks = self.tasks_to_do(state)

                            if tasks[0][1] == 'fetch_onion':  # tasks[0][1] is the next task (when there's one pot)
                                logging.info('Next task: deliver onion')
                                motion_goals = default_motion_goals

                            elif tasks[0][1] == 'fetch_dish':
                                logging.info('Next task: dish needed')
                                motion_goals = am.place_obj_on_counter_actions(state)

                        else:
                            raise ValueError('More pots than expected')

                    else:
                        # Deliver onion
                        motion_goals = default_motion_goals


                elif soups_need_onions == 2:
                    # Deliver onion regardless of other player
                    motion_goals = am.put_onion_in_pot_actions(pot_states_dict)

                if random.random() > self.teamwork:
                    # Overwrite consideration of other player
                    logging.info('Overwriting motion_goal with the greed option')
                    if soups_need_onions == 0:
                        # Player has an onion but there are no soups to put it in, then drop the onion
                        motion_goals = am.place_obj_on_counter_actions(state)
                    else:
                        motion_goals = am.put_onion_in_pot_actions(pot_states_dict)


            elif player_obj.name == 'dish':

                default_motion_goals = am.pickup_soup_with_dish_actions(pot_states_dict, only_nearly_ready=True)

                if count_soups_nearly_ready == 0:

                    if (number_of_pots == 1) and (self.dont_drop == True):
                            # Don't drop the dish
                            logging.info('Got dish because its the next task: just do nothing')
                            #TODO: unrealistic to do nothing, but the alternatives are hard to code
                            motion_goals = state.players[self.agent_index].pos_and_or
                            temp_dont_drop = True
                    else:
                        # Got dish but no soups to fetch
                        logging.info('Got dish but no soups to fetch')
                        motion_goals = am.place_obj_on_counter_actions(state)

                elif count_soups_nearly_ready == 1:
                    # If closer than the other player, then fetch the soup

                    # Remove invalid goals:
                    own_motion_goals = list(
                        filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, goal),
                               default_motion_goals))
                    others_motion_goals = list(
                        filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(other_player.pos_and_or, goal),
                               default_motion_goals))

                    own_min_cost = self.find_min_plan_cost(own_motion_goals, state, self.agent_index)
                    others_min_cost = self.find_min_plan_cost(others_motion_goals, state, 1 - self.agent_index)

                    if other_player.has_object() and (other_player.get_object().name == 'dish') and (others_min_cost < own_min_cost):

                        logging.info('Other player has dish and is closer to a nearly ready soup')

                        #TODO: this if loop not needed
                        if number_of_pots == 2:
                            # Two pots but only 1 needs dish: so get an onion!
                            logging.info('Two pots but only 1 needs dish: this means an onion is needed!')
                            motion_goals = am.place_obj_on_counter_actions(state)

                        elif number_of_pots == 1:
                            logging.info('Only one pot, which needs a dish, but other is closer. Next task is always onion')
                            motion_goals = am.place_obj_on_counter_actions(state)

                        else:
                            raise ValueError('More pots than expected')

                    else:
                        motion_goals = default_motion_goals

                elif count_soups_nearly_ready == 2:
                    # Collect soup regardless
                    motion_goals = default_motion_goals

                else:
                    raise ValueError()

                if random.random() > self.teamwork:
                    # Overwrite consideration of other player
                    logging.info('Overwriting motion_goal with the greed option')
                    if not soup_nearly_ready:
                        # Player has a dish but there are no longer any "nearly ready" soups: drop the dish
                        motion_goals = am.place_obj_on_counter_actions(state)
                    elif soup_nearly_ready:
                        motion_goals = am.pickup_soup_with_dish_actions(pot_states_dict, only_nearly_ready=True)


            elif player_obj.name == 'soup':
                # Deliver soup whatever the other player is doing
                motion_goals = am.deliver_soup_actions()

            else:
                raise ValueError()
        else:
            raise ValueError('Player has AND does not have an object!')



        # OLd code:
        #
        # # Determine if any soups are "nearly ready", meaning that they are cooking or ready
        # if next_order == 'any':
        #     ready_soups = pot_states_dict['onion']['ready'] + pot_states_dict['tomato']['ready']
        #     cooking_soups = pot_states_dict['onion']['cooking'] + pot_states_dict['tomato']['cooking']
        # else:
        #     ready_soups = pot_states_dict[next_order]['ready']
        #     cooking_soups = pot_states_dict[next_order]['cooking']
        #
        # soup_nearly_ready = len(ready_soups) > 0 or len(cooking_soups) > 0
        # #other_has_dish = other_player.has_object() and other_player.get_object().name == 'dish'  <-- no longer used

        # if not player.has_object():
        #
        #     if soup_nearly_ready:  # PK removed "and not other_has_dish"
        #         motion_goals = am.pickup_dish_actions(state, counter_objects)
        #     else:
        #         next_order = None
        #         if len(state.order_list) > 1:
        #             next_order = state.order_list[1]
        #
        #         if next_order == 'onion':
        #             motion_goals = am.pickup_onion_actions(state, counter_objects)
        #         elif next_order == 'tomato':
        #             motion_goals = am.pickup_tomato_actions(state, counter_objects)
        #         elif next_order is None or next_order == 'any':
        #             motion_goals = am.pickup_onion_actions(state, counter_objects) + am.pickup_tomato_actions(state, counter_objects)

            # # If there's a soup on the counter, then override other goals and get the soup
            # if 'soup' in counter_objects:
            #     motion_goals = am.pickup_counter_soup_actions(state, counter_objects)

        # else:
        #
        #     # Determine if any soups need onions
        #     pots_empty = len(pot_states_dict['empty'])
        #     pots_partially_full = len(pot_states_dict['onion']['partially_full'])
        #     soup_needs_onions = pots_empty > 0 or pots_partially_full > 0
        #
        #     player_obj = player.get_object()
        #
        #     if player_obj.name == 'onion':
        #         if not soup_needs_onions:
        #             # If player has an onion but there are no soups to put it in, then drop the onion!
        #             motion_goals = am.place_obj_on_counter_actions(state)
        #         else:
        #             motion_goals = am.put_onion_in_pot_actions(pot_states_dict)
        #
        #     elif player_obj.name == 'tomato':
        #         motion_goals = am.put_tomato_in_pot_actions(pot_states_dict)
        #
        #     elif player_obj.name == 'dish':
        #         # If player has a dish but there are no longer any "nearly ready" soups, then drop the dish!
        #         if not soup_nearly_ready:
        #             motion_goals = am.place_obj_on_counter_actions(state)
        #         elif soup_nearly_ready:
        #             motion_goals = am.pickup_soup_with_dish_actions(pot_states_dict, only_nearly_ready=True)
        #
        #     elif player_obj.name == 'soup':
        #         motion_goals = am.deliver_soup_actions()
        #
        #     else:
        #         raise ValueError()

        # Remove invalid goals:
        motion_goals = list(filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, goal), motion_goals))

        # If no goals, then just go to nearest feature
        if len(motion_goals) == 0:
            motion_goals = am.go_to_closest_feature_actions(player)
            motion_goals = list(filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, goal), motion_goals))
            assert len(motion_goals) != 0

        if temp_dont_drop == True:
            self.dont_drop = True
        else:
            self.dont_drop = False

        return motion_goals

    def take_alternative_action(self):
        """This first gives Prob(taking alternative action)=1 if perseverance=0 and Prob=0 if perseverance=1. Otherwise,
        e.g. perseverance=_, num_items, _ = state.get_object(self.mlp.mdp.get_pot_locations()[0]).state
            #current_list = full_list[num_items:number_of_tasks]

        # if number_of_pots == 2:
            #0.5, for the first timestep Prob~0.5, then Prob-->1 quickly as time increases. Then using this prob
        it randomly determines if the player should take an alternative action."""

        Prob_taking_alternative = 1 - (1 / (np.exp(self.timesteps_stuck*10))) ** (1 - self.perseverance ** (1 / 10))
        rand = random.random()
        take_alternative = rand < Prob_taking_alternative

        # logging.info('time stuck: {}, perserv: {}, Prob_taking_alt: {}, Taking alt: {}'.format(self.timesteps_stuck, self.perseverance, Prob_taking_alternative, take_alternative))

        return take_alternative

    def tasks_to_do(self, state):
        """
        :return: tasks = a list of list of tasks to be done, where tasks[i] is a list of what need doing for pot i (where
        pot i is the pot at location get_pot_locations()[i])
        """

        # OLD
        # if number_of_pots == 1:
        #     order_length = state.order_list.__len__()
        #     full_list = ['fetch_onion','fetch_onion','fetch_onion','fetch_dish']*order_length
        #     pot_pos = self.mlp.mdp.get_pot_locations()[0]
        #     if not state.has_object(pot_pos):
        #         num_items = 0
        #     else:
        #         _, num_items, _ = state.get_object(self.mlp.mdp.get_pot_locations()[0]).state
        #     current_list = full_list[num_items:num_items+number_of_tasks]


        number_of_pots = self.mlp.mdp.get_pot_locations().__len__()
        order_length = state.order_list.__len__()
        initial_list_for_each_pot = ['fetch_onion', 'fetch_onion', 'fetch_onion', 'fetch_dish'] * order_length

        # Find how many items in each pot, then make a new list of lists of what actually needs doing for each pot
        tasks = []
        for pot in range(number_of_pots):
            # Currently state.get_object is false if the pot doesn't have any onions! #TODO: Rectify this??
            pot_pos = self.mlp.mdp.get_pot_locations()[pot]
            if not state.has_object(pot_pos):
                num_items = 0
            else:
                _, num_items, _ = state.get_object(pot_pos).state

            tasks.append(initial_list_for_each_pot[num_items:])

        return tasks

    def find_min_plan_cost(self, motion_goals, state, temp_player_index):
        """
        Given some motion goals, find the cost for the lowest cost goal
        :param motion_goals:
        :param state:
        :param temp_player_index:
        :return:
        """

        start_pos_and_or = state.players_pos_and_or[temp_player_index]

        if len(motion_goals) == 0:
            min_cost = np.Inf
        else:
            min_cost = np.Inf
            # best_action = None
            for goal in motion_goals:
                _, _, plan_cost = self.mlp.mp.get_plan(start_pos_and_or, goal)
                if plan_cost < min_cost:
                    # best_action = action_plan[0]
                    min_cost = plan_cost
                    # best_goal = goal

        return min_cost


class AdvancedComplementaryModel(Agent):
    """
    Built on greedy human model
    - added a simple heuristic that's used to factor in other player's expected moves
    - motion goals can be retained rather than re-finding goals every timestep
    - make the wrong decision about a motion goal with prob: wrong_decision
    - thinking_time
    - with prob path_teamwork, factor in the other player (assuming Greedy) when finding the best path
    Adding:
    - Take a sub-optimal first action with Boltzmann rational probability
    """

    def __init__(self, mlp, player_index,
                 perseverance=0.5, teamwork=0.8, retain_goals=0.8, wrong_decisions=0.02, thinking_prob=0.8,
                 path_teamwork=0.8, rationality_coefficient=3, prob_pausing=0.7):
        self.mlp = mlp
        self.agent_index = player_index
        self.mdp = self.mlp.mdp
        self.prev_state = None
        self.timesteps_stuck = 0  # Count how many times there's a clash with the other player
        self.dont_drop = False
        self.prev_motion_goal = None
        self.prev_best_action = None
        self.GHM = GreedyHumanModelv2(self.mlp, player_index=1-self.agent_index)  # For factoring in the other players path

        self.human_model = True

        # "Personality" parameters (within 0 to 1, except rationality_coefficient)
        self.perseverance = perseverance  # perseverance = 1 means the agent always tries to go where it want to do, even when it's stuck
        self.teamwork = teamwork  # teamwork = 0 should make this agent v similar to GreedyHuman
        self.retain_goals = retain_goals  # Prob of keeping the previous goal each timestep (rather than re-calculating)
        self.wrong_decisions = wrong_decisions  # Prob of making a wrong decision about the motion_goal. I.e. they should
        # get onion but they get dish instead. Note: this should only work well if retain_goals is large
        self.thinking_prob = thinking_prob  # After achieving a goal (e.g. fetching onion) the agent waits to "think".
        # thinking_prob is the probability (p) of moving on during this "thinking time". Expected value of a geometrically
        # distributed random variable is 1/p, so set p to e.g. 0.25?
        self.path_teamwork = path_teamwork  # Prob of considering the other agent's location in the path choice
        self.rationality_coefficient = rationality_coefficient  # Setting to 0 means random actions; inf means always takes
        # lowest cost path. In practice inf ~ 100
        self.prob_pausing = prob_pausing # Probability of pausing on a given timestep, instead of acting. From a
        # quick initial look at the human data, the humans pause approx 70% of the time

    # def direct_action(self, observation):
    #     """Required for running with pbt. Each observation is a 25 (?) layered "mask" that is sent to the CNN for the ppo
    #     agents. The first dimension of observation is n=SIM_THREADS.
    #     :return: n actions"""
    #     # to do: Check that the first dimension of observation is indeed SIM_THREADS
    #     actions = []
    #     for i in range(observation.shape[0]):  # for each SIM_THREAD
    #         obs = observation[i, :, :, :]  # Select the SIM THREAD observation
    #         state = self.mdp.state_from_observation(obs, self.agent_index)  # Find the state
    #         # According to Overcooked.step in overcooked_env.py, the action should be "in index format":
    #         this_action = self.action(state)
    #         this_action = Action.ACTION_TO_INDEX[this_action]
    #         actions.append(this_action)
    #
    #     return np.array(actions)

    # NO LONGER NEED THIS (we want each agent to do one action, not one agent to do different actions for differnet
    # states!)
    # def multiple_thread_action(self, multi_thread_state):
    #     """Takes multiple states and outputs multiple actions"""
    #     actions=[]
    #     for i in range(multi_thread_state.__len__()):
    #         actions.append(self.action(multi_thread_state[i]))
    #     return actions

    def action(self, state):

        # Display the game during training:
        try:
            self.display
            overcooked_env = OvercookedEnv(self.mdp)
            overcooked_env.state = state
            print('TRAINING GAME WITH HM. HM index: {}'.format(self.agent_index))
            print(overcooked_env)
        except:
            AttributeError # self.display = False

        # To determine whether the same HM is used for several different envs during multiple_thread_action when
        # training the ppo using sim_threads. Answer: SAME HM is used accross DIFFERENT states, which is wrong!
        # try:
        #     self.stamp
        # except AttributeError:
        #     self.stamp = 0
        # print('...next iter...')
        # print('Previous best action: {}'.format(self.prev_best_action))
        # print('Previous state:')
        # print(self.prev_state)
        # print('Current state:')
        # print(state)
        # print('Previous stamp: {}; index: {}'.format(self.stamp, self.agent_index))
        # self.stamp = round(np.random.rand()*10**4)
        # print('Next stamp....: {}'.format(self.stamp))


        # The agent sometimes just pauses instead of acting:
        if random.random() > self.prob_pausing:
            logging.info('Agent not pausing')

            logging.info('Player: {}'.format(self.agent_index))
            logging.info('Other GHM: {}'.format(self.GHM.agent_index))
            # Get motion_goals if i) self.prev_motion_goal == None; ii) with prob = 1 - self.retain_goals;
            # iii) if stuck for 2 steps (??); iv) if reached goal
            # TODO: modify the getting stuck condition, e.g. include the goal to move out the way as a (different type of)
            #  goal to be saved (see notes)
            rand = random.random()
            logging.info('Prev ac plan: {}'.format(self.prev_best_action))
            if self.prev_best_action == None or rand > self.retain_goals or \
                    self.prev_best_action == 'interact' or self.prev_best_action == (0,0):

                if (self.prev_best_action == 'interact' or self.prev_best_action == (0,0)) \
                        and (random.random() > self.thinking_prob):

                    logging.info('Thinking time...')
                    # # Keep self.prev_action_plan == ['interact'], so that the agent also rethinks the next action
                    # action_plan = self.prev_action_plan
                    best_action = (0,0)

                else:

                    logging.info('Getting new goal')
                    # logging.info('rand: {}, retain_goals: {}, stuck: {}, current goal: {}, current pos or: {}'
                    #       .format(rand, self.retain_goals, self.timesteps_stuck,
                    #               self.prev_motion_goal, state.players_pos_and_or[self.agent_index]))

                    motion_goals = self.ml_action(state)

                    # Find plan; with Prob = self.path_teamwork factor in the other player
                    if random.random() < self.path_teamwork:
                        best_action, best_goal = self.find_plan_boltz_rat_inc_other(state, motion_goals)
                        logging.info('Choosing path inc other player. Best act: {}, goal: {}'.format(best_action, best_goal))

                        # If the plan that included the other player has infinite cost, then ignore them or do a random action
                        if best_action == None:
                            # Get temp action and goal by ignoring the other player:
                            best_action, best_goal = self.find_plan_boltz_rational(state, motion_goals)
                            # if random.random() < self.perseverance:
                            # Ignore the other player:
                            # best_action = best_action_temp
                            # best_goal = best_goal_temp
                            logging.info('No path with finite cost... ignoring other player now. Best act: {}, goal: {}'.format(best_action, best_goal))

                            # Removed this because Boltz rationality will take random actions
                            # else:
                            #     # Move in random direction, but keep goal as the best goal if ignoring the other agent
                            #     all_actions = Action.ALL_ACTIONS
                            #     best_action = all_actions[np.random.choice(len(all_actions))]
                            #     best_goal = best_goal_temp
                            #     logging.info('No path with finite cost... Random action. Best act: {}, goal: {}'.format(best_action, best_goal))

                    else:
                        best_action, best_goal = self.find_plan_boltz_rational(state, motion_goals)
                        logging.info('Choosing path ignoring other player. Best act: {}, goal: {}'.format(best_action, best_goal))

                    # Save motion goal:
                    self.prev_motion_goal = [best_goal]

            else:

                logging.info('Keeping old goal:')
                # Use previous goal:
                motion_goals = self.prev_motion_goal
                # Find action for this goal:
                # Find plan; with Prob = self.path_teamwork factor in the other player
                if random.random() < self.path_teamwork:
                    best_action, _ = self.find_plan_boltz_rat_inc_other(state, motion_goals)

                    # If the plan that included the other player has infinite cost, then ignore them or do random action
                    if best_action == None:
                        # Get temp action and goal by ignoring the other player:
                        best_action, _ = self.find_plan_boltz_rational(state, motion_goals)
                        # if random.random() < self.perseverance:
                        # Ignore the other player:
                        # best_action = best_action_temp

                        # Removed this because Boltz rationality will take random actions
                        # else:
                        #     # Move in random direction
                        #     all_actions = Action.ALL_ACTIONS
                        #     best_action = all_actions[np.random.choice(len(all_actions))]

                else:
                    best_action, _ = self.find_plan_boltz_rational(state, motion_goals)

            # Before setting self.prev_best_action = best_action, we need to determine if the prev_best_action was (0,0),
            # in which case the agent isn't stuck:
            if self.prev_best_action == (0,0):
                agent_chose_stationary = True
            else:
                agent_chose_stationary = False

            self.prev_best_action = best_action

            """If the agent is stuck, then take an alternative action with a probability based on the time stuck and the
            agent's "perseverance". Note: We consider an agent stuck if their whole state is unchanged (but this misses the
            case when they try to move and do change direction but can't move <-- here the state changes & they're stuck).
            Also, there exist cases when the state doesn't change but they're not stuck, they just can't complete their action
            (e.g. on unident they could try to use an onion but the other player has already filled the pot)"""
            #Todo: Check this more thoroughly: Later change to ensure previous action != (0, 0), because in this
            # cause the agent isn't stuck
            if self.prev_state is not None \
                    and state.players[self.agent_index] == self.prev_state.players[self.agent_index] \
                    and not agent_chose_stationary:

                self.timesteps_stuck += 1
                take_alternative = self.take_alternative_action()
                # logging.info('Player {} timesteps stuck: {}'.format(self.agent_index, self.timesteps_stuck))
                if take_alternative:
                     logging.info('Stuck, and taking alternative action')
                     # If the agent is stuck, set self.prev_action_plan = None, then they always re-think their goal
                     # TODO: This is the place to put a more thorough avoiding action, e.g. 2 steps rather than 1?
                     # self.prev_action_plan = ['interact']  # Trialling this, because it means the agent sometimes pauses after
                     # taking an avoiding action, which measn 2 agents won't get stuck so much
                     self.prev_best_action = None

                     # logging.info('Taking alternative action!')
                     # Select an action at random that would change the player positions if the other player were not to move
                     if self.agent_index == 0:
                         joint_actions = list(itertools.product(Action.ALL_ACTIONS, [Action.STAY]))
                     elif self.agent_index == 1:
                         joint_actions = list(itertools.product([Action.STAY], Action.ALL_ACTIONS))
                     else:
                         raise ValueError("Player index not recognized")

                     unblocking_joint_actions = []
                     for j_a in joint_actions:
                         new_state, _, _ = self.mlp.mdp.get_state_transition(state, j_a)
                         if new_state.player_positions != self.prev_state.player_positions:
                             unblocking_joint_actions.append(j_a)

                     """Prefer adjacent actions if available:"""
                     # Adjacent actions only exist if best_action is N, S, E, W
                     if best_action in Direction.ALL_DIRECTIONS:

                         # Find the adjacent actions:
                         if self.agent_index == 0:
                             joint_adjacent_actions = list(itertools.product(Direction.get_adjacent_directions(best_action),
                                                                             [Action.STAY]))
                         elif self.agent_index == 1:
                             joint_adjacent_actions = list(itertools.product([Action.STAY],
                                                                             Direction.get_adjacent_directions(best_action)))
                         else:
                             raise ValueError("Player index not recognized")

                         # If at least one of the adjacent actions is in the set of unblocking_joint_actions, then select these:
                         if (joint_adjacent_actions[0] in unblocking_joint_actions
                                 or joint_adjacent_actions[1] in unblocking_joint_actions):
                             preferred_unblocking_joint_actions = []
                             # There are only ever two adjacent actions:
                             if joint_adjacent_actions[0] in unblocking_joint_actions:
                                 preferred_unblocking_joint_actions.append(joint_adjacent_actions[0])
                             if joint_adjacent_actions[1] in unblocking_joint_actions:
                                 preferred_unblocking_joint_actions.append(joint_adjacent_actions[1])
                         elif (joint_adjacent_actions[0] not in unblocking_joint_actions
                               and joint_adjacent_actions[1] not in unblocking_joint_actions):
                             # No adjacent actions in the set of unblocking_joint_actions, so keep these actions
                             preferred_unblocking_joint_actions = unblocking_joint_actions
                         else:
                             raise ValueError("Binary truth value is neither true nor false")

                     # If adjacent actions don't exist then keep unblocking_joint_actions as it is
                     else:
                         preferred_unblocking_joint_actions = unblocking_joint_actions

                     best_action = preferred_unblocking_joint_actions[
                         np.random.choice(len(preferred_unblocking_joint_actions))][self.agent_index]
                     # Note: np.random isn't actually random!


            else:
                self.timesteps_stuck = 0  # Reset to zero if prev & current player states aren't the same (they're not stuck)

            # NOTE: Assumes that calls to action are sequential
            self.prev_state = state

        # The agent sometimes just pauses instead of acting:
        else:
            logging.info('Agent pausing')
            best_action = (0,0)

        return best_action


    def ml_action(self, state):
        """Selects a medium level action for the current state"""

        player = state.players[self.agent_index]
        other_player = state.players[1 - self.agent_index]
        am = self.mlp.ml_action_manager

        counter_objects = self.mlp.mdp.get_counter_objects_dict(state, list(self.mlp.mdp.terrain_pos_dict['X']))
        pot_states_dict = self.mlp.mdp.get_pot_states(state)

        #next_order = state.order_list[0]  # Assuming soup only for now

        # (removed tomato)
        ready_soups = pot_states_dict['onion']['ready']
        cooking_soups = pot_states_dict['onion']['cooking']

        soup_nearly_ready = len(ready_soups) > 0 or len(cooking_soups) > 0
        count_soups_nearly_ready = len(ready_soups) + len(cooking_soups)
        other_has_dish = other_player.has_object() and other_player.get_object().name == 'dish'
        other_has_onion = other_player.has_object() and other_player.get_object().name == 'onion'

        number_of_pots = self.mlp.mdp.get_pot_locations().__len__()

        temp_dont_drop = False

        if not player.has_object():

            # Only get the dish if the soup is nearly ready
            if soup_nearly_ready:

                """Get the dish... unless other player is better placed"""

                default_motion_goals = am.pickup_dish_actions(state, counter_objects)

                # Remove invalid goals:
                own_motion_goals = list(filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, goal), default_motion_goals))
                others_motion_goals = list(filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(other_player.pos_and_or, goal), default_motion_goals))

                own_min_cost = self.find_min_plan_cost(own_motion_goals, state, self.agent_index)
                others_min_cost = self.find_min_plan_cost(others_motion_goals, state, 1 - self.agent_index)

                # logging.info('Player {}'.format(self.agent_index))
                # logging.info('own min cost: {}'.format(own_min_cost))
                # logging.info('other\'s min cost: {}'.format(others_min_cost))

                # Get the dish if the other player doesn't have a dish or they do have a dish but are further away
                if (not other_has_dish) and (others_min_cost >= own_min_cost):  #TODO: consider having >

                    logging.info('Getting the dish: assume in better position that other player')
                    motion_goals = default_motion_goals

                    # Wrong decision:
                    if random.random() < self.wrong_decisions:
                        motion_goals = am.pickup_onion_actions(state, counter_objects)
                        logging.info('Making the wrong decision!')

                elif other_has_dish or ((not other_has_dish) and (others_min_cost < own_min_cost)):
                    # Now find the *next* task that needs to be done, and do that
                    # Simple way of doing this: IF only one nearly_ready pot at the top of the list THEN get onion
                    # IF both lists start with pot, THEN get a dish (worry about where to put it later!)

                    logging.info('Assume other player is in better position, so do the next task on the list')
                    tasks = self.tasks_to_do(state)

                    # ASSUMING TWO POTS MAX
                    number_of_pots = self.mlp.mdp.get_pot_locations().__len__()
                    if number_of_pots == 1:
                        # Next task will always be onion
                        logging.info('Next task: onion')
                        motion_goals = am.pickup_onion_actions(state, counter_objects)

                        # Wrong decision:
                        if random.random() < self.wrong_decisions:
                            motion_goals = am.pickup_dish_actions(state, counter_objects)
                            logging.info('Making the wrong decision!')

                        logging.info('Onion shouldnt be dropped, so set temp_dont_drop = True')
                        temp_dont_drop = True

                    elif number_of_pots == 2:

                        if (tasks[0][0] == 'fetch_onion') or (tasks[1][0] == 'fetch_onion'):

                            logging.info('Next task: onion')
                            # Next task will be onion
                            motion_goals = am.pickup_onion_actions(state, counter_objects)

                            # Wrong decision:
                            if random.random() < self.wrong_decisions:
                                motion_goals = am.pickup_dish_actions(state, counter_objects)
                                logging.info('Making the wrong decision!')

                        elif (tasks[0][0] == 'fetch_dish') and (tasks[1][0] == 'fetch_dish'):

                            logging.info('Next task: dish')
                            # Next task will be dish
                            motion_goals = am.pickup_dish_actions(state, counter_objects)

                            # Wrong decision:
                            if random.random() < self.wrong_decisions:
                                motion_goals = am.pickup_onion_actions(state, counter_objects)
                                logging.info('Making the wrong decision!')

                        else:
                            raise ValueError('Failed logic')
                            #TODO: is there a special type of logical error??

                else:
                    raise ValueError('Failed logic')

                if random.random() > self.teamwork:
                    # Overwrite consideration of other player
                    logging.info('Overwriting motion_goal with the greedy option')
                    motion_goals = default_motion_goals

                    # Wrong decision:
                    if random.random() < self.wrong_decisions:
                        motion_goals = am.pickup_onion_actions(state, counter_objects)
                        logging.info('Making the wrong decision!')


            elif not soup_nearly_ready:

                """If there are two pots, then we can only get inside this 'if' when both pots have onion first on the list
                of tasks, so we can always get an onion! For 1 pot then just look what's next on the list"""

                default_motion_goals = am.pickup_onion_actions(state, counter_objects)

                # ASSUMING TWO POTS MAX THROUGHOUT
                if number_of_pots == 1:

                    # Remove invalid goals of getting an onion:
                    own_motion_goals = list(
                        filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, goal),
                               default_motion_goals))
                    others_motion_goals = list(
                        filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(other_player.pos_and_or, goal),
                               default_motion_goals))

                    own_min_cost = self.find_min_plan_cost(own_motion_goals, state, self.agent_index)
                    others_min_cost = self.find_min_plan_cost(others_motion_goals, state, 1 - self.agent_index)

                    # logging.info('Player {}'.format(self.agent_index))
                    # logging.info('own min cost: {}'.format(own_min_cost))
                    # logging.info('other\'s min cost: {}'.format(others_min_cost))

                    # Get the onion if the other player doesn't have an onion or they do have an onion but are further away
                    if (not other_has_onion) and (others_min_cost >= own_min_cost):  # TODO: make it random if others_min_cost = own_min_cost ?

                        logging.info('Getting the onion: assume in better position that other player')
                        motion_goals = default_motion_goals

                        # Wrong decision:
                        if random.random() < self.wrong_decisions:
                            motion_goals = am.pickup_dish_actions(state, counter_objects)
                            logging.info('Making the wrong decision!')

                    elif other_has_onion or ((not other_has_onion) and (others_min_cost < own_min_cost)):
                        # Now find the *next* task that needs to be done, and do that

                        logging.info('Assume other player is in better position, so do the next task on the list')
                        tasks = self.tasks_to_do(state)

                        if tasks[0][1] == 'fetch_onion':  # tasks[0][1] is the next task (when there's one pot)
                            logging.info('Next task: onion')
                            motion_goals = am.pickup_onion_actions(state, counter_objects)

                            # Wrong decision:
                            if random.random() < self.wrong_decisions:
                                motion_goals = am.pickup_dish_actions(state, counter_objects)
                                logging.info('Making the wrong decision!')

                        elif tasks[0][1] == 'fetch_dish':
                            logging.info('Next task: dish')
                            motion_goals = am.pickup_dish_actions(state, counter_objects)

                            # Wrong decision:
                            if random.random() < self.wrong_decisions:
                                motion_goals = am.pickup_onion_actions(state, counter_objects)
                                logging.info('Making the wrong decision!')

                            logging.info('Dish shouldnt be dropped, so set temp_dont_drop = True')
                            temp_dont_drop = True

                elif number_of_pots == 2:
                    # Get an onion regardless: there's always a pot needing an onion (because 'not soup_nearly_ready')
                    motion_goals = default_motion_goals

                    # Wrong decision:
                    if random.random() < self.wrong_decisions:
                        motion_goals = am.pickup_dish_actions(state, counter_objects)
                        logging.info('Making the wrong decision!')

                else:
                    raise ValueError('Assuming 1 or 2 pots, but there are more??')

                if random.random() > self.teamwork:
                    # Overwrite consideration of other player
                    logging.info('Overwriting motion_goal with the greed option')
                    motion_goals = default_motion_goals

                    # Wrong decision:
                    if random.random() < self.wrong_decisions:
                        motion_goals = am.pickup_dish_actions(state, counter_objects)
                        logging.info('Making the wrong decision!')

            else:
                raise ValueError('Failed logic')

            # If there's a soup on the counter, then override other goals and get the soup
            # UNLESS other player (strictly) is closer
            if 'soup' in counter_objects:

                default_motion_goals = am.pickup_counter_soup_actions(state, counter_objects)

                # Remove invalid goals of getting an onion:
                own_motion_goals = list(
                    filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, goal),
                           default_motion_goals))
                others_motion_goals = list(
                    filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(other_player.pos_and_or, goal),
                           default_motion_goals))

                own_min_cost = self.find_min_plan_cost(own_motion_goals, state, self.agent_index)
                others_min_cost = self.find_min_plan_cost(others_motion_goals, state, 1 - self.agent_index)

                if others_min_cost >= own_min_cost:

                    logging.info('Soup on counter and other player is futher away: get soup')
                    motion_goals = default_motion_goals

                    # Wrong decision:
                    if random.random() < self.wrong_decisions:
                        # Get dish or onion with 50% prob each
                        if random.random() > 0.5:
                            motion_goals = am.pickup_dish_actions(state, counter_objects)
                        else:
                            motion_goals = am.pickup_onion_actions(state, counter_objects)
                        logging.info('Making the wrong decision!')

                elif others_min_cost < own_min_cost:

                    logging.info('Soup on counter but other closer')
                    # logging.info('Own cost = {}'.format(own_min_cost))



        elif player.has_object():

            # Determine if any soups need onions
            pots_empty = len(pot_states_dict['empty'])
            pots_partially_full = len(pot_states_dict['onion']['partially_full'])
            soups_need_onions = pots_empty + pots_partially_full

            player_obj = player.get_object()

            if player_obj.name == 'onion':

                default_motion_goals = am.put_onion_in_pot_actions(pot_states_dict)

                if soups_need_onions == 0:

                    if self.dont_drop == True:
                        # Don't drop the onion
                        logging.info('Got onion because its the next task: just do nothing')
                        #TODO: unrealistic to do nothing, but the alternatives are hard to code
                        motion_goals = state.players[self.agent_index].pos_and_or
                        temp_dont_drop = True

                        # Wrong decision:
                        if random.random() < self.wrong_decisions:
                            if random.random() > 0.5:
                                motion_goals = am.place_obj_on_counter_actions(state)
                            else:
                                motion_goals = am.pickup_soup_with_dish_actions(pot_states_dict)
                            logging.info('Making the wrong decision!')

                    else:
                        # Drop onion
                        logging.info('Onion not needed: drop it')
                        motion_goals = am.place_obj_on_counter_actions(state)

                        # Wrong decision:
                        if random.random() < self.wrong_decisions:
                            # Treating the onion as if it were a dish!
                            motion_goals = am.pickup_soup_with_dish_actions(pot_states_dict)
                            logging.info('Making the wrong decision!')

                elif soups_need_onions == 1:
                    # If closer than the other player, then deliver the onion

                    # Remove invalid goals of delivering an onion:
                    own_motion_goals = list(
                        filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, goal),
                               default_motion_goals))
                    others_motion_goals = list(
                        filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(other_player.pos_and_or, goal),
                               default_motion_goals))

                    own_min_cost = self.find_min_plan_cost(own_motion_goals, state, self.agent_index)
                    others_min_cost = self.find_min_plan_cost(others_motion_goals, state, 1 - self.agent_index)

                    if other_player.has_object() and (other_player.get_object().name == 'onion') and (others_min_cost < own_min_cost):

                        if number_of_pots == 2:
                            # Two pots but only 1 needs onion: this means a dish is needed!
                            logging.info('Two pots but only 1 needs onion: this means a dish is needed!')
                            motion_goals = am.place_obj_on_counter_actions(state)

                            # Wrong decision:
                            if random.random() < self.wrong_decisions:
                                motion_goals = am.put_onion_in_pot_actions(pot_states_dict)
                                logging.info('Making the wrong decision!')

                        elif number_of_pots == 1:
                            logging.info('Only one pot, which needs an onion, but other is closer. So do the next task')
                            # Do the next task
                            tasks = self.tasks_to_do(state)

                            if tasks[0][1] == 'fetch_onion':  # tasks[0][1] is the next task (when there's one pot)
                                logging.info('Next task: deliver onion')
                                motion_goals = default_motion_goals

                                # Wrong decision:
                                if random.random() < self.wrong_decisions:
                                    motion_goals = am.place_obj_on_counter_actions(state)
                                    logging.info('Making the wrong decision!')

                            elif tasks[0][1] == 'fetch_dish':
                                logging.info('Next task: dish needed')
                                motion_goals = am.place_obj_on_counter_actions(state)

                                # Wrong decision:
                                if random.random() < self.wrong_decisions:
                                    motion_goals = am.put_onion_in_pot_actions(pot_states_dict)
                                    logging.info('Making the wrong decision!')

                        else:
                            raise ValueError('More pots than expected')

                    else:
                        # Deliver onion
                        motion_goals = default_motion_goals

                        # Wrong decision:
                        if random.random() < self.wrong_decisions:
                            motion_goals = am.place_obj_on_counter_actions(state)
                            logging.info('Making the wrong decision!')

                elif soups_need_onions == 2:
                    # Deliver onion regardless of other player
                    motion_goals = am.put_onion_in_pot_actions(pot_states_dict)

                    # Wrong decision:
                    if random.random() < self.wrong_decisions:
                        motion_goals = am.place_obj_on_counter_actions(state)
                        logging.info('Making the wrong decision!')

                if random.random() > self.teamwork:
                    # Overwrite consideration of other player
                    logging.info('Overwriting motion_goal with the greed option')
                    if soups_need_onions == 0:
                        # Player has an onion but there are no soups to put it in, then drop the onion
                        motion_goals = am.place_obj_on_counter_actions(state)

                        # Wrong decision:
                        if random.random() < self.wrong_decisions:
                            # Treating the onion as if it were a dish!
                            motion_goals = am.pickup_soup_with_dish_actions(pot_states_dict)
                            logging.info('Making the wrong decision!')

                    else:
                        motion_goals = am.put_onion_in_pot_actions(pot_states_dict)

                        # Wrong decision:
                        if random.random() < self.wrong_decisions:
                            motion_goals = am.place_obj_on_counter_actions(state)
                            logging.info('Making the wrong decision!')


            elif player_obj.name == 'dish':

                default_motion_goals = am.pickup_soup_with_dish_actions(pot_states_dict, only_nearly_ready=True)

                if count_soups_nearly_ready == 0:

                    if (number_of_pots == 1) and (self.dont_drop == True):
                            # Don't drop the dish
                            logging.info('Got dish because its the next task: just do nothing')
                            #TODO: unrealistic to do nothing, but the alternatives are hard to code
                            motion_goals = state.players[self.agent_index].pos_and_or

                            # Wrong decision:
                            if random.random() < self.wrong_decisions:
                                if random.random() > 0.5:
                                    motion_goals = am.place_obj_on_counter_actions(state)
                                else:
                                    motion_goals = am.put_onion_in_pot_actions(pot_states_dict)
                                logging.info('Making the wrong decision!')
                            temp_dont_drop = True
                    else:
                        # Got dish but no soups to fetch
                        logging.info('Got dish but no soups to fetch')
                        motion_goals = am.place_obj_on_counter_actions(state)

                        # Wrong decision:
                        if random.random() < self.wrong_decisions:
                            motion_goals = am.put_onion_in_pot_actions(pot_states_dict)
                            logging.info('Making the wrong decision!')

                elif count_soups_nearly_ready == 1:
                    # If closer than the other player, then fetch the soup

                    # Remove invalid goals:
                    own_motion_goals = list(
                        filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, goal),
                               default_motion_goals))
                    others_motion_goals = list(
                        filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(other_player.pos_and_or, goal),
                               default_motion_goals))

                    own_min_cost = self.find_min_plan_cost(own_motion_goals, state, self.agent_index)
                    others_min_cost = self.find_min_plan_cost(others_motion_goals, state, 1 - self.agent_index)

                    if other_player.has_object() and (other_player.get_object().name == 'dish') and (others_min_cost < own_min_cost):

                        logging.info('Other player has dish and is closer to a nearly ready soup')

                        #TODO: this if loop not needed
                        if number_of_pots == 2:
                            # Two pots but only 1 needs dish: so get an onion!
                            logging.info('Two pots but only 1 needs dish: this means an onion is needed!')
                            motion_goals = am.place_obj_on_counter_actions(state)

                            # Wrong decision:
                            if random.random() < self.wrong_decisions:
                                motion_goals = default_motion_goals
                                logging.info('Making the wrong decision!')

                        elif number_of_pots == 1:
                            logging.info('Only one pot, which needs a dish, but other is closer. Next task is always onion')
                            motion_goals = am.place_obj_on_counter_actions(state)

                            # Wrong decision:
                            if random.random() < self.wrong_decisions:
                                motion_goals = default_motion_goals
                                logging.info('Making the wrong decision!')

                        else:
                            raise ValueError('More pots than expected')

                    else:
                        motion_goals = default_motion_goals

                        # Wrong decision:
                        if random.random() < self.wrong_decisions:
                            motion_goals = am.place_obj_on_counter_actions(state)
                            logging.info('Making the wrong decision!')

                elif count_soups_nearly_ready == 2:
                    # Collect soup regardless
                    motion_goals = default_motion_goals

                    # Wrong decision:
                    if random.random() < self.wrong_decisions:
                        motion_goals = am.place_obj_on_counter_actions(state)
                        logging.info('Making the wrong decision!')

                else:
                    raise ValueError()

                if random.random() > self.teamwork:
                    # Overwrite consideration of other player
                    logging.info('Overwriting motion_goal with the greed option')
                    if not soup_nearly_ready:
                        # Player has a dish but there are no longer any "nearly ready" soups: drop the dish
                        motion_goals = am.place_obj_on_counter_actions(state)

                        # Wrong decision:
                        if random.random() < self.wrong_decisions:
                            motion_goals = am.put_onion_in_pot_actions(pot_states_dict)
                            logging.info('Making the wrong decision!')

                    elif soup_nearly_ready:
                        motion_goals = am.pickup_soup_with_dish_actions(pot_states_dict, only_nearly_ready=True)

                        # Wrong decision:
                        if random.random() < self.wrong_decisions:
                            motion_goals = am.place_obj_on_counter_actions(state)
                            logging.info('Making the wrong decision!')

            elif player_obj.name == 'soup':
                # Deliver soup whatever the other player is doing
                motion_goals = am.deliver_soup_actions()

                # Wrong decision:
                if random.random() < self.wrong_decisions:
                    motion_goals = am.place_obj_on_counter_actions(state)
                    logging.info('Making the wrong decision!')

            else:
                raise ValueError()
        else:
            raise ValueError('Player has AND does not have an object!')



        # OLd code:
        #
        # # Determine if any soups are "nearly ready", meaning that they are cooking or ready
        # if next_order == 'any':
        #     ready_soups = pot_states_dict['onion']['ready'] + pot_states_dict['tomato']['ready']
        #     cooking_soups = pot_states_dict['onion']['cooking'] + pot_states_dict['tomato']['cooking']
        # else:
        #     ready_soups = pot_states_dict[next_order]['ready']
        #     cooking_soups = pot_states_dict[next_order]['cooking']
        #
        # soup_nearly_ready = len(ready_soups) > 0 or len(cooking_soups) > 0
        # #other_has_dish = other_player.has_object() and other_player.get_object().name == 'dish'  <-- no longer used

        # if not player.has_object():
        #
        #     if soup_nearly_ready:  # PK removed "and not other_has_dish"
        #         motion_goals = am.pickup_dish_actions(state, counter_objects)
        #     else:
        #         next_order = None
        #         if len(state.order_list) > 1:
        #             next_order = state.order_list[1]
        #
        #         if next_order == 'onion':
        #             motion_goals = am.pickup_onion_actions(state, counter_objects)
        #         elif next_order == 'tomato':
        #             motion_goals = am.pickup_tomato_actions(state, counter_objects)
        #         elif next_order is None or next_order == 'any':
        #             motion_goals = am.pickup_onion_actions(state, counter_objects) + am.pickup_tomato_actions(state, counter_objects)

            # # If there's a soup on the counter, then override other goals and get the soup
            # if 'soup' in counter_objects:
            #     motion_goals = am.pickup_counter_soup_actions(state, counter_objects)

        # else:
        #
        #     # Determine if any soups need onions
        #     pots_empty = len(pot_states_dict['empty'])
        #     pots_partially_full = len(pot_states_dict['onion']['partially_full'])
        #     soup_needs_onions = pots_empty > 0 or pots_partially_full > 0
        #
        #     player_obj = player.get_object()
        #
        #     if player_obj.name == 'onion':
        #         if not soup_needs_onions:
        #             # If player has an onion but there are no soups to put it in, then drop the onion!
        #             motion_goals = am.place_obj_on_counter_actions(state)
        #         else:
        #             motion_goals = am.put_onion_in_pot_actions(pot_states_dict)
        #
        #     elif player_obj.name == 'tomato':
        #         motion_goals = am.put_tomato_in_pot_actions(pot_states_dict)
        #
        #     elif player_obj.name == 'dish':
        #         # If player has a dish but there are no longer any "nearly ready" soups, then drop the dish!
        #         if not soup_nearly_ready:
        #             motion_goals = am.place_obj_on_counter_actions(state)
        #         elif soup_nearly_ready:
        #             motion_goals = am.pickup_soup_with_dish_actions(pot_states_dict, only_nearly_ready=True)
        #
        #     elif player_obj.name == 'soup':
        #         motion_goals = am.deliver_soup_actions()
        #
        #     else:
        #         raise ValueError()

        # Remove invalid goals:
        motion_goals = list(filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, goal), motion_goals))

        # If no goals, then just go to nearest feature
        if len(motion_goals) == 0:
            motion_goals = am.go_to_closest_feature_actions(player)
            motion_goals = list(filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, goal), motion_goals))
            assert len(motion_goals) != 0

        if temp_dont_drop == True:
            self.dont_drop = True
        else:
            self.dont_drop = False

        return motion_goals

    def take_alternative_action(self):
        """This first gives Prob(taking alternative action)=1 if perseverance=0 and Prob=0 if perseverance=1. Otherwise,
        e.g. perseverance=_, num_items, _ = state.get_object(self.mlp.mdp.get_pot_locations()[0]).state
            #current_list = full_list[num_items:number_of_tasks]

        # if number_of_pots == 2:
            #0.5, for the first timestep Prob~0.5, then Prob-->1 quickly as time increases. Then using this prob
        it randomly determines if the player should take an alternative action."""

        Prob_taking_alternative = 1 - (1 / (np.exp(self.timesteps_stuck*10))) ** (1 - self.perseverance ** (1 / 10))
        rand = random.random()
        take_alternative = rand < Prob_taking_alternative

        # logging.info('time stuck: {}, perserv: {}, Prob_taking_alt: {}, Taking alt: {}'.format(self.timesteps_stuck, self.perseverance, Prob_taking_alternative, take_alternative))

        return take_alternative

    def tasks_to_do(self, state):
        """
        :return: tasks = a list of list of tasks to be done, where tasks[i] is a list of what need doing for pot i (where
        pot i is the pot at location get_pot_locations()[i])
        """

        # OLD
        # if number_of_pots == 1:
        #     order_length = state.order_list.__len__()
        #     full_list = ['fetch_onion','fetch_onion','fetch_onion','fetch_dish']*order_length
        #     pot_pos = self.mlp.mdp.get_pot_locations()[0]
        #     if not state.has_object(pot_pos):
        #         num_items = 0
        #     else:
        #         _, num_items, _ = state.get_object(self.mlp.mdp.get_pot_locations()[0]).state
        #     current_list = full_list[num_items:num_items+number_of_tasks]


        number_of_pots = self.mlp.mdp.get_pot_locations().__len__()
        order_length = state.order_list.__len__()
        initial_list_for_each_pot = ['fetch_onion', 'fetch_onion', 'fetch_onion', 'fetch_dish'] * order_length

        # Find how many items in each pot, then make a new list of lists of what actually needs doing for each pot
        tasks = []
        for pot in range(number_of_pots):
            # Currently state.get_object is false if the pot doesn't have any onions! #TODO: Rectify this??
            pot_pos = self.mlp.mdp.get_pot_locations()[pot]
            if not state.has_object(pot_pos):
                num_items = 0
            else:
                _, num_items, _ = state.get_object(pot_pos).state

            num_items = int(num_items)
            tasks.append(initial_list_for_each_pot[num_items:])

        return tasks

    def find_min_plan_cost(self, motion_goals, state, temp_player_index):
        """
        Given some motion goals, find the cost for the lowest cost goal
        :param motion_goals:
        :param state:
        :param temp_player_index:
        :return:
        """

        start_pos_and_or = state.players_pos_and_or[temp_player_index]

        if len(motion_goals) == 0:
            min_cost = np.Inf
        else:
            min_cost = np.Inf
            # best_action = None
            for goal in motion_goals:
                _, _, plan_cost = self.mlp.mp.get_plan(start_pos_and_or, goal)
                if plan_cost < min_cost:
                    # best_action = action_plan[0]
                    min_cost = plan_cost
                    # best_goal = goal

        return min_cost

    def find_plan(self, state, motion_goals):

        # Find valid actions:
        valid_next_pos_and_ors = self.find_valid_next_pos_and_ors(state)

        start_pos_and_or = state.players_pos_and_or[self.agent_index]
        min_cost = np.Inf
        best_action = None
        for goal in motion_goals:
            action_plan, _, plan_cost = self.mlp.mp.get_plan(start_pos_and_or, goal)
            if plan_cost < min_cost:
                best_action = action_plan[0]
                min_cost = plan_cost
                best_goal = goal

        return best_action, best_goal, action_plan

    def find_valid_next_pos_and_ors(self, state):
        # Assuming the other agent doesn't move
        if self.agent_index == 0:
            joint_actions = list(itertools.product(Action.ALL_ACTIONS, [Action.STAY]))
        elif self.agent_index == 1:
            joint_actions = list(itertools.product([Action.STAY], Action.ALL_ACTIONS))
        else:
            raise ValueError("Player index not recognized")

        valid_next_pos_and_ors = []
        for j_a in joint_actions:
            new_state, _, _ = self.mlp.mdp.get_state_transition(state, j_a)
            if new_state.player_positions != state.player_positions:
                valid_next_pos_and_ors.append(new_state.players_pos_and_or[self.agent_index])

        return valid_next_pos_and_ors

    def find_plan_from_start_pos_and_or(self, start_pos_and_or, motion_goals):

        min_cost = np.Inf
        # best_action = None
        for goal in motion_goals:
            _, _, plan_cost = self.mlp.mp.get_plan(start_pos_and_or, goal)
            if plan_cost < min_cost:
                # best_action = action_plan[0]
                min_cost = plan_cost
                best_goal = goal
            elif plan_cost == min_cost and plan_cost != np.Inf and random.random() > 0.5:
                # If the cost is the same, then pick randomly
                best_goal = goal

        return best_goal, min_cost

    def boltz_rationality(self, x):
        """Compute softmax values for each sets of scores in x."""
        temperature = self.rationality_coefficient
        exponent = [-y*temperature for y in x]
        #TODO: We take -ve exponents because we want to prioritise the smallest costs. Is taking -ve a good way to do it??

        # if np.isnan(np.sum(np.exp(exponent), axis=0)) or np.sum(np.exp(exponent), axis=0) == 0:
            # print('Dividing by zero or NaN in function boltz_rationality')

        return np.exp(exponent) / np.sum(np.exp(exponent), axis=0)

    def find_plan_boltz_rational(self, state, motion_goals):

        player_pos_and_or = state.players_pos_and_or[self.agent_index]

        # Find valid actions: (INCLUDING NOT MOVING)
        valid_next_pos_and_ors = self.find_valid_next_pos_and_ors(state) + [player_pos_and_or]

        # action_plans = []
        plan_costs = []
        plan_goals = []

        for start_pos_and_or in valid_next_pos_and_ors:

            plan_goal, plan_cost = self.find_plan_from_start_pos_and_or(start_pos_and_or, motion_goals)

            # action_plans.append(action_plan)  # <-- this is the action plan from the next position, not current position
            plan_costs.append(plan_cost)
            plan_goals.append(plan_goal)

        # TODO: Just adding the current pos_or to valid_next_... misses the option of staying in the same position but changing
        #  orientation. The result is that the plan_cost for this is 1 more than it should be. So this method over-inflates
        #  the cost of just changing direction. To recify this we reduce the cost by 1. BUT this under-inflates the cost
        #  of being in one place but not needing to change direction! Anyhow, if a human was in the right place it's v
        #  unlikely they'd randomly step away?!
        if plan_goals[len(plan_goals)-1][0] == player_pos_and_or[0]:
            plan_costs[len(plan_costs)-1] -= 1

        # Next: convert costs into probability distributions
        plan_probs = self.boltz_rationality(plan_costs)

        # Random choice from prob dist
        chosen_index = random.choices(range(len(plan_goals)), plan_probs)[0]
        # chosen_action_plan = action_plans[chosen_index]
        chosen_goal = plan_goals[chosen_index]
        # chosen_action = chosen_action_plan[0]

        # Now we have a chosen goal, we need to choose the chosen NEXT POS_OR from the valid ones:
        chosen_next_pos_and_or = valid_next_pos_and_ors[chosen_index]

        if chosen_index == (len(valid_next_pos_and_ors)-1) and chosen_next_pos_and_or[0] == chosen_goal[0]:
            # In this case we choose the action that leaves us in the same location. BUT we might want to change direction,
            # so we need to find whether the action to reach the chosen goal just requires a change of direction. If so,
            # then take this action.
            action_plan, _, plan_cost_test = self.mlp.mp.get_plan(player_pos_and_or, chosen_goal)
            if plan_cost_test > 2:  # At most the plan should be to change direction and interact
                raise ValueError('Incorrect action plan chosen')
            chosen_action = action_plan[0]

        else:
            # Now find which action gets to that state!
            # Note: We can't use get_plan because goals must face an object, whereas some of the valid_pos_and_or don't
            action_plan, _, plan_cost_test = self.mlp.mp.action_plan_from_positions(chosen_next_pos_and_or,
                                                                                    player_pos_and_or,
                                                                                    chosen_next_pos_and_or)
            if plan_cost_test != 2:  # Plan should be 1 step then interact
                if action_plan != ['interact']:
                    raise ValueError('Incorrect action plan chosen')
            chosen_action = action_plan[0]

        return chosen_action, chosen_goal

    def find_joint_plan_from_start_pos_and_or(
            self, start_pos_and_or, start_pos_and_or_other, motion_goals, others_predicted_goal):

        # Now find their own best goal, assuming the other agent is Greedy:
        min_cost = np.Inf
        # best_action = None

        for goal in motion_goals:

            if self.agent_index == 0:
                start_jm_state = (start_pos_and_or, start_pos_and_or_other)
                goal_jm_state = (goal, others_predicted_goal)
            elif self.agent_index == 1:
                start_jm_state = (start_pos_and_or_other, start_pos_and_or)
                goal_jm_state = (others_predicted_goal, goal)
            else:
                raise ValueError('Index error')

            _, _, plan_lengths = self.mlp.jmp.get_low_level_action_plan(start_jm_state, goal_jm_state)
            if plan_lengths[self.agent_index] < min_cost:
                # best_action = joint_action_plan[0][self.agent_index]
                min_cost = plan_lengths[self.agent_index]
                best_goal = goal
            elif plan_lengths[self.agent_index] == min_cost and plan_lengths[self.agent_index] != np.Inf and random.random() > 0.5:
                # If the cost is the same, then pick randomly
                best_goal = goal

        if min_cost == np.Inf:
            # Then there is no finite-cost goal
            best_goal = None

        return best_goal, min_cost

    def find_plan_boltz_rat_inc_other(self, state, motion_goals):

        player_pos_and_or = state.players_pos_and_or[self.agent_index]

        # Find valid actions: (INCLUDING NOT MOVING). This assumes the other player doesn't move
        #TODO: Assume the other moves to take a greedy action?
        valid_next_pos_and_ors = self.find_valid_next_pos_and_ors(state) + [player_pos_and_or]

        # Assume other player is GreedyHumanModel and find what action they would do:
        other_player_pos_and_or = state.players_pos_and_or[1 - self.agent_index]
        others_predicted_goals = self.GHM.ml_action(state)
        # Find their closest goal
        #TODO: Can use helper function find_plan, after some modification
        min_cost = np.Inf
        others_predicted_goal = None
        for goal in others_predicted_goals:
            _, _, others_plan_cost = self.mlp.mp.get_plan(other_player_pos_and_or, goal)
            if others_plan_cost < min_cost:
                min_cost = others_plan_cost
                others_predicted_goal = goal

        # For each valid start position find goals and costs, using joint planner
        plan_costs = []
        plan_goals = []

        for start_pos_and_or in valid_next_pos_and_ors:
            #TODO: Currently i) assuming other player doesn't move initially and ii) finding joint optimal action... this
            # is inconsistent -- we should assume first step of other is Greedy. Anyway, we're simulating sub-optimal
            # trajectories so it's not too bad to do this
            plan_goal, plan_cost = self.find_joint_plan_from_start_pos_and_or(
                start_pos_and_or, other_player_pos_and_or, motion_goals, others_predicted_goal)

            # action_plans.append(action_plan)  # <-- this is the action plan from the next position, not current position
            plan_costs.append(plan_cost)
            plan_goals.append(plan_goal)

        # TODO: Just adding the current pos_or to valid_next_... misses the option of staying in the same position but changing
        #  orientation. The result is that the plan_cost for this is 1 more than it should be. So this method over-inflates
        #  the cost of just changing direction. To recify this we reduce the cost by 1. BUT this under-inflates the cost
        #  of being in one place but not needing to change direction! Anyhow, if a human was in the right place it's v
        #  unlikely they'd randomly step away anyway?!
        if plan_goals[len(plan_goals)-1] != None and plan_goals[len(plan_goals)-1][0] == player_pos_and_or[0]:
            plan_costs[len(plan_costs)-1] -= 1

        # Next: convert costs into probability distributions
        plan_probs = self.boltz_rationality(plan_costs)

        # Random choice from prob dist
        chosen_index = random.choices(range(len(plan_goals)), plan_probs)[0]
        chosen_goal = plan_goals[chosen_index]

        # Now we have a chosen goal, we need to choose the chosen NEXT POS_OR from the valid ones:
        chosen_next_pos_and_or = valid_next_pos_and_ors[chosen_index]

        if chosen_goal == None:
            chosen_action = None

        elif chosen_index == (len(valid_next_pos_and_ors)-1) and chosen_next_pos_and_or[0] == chosen_goal[0]:
            # In this case we choose the action that leaves us in the same location. BUT we might want to change direction,
            # so we need to find whether the action to reach the chosen goal just requires a change of direction. If so,
            # then take this action.
            action_plan, _, plan_cost_test = self.mlp.mp.get_plan(player_pos_and_or, chosen_goal)
            if plan_cost_test > 2:  # At most the plan should be to change direction and interact
                raise ValueError('Incorrect action plan chosen')
            chosen_action = action_plan[0]

        else:
            # Find which action gets to the chosen next pos and or
            # Note: We can't use get_plan because goals must face an object, whereas some of the valid_pos_and_or don't
            action_plan, _, plan_cost_test = self.mlp.mp.action_plan_from_positions(chosen_next_pos_and_or,
                                                                                    player_pos_and_or,
                                                                                    chosen_next_pos_and_or)
            if plan_cost_test != 2:  # Plan should be 1 step then interact or just interact (if other is in the way)
                if action_plan != ['interact']:
                    raise ValueError('Incorrect action plan chosen')
            chosen_action = action_plan[0]
            if chosen_action == 'interact':
                # If the chosen next step is to not move, then the action will be 'interact', so there's a risk of randomly
                # dropping objects. So instead, just don't move.
                chosen_action = (0, 0)

        return chosen_action, chosen_goal

    def find_plan_including_other(self, state, motion_goals):


        start_pos_and_or = state.players_pos_and_or[self.agent_index]
        start_pos_and_or_other = state.players_pos_and_or[1-self.agent_index]
        # # Assume other player is going to closest feature (otherwise it's not a valid motion goal!!)
        # closet_feature_for_other = self.mlp.ml_action_manager.go_to_closest_feature_actions(state.players[1-self.agent_index])[0]
        # Assume other player is GreedyHumanModel and find what action they would do:
        others_predicted_goals = self.GHM.ml_action(state)
        # Find their closest goal
        min_cost = np.Inf
        others_predicted_goal = None
        for goal in others_predicted_goals:
            _, _, plan_cost = self.mlp.mp.get_plan(start_pos_and_or_other, goal)
            if plan_cost < min_cost:
                min_cost = plan_cost
                others_predicted_goal = goal

        # Now find their own best goal/action, assuming the other agent is Greedy:
        min_cost = np.Inf
        best_action = None
        for goal in motion_goals:

            if self.agent_index == 0:
                start_jm_state = (start_pos_and_or, start_pos_and_or_other)
                goal_jm_state = (goal, others_predicted_goal)
            elif self.agent_index == 1:
                start_jm_state = (start_pos_and_or_other, start_pos_and_or)
                goal_jm_state = (others_predicted_goal, goal)
            else:
                raise ValueError('Index error')

            joint_action_plan, end_jm_state, plan_lengths = self.mlp.jmp.get_low_level_action_plan(start_jm_state, goal_jm_state)
            if plan_lengths[self.agent_index] < min_cost:
                best_action = joint_action_plan[0][self.agent_index]
                min_cost = plan_lengths[self.agent_index]
                best_goal = goal

        if best_action == None:
            best_goal = None
            joint_action_plan = None

        return best_action, best_goal, joint_action_plan


# class GreedyHeuristicAgent(Agent):
#     """
#     (DEPRECATED)

#     Agent that takes actions that greedily maximize the
#     heuristic function for the MediumLevelPlanner mlp
#     """

#     def __init__(self, mlp, player_index, heuristic=None):
#         self.mlp = mlp
#         self.player_index = player_index
#         self.heuristic = heuristic if heuristic is not None else Heuristic(mlp.mp).hard_heuristic

#     def action(self, state):
#         best_h_plus_cost = np.Inf
#         best_action = None
#         for motion_goal, successor_state, delta_cost in self.mlp.get_successor_states(state):
#             curr_h = self.heuristic(successor_state)
#             curr_h_plus_cost = curr_h + delta_cost
#             if curr_h_plus_cost <= best_h_plus_cost:
#                 best_action = motion_goal
#                 best_h_plus_cost = curr_h_plus_cost

#         start_jm_state = state.players_pos_and_or
#         joint_action_plan, _, _ = self.mlp.jmp.get_low_level_action_plan(start_jm_state, best_action)
#         return joint_action_plan[0][self.player_index] if len(joint_action_plan) > 0 else None

# class TheoryOfMindHumanModel(Agent):
#     """
#     A human model that is parametrized by a linear combination of features,
#     and chooses actions thanks to a matrix of weights and performing a softmax
#     on logits.
#     """

#     def __init__(self, mdp):
#         self.mdp = mdp

#         dummy_state = None
#         dummy_feats = self.extract_features(dummy_state)
#         self.num_actions = len(Action.ALL_ACTIONS)
#         self.weights = np.zeros((self.num_actions, len(dummy_feats)))

#     def action(self, state):
#         feature_vector = self.extract_features(state)
#         logits = self.weights @ feature_vector
#         probs = scipy.special.softmax(logits)
#         return np.random.choice(self.num_actions, p=probs)

#     def extract_features(self, state):
#         counter_objects = self.mdp.get_counter_objects_dict(state)
#         pot_states_dict = self.mdp.get_pot_states(state)

#         return []
