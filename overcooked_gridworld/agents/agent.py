import scipy
import itertools
import numpy as np
from collections import defaultdict

from overcooked_gridworld.mdp.overcooked_mdp import Action, Direction, OvercookedState
from overcooked_gridworld.planning.planners import MediumLevelPlanner, Heuristic


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
        except AttributeError:
            raise AttributeError("Need to set the agent_index or mdp of the Agent before using it")

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
        return Direction.STAY

    def direct_action(self, obs):
        return [Action.ACTION_TO_INDEX[Direction.STAY]] * self.sim_threads


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
            return Direction.CARDINAL[np.random.randint(4)]
        return joint_action_plan[0][self.agent_index] if len(joint_action_plan) > 0 else None


class EmbeddedPlanningAgent(Agent):

    def __init__(self, other_agent, mlp, delivery_horizon=2):
        """other_agent_policy returns highest prob action"""
        self.other_agent = other_agent
        self.delivery_horizon = delivery_horizon
        self.mlp = mlp
        self.h_fn = Heuristic(mlp.mp).simple_heuristic

    def action(self, state):
        from overcooked_gridworld.planning.search import SearchTree
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

class GreedyHumanModel(Agent):
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
                joint_actions = list(itertools.product(Action.ALL_ACTIONS, [Direction.STAY]))
            elif self.agent_index == 1:
                joint_actions = list(itertools.product([Direction.STAY], Action.ALL_ACTIONS))
            else:
                raise ValueError("Player index not recognized")

            unblocking_joint_actions = []
            for j_a in joint_actions:
                new_state, reward = self.mlp.mdp.get_transition_states_and_probs(state, j_a)[0][0]
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
