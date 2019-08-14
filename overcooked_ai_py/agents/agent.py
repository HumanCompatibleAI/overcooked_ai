import itertools
import numpy as np

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.planning.planners import Heuristic
from overcooked_ai_py.planning.search import SearchTree


class Agent(object):

    def action(self, state):
        return NotImplementedError()

    def set_agent_index(self, agent_index):
        self.agent_index = agent_index

    def set_mdp(self, mdp):
        self.mdp = mdp

    def reset(self):
        pass


class AgentGroup(object):
    """
    AgentGroup is a group of N agents used to sample 
    joint actions in the context of an OvercookedEnv instance.
    """

    def __init__(self, *agents, allow_duplicate_agents=False):
        self.agents = agents
        self.n = len(self.agents)
        for i, agent in enumerate(self.agents):
            agent.set_agent_index(i)

        if not all(a0 is not a1 for a0, a1 in itertools.combinations(agents, 2)):
            assert allow_duplicate_agents, "All agents should be separate instances, unless allow_duplicate_agents is set to true"

    def joint_action(self, state):
        return tuple(a.action(state) for a in self.agents)

    def set_mdp(self, mdp):
        for a in self.agents:
            a.set_mdp(mdp)

    def reset(self):
        for a in self.agents:
            a.reset()


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

        if type(self.a0) is CoupledPlanningAgent and type(self.a1) is CoupledPlanningAgent:
            print("If the two planning agents have same params, consider using CoupledPlanningPair instead to reduce computation time by a factor of 2")

    def joint_action(self, state):
        if self.a0 is self.a1:
            # When using the same instance of an agent for self-play, 
            # reset agent index at each turn to prevent overwriting it
            self.a0.set_agent_index(0)
            action_0 = self.a0.action(state)
            self.a1.set_agent_index(1)
            action_1 = self.a1.action(state)
            return (action_0, action_1)
        else:
            return super().joint_action(state)


class CoupledPlanningPair(AgentPair):
    """
    Pair of identical coupled planning agents. Enables to search for optimal
    action once rather than repeating computation to find action of second agent
    """

    def __init__(self, agent):
        super().__init__(agent, agent, allow_duplicate_agents=True)

    def joint_action(self, state):
        # Reduce computation by half if both agents are coupled planning agents
        joint_action_plan = self.a0.mlp.get_low_level_action_plan(state, self.a0.heuristic, delivery_horizon=self.a0.delivery_horizon, goal_info=True)
        return joint_action_plan[0] if len(joint_action_plan) > 0 else (None, None)


class AgentFromPolicy(Agent):
    """
    Defines an agent from a `state_policy` and `direct_policy` functions
    """
    
    def __init__(self, state_policy, direct_policy, stochastic=True, action_probs=False):
        """
        state_policy (fn): a function that takes in an OvercookedState instance and returns corresponding actions
        direct_policy (fn): a function that takes in a preprocessed OvercookedState instances and returns actions
        stochastic (Bool): Whether the agent should sample from policy or take argmax
        action_probs (Bool): Whether agent should return action probabilities or a sampled action
        """
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
        return Action.STAY

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
            return Action.STAY
        curr_action = self.plan[self.i]
        self.i += 1
        return curr_action
    
    def reset(self):
        self.i = 0


class CoupledPlanningAgent(Agent):
    """
    An agent that uses a joint planner (mlp, a MediumLevelPlanner) to find near-optimal
    plans. At each timestep the agent re-plans under the assumption that the other agent 
    is also a CoupledPlanningAgent, and then takes the first action in the plan.
    """

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
    """
    An agent that uses A* search to find an optimal action based on a model of the other agent,
    `other_agent`. This class approximates the other agent as being deterministic even though it
    might be stochastic in order to perform the search.
    """

    def __init__(self, other_agent, mlp, env, delivery_horizon=2, logging_level=0):
        """mlp is a MediumLevelPlanner"""
        self.other_agent = other_agent
        self.delivery_horizon = delivery_horizon
        self.mlp = mlp
        self.env = env
        self.h_fn = Heuristic(mlp.mp).simple_heuristic
        self.logging_level = logging_level

    def action(self, state):
        start_state = state.deepcopy()
        order_list = start_state.order_list if start_state.order_list is not None else ["any", "any"]
        start_state.order_list = order_list[:self.delivery_horizon]
        other_agent_index = 1 - self.agent_index
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

        # Check estimated cost of the plan equals 
        # the sum of the costs of each medium-level action
        assert sum([len(item[0]) for item in ml_s_a_plan[1:]]) == cost

        # In this case medium level actions are tuples of low level actions
        # We just care about the first low level action of the first med level action
        first_s_a = ml_s_a_plan[1]

        # Print what the agent is expecting to happen
        if self.logging_level >= 2:
            self.env.state = start_state
            for joint_a in first_s_a[0]:
                print(self.env)
                print(joint_a)
                self.env.step(joint_a)
            print(self.env)
            print("======The End======")

        self.env.state = initial_env_state

        first_joint_action = first_s_a[0][0]
        if self.logging_level >= 1: 
            print("expected joint action", first_joint_action)
        action = first_joint_action[self.agent_index]
        return action


class GreedyHumanModel(Agent):
    """
    Agent that at each step selects a medium level action corresponding
    to the most intuitively high-priority thing to do
    
    NOTE: MIGHT NOT WORK IN ALL ENVIRONMENTS, for example random1.layout,
    in which an individual agent cannot complete the task on their own.
    """

    def __init__(self, mlp, hl_boltzmann_rational=False, ll_boltzmann_rational=False, hl_temp=1, ll_temp=1):
        self.mlp = mlp
        self.mdp = self.mlp.mdp

        # Bool for perfect rationality vs Boltzmann rationality for high level and low level action selection
        self.hl_boltzmann_rational = hl_boltzmann_rational # For choices among high level goals of same type
        self.ll_boltzmann_rational = ll_boltzmann_rational # For choices about low level motion

        # Coefficient for Boltzmann rationality for high level action selection
        self.hl_temperature = hl_temp
        self.ll_temperature = ll_temp

        self.reset()

    def reset(self):
        self.prev_state = None
        self.optimal_action_hist = []

    def actions(self, states, agent_indices):
        actions = []
        for state, agent_idx in zip(states, agent_indices):
            self.set_agent_index(agent_idx)
            self.reset()
            actions.append(self.action(state))
        return actions

    def action(self, state):
        possible_motion_goals = self.ml_action(state)

        # Once we have identified the motion goals for the medium
        # level action we want to perform, select the one with lowest cost
        start_pos_and_or = state.players_pos_and_or[self.agent_index]

        chosen_goal, chosen_goal_action = self.choose_motion_goal(start_pos_and_or, possible_motion_goals)

        if not self.ll_boltzmann_rational or chosen_goal[0] == start_pos_and_or[0]:
            chosen_action = chosen_goal_action
        else:
            chosen_action = self.boltzmann_rational_ll_action(start_pos_and_or, chosen_goal)
        
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

    def choose_motion_goal(self, start_pos_and_or, motion_goals):
        """Returns chosen motion goal (either boltzmann rationally or rationally), and corresponding action"""
        if self.hl_boltzmann_rational:
            possible_plans = [self.mlp.mp.get_plan(start_pos_and_or, goal) for goal in motion_goals]
            plan_costs = [plan[2] for plan in possible_plans]
            goal_idx = self.get_boltzmann_rational_action_idx(plan_costs, self.hl_temperature)
            chosen_goal = motion_goals[goal_idx]
            chosen_goal_action = possible_plans[goal_idx][0][0]
        else:
            chosen_goal, chosen_goal_action = self.get_lowest_cost_action_and_goal(start_pos_and_or, motion_goals)
        
        return chosen_goal, chosen_goal_action

    def get_boltzmann_rational_action_idx(self, costs, temperature):
        """Chooses index based on softmax probabilities obtained from cost array"""
        costs = np.array(costs)
        softmax_probs = np.exp(-costs * temperature) / np.sum(np.exp(-costs * temperature))
        action_idx = np.random.choice(len(costs), p=softmax_probs)
        optimal_action = action_idx == np.argmin(costs)
        self.optimal_action_hist.append(optimal_action)
        return action_idx

    def get_lowest_cost_action_and_goal(self, start_pos_and_or, motion_goals):
        """Returns action and goal that correspond to the cheapest plan among possible motion goals"""
        min_cost = np.Inf
        best_action, best_goal = None, None
        for goal in motion_goals:
            action_plan, _, plan_cost = self.mlp.mp.get_plan(start_pos_and_or, goal)
            if plan_cost < min_cost:
                best_action = action_plan[0]
                min_cost = plan_cost
                best_goal = goal
        return best_goal, best_action

    def boltzmann_rational_ll_action(self, start_pos_and_or, goal):
        future_costs = []
        for action in Action.ALL_ACTIONS:
            pos, orient = start_pos_and_or
            new_pos_and_or = self.mdp._move_if_direction(pos, orient, action)
            _, _, plan_cost = self.mlp.mp.get_plan(new_pos_and_or, goal)
            future_costs.append(plan_cost)

        action_idx = self.get_boltzmann_rational_action_idx(future_costs, self.ll_temperature)
        return Action.ALL_ACTIONS[action_idx]

    def ml_action(self, state):
        """Selects a medium level action for the current state"""
        player = state.players[self.agent_index]
        other_player = state.players[1 - self.agent_index]
        am = self.mlp.ml_action_manager
        
        counter_objects = self.mlp.mdp.get_counter_objects_dict(state, list(self.mlp.mdp.terrain_pos_dict['X']))
        pot_states_dict = self.mlp.mdp.get_pot_states(state)

        # NOTE: this most likely will fail in some tomato scenarios
        curr_order = state.curr_order

        if not player.has_object():

            if curr_order == 'any':
                ready_soups = pot_states_dict['onion']['ready'] + pot_states_dict['tomato']['ready']
                cooking_soups = pot_states_dict['onion']['cooking'] + pot_states_dict['tomato']['cooking']
            else:
                ready_soups = pot_states_dict[curr_order]['ready']
                cooking_soups = pot_states_dict[curr_order]['cooking']
            
            soup_nearly_ready = len(ready_soups) > 0 or len(cooking_soups) > 0
            other_has_dish = other_player.has_object() and other_player.get_object().name == 'dish'
            
            if soup_nearly_ready and not other_has_dish:
                motion_goals = am.pickup_dish_actions(state, counter_objects)
            else:
                next_order = None
                if state.num_orders_remaining > 1:
                    next_order = state.next_order
                
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
        
        motion_goals = [mg for mg in motion_goals if self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, mg)]

        if len(motion_goals) == 0:
            motion_goals = am.go_to_closest_feature_actions(player)
            motion_goals = [mg for mg in motion_goals if self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, mg)]
            assert len(motion_goals) != 0

        return motion_goals

