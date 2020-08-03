
from overcooked_ai_py.agents.agent import Agent, AgentPair
import itertools, math, copy
import numpy as np
from collections import defaultdict
import random
import logging

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelPlanner, Heuristic
from overcooked_ai_py.planning.search import SearchTree

"""This file contains the agents used for the project human_ai_robustness"""


class GreedyHumanModel_pk(Agent):
    """
    This is Paul's GreedyHumanModel, which is slightly different Micah's (which is overcooked_ai_py.agents.agent.GreedyHumanModel)

    Agent that at each step selects a medium level action corresponding to the most intuitively high-priority thing to do

    Enhancements added to v2:
    1) Drop onion if a dish is needed instead
    2) Drop dish if an onion is needed instead
    3) If soup on the counter then pick it up
    3) Added parameter "perseverance", then added mechanism for resolving crashes: each agent steps out of the way with
    a certain probability, which depends on their perseverance and how many timesteps they've been stuck.

    """

    def __init__(self, mlp, perseverance=0.5):
        self.mlp = mlp
        self.mdp = self.mlp.mdp
        self.prev_state = None
        self.timesteps_stuck = 0  # Count how many times there's a clash with the other player
        self.perseverance = perseverance

    def reset(self):
        self.timesteps_stuck = 0
        self.prev_state = None

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
        return best_action, {}

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
                motion_goals = am.pickup_dish_actions(counter_objects)
            else:
                next_order = None
                #TODO: This seems to look at the next-but-one order? Should it be order_list[0]? Check this, and modify if needed
                if len(state.order_list) > 1:
                    next_order = state.order_list[1]

                if next_order == 'onion':
                    motion_goals = am.pickup_onion_actions(counter_objects)
                elif next_order == 'tomato':
                    motion_goals = am.pickup_tomato_actions(counter_objects)
                elif next_order is None or next_order == 'any':
                    motion_goals = am.pickup_onion_actions(counter_objects) + am.pickup_tomato_actions(counter_objects)

            # If there's a soup on the counter, then override other goals and get the soup
            #TODO: This can cause issues in unident <-- fix it
            if 'soup' in counter_objects:
                motion_goals = am.pickup_counter_soup_actions(counter_objects)

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

# Depreciated:
# def make_tom_model(env, mlp, hm_number, agent_index):
#
#     perseverance = env.tom_params[hm_number]["PERSEVERANCE_HM{}".format(hm_number)]
#     teamwork = env.tom_params[hm_number]["TEAMWORK_HM{}".format(hm_number)]
#     retain_goals = env.tom_params[hm_number]["RETAIN_GOALS_HM{}".format(hm_number)]
#     wrong_decisions = env.tom_params[hm_number]["WRONG_DECISIONS_HM{}".format(hm_number)]
#     thinking_prob = env.tom_params[hm_number]["THINKING_PROB_HM{}".format(hm_number)]
#     path_teamwork = env.tom_params[hm_number]["PATH_TEAMWORK_HM{}".format(hm_number)]
#     rationality_coeff = env.tom_params[hm_number]["RATIONALITY_COEFF_HM{}".format(hm_number)]
#     prob_pausing = env.tom_params[hm_number]["PROB_PAUSING_HM{}".format(hm_number)]
#
#     return ToMModel(mlp=mlp, player_index=agent_index, perseverance=perseverance,
#                     teamwork=teamwork, retain_goals=retain_goals,
#                     wrong_decisions=wrong_decisions, thinking_prob=thinking_prob,
#                     path_teamwork=path_teamwork, rationality_coefficient=rationality_coeff,
#                     prob_pausing=prob_pausing)

# TODO: Eventually remove the alternate names altogether
INIT_PARAMS_TO_ALTERNATE_PARAMS_NAMES = {
    "compliance": "COMPLIANCE_TOM",
    "retain_goals": "RETAIN_GOALS_TOM",
    "prob_thinking_not_moving": "PROB_THINKING_NOT_MOVING_TOM",
    "path_teamwork": "PATH_TEAMWORK_TOM",
    "rationality_coefficient": "RAT_COEFF_TOM",
    "prob_pausing": "PROB_PAUSING_TOM",
    "prob_greedy": "PROB_GREEDY_TOM",
    "prob_obs_other": "PROB_OBS_OTHER_TOM",
    "look_ahead_steps": "LOOK_AHEAD_STEPS_TOM"
}
ALTERNATE_PARAM_TO_INIT_PARAMS_NAMES = { v:k  for k, v in INIT_PARAMS_TO_ALTERNATE_PARAMS_NAMES.items() }


class ToMModel(Agent):
    """
    #TODO: need to update this description:
    Built on greedy human model
    - added a simple heuristic that's used to factor in other player's expected moves
    - motion goals can be retained rather than re-finding goals every timestep
    - make the wrong decision about a motion goal with prob: wrong_decision
    - thinking_time
    - with prob path_teamwork, factor in the other player (assuming Greedy) when finding the best path
    Adding:
    - Take a sub-optimal first action with Boltzmann rational probability
    """

    def __init__(self, mlp, prob_random_action=0,
                 compliance=0.5, teamwork=0.8, retain_goals=0.8, wrong_decisions=0.02, prob_thinking_not_moving=0.2,
                 path_teamwork=0.8, rationality_coefficient=3, prob_pausing=0.5, use_OLD_ml_action=False,
                 prob_greedy=0, prob_obs_other=0, look_ahead_steps=4):
        self.mlp = mlp
        self.mdp = self.mlp.mdp
        self.GHM = GreedyHumanModel_pk(self.mlp)  # For ToM of other players
        self.reset()
        self.human_model = True

        ## "Personality" parameters ##
        # Motion-level parmas:
        self.compliance = compliance  # compliance = 0 means it always tries to go where it wants, even when stuck
        #TODO: Remove perseverance once I stop using the old fitted params:
        self.perseverance = 1 - self.compliance
        self.path_teamwork = path_teamwork  # Prob of considering the other agent's location in the path choice
        self.rationality_coefficient = rationality_coefficient  # Setting to 0 means random actions; inf means always takes
        # lowest cost path. In practice choose 20 for fully rational (if it's too large, e.g. 100, we get rounding errors)
        self.prob_pausing = prob_pausing # Probability of pausing on a given timestep, instead of acting. From a
        # quick initial look at the human data, the humans pause very approx 50% of the time
        self.prob_thinking_not_moving = prob_thinking_not_moving  # After achieving a goal (e.g. fetching onion) the agent
        # waits to "think". self.prob_thinking_not_moving is the probability of thinking instead of moving on
        # during this "thinking time". PARAM HAS CHANGED: prob_thinking_not_moving = 1 - thinking_prob
        self.prob_random_action = prob_random_action  # Randomly return one of the 6 actions, with this prob

        # Higher level strategy:  (Note: prob_greedy and prob_obs_other will be converted into personality types A-D)
        #TODO: retain_goals should be renamed to "inertia"?
        self.retain_goals = retain_goals  # Prob of keeping the previous goal each timestep (rather than re-calculating)
        self.prob_greedy = prob_greedy  # Prob of having greedy personality type (as opposed to finding team strategies)
        self.prob_obs_other = prob_obs_other  # Prob of factoring in the other's likely goals during decision making
        self.look_ahead_steps = look_ahead_steps  # How many steps to look ahead when planning. Choose 2 or 4 (1
        # makes A=C, 3~2, and >4 pointless? Should be >1 (required by function remove_others_current_task_from_list)

        # Unused, or related to the old TOM:
        self.focus_most_full_pot = True  # Not actually using this yet...
        self.wrong_decisions = wrong_decisions  # Prob of making a wrong decision about the motion_goal. I.e. they should
        # get onion but they get dish instead. Note: this should only work well if retain_goals is large
        self.teamwork = teamwork  # teamwork = 0 should make this agent similar to GreedyHuman
        self.use_OLD_ml_action = use_OLD_ml_action

    @staticmethod
    def get_stationary_ToM(mlp):
        """Make a TOM agent that doesn't move: (prob_pausing == 1, prob_random_action=0 (all other params are irrelevant))"""
        return ToMModel(mlp=mlp, prob_random_action=0, prob_pausing=1)

    @staticmethod
    def from_alternate_names_params_dict(mlp, alternate_names_params):
        """Create a ToM model from a params dict that follows the key-value pairs of the `set_tom_params` method"""
        params = { init_param : alternate_names_params[alternate_param_name] for init_param, alternate_param_name in INIT_PARAMS_TO_ALTERNATE_PARAMS_NAMES.items() }
        return ToMModel(mlp, **params)

    def set_agent_index(self, agent_index):
        super().set_agent_index(agent_index)
        self.GHM.set_agent_index(1 - agent_index)

    def reset(self):
        # Reset agent -- wipe it's history
        self.prev_state = None
        self.timesteps_stuck = 0
        self.dont_drop = False
        self.prev_motion_goal = None
        self.prev_best_action = None
        self.only_take_dispenser_onions = False
        self.only_take_dispenser_dishes = False
        self.GHM.reset()

    def set_tom_params(self, num_toms, other_agent_idx, tom_params, tom_params_choice=None):
        """
        If tom_params_choice is not None, set the params of the TOM using the element tom_params_choice of tom_params
        If tom_params_choice == None: Randomly choose which TOM params to use for this agent. Choose integer up to num_toms, which corresponds to
        params in tom_params.
        """
        if tom_params_choice == None:
            tom_params_choice = np.random.randint(0, num_toms)

        self.compliance = tom_params[tom_params_choice]["COMPLIANCE_TOM"]
        self.retain_goals = tom_params[tom_params_choice]["RETAIN_GOALS_TOM"]
        self.prob_thinking_not_moving = tom_params[tom_params_choice]["PROB_THINKING_NOT_MOVING_TOM"]
        self.path_teamwork = tom_params[tom_params_choice]["PATH_TEAMWORK_TOM"]
        self.rationality_coefficient = tom_params[tom_params_choice]["RAT_COEFF_TOM"]
        self.prob_pausing = tom_params[tom_params_choice]["PROB_PAUSING_TOM"]
        self.prob_greedy = tom_params[tom_params_choice]["PROB_GREEDY_TOM"]
        self.prob_obs_other = tom_params[tom_params_choice]["PROB_OBS_OTHER_TOM"]
        self.look_ahead_steps = round(tom_params[tom_params_choice]["LOOK_AHEAD_STEPS_TOM"])

        # Reset the "history" of the agent, and set the index:
        self.reset()
        if not other_agent_idx is None:
            self.set_agent_index(other_agent_idx)
        return tom_params_choice

    #TODO: Perhaps add a function to output a prob dist over actions (i.e. move the code from fit_TOMs_to_data --
    # although we can't directly use that code because that has a pop of parallel agents, to respect the different
    # distories that might have arisen!)
    def action(self, state):

        self.display_game_during_training(state)

        # With a given prob the agent will either act or pause for one timestep:
        if random.random() > self.prob_pausing:
            logging.info('Agent not pausing; Player index: {}'.format(self.agent_index))
            self.fix_invalid_prev_motion_goal(state)

            # Get new motion_goals if i) There is no previous goal (i.e. self.prev_best_action == None); OR ii) with
            # prob = 1 - self.retain_goals (i.e. there's a certain probability that the agent just keeps its old goal);
            # OR iii) if the agent is stuck for >1 timestep; OR iv) if reached a goal (then a new one is needed)
            rand = random.random()
            if self.prev_best_action == None or rand > self.retain_goals or \
                    self.prev_best_action == 'interact' or self.prev_best_action == (0,0):

                logging.info('Getting a new motion goal...')

                if (self.prev_best_action == 'interact' or self.prev_best_action == (0,0)) \
                        and (random.random() < self.prob_thinking_not_moving):

                    logging.info('Agent is pausing to "think"...')
                    best_action = (0,0)

                else:

                    logging.info('Getting new goal')

                    #TODO: Remove this once the old ml_action is no longer used:
                    if not self.use_OLD_ml_action:
                        motion_goals = self.ml_action(state)
                    else:
                        motion_goals = self.OLD_ml_action(state)

                    best_action = self.choose_best_action(state, motion_goals)

            else:

                logging.info('Keeping previous goal (instead of choosing a new goal)')
                # Use previous goal:
                motion_goals = self.prev_motion_goal
                best_action = self.choose_best_action(state, motion_goals)

            # If stuck, take avoiding action:
            best_action = self.take_alternative_action_if_stuck(best_action, state)

        # The agent sometimes just pauses instead of acting:
        else:
            logging.info('Agent pausing')
            best_action = (0,0)

        best_action = self.choose_random_action(self.prob_random_action, best_action)

        return best_action, {}


    def ml_action(self, state):
        """Choose a higher-level task (e.g. fetch an onion), which gives a set of motion_goals to achieve that task."""

        #TODO: The helper functions below at times have some shared-functionality --> it may be possible to combine some
        # of them together

        # For noting if the task you're doing is/is not the top priority task:
        self.doing_lower_priority_task = None

        # TODO: modify and merge get_info_for_making_decisions and get_extra_info_for_making_decisions so it returns
        #  'info', SEE DIRECTLY BELOW! AND add get_extra_info_for_making_decisions to get_info_for_making_decisions?
        player, other_player, am, counter_objects, pot_states_dict, soup_nearly_ready, count_soups_nearly_ready, \
        other_has_dish, other_has_onion, number_of_pots, temp_dont_drop = self.get_info_for_making_decisions(state)
        info = {'player': player, 'other_player': other_player, 'am': am, 'counter_objects': counter_objects,
                'pot_states_dict': pot_states_dict, 'other_has_dish': other_has_dish,
                'other_has_onion': other_has_onion, 'number_of_pots': number_of_pots, 'temp_dont_drop': temp_dont_drop}
        if info['player'].has_object():
            soups_need_onions, player_obj = self.get_extra_info_for_making_decisions(pot_states_dict, player)
            info.update({'soups_need_onions': soups_need_onions, 'player_obj': player_obj})

        # Make the list of tasks by priority:
        task_priority_list = self.make_priority_list_of_tasks(self.look_ahead_steps, info)

        # Special case for all agents: if holding a soup then deliver it. Ignore costs and tasks etc, just deliver it!
        if info['player'].has_object() and info['player_obj'].name == 'soup':
            motion_goals = info['am'].deliver_soup_actions()

        else:
            # Construct probabilities of being one of the four personality type, A-D:
            self.personality_type = [self.prob_greedy*(1-self.prob_obs_other),
                                     self.prob_greedy*self.prob_obs_other,
                                     (1-self.prob_greedy)*(1-self.prob_obs_other),
                                     (1-self.prob_greedy)*self.prob_obs_other]
            personality_type_to_use = np.random.choice(['A', 'B', 'C', 'D'], 1, p=self.personality_type)
            #TODO: Make this into a helper function, or just find a more elegant way to do this!:
            if personality_type_to_use == 'A':
                motion_goals = self.choose_goals_type_A(state, task_priority_list, info)
            elif personality_type_to_use == 'B':
                motion_goals = self.choose_goals_type_B(state, task_priority_list, info)
            elif personality_type_to_use == 'C':
                motion_goals = self.choose_goals_type_C(state, task_priority_list, info, self.look_ahead_steps)
            elif personality_type_to_use == 'D':
                motion_goals = self.choose_goals_type_D(state, task_priority_list, info, self.look_ahead_steps)
            else:
                raise ValueError('Unavailable personality type selected')

        motion_goals = self.remove_invalid_goals_and_clean_up(player, motion_goals, am, temp_dont_drop)

        return motion_goals

#============== Sub-methods for the new ToM model, with personality types A-D ======================#

    def make_priority_list_of_tasks(self, look_ahead_steps, info):
        """Make a list of the tasks to be done next.
        Returns: task_priority_list is a list of lists of dictionaries: element 0 is a list of the top priority
        tasks. Each element of this list is a dictionary, {'task_name': location_of_task}.
        """

        # "Simulation" of the relevant info needed to work out the next priority task
        sim_pot_states_dict = copy.deepcopy(info['pot_states_dict'])
        sim_counter_objects = copy.deepcopy(info['counter_objects'])

        task_priority_list = []

        for priority in range(look_ahead_steps):

            # Return list of the next priority tasks. And simulate completing these tasks, which leads to a new sim_info
            tasks_this_priority, sim_pot_states_dict, sim_counter_objects =\
                    self.calculate_next_priority_tasks(sim_pot_states_dict, sim_counter_objects)
            task_priority_list.append(tasks_this_priority)

        logging.info('Task list: ', task_priority_list)
        return task_priority_list

    def calculate_next_priority_tasks(self, sim_pot_states_dict, sim_counter_objects):
        """
        Work out what task(s) is the highest priority, given the state as represented by sim_info. Then modify
        sim_info to simulate the task(s) being completed.

        Returns: tasks_this_priority: a list of tasks that have equal next priority to be done next. Each "task" in
        the list is a dictionary, {'name of task': location}
        """
        tasks_this_priority = []

        ready_soups = sim_pot_states_dict['onion']['ready']
        cooking_soups = sim_pot_states_dict['onion']['cooking']
        count_soups_nearly_ready = len(ready_soups) + len(cooking_soups)
        pots_empty = sim_pot_states_dict['empty']
        pots_partially_full = sim_pot_states_dict['onion']['partially_full']
        soups_need_onions = len(pots_empty) + len(pots_partially_full)

        #TODO: Sort out and put back this next bit (and changes therein) on imminently_ready_soups etc

        # ready_soups = []
        # early_cooking_soups = []
        # imminently_ready_soups = []
        # # For each soup state
        # for i, location in enumerate(sim_state.all_objects_by_type['soup']):
        #     _, _, cook_time = sim_state.all_objects_by_type['soup'][i].state
        #     if cook_time > self.imminently_ready_threshold:
        #         imminently_ready_soups.append(location)
        #     elif cook_time > 0:
        #         early_cooking_soups.append(location)
        #     elif cook_time == 0:
        #         ready_soups.append(location)
        #     else:
        #         raise ValueError('Error')
        # assert len(cooking_soups) == len(imminently_ready_soups) + len(early_cooking_soups)
        # assert ready_soups is sim_pot_states_dict['onion']['ready']
        # assert len(sim_state.all_objects_by_type['soup']) == len(imminently_ready_soups) + len(early_cooking_soups) \
        #        + len(ready_soups)

        # If soup on counter:
        if 'soup' in sim_counter_objects:

            # For each soup on counter add to the list of tasks for this priority
            for location in sim_counter_objects['soup']:

                tasks_this_priority.append({'soup_from_counter': location})

                # Remove this soup from the counter objects, and remove the soup dict completely if it's empty:
                sim_counter_objects['soup'].remove(location)
                if len(sim_counter_objects['soup']) == 0:
                    sim_counter_objects.pop('soup')

        # elif len(imminently_ready_soups) > 0:
        #
        #     for i, location in enumerate(imminently_ready_soups):
        #         tasks_this_priority.append({'deliver_soup': location})
        #
        #         # Ammend simulation to factor this in:
        #         for i, soup_state in enumerate(sim_state.all_objects_by_type['soup']):
        #             if soup_state.position == location:
        #                 sim_state.all_objects_by_type['soup'][i] = cook_time
        #

        elif count_soups_nearly_ready > 0:
            locations = []  # This method of creating 'locations' then using this to remove/append the pot states dict
            # is needed because otherwise ready_soups updates as we append/remove the pot states dict, which means the
            # 'for location' loop doesn't complete as it should do
            for location in ready_soups:
                tasks_this_priority.append({'deliver_soup': location})
                locations.append(location)
            [sim_pot_states_dict['onion']['ready'].remove(location) for location in locations]
            [sim_pot_states_dict['empty'].append(location) for location in locations]

            locations = []
            for location in cooking_soups:
                tasks_this_priority.append({'deliver_soup': location})
                locations.append(location)
            [sim_pot_states_dict['onion']['cooking'].remove(location) for location in locations]
            [sim_pot_states_dict['empty'].append(location) for location in locations]

        elif soups_need_onions > 0:

            if self.focus_most_full_pot:

                #TODO: Factor into helper functions?:

                # Focus on the pot with more onions in. So we look for pots with 2 onions in first, then 2, then 1:

                if sim_pot_states_dict['onion']['2_items']:
                    locations = []
                    for location in sim_pot_states_dict['onion']['2_items']:
                        tasks_this_priority.append({'deliver_onion': location})
                        locations.append(location)
                    [sim_pot_states_dict['onion']['2_items'].remove(location) for location in locations]
                    # [sim_pot_states_dict['onion']['3_items'].append(location) for location in locations]
                    [sim_pot_states_dict['onion']['partially_full'].remove(location) for location in locations]
                    [sim_pot_states_dict['onion']['cooking'].append(location) for location in locations]


                elif sim_pot_states_dict['onion']['1_items']:
                    locations = []
                    for location in sim_pot_states_dict['onion']['1_items']:
                        tasks_this_priority.append({'deliver_onion': location})
                        locations.append(location)
                    [sim_pot_states_dict['onion']['1_items'].remove(location) for location in locations]
                    [sim_pot_states_dict['onion']['2_items'].append(location) for location in locations]

                elif sim_pot_states_dict['empty']:
                    locations = []
                    for location in sim_pot_states_dict['empty']:
                        tasks_this_priority.append({'deliver_onion': location})
                        locations.append(location)
                    [sim_pot_states_dict['empty'].remove(location) for location in locations]
                    [sim_pot_states_dict['onion']['partially_full'].append(location) for location in locations]
                    [sim_pot_states_dict['onion']['1_items'].append(location) for location in locations]
                else:
                    raise ValueError('Error')
            else:
                raise ValueError('NOT FINISHED THIS PART YET!!')

        else:
            raise ValueError('Error')

        return tasks_this_priority, sim_pot_states_dict, sim_counter_objects

    def choose_goals_type_A(self, state, task_priority_list, info):
        """This agent does the first task on the list, regardless of what the other agent is doing"""
        tasks_to_do = task_priority_list[0]  # This could be a list of several tasks
        task_to_do = self.find_lowest_cost_task(tasks_to_do, info, state)
        logging.info('Chosen task to do: {}'.format(task_to_do))
        motion_goals = self.find_motion_goals_for_task(state, info, task_to_do)
        logging.info('Chosen motion_goals (agent type A): {}'.format(motion_goals))

        return motion_goals

    def choose_goals_type_B(self, state, task_priority_list, info):
        """This agent identifies the other agent’s current task and plans in isolation after that"""

        # If other player has soup, then just assume they are delivering it, and leave the task list unchanged
        if not (info['other_player'].has_object() and info['other_player'].get_object().name == 'soup'):
            task_priority_list, _ = self.remove_others_current_task_from_list(task_priority_list, info, state)
            logging.info('Revised task priority list: {}'.format(task_priority_list))

        # Now we have a revised task list, do the greedy action:
        tasks_to_do = task_priority_list[0]  # This could be a list of several tasks
        task_to_do = self.find_lowest_cost_task(tasks_to_do, info, state)
        logging.info('Chosen task to do: {}'.format(task_to_do))
        motion_goals = self.find_motion_goals_for_task(state, info, task_to_do)
        logging.info('Chosen motion_goals (agent type B): {}'.format(motion_goals))

        return motion_goals

    def choose_goals_type_C(self, state, task_priority_list, info, look_ahead_steps):
        """This agent works out the lowest-cost team strategy, then executes their first move for this strategy"""

        others_cost = 0
        if info['other_player'].has_object():
            others_sim_held_object = info['other_player'].get_object()
        else:
            others_sim_held_object = None
        others_sim_pos_and_or = info['other_player'].pos_and_or
        sim_counter_objects = info['counter_objects']

        # Special case if they're holding a soup: they should deliver it
        if info['other_player'].has_object() and info['other_player'].get_object().name == 'soup':
            others_cost, others_sim_pos_and_or = self.cost_serve_soup(info["am"], others_sim_pos_and_or)
            others_sim_held_object = None

        # Execute the lowest cost team strategy:
        motion_goals = self.find_motion_goal_for_best_team_strategy(task_priority_list, info, state, look_ahead_steps,
                                                                others_cost, others_sim_pos_and_or,
                                                                others_sim_held_object, sim_counter_objects)

        logging.info('Chosen motion_goals (agent type C): {}'.format(motion_goals))
        return motion_goals

    def choose_goals_type_D(self, state, task_priority_list, info, look_ahead_steps):
        """This agent first works out what the other player is doing and crosses this task off the list (as agent B
        does). It then works out the lowest-cost team strategy, then executes their first move for this strategy (as
        agent C does)."""

        task_priority_list, task_removed = self.remove_others_current_task_from_list(task_priority_list, info, state,
                                                                                     look_ahead_steps=look_ahead_steps)
        logging.info('Task removed: {}; Revised task priority list: {}'.format(task_removed, task_priority_list))

        if task_removed is not None:
            # Find min cost and final pos_or of other agent doing the task_removed:
            others_cost, others_sim_pos_and_or, sim_counter_objects, _ = self.find_min_cost_of_task(task_removed,
                                        info["am"], find_own_cost=False, first_action_info=[state, info])
            others_sim_held_object = None
        else:
            others_cost = 0
            if info['other_player'].has_object():
                others_sim_held_object = info['other_player'].get_object()
            else:
                others_sim_held_object = None
            others_sim_pos_and_or = info['other_player'].pos_and_or
            sim_counter_objects = info['counter_objects']

        # Assume other player has now done the task_removed, work out the lowest cost team strategy:
        motion_goals =  self.find_motion_goal_for_best_team_strategy(task_priority_list, info, state, look_ahead_steps,
                                    others_cost, others_sim_pos_and_or, others_sim_held_object, sim_counter_objects)

        logging.info('Chosen motion_goals (agent type D): {}'.format(motion_goals))
        return motion_goals

    def find_motion_goal_for_best_team_strategy(self, task_priority_list, info, state, look_ahead_steps,
                                                                        others_cost, others_sim_pos_and_or,
                                                                        others_sim_held_object, sim_counter_objects):
        """Given the task_priority_list and the current state (or current simulated state, if others_cost>0), work out
        the optimal team strategy. From this, deduce the best motion_goal to take."""

        am = info["am"]

        for i in range(len(task_priority_list)):

            # Who has lower cost: me for doing the i'th task; or other player for doing ALL tasks from 0 to i'th?

            # Own cost for doing i'th task. If more than one i'th task, work out cost for both:
            own_min_cost = np.Inf
            for task in task_priority_list[i]:
                cost, _, _, task_goal = self.find_min_cost_of_task(task, am, find_own_cost=True,
                                        sim_counter_objects=sim_counter_objects, first_action_info=[state, info])
                if cost < own_min_cost:
                    own_min_cost = cost
                    own_task = task

            # Others cost for doing tasks 0 to i'th:
            others_total_cost = others_cost
            for j in range(i+1):
                for task in task_priority_list[j]:
                    cost, others_sim_pos_and_or, sim_counter_objects, _ = self.find_min_cost_of_task(task,
                                            am, find_own_cost=False, sim_counter_objects=sim_counter_objects,
                                            subsequent_action_info=[others_sim_pos_and_or, others_sim_held_object, am])
                    others_sim_held_object = None  # After doing a full task, they won't be holding an object
                    others_total_cost += cost

            # If own cost is less than for the other player doing ALL tasks up to this point, then do the task:
            if (own_min_cost < np.Inf) and (own_min_cost <= others_total_cost):
                if i > 0:
                    self.doing_lower_priority_task = True
                elif i == 0:
                    self.doing_lower_priority_task = None  # Reset here just in case
                logging.info('Chosen task to do: {} (task #{})'.format(own_task, i))
                return self.find_motion_goals_for_task(state, info, own_task)

        # If the other player always has a lower cost, then motion_goals=[]:
        return []

    def find_min_cost_of_task(self, task, am, find_own_cost, sim_counter_objects=None,
                              first_action_info=False, subsequent_action_info=False):
        """Task removed might have several task_goals, so here we find the goal with the lowest cost, and return the
        goal and the cost and simulated pos/or and counter objects."""

        # Special case if they're holding soup:
        if list(task.keys())[0] == 'special_case_deliver_soup':
            others_pos_and_or = first_action_info[1]["other_player"].pos_and_or
            cost, others_motion_goal = self.cost_serve_soup(am, others_pos_and_or)
            return cost, others_motion_goal, sim_counter_objects, None

        lowest_cost = np.Inf
        task_goals = am._get_ml_actions_for_positions([list(task.values())[0]])
        for task_goal in task_goals:
            cost, sim_pos_and_or_temp, sim_counter_objects_temp = self.find_cost_of_single_task(task,
                                task_goal, find_own_cost=find_own_cost, first_action_info=first_action_info,
                                subsequent_action_info=subsequent_action_info, sim_counter_objects=sim_counter_objects)

            if cost < lowest_cost:
                lowest_cost = cost
                sim_pos_and_or = sim_pos_and_or_temp
                sim_counter_objects = sim_counter_objects_temp
                lowest_cost_task_goal = task_goal
        if lowest_cost < np.Inf:
            return lowest_cost, sim_pos_and_or, sim_counter_objects, lowest_cost_task_goal
        else:
            #TODO: This shouldn't be needed (try to fix it)... but we're setting sim_pos_and_or for inf cost tasks
            pos_and_or = self.find_pos_and_or(find_own_cost, first_action_info, subsequent_action_info)
            return lowest_cost, pos_and_or, defaultdict(list, {'onion': []}), None

    def find_motion_goals_for_task(self, state, info, task_to_do):
        """Find the motion goals for doing the task_to_do"""

        if task_to_do:
            task_name = list(task_to_do.keys())[0]
        else:
            task_name = None
        if task_name == 'soup_from_counter':
            motion_goals = self.find_subgoal_for_counter_soup(state, info,
                                                              soup_location=task_to_do['soup_from_counter'])
        elif task_name == 'deliver_onion':
            motion_goals = self.find_subgoal_for_delivering_onion(state, info,
                                                                  pot_location=task_to_do['deliver_onion'])
        elif task_name == 'deliver_soup':
            motion_goals = self.find_subgoal_for_delivering_soup(state, info,
                                                                 pot_location=task_to_do['deliver_soup'])
        elif task_name == None:
            motion_goals = []
        else:
            raise ValueError('Chosen task not valid')

        return motion_goals

    def find_lowest_cost_task(self, tasks_to_do, info, state, find_own_cost=True):
        """If there are several top-priority tasks, pick the one with the lowest cost for this player
        return: tasks_to_do
        """
        task_to_do = None
        lowest_cost = np.inf
        for task in tasks_to_do:
            # Each task might have more than one goal
            task_goals = info["am"]._get_ml_actions_for_positions([list(task.values())[0]])
            for task_goal in task_goals:
                cost, _, _ = self.find_cost_of_single_task(task, task_goal, find_own_cost=find_own_cost,
                                                           first_action_info=[state, info])
                if cost < lowest_cost:
                    lowest_cost = cost
                    task_to_do = task

        return task_to_do

    def find_subgoal_for_counter_soup(self, state, info, soup_location):
        """The task is to deliver an counter soup. Depending on the player's current state, work out the best way to do
        this"""

        if info['player'].has_object():

            if info['player_obj'].name == 'soup':

                motion_goals = info['am'].deliver_soup_actions()

            else:

                motion_goals = info['am'].place_obj_on_counter_actions(state)

        elif not info['player'].has_object():

            # The soup location directly gives the motion goal
            motion_goals = info["am"]._get_ml_actions_for_positions([soup_location])

        return motion_goals

    def find_subgoal_for_delivering_onion(self, state, info, pot_location):
        """The task is to deliver an onion. Depending on the player's current state, work out the best way to do this"""

        if info['player'].has_object():

            if info['player_obj'].name == 'onion':

                motion_goals = info["am"]._get_ml_actions_for_positions([pot_location])

                # Special case: If not doing the top priority task, then don't go all the way to the goal.
                # BUT in asymmetric_advantages we can't get in the way, so ignore this:
                if self.doing_lower_priority_task and self.mdp.layout_name != 'asymmetric_advantages':
                    # Move to the nearest counter/feature to the goal:
                    return self.move_close_to_goal(motion_goals, pot_location, info['player'].pos_and_or, info["am"])

            elif info['player_obj'].name == 'dish':

                motion_goals = info['am'].place_obj_on_counter_actions(state)

            elif info['player_obj'].name == 'soup':
                motion_goals = info['am'].deliver_soup_actions()

        elif not info['player'].has_object():

            # We haven't specified WHICH onion to pick up; just which pot to put the onion in!
            motion_goals = info['am'].pickup_onion_actions(info['counter_objects'])

        return motion_goals

    def find_subgoal_for_delivering_soup(self, state, info, pot_location):
        """The task is to deliver the soup. Depending on the player's current state, work out the best way to do this"""

        if info['player'].has_object():

            if info['player_obj'].name == 'dish':

                motion_goals = info["am"]._get_ml_actions_for_positions([pot_location])

                # Special case: If not doing the top priority task, then don't go all the way to the goal:
                # BUT in asymmetric_advantages we can't get in the way, so ignore this:
                if self.doing_lower_priority_task and self.mdp.layout_name != 'asymmetric_advantages':
                    # This will return positions close to the final motion_goal
                    return self.move_close_to_goal(motion_goals, pot_location, info['player'].pos_and_or, info["am"])

            elif info['player_obj'].name == 'soup':

                motion_goals = info['am'].deliver_soup_actions()

            elif info['player_obj'].name == 'onion':

                motion_goals = info['am'].place_obj_on_counter_actions(state)

            else:
                raise ValueError('Object not recognised')

        elif not info['player'].has_object():

            # We haven't specified WHICH dish to pick up; just which pot to fetch!
            motion_goals = info['am'].pickup_dish_actions(info['counter_objects'])

        return motion_goals

    def move_close_to_goal(self, motion_goals, goal_location, player_pos_and_or, am):
        """Instead of moving to the motion_goals, move to an adjacent location"""
        logging.info("Other is doing the 1st task on the list, so we're heading near to the motion_goal, so we don't obstruct")

        # First find closest motion goal to the player:
        _, closest_motion_goal = self.find_min_plan_cost_from_pos_or(motion_goals, player_pos_and_or)

        return am.go_to_closest_feature_or_counter_to_goal(closest_motion_goal, goal_location)

    def find_pos_and_or(self, find_own_cost, first_action_info, subsequent_action_info):
        """Return the pos_and_or given in either first_ or subsequent_ action_info"""
        if first_action_info:
            _, info = first_action_info
            if find_own_cost:
                player = info['player']
            else:
                player = info['other_player']
            pos_and_or = player.pos_and_or
        elif subsequent_action_info:
            pos_and_or, _, _ = subsequent_action_info
        return pos_and_or

    def find_cost_of_single_task(self, task, task_goal, find_own_cost=True,
                                 first_action_info=False, subsequent_action_info=False,
                                 sim_counter_objects=None):
        """
        Find the min cost, and the final state that you would have ended up in, for doing a single task.
        first_action_info = [state, info], which is the actual state and actual information about the system.
        subsequent_action_info = [sim_pos_and_or, sim_held_object, am], which is your simulated
        position/orientation etc.
        """

        #TODO: SPLIT THIS INTO HELPER FUNCTIONS!

        if first_action_info:
            # If this is the first action, we need to create simulated ("sim") information from state and info. We
            # will also need some extra things like state and player.

            state, info = first_action_info
            am = info["am"]

            if find_own_cost:
                player = info['player']
            else:
                player = info['other_player']

            # Needed to simulate where the player will end up, and what they'll be carrying, after each subgoal
            sim_pos_and_or = copy.deepcopy(player.pos_and_or)
            sim_held_object = copy.deepcopy(player.held_object)
            sim_counter_objects = copy.deepcopy(info["counter_objects"]) if sim_counter_objects == None \
                                                                                            else sim_counter_objects

        elif subsequent_action_info:
            # If this is not the first action, then subsequent_action_info will contain the simulated information:
            sim_pos_and_or, sim_held_object, am = subsequent_action_info
            sim_counter_objects = defaultdict(list, {'onion': []}) if sim_counter_objects == None \
                                                                                            else sim_counter_objects

        else:
            raise ValueError


        task_name = list(task.keys())[0]
        # If task goal is valid then calculate cost, if not then give cost = Inf
        if not self.mlp.mp.is_valid_motion_start_goal_pair(sim_pos_and_or, task_goal):
            cost = np.Inf
            final_pos_and_or = sim_pos_and_or

        else:

            if task_name == 'deliver_onion':
                object_wanted = 'onion'
            elif task_name == 'deliver_soup':
                object_wanted = 'dish'
            elif task_name == 'soup_from_counter':
                object_wanted = 'soup'

            # If the player is holding the wrong object, then find the cost and final location for dropping it on the
            # closest counter.
            if sim_held_object and sim_held_object.name != object_wanted:
                assert sim_held_object.name != "soup", "Holding the soup is a special case and shouldn't reach here"
                # Wrong object, so drop it first:
                if first_action_info:
                    motion_goals = info["am"].place_obj_on_counter_actions(state)
                #TODO: This might be causing issues; and it's not an ideal solution:
                else:  # Here we don't have the state info, so we simplify by considering all counters:
                    all_counters = self.mdp.get_counter_locations()
                    motion_goals = am._get_ml_actions_for_positions(all_counters)
                cost_to_drop, best_goal = self.find_min_plan_cost_from_pos_or(motion_goals, sim_pos_and_or)
                sim_pos_and_or = best_goal
                sim_held_object = None
            else:
                cost_to_drop = 0

            if task_name == 'soup_from_counter':

                # Cost to pick up the soup:
                cost_to_pick_up = self.find_plan_cost_inc_inf(sim_pos_and_or, task_goal)
                #TODO: Note that this will be set even if the cost is infinite. Change to only set this if cost is finite?
                sim_pos_and_or = task_goal

                # Cost to deliver the soup:
                motion_goals = am.deliver_soup_actions()
                cost_to_deliver, final_pos_and_or = self.find_min_plan_cost_from_pos_or(motion_goals, sim_pos_and_or)

                cost = cost_to_drop + cost_to_pick_up + cost_to_deliver

            elif task_name == 'deliver_onion':

                # If not holding an object, then get an onion. Consider ALL available onions
                if not sim_held_object:
                    cost_to_pickup, sim_pos_and_or_list, pos_and_or_of_pickup = self.get_onion(am,
                                                                                sim_counter_objects, sim_pos_and_or)
                else:
                    assert sim_held_object.name == "onion"
                    cost_to_pickup = [0]
                    sim_pos_and_or_list = [sim_pos_and_or]
                    pos_and_or_of_pickup = [None]

                cost_to_deliver = self.find_cost_deliver_onion(sim_pos_and_or_list, task_goal)

                min_cost, pickup_location = self.find_min_total_cost_onion(cost_to_pickup, cost_to_drop,
                                                                           cost_to_deliver, pos_and_or_of_pickup)

                # If we picked up from a counter, then remove this location from the sim_counter_objects
                if pickup_location in sim_counter_objects['onion']:
                    sim_counter_objects['onion'].remove(pickup_location)

                final_pos_and_or = task_goal
                cost = min_cost

            elif task_name == 'deliver_soup':

                # TODO: Factor into helper functions, as I did with deliver_onion. It's similar to onion, at least until
                #  we get to delivering the soup

                # If not holding an object, then get a dish. Consider ALL available dishes
                if not sim_held_object:

                    cost_to_pickup_dish, sim_pos_and_or_list, pos_and_or_of_dish_pickup = self.get_dish(am,
                                                                        sim_counter_objects, sim_pos_and_or)

                else:
                    assert sim_held_object.name == "dish"
                    cost_to_pickup_dish = [0]
                    sim_pos_and_or_list = [sim_pos_and_or]
                    pos_and_or_of_dish_pickup = None

                cost_to_fetch_soup = self.find_cost_fetch_soup(sim_pos_and_or_list, task_goal)

                min_cost, dish_pickup_location = self.find_min_partial_cost_dish(cost_to_pickup_dish, cost_to_drop,
                                                        cost_to_fetch_soup, sim_held_object, pos_and_or_of_dish_pickup)

                # If we picked up from a counter, then remove this location from the sim_counter_objects
                if dish_pickup_location in sim_counter_objects['dish']:
                    sim_counter_objects.remove(dish_pickup_location)

                sim_pos_and_or = task_goal
                cost_so_far = min_cost

                serving_cost, valid_motion_goals = self.cost_serve_soup(am, sim_pos_and_or)

                # Work out total cost and final postion:
                cost = serving_cost + cost_so_far
                final_pos_and_or = valid_motion_goals

            else:
                raise ValueError('Chosen task not valid')

        return cost, final_pos_and_or, sim_counter_objects

    # TODO: Remove:
    #
    # def create_sim_information(self, first_action_info, find_own_cost):
    #     """If this is the first action, we need to create simulated ("sim") information from state and info. The
    #     simulated info is then used to record what would happen if the agent did x, y or z..."""
    #
    #     state, info = first_action_info
    #
    #     if find_own_cost:
    #         player = info['player']
    #     else:
    #         player = info['other_player']
    #
    #     # Needed to simulate where the player will end up, and what they'll be carrying, after each subgoal
    #     sim_pos_and_or = copy.deepcopy(player.pos_and_or)
    #     sim_held_object = copy.deepcopy(player.held_object)
    #     sim_counter_objects = copy.deepcopy(info["counter_objects"])
    #
    #     return sim_pos_and_or, sim_held_object, sim_counter_objects, info["am"]

    # def place_obj_on_counter_actions_no_state(self, sim_counter_objects, am):
    #     """MediumLevelActionManager.place_obj_on_counter_actions requires the state. But the state isn't needed for
    #     this function, so here we make the equivalent function but without using the state."""
    #
    #     counter_locations = self.mdp.get_counter_locations()
    #     all_empty_counters = [pos for pos in counter_locations if not (pos in (sim_counter_objects['onion'] or
    #                                                                            sim_counter_objects['dish'] or
    #                                                                            sim_counter_objects['soup']))]
    #     valid_empty_counters = [c_pos for c_pos in am.counter_drop if c_pos in all_empty_counters]
    #     return am._get_ml_actions_for_positions(valid_empty_counters)

    def cost_serve_soup(self, am, sim_pos_and_or):
        """Work out the cost and final pos/or for serving the soup"""
        motion_goals = am.deliver_soup_actions()
        valid_motion_goals = list(
            filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(sim_pos_and_or, goal),
                   motion_goals))

        #TODO: This next bit should be made briefer/neater:

        # Cost of first valid goal:
        valid_motion_goal = valid_motion_goals[0]
        serving_cost0 = self.find_plan_cost_inc_inf(sim_pos_and_or, valid_motion_goal)

        # If there's a second valid goal, then choose the lowest cost one:
        if len(valid_motion_goals) == 2:
            serving_cost1 = self.find_plan_cost_inc_inf(sim_pos_and_or, valid_motion_goals[1])
            if serving_cost1 < serving_cost0:
                return serving_cost1, valid_motion_goals[1]

        return serving_cost0, valid_motion_goal

    def get_dish(self, am, sim_counter_objects, sim_pos_and_or):
        """Find cost and other details of getting the dish"""

        motion_goals = am.pickup_dish_actions(sim_counter_objects)
        cost_to_pickup_dish = []
        sim_pos_and_or_list = []
        pos_and_or_of_dish_pickup = []

        for goal in motion_goals:
            plan_cost = self.find_plan_cost_inc_inf(sim_pos_and_or, goal)
            cost_to_pickup_dish.append(plan_cost)
            sim_pos_and_or_list.append(goal)
            pos_and_or_of_dish_pickup.append(goal)  # Needed to work out if the pickup location was a counter

        return cost_to_pickup_dish, sim_pos_and_or_list, pos_and_or_of_dish_pickup

    def get_onion(self, am, sim_counter_objects, sim_pos_and_or):
        """Find cost and other details of getting the onion"""

        motion_goals = am.pickup_onion_actions(sim_counter_objects)
        cost_to_pickup = []
        sim_pos_and_or_list = []
        pos_and_or_of_pickup = []

        for goal in motion_goals:
            plan_cost = self.find_plan_cost_inc_inf(sim_pos_and_or, goal)
            cost_to_pickup.append(plan_cost)
            sim_pos_and_or_list.append(goal)
            # Convert goal to actual the location of the object being picked up, then record the pickup location
            goal_location = self.find_goal_location_from_motion_goal(goal)
            pos_and_or_of_pickup.append(goal_location)

        return cost_to_pickup, sim_pos_and_or_list, pos_and_or_of_pickup

    def find_cost_fetch_soup(self, sim_pos_and_or_list, task_goal):
        cost_to_fetch_soup = []
        # Now we have a dish! Work out the cost to collect the soup, for each item in sim_pos_and_or_list
        for pos_and_or in sim_pos_and_or_list:
            plan_cost = self.find_plan_cost_inc_inf(pos_and_or, task_goal)
            cost_to_fetch_soup.append(plan_cost)
        return cost_to_fetch_soup

    def find_cost_deliver_onion(self, sim_pos_and_or_list, task_goal):
        """Find cost of delivering the onion"""
        cost_to_deliver = []
        # Now we have an onion! Work out the cost to deliver it, for each item in sim_pos_and_or_list
        for pos_and_or in sim_pos_and_or_list:
            plan_cost = self.find_plan_cost_inc_inf(pos_and_or, task_goal)
            cost_to_deliver.append(plan_cost)
        return cost_to_deliver

    def find_min_partial_cost_dish(self, cost_to_pickup_dish, cost_to_drop, cost_to_fetch_soup,
                                   sim_held_object, pos_and_or_of_dish_pickup):
        """Work out min total cost so far. NOTE: We can do this now because each layout only has 1 available
        serving location for each player. So we can work out the min cost now, then add on the serving cost"""
        min_cost = np.Inf
        dish_pickup_location = None
        for i, cost in enumerate(cost_to_pickup_dish):
            total_cost_so_far = cost_to_drop + cost + cost_to_fetch_soup[i]
            if total_cost_so_far < min_cost:
                min_cost = total_cost_so_far
                if not sim_held_object:
                    dish_pickup_location = pos_and_or_of_dish_pickup[i]
        return min_cost, dish_pickup_location

    def find_min_total_cost_onion(self, cost_to_pickup, cost_to_drop, cost_to_deliver, pos_and_or_of_pickup):
        """Work out total min cost of delivering onion"""
        min_cost = np.Inf
        pickup_location = None
        for i, cost in enumerate(cost_to_pickup):
            total_cost = cost_to_drop + cost + cost_to_deliver[i]
            if total_cost < min_cost:
                min_cost = total_cost
                pickup_location = pos_and_or_of_pickup[i]
        return min_cost, pickup_location

    def find_plan_cost_inc_inf(self, start_pos_and_or, goal):
        """self.mlp.mp.get_plan doesn't allow for invalid goals -- here we say invalid goals have infinite cost"""
        if self.mlp.mp.is_valid_motion_start_goal_pair(start_pos_and_or, goal):
            _, _, plan_cost = self.mlp.mp.get_plan(start_pos_and_or, goal)
        else:
            plan_cost = np.Inf

        return plan_cost

    def find_goal_location_from_motion_goal(self, motion_goal):
        """Find the coordinates of a goal location from the motion_goal"""
        goal_location = tuple([sum(x) for x in zip(list(motion_goal[0]), list(motion_goal[1]))])
        return goal_location

    def remove_others_current_task_from_list(self, task_priority_list, info, state, look_ahead_steps=1):
        """Take the task list, then observe what the other agent is likely doing and remove this task from the list."""

        other_player = info['other_player']

        if other_player.has_object():

            # Special case: If other is holding a soup, assume they'll deliver it, and leave list unchanged:
            if other_player.get_object().name == 'soup':
                task = {'special_case_deliver_soup': None}
                return task_priority_list, task

            elif look_ahead_steps == 1:
                # Returning "None" because if look_ahead=1 then we will do the greedy task regardless (look ahead = 1
                # only for agent B)
                return self.cross_task_off_list_if_appears_first(other_player, task_priority_list, info, state), None
            elif look_ahead_steps > 1:
                return self.cross_task_off_list_if_appears_anywhere(other_player, task_priority_list, info, state,
                                                                     look_ahead_steps)
            else:
                raise ValueError

        else:
            # See if any of the 1st tasks on list are within the field of view of the agent; if so, cross off list:
            object_goals, object_tasks = self.find_objects_for_tasks(task_priority_list[0], info)
            # If they don't have an object 1st on the list, but we do, then don't assume they're doing that task:
            object_goals, object_tasks = self.remove_tasks_if_were_holding_object(object_goals, object_tasks, info)
            task = self.find_task_for_object_in_field_of_view(other_player, object_goals, object_tasks)
            return self.cross_task_off_list(task_priority_list, task), task

    def cross_task_off_list_if_appears_first(self, other_player, task_priority_list, info, state):
        # Find if object is 1st on list and if it only appears once on the list
        count_object_first_on_list, tasks_with_object = self.find_if_object_in_task_list(other_player.get_object(),
                                                                                         task_priority_list[0])
        if count_object_first_on_list == 0:
            # The object held by the other player is not first on the list, so we can just keep the list as it is.
            logging.info("The object held by the other player is not first on the list, so we can just keep the list as it is")
            return task_priority_list

        elif count_object_first_on_list == 1:
            # Assume other agent is doing this task, so cross it off the list
            logging.info("Assume other agent is doing the 1st task on list, so cross it off the list")
            return self.cross_task_off_list(task_priority_list, tasks_with_object[0])

        elif count_object_first_on_list == 2:
            # Find which task in tasks_with_object has a lower cost
            logging.info("2 options for which task other could be doing: cross their lowest cost one off the list")
            lowest_cost_task = self.find_lowest_cost_task(tasks_with_object, info, state, find_own_cost=False)
            return self.cross_task_off_list(task_priority_list, lowest_cost_task)

    def cross_task_off_list_if_appears_anywhere(self, other_player, task_priority_list, info, state, look_ahead_steps):

        for i in range(look_ahead_steps):

            count_object_on_list, tasks_with_object = self.find_if_object_in_task_list(other_player.get_object(),
                                                                                             task_priority_list[i])
            if count_object_on_list == 1:

                return self.cross_task_off_list(task_priority_list, tasks_with_object[0]), tasks_with_object[0]

            elif count_object_on_list == 2:
                lowest_cost_task = self.find_lowest_cost_task(tasks_with_object, info, state, find_own_cost=False)
                return self.cross_task_off_list(task_priority_list, lowest_cost_task), lowest_cost_task

        # The object held by other_player is not on the task list up to look_ahead_steps, so keep the list as it is:
        return task_priority_list, None


    def remove_tasks_if_were_holding_object(self, object_goals, object_tasks, info):
        """If we're already holding one of the objects, then ignore the other player"""
        object_goals_new, object_tasks_new = [], []
        for i, task in enumerate(object_tasks):
            task_name = list(task.keys())[0]

            if info['player'].has_object():
                if task_name == 'deliver_onion':
                    if not info['player'].get_object().name == 'onion':
                        object_goals_new.append(object_goals[i])
                        object_tasks_new.append(task)
                elif task_name == 'deliver_soup':
                    if not info['player'].get_object().name == 'dish':
                        object_goals_new.append(object_goals[i])
                        object_tasks_new.append(task)
            else:
                object_goals_new.append(object_goals[i])
                object_tasks_new.append(task)

        return object_goals_new, object_tasks_new

    def find_task_for_object_in_field_of_view(self, player, object_goals, objects_tasks):
        """Find which of the objects are within player's 180 degree field of view, and return the associated task"""
        min_cost = np.Inf
        task_to_do = None
        for i, object_goal in enumerate(object_goals):
            within_field_of_view = False

            # Manually find if each object is within 180 degree view of player:
            if player.orientation == (1, 0):
                # Facing Right. So we need the object's goal location's first coordinate to be > the player's first
                # position coordinate.
                if object_goal[0][0] >= player.position[0]:
                    within_field_of_view = True
            elif player.orientation == (-1, 0):  # Facing L
                if object_goal[0][0] <= player.position[0]:
                    within_field_of_view = True
            elif player.orientation == (0, 1):  # Facing D
                if object_goal[0][1] >= player.position[1]:
                    within_field_of_view = True
            elif player.orientation == (0, -1):  # Facing U
                if object_goal[0][1] <= player.position[1]:
                    within_field_of_view = True
            else:
                raise ValueError("Orientation not recognised")

            if within_field_of_view:
                # Now check if the object is the closest:

                cost = self.find_plan_cost_inc_inf(player.pos_and_or, object_goals[i])
                if cost < min_cost:
                    min_cost = cost
                    task_to_do = objects_tasks[i]

        return task_to_do

    def find_objects_for_tasks(self, task_list, info):
        """Return all objects and their "goals" for all tasks on task_list"""
        object_goals, object_tasks = [], []
        for task in task_list:
            task_name = list(task.keys())[0]
            if task_name == 'deliver_onion':
                motion_goals = info["am"].pickup_onion_actions(info["counter_objects"])
                for goal in motion_goals:
                    object_goals.append(goal)
                    object_tasks.append(task)
            elif task_name == 'deliver_soup':
                motion_goals = info["am"].pickup_dish_actions(info["counter_objects"])
                for goal in motion_goals:
                    object_goals.append(goal)
                    object_tasks.append(task)
            # TODO: The elif below is sloppy: it returns all motion goals for ALL soups on the counter, not just the specific one given by 'task'. E.g.
            #  if the task is soup at (1, 0), then below will also return goals for any other soup!
            elif task_name == 'soup_from_counter':
                motion_goals = info["am"].pickup_counter_soup_actions(info["counter_objects"])
                for goal in motion_goals:
                    object_goals.append(goal)
                    object_tasks.append(task)
            else:
                raise ValueError("Task not recognised")

        return object_goals, object_tasks

    def cross_task_off_list(self, task_priority_list, task):
        """Cross task off the task_priority list then return the amended list"""
        for i, tasks in enumerate(task_priority_list):
            if task in tasks:
                task_priority_list[i].remove(task)
                if task_priority_list[i] == []:  # If this priority task is now empty, then remove from task list:
                    task_priority_list.remove([])

                    # If this priority task is now empty, and i==0, then we can't do the top priority task:
                    if i == 0:
                        self.doing_lower_priority_task = True

                # We want to remove the task from the list only once, so exit here:
                return task_priority_list

        return task_priority_list

    def find_if_object_in_task_list(self, object, task_list):
        """
        Returns:    Number of times object is on the list of objects associated with a task on task_list
                    List of tasks for which the object appears
        """
        tasks_with_object = []

        for task in task_list:
            task_name = list(task.keys())[0]
            if task_name == 'deliver_onion':
                task_objects = ['onion']
                if object.name in task_objects:
                    tasks_with_object.append(task)
            elif task_name == 'deliver_soup':
                task_objects = ['dish']
                if object.name in task_objects:
                    tasks_with_object.append(task)
            elif task_name == 'soup_from_counter':
                task_objects = ['soup']
                if object.name in task_objects:
                    raise ValueError('We should never get here... Also, if theyre holding a soup then they cant deliver '
                                     'a counter soup without dropping the soup theyre holding!')

        return len(tasks_with_object), tasks_with_object

    def choose_random_action(self, prob_random_action, action):
        """Choose a random action from the 6 possible actions (ignore whether actions are possible or not)"""
        if np.random.rand() < prob_random_action:
            return np.random.choice(Action.ALL_ACTIONS)
        else:
            return action


#----------------------------------------------------------------------------------------------------------------------#

    def OLD_ml_action(self, state):
        """Selects a medium level action for the current state"""

        # TODO: Here, or in future codes, I should use dictionaries instead of return a long list of variables. E.g.
        # make a dictionary info_for_decisions, with name:variable pairs. Then pass that dict in to other methods
        # that need info, and extract the info from the dict only when you need it...
        player, other_player, am, counter_objects, pot_states_dict, soup_nearly_ready, count_soups_nearly_ready, \
        other_has_dish, other_has_onion, number_of_pots, temp_dont_drop = self.get_info_for_making_decisions(state)

        if not player.has_object():

            # Only get the dish if the soup is nearly ready:
            if soup_nearly_ready:

                default_motion_goals = am.pickup_dish_actions(counter_objects)  # (Exclude
                # self.only_take_dispenser_dishes here because we are considering the goals for both players)
                motion_goals, temp_dont_drop = self.revise_pickup_dish_goal_considering_other_player_and_noise(
                    player, default_motion_goals, other_player, state,
                    other_has_dish, am, counter_objects, temp_dont_drop)
                motion_goals = self.sometimes_overwrite_goal_with_greedy(motion_goals, am, state, counter_objects,
                                                                         greedy_goal='pickup_dish')

            # Otherwise get onion:
            elif not soup_nearly_ready:

                default_motion_goals = am.pickup_onion_actions(counter_objects)  # (Exclude
                # self.only_take_dispenser_onions here because we are considering the goals for both players)
                motion_goals, temp_dont_drop = self.revise_pickup_onion_goal_considering_other_player_and_noise(
                    number_of_pots, player, default_motion_goals, other_player, state,
                    other_has_onion, counter_objects, am, temp_dont_drop)
                motion_goals = self.sometimes_overwrite_goal_with_greedy(motion_goals, am, state, counter_objects,
                                                                         greedy_goal='pickup_onion')
            else:
                raise ValueError('Failed logic')

            motion_goals = self.overwrite_goal_if_soup_on_counter(motion_goals, counter_objects, am, state, player,
                                                                  other_player)

        elif player.has_object():

            soups_need_onions, player_obj = self.get_extra_info_for_making_decisions(pot_states_dict, player)

            if player_obj.name == 'onion':

                default_motion_goals = am.put_onion_in_pot_actions(pot_states_dict)
                motion_goals, temp_dont_drop = self.revise_use_onion_goal_considering_other_player_and_noise(
                    soups_need_onions, state, temp_dont_drop, counter_objects, am, pot_states_dict, player,
                    default_motion_goals, other_player, number_of_pots)
                motion_goals = self.sometimes_overwrite_goal_with_greedy(motion_goals, am, state, counter_objects,
                                                                         greedy_goal='drop_or_use_onion',
                                                                         soups_need_onions=soups_need_onions,
                                                                         pot_states_dict=pot_states_dict)
                # If we're on forced_coord then we need to consider goals that pass the object to the other player:
                motion_goals = self.special_onion_motion_goals_for_forced_coord(soups_need_onions, player,
                                                                                motion_goals, state)

            elif player_obj.name == 'dish':

                default_motion_goals = am.pickup_soup_with_dish_actions(pot_states_dict, only_nearly_ready=True)
                motion_goals, temp_dont_drop = self.revise_pickup_soup_goal_considering_other_player_and_noise(
                    count_soups_nearly_ready, number_of_pots, temp_dont_drop, state, counter_objects, am,
                    pot_states_dict, player, default_motion_goals, other_player)
                motion_goals = self.sometimes_overwrite_goal_with_greedy(motion_goals, am, state, counter_objects,
                                                                         greedy_goal='drop_or_use_dish',
                                                                         pot_states_dict=pot_states_dict,
                                                                         soup_nearly_ready=soup_nearly_ready)
                # If we're on forced_coord then we need to consider goals that pass the object to the other player:
                motion_goals = self.special_dish_motion_goals_for_forced_coord(count_soups_nearly_ready, player,
                                                                               motion_goals, state)

            elif player_obj.name == 'soup':
                # Deliver soup whatever the other player is doing
                motion_goals = am.deliver_soup_actions()
                motion_goals = self.sometimes_make_wrong_decision(motion_goals, state, counter_objects, am,
                                                                  current_goal='use_dish')
            else:
                raise ValueError()
        else:
            raise ValueError('Player has AND does not have an object!')

        motion_goals = self.remove_invalid_goals_and_clean_up(player, motion_goals, am, temp_dont_drop)

        return motion_goals

#============== Sub-methods (mainly for ToMModel) ======================#

    def take_alternative_action(self):

        #TODO: Remove the old method once it's no longer in use:
        # NEW METHOD:
        if not self.use_OLD_ml_action:
            return random.random() < self.compliance

        # OLD METHOD:
        else:
            """This first gives Prob(taking alternative action)=1 if perseverance=0 and Prob=0 if perseverance=1. Otherwise,
                    e.g. perseverance=_, num_items, _ = state.get_object(self.mlp.mdp.get_pot_locations()[0]).state
                        #current_list = full_list[num_items:number_of_tasks]

                    # if number_of_pots == 2:
                        #0.5, for the first timestep Prob~0.5, then Prob-->1 quickly as time increases. Then using this prob
                    it randomly determines if the player should take an alternative action"""

            prob_taking_alternative = 1 - (1 / (np.exp(self.timesteps_stuck*10))) ** (1 - self.perseverance ** (1 / 10))
            rand = random.random()
            take_alternative = rand < prob_taking_alternative
            # logging.info('time stuck: {}, perserv: {}, Prob_taking_alt: {}, Taking alt: {}'.format(self.timesteps_stuck, self.perseverance, Prob_taking_alternative, take_alternative))
            return take_alternative

    def OLD_tasks_to_do(self, state):
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


        number_of_pots = len(self.mlp.mdp.get_pot_locations())
        order_length = len(state.order_list)
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
                plan_cost = self.find_plan_cost_inc_inf(start_pos_and_or, goal)
                if plan_cost < min_cost:
                    # best_action = action_plan[0]
                    min_cost = plan_cost
                    # best_goal = goal

        return min_cost

    def find_min_plan_cost_from_pos_or(self, motion_goals, pos_and_or):
        """
        Given some motion goals, find the cost for the lowest cost goal, from pos_and_or to each motion_goal
        """
        min_cost = np.Inf
        best_goal = pos_and_or  # This is needed in case all motion_goals have inf cost, in which case we need a
        # best_goal but it doesn't matter what it is because if the cost in inf then best_goal will not be used
        if len(motion_goals) != 0:
            for goal in motion_goals:
                plan_cost = self.find_plan_cost_inc_inf(pos_and_or, goal)
                if plan_cost < min_cost:
                    # best_action = action_plan[0]
                    min_cost = plan_cost
                    best_goal = goal

        return min_cost, best_goal

    def find_plan(self, state, motion_goals):
        """Find the lowest cost plan (best_action, best_goal, action_plan) from a set of motion_goals"""

        # Find valid actions:
        # valid_next_pos_and_ors = self.find_valid_next_pos_and_ors(state)

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
        """Find the lowest cost plan from the set of motion goals from start_pos_and_or. Return the cost of the lowest
        cost plan."""

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

        if np.inf in x:
            for i, y in enumerate(x):
                if y == np.inf:
                    exponent[i] = -1000
        # More mathematically stable version that what I had before:
        e_x = np.exp(exponent - np.max(exponent))
        return e_x / e_x.sum()

    def find_plan_boltz_rational(self, state, motion_goals):
        # TODO: Needs a description!

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
            if chosen_action == 'interact':
                # If the chosen next step is to not move, then the action will be 'interact', so there's a risk of randomly
                # dropping objects. So instead, just don't move.
                chosen_action = (0, 0)

        return chosen_action, chosen_goal

    def find_joint_plan_from_start_pos_and_or(
            self, start_pos_and_or, start_pos_and_or_other, motion_goals, others_predicted_goal):
        # TODO: Needs a description!

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
        #TODO: Needs a description!

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
        # TODO: Needs a description!

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

    # === Sub-methods added during code cleanup === #

    # Cleaning up action function:

    def display_game_during_training(self, state):
        # Display the game during training:
        try:
            self.display
            overcooked_env = OvercookedEnv(self.mdp)
            overcooked_env.state = state
            print('TRAINING GAME WITH TOM. TOM index: {}'.format(self.agent_index))
            print(overcooked_env)
        except:
            AttributeError  # self.display = False

    def display_game_state(self, state):
        overcooked_env = OvercookedEnv(self.mdp)
        overcooked_env.state = state
        print(overcooked_env)

    def fix_invalid_prev_motion_goal(self, state):
        # Check motion goal is valid; if not, set to None AND set best action to None:
        if self.prev_motion_goal is not None and not self.mlp.mp.is_valid_motion_start_goal_pair \
                    (state.players_pos_and_or[self.agent_index], self.prev_motion_goal[0]):
            self.prev_motion_goal = None
            self.prev_best_action = None
        # TODO: We only need this very hacky 'check motion goals' cos the HM agent isn't reset at the end of an
        #  episode, so in unident it's possible that the agent has changed sides, so that the prev motion goal
        #  is no longer valid. A better solution is to reset the HM after each episode. (But I'm not sure how?
        #  Perhaps on line 386 of ppo2.py??)

    def choose_best_action(self, state, motion_goals):
        # Find plan; with Prob = self.path_teamwork factor in the other player
        if random.random() < self.path_teamwork:
            best_action, best_goal = self.find_plan_boltz_rat_inc_other(state, motion_goals)
            logging.info('Choosing path that factors in the other player. Best act: {}, goal: {}'
                         .format(best_action, best_goal))
            # If the plan that included the other player has inf cost, then ignore them / do a random action
            if best_action == None:
                # Get temp action and goal by ignoring the other player:
                best_action, best_goal = self.find_plan_boltz_rational(state, motion_goals)
                logging.info('No path with finite cost... ignoring other player now. Best act: {}, goal: {}'
                             .format(best_action, best_goal))
        else:
            best_action, best_goal = self.find_plan_boltz_rational(state, motion_goals)
            logging.info('Choosing path that ignores the other player. Best act: {}, goal: {}'
                         .format(best_action, best_goal))
        # Save motion goal:
        self.prev_motion_goal = [best_goal]

        return best_action

    def take_alternative_action_if_stuck(self, best_action, state):
        #TODO: Needs a description

        # Before setting self.prev_best_action = best_action, we need to determine if the prev_best_action was (0,0),
        # in which case the agent isn't stuck:
        if self.prev_best_action == (0, 0):
            agent_chose_stationary = True
        else:
            agent_chose_stationary = False

        self.prev_best_action = best_action

        """If the agent is stuck, then take an alternative action with a probability based on the time stuck and the
        agent's "perseverance". Note: We consider an agent stuck if their whole state is unchanged (but this misses the
        case when they try to move and do change direction but can't move <-- here the state changes & they're stuck).
        Also, there exist cases when the state doesn't change but they're not stuck, they just can't complete their action
        (e.g. on unident they could try to use an onion but the other player has already filled the pot)"""

        if self.prev_state is not None \
                and state.players[self.agent_index] == self.prev_state.players[self.agent_index] \
                and not agent_chose_stationary:

            self.timesteps_stuck += 1
            take_alternative = self.take_alternative_action()
            # logging.info('Player {} timesteps stuck: {}'.format(self.agent_index, self.timesteps_stuck))
            if take_alternative:
                logging.info('Stuck, and taking alternative action')
                # If the agent is stuck, set self.prev_best_action = None, then they always re-think their goal
                self.prev_best_action = None
                # TODO: This is the place to put a more thorough avoiding action, e.g. taking 2 steps to
                #  mavouver round the other player

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

                best_action = self.prefer_adjacent_actions_if_available(best_action, unblocking_joint_actions)

        else:
            self.timesteps_stuck = 0  # Reset to zero if prev & current player states aren't the same (they're not stuck)

        # NOTE: Assumes that calls to action are sequential
        self.prev_state = state

        return best_action

    def prefer_adjacent_actions_if_available(self, best_action, unblocking_joint_actions):
        ## Prefer to take adjacent actions if available:
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

        return best_action

    # Cleaning up ml_action function:

    def get_info_for_making_decisions(self, state):
        # All this info is used when the agent makes decisions:
        player = state.players[self.agent_index]
        other_player = state.players[1 - self.agent_index]
        am = self.mlp.ml_action_manager

        counter_objects = self.mlp.mdp.get_counter_objects_dict(state, list(self.mlp.mdp.terrain_pos_dict['X']))
        pot_states_dict = self.mlp.mdp.get_pot_states(state)

        ready_soups = pot_states_dict['onion']['ready']
        cooking_soups = pot_states_dict['onion']['cooking']

        soup_nearly_ready = len(ready_soups) > 0 or len(cooking_soups) > 0
        count_soups_nearly_ready = len(ready_soups) + len(cooking_soups)
        other_has_dish = other_player.has_object() and other_player.get_object().name == 'dish'
        other_has_onion = other_player.has_object() and other_player.get_object().name == 'onion'

        number_of_pots = len(self.mlp.mdp.get_pot_locations())
        temp_dont_drop = False

        return player, other_player, am, counter_objects, pot_states_dict, soup_nearly_ready, \
               count_soups_nearly_ready, other_has_dish, other_has_onion, number_of_pots, temp_dont_drop

    def get_extra_info_for_making_decisions(self, pot_states_dict, player):
        # Determine if any soups need onions
        pots_empty = len(pot_states_dict['empty'])
        pots_partially_full = len(pot_states_dict['onion']['partially_full'])
        soups_need_onions = pots_empty + pots_partially_full
        player_obj = player.get_object()
        return soups_need_onions, player_obj

    def remove_invalid_goals_and_clean_up(self, player, motion_goals, am, temp_dont_drop):

        # Remove invalid goals:
        motion_goals = list(filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or,
                                                                                                goal), motion_goals))

        # If no goals, then just go to nearest feature
        if len(motion_goals) == 0:
            motion_goals = am.go_to_closest_feature_actions(player)
            motion_goals = list(
                filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, goal), motion_goals))
            assert len(motion_goals) != 0

        if temp_dont_drop == True:
            self.dont_drop = True
        else:
            self.dont_drop = False

        return motion_goals

    def special_onion_motion_goals_for_forced_coord(self, soups_need_onions, player, motion_goals, state):
        """At this stage there should always be a valid motion_goal, unless the player can't reach any goal.
        ASSUME that this only happens if there is no free pot OR if they're on random0 / Forced Coord."""
        if (soups_need_onions > 0) and list(filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(
                player.pos_and_or, goal), motion_goals)) == []:
            # No goal is reachable, and there is a pot to be filled. Therefore put the onion somewhere where
            # the other player can reach it!
            free_counters_valid_for_both = \
                self.mlp.mdp.find_free_counters_valid_for_both_players(state, self.mlp)
            # TODO: Must be a more efficent way than these next few lines:
            new_goals = [self.mlp.mp._get_possible_motion_goals_for_feature(counter) for counter in
                         free_counters_valid_for_both]
            for i in range(len(new_goals)):
                for j in range(len(new_goals[0])):
                    motion_goals.append(new_goals[i][j])
            # (Non-valid motion goals are removed later)

            # From now on, only pick onions up from the dispenser
            # Todo: What we really want is to make the agent not pick up the specific onion that it dropped,
            #  but this is long-winded to code and the end result won't be much different...
            self.only_take_dispenser_onions = True

        return motion_goals

    def special_dish_motion_goals_for_forced_coord(self, count_soups_nearly_ready, player, motion_goals, state):
        """At this stage there should always be a valid motion_goal, unless the player can't reach any goal.
        # ASSUME that this only happens if there is no cooked soup OR if they're on random0 / Forced Coord."""
        if (count_soups_nearly_ready > 0) and list(
                filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(
                    player.pos_and_or, goal), motion_goals)) == []:
            # No goal is reachable, and there is a ready/cooking soup. Therefore, put the dish somewhere where
            # the other player can reach it:
            free_counters_valid_for_both = \
                self.mlp.mdp.find_free_counters_valid_for_both_players(state, self.mlp)
            # TODO: Must be a more efficent way than these next few lines:
            new_goals = [self.mlp.mp._get_possible_motion_goals_for_feature(counter) for counter in
                         free_counters_valid_for_both]
            for i in range(len(new_goals)):
                for j in range(len(new_goals[0])):
                    motion_goals.append(new_goals[i][j])
            # Non-valid motion goals are removed at the end...

            # From now on, only pick dishes up from the dispenser
            # Todo: What we really want is to make the agent not pick up the specific dish that it dropped,
            #  but this is long-winded to code and the end result won't be much different...
            self.only_take_dispenser_dishes = True

        return motion_goals

    #TODO: An alternative way for the agent to make the wrong decision is just to take a random action at the end of
    # the section "if not player.has_object()" then again after section "elif player.has_object()". This will
    # sometimes give goals that can't be achieved, but will neaten the code up.
    def sometimes_make_wrong_decision(self, motion_goals, state, counter_objects, am, current_goal=None,
                                      pot_states_dict_temp=None):
        if random.random() < self.wrong_decisions:
            logging.info('Making the wrong decision!')

            if current_goal == 'dish':
                motion_goals = am.pickup_onion_actions(counter_objects)
            elif current_goal == 'onion':
                motion_goals = am.pickup_dish_actions(counter_objects)
            elif current_goal == 'soup':
                if random.random() > 0.5:
                    motion_goals = am.pickup_dish_actions(counter_objects)
                else:
                    motion_goals = am.pickup_onion_actions(counter_objects)

            elif current_goal == 'hold_onion':
                if random.random() > 0.5:
                    motion_goals = am.place_obj_on_counter_actions(state)
                else:
                    motion_goals = am.pickup_soup_with_dish_actions(pot_states_dict_temp)
            elif current_goal == 'hold_dish':
                if random.random() > 0.5:
                    motion_goals = am.place_obj_on_counter_actions(state)
                else:
                    motion_goals = am.put_onion_in_pot_actions(pot_states_dict_temp)

            elif current_goal == 'drop_onion':
                # Treating onion as if it were a dish:
                motion_goals = am.pickup_soup_with_dish_actions(pot_states_dict_temp)
            elif current_goal == 'drop_onion_alt':
                motion_goals = am.put_onion_in_pot_actions(pot_states_dict_temp)
            elif current_goal == 'drop_dish':
                motion_goals = am.put_onion_in_pot_actions(pot_states_dict_temp)
            elif current_goal == 'drop_dish_alt':
                am.pickup_soup_with_dish_actions(pot_states_dict_temp, only_nearly_ready=True)

            elif current_goal == 'use_onion':
                motion_goals = am.place_obj_on_counter_actions(state)
            elif current_goal == 'use_dish':
                motion_goals = am.place_obj_on_counter_actions(state)

        if current_goal == None:
            raise ValueError('current_goal needs to be specified')

        return motion_goals

    def sometimes_overwrite_goal_with_greedy(self, motion_goals, am, state, counter_objects, greedy_goal=None,
                                soups_need_onions=None, pot_states_dict=None, soup_nearly_ready=None):
        if random.random() > self.teamwork:
            logging.info('Overwriting motion_goal with the greedy option')
            if greedy_goal == 'pickup_dish':
                motion_goals = am.pickup_dish_actions(counter_objects, self.only_take_dispenser_dishes)
                motion_goals = self.sometimes_make_wrong_decision(motion_goals, state, counter_objects, am,
                                                                  current_goal='dish')
            elif greedy_goal == 'pickup_onion':
                motion_goals = am.pickup_onion_actions(counter_objects, self.only_take_dispenser_onions)
                motion_goals = self.sometimes_make_wrong_decision(motion_goals, state, counter_objects, am,
                                                                  current_goal='onion')
            elif greedy_goal == 'drop_or_use_onion':
                if soups_need_onions == 0:
                    # Player has an onion but there are no soups to put it in, then drop the onion
                    motion_goals = am.place_obj_on_counter_actions(state)
                    motion_goals = self.sometimes_make_wrong_decision(motion_goals, state, counter_objects, am,
                                      current_goal='drop_onion', pot_states_dict_temp=pot_states_dict)
                else:
                    motion_goals = am.put_onion_in_pot_actions(pot_states_dict)
                    motion_goals = self.sometimes_make_wrong_decision(motion_goals, state, counter_objects, am,
                                                                      current_goal='use_onion')
            elif greedy_goal == 'drop_or_use_dish':
                if not soup_nearly_ready:
                    # Player has a dish but there are no longer any "nearly ready" soups: drop the dish
                    motion_goals = am.place_obj_on_counter_actions(state)
                    motion_goals = self.sometimes_make_wrong_decision(motion_goals, state, counter_objects, am,
                                                current_goal='drop_dish', pot_states_dict_temp=pot_states_dict)
                elif soup_nearly_ready:
                    motion_goals = am.pickup_soup_with_dish_actions(pot_states_dict, only_nearly_ready=True)
                    motion_goals = self.sometimes_make_wrong_decision(motion_goals, state, counter_objects, am,
                                                                      current_goal='use_dish')
            elif greedy_goal == None:
                raise ValueError('greedy_goal not specified')

        return motion_goals

    def find_min_cost_of_achieving_goal(self, player, default_motion_goals, other_player, state):
        # Remove invalid goals:
        own_motion_goals = list(
            filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, goal),
                   default_motion_goals))
        others_motion_goals = list(
            filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(other_player.pos_and_or, goal),
                   default_motion_goals))
        # Find costs:
        own_min_cost = self.find_min_plan_cost(own_motion_goals, state, self.agent_index)
        others_min_cost = self.find_min_plan_cost(others_motion_goals, state, 1 - self.agent_index)

        return own_min_cost, others_min_cost

    # Functions specific to each part of the decision tree:

    def revise_pickup_onion_goal_considering_other_player_and_noise(self, number_of_pots, player,
                    default_motion_goals, other_player, state, other_has_onion, counter_objects, am, temp_dont_drop):
        """Revise what the best goal is, depending on whether the other player is in a better/worse position to
        achieve the default goal. Noise refers to making wrong decisions with a given prob.
        Note: If there are two pots, then we can only get inside this 'if' when both pots have onion first on the
        list of tasks to do, so we can always get an onion. For 1 pot then just look what's next on the list"""

        # ASSUMING TWO POTS MAX THROUGHOUT
        if number_of_pots == 1:

            own_min_cost, others_min_cost = self.find_min_cost_of_achieving_goal(player,
                                                                                 default_motion_goals, other_player,
                                                                                 state)

            # Get the onion if the other player doesn't have an onion or they do have an onion but are further away
            if (not other_has_onion) and (
                    others_min_cost >= own_min_cost):  # TODO: make it random if others_min_cost = own_min_cost ?
                logging.info('Getting an onion: assume in better position that other player')
                motion_goals = am.pickup_onion_actions(counter_objects, self.only_take_dispenser_onions)
                motion_goals = self.sometimes_make_wrong_decision(motion_goals, state, counter_objects, am,
                                                                  current_goal='onion')

            elif other_has_onion or ((not other_has_onion) and (others_min_cost < own_min_cost)):
                # Now find the *next* task that needs to be done, and do that
                logging.info('Assume other player is in better position, so do the next task on the list')
                tasks = self.OLD_tasks_to_do(state)

                if tasks[0][1] == 'fetch_onion':  # tasks[0][1] is the next task (when there's one pot)
                    logging.info('Next task: onion')
                    motion_goals = am.pickup_onion_actions(counter_objects, self.only_take_dispenser_onions)
                    motion_goals = self.sometimes_make_wrong_decision(motion_goals, state, counter_objects, am,
                                                                      current_goal='onion')

                elif tasks[0][1] == 'fetch_dish':
                    logging.info('Next task: dish')
                    motion_goals = am.pickup_dish_actions(counter_objects, self.only_take_dispenser_dishes)
                    motion_goals = self.sometimes_make_wrong_decision(motion_goals, state, counter_objects, am,
                                                                      current_goal='dish')
                    logging.info('Dish shouldnt be dropped, so set temp_dont_drop = True')
                    temp_dont_drop = True

        elif number_of_pots == 2:
            # Get an onion regardless: there's always a pot needing an onion (because 'not soup_nearly_ready')
            motion_goals = am.pickup_onion_actions(counter_objects, self.only_take_dispenser_onions)
            motion_goals = self.sometimes_make_wrong_decision(motion_goals, state, counter_objects, am,
                                                              current_goal='onion')
        else:
            raise ValueError('Assuming 1 or 2 pots, but there are more??')

        return motion_goals, temp_dont_drop

    def revise_pickup_dish_goal_considering_other_player_and_noise(self, player, default_motion_goals,
                                                                   other_player, state, other_has_dish, am,
                                                                   counter_objects, temp_dont_drop):
        """Revise what the best goal is, depending on whether the other player is in a better/worse position to
        achieve the default goal. Noise refers to making wrong decisions with a given prob."""

        own_min_cost, others_min_cost = self.find_min_cost_of_achieving_goal(player, default_motion_goals,
                                                                             other_player, state)

        # Get the dish if the other player doesn't have a dish or they do have a dish but are further away
        if (not other_has_dish) and (others_min_cost >= own_min_cost):

            logging.info('Getting the dish: assume in better position that other player')
            motion_goals = am.pickup_dish_actions(counter_objects, self.only_take_dispenser_dishes)
            motion_goals = self.sometimes_make_wrong_decision(motion_goals, state, counter_objects, am,
                                                              current_goal='dish')

        elif other_has_dish or ((not other_has_dish) and (others_min_cost < own_min_cost)):
            # Now find the *next* task that needs to be done, and do it. Simple way of doing this: IF only
            # one nearly_ready pot at the top of the list THEN get onion. IF both lists start with pot,
            # THEN get a dish (worry about where to put it later!)
            logging.info('Assume other player is in better position, so do the next task on the list')
            tasks = self.OLD_tasks_to_do(state)

            # ASSUMING TWO POTS MAX
            number_of_pots = len(self.mlp.mdp.get_pot_locations())
            if number_of_pots == 1:
                # Next task will always be onion
                logging.info('Next task: onion')
                motion_goals = am.pickup_onion_actions(counter_objects, self.only_take_dispenser_onions)
                motion_goals = self.sometimes_make_wrong_decision(motion_goals, state, counter_objects, am,
                                                                  current_goal='onion')
                logging.info('Onion shouldnt be dropped, so set temp_dont_drop = True')
                temp_dont_drop = True

            elif number_of_pots == 2:

                if (tasks[0][0] == 'fetch_onion') or (tasks[1][0] == 'fetch_onion'):
                    logging.info('Next task: onion')
                    # Next task will be onion
                    motion_goals = am.pickup_onion_actions(counter_objects, self.only_take_dispenser_onions)
                    motion_goals = self.sometimes_make_wrong_decision(motion_goals, state, counter_objects, am,
                                                                      current_goal='onion')

                elif (tasks[0][0] == 'fetch_dish') and (tasks[1][0] == 'fetch_dish'):
                    logging.info('Next task: dish')
                    # Next task will be dish
                    motion_goals = am.pickup_dish_actions(counter_objects, self.only_take_dispenser_dishes)
                    motion_goals = self.sometimes_make_wrong_decision(motion_goals, state, counter_objects, am,
                                                                      current_goal='dish')
                else:
                    raise ValueError('Failed logic')
        else:
            raise ValueError('Failed logic')

        return motion_goals, temp_dont_drop

    def overwrite_goal_if_soup_on_counter(self, motion_goals, counter_objects, am, state, player, other_player):
        """If there's a soup on the counter, then override other goals and get the soup, unless other player is (
        # strictly) closer"""
        if 'soup' in counter_objects:

            default_motion_goals = am.pickup_counter_soup_actions(counter_objects)
            own_min_cost, others_min_cost = self.find_min_cost_of_achieving_goal(player, default_motion_goals,
                                                                                 other_player, state)
            if others_min_cost >= own_min_cost:
                logging.info('Soup on counter and other player is futher away: get soup')
                motion_goals = default_motion_goals
                motion_goals = self.sometimes_make_wrong_decision(motion_goals, state, counter_objects, am,
                                                                  current_goal='soup')
            elif others_min_cost < own_min_cost:
                logging.info('Soup on counter but other closer')

        return motion_goals

    def revise_use_onion_goal_considering_other_player_and_noise(self,
                                                                 soups_need_onions, state, temp_dont_drop,
                                                                 counter_objects, am, pot_states_dict, player,
                                                                 default_motion_goals, other_player, number_of_pots):
        """Revise what the best goal is, depending on whether the other player is in a better/worse position to
        achieve the default goal. Noise refers to making wrong decisions with a given prob."""
        if soups_need_onions == 0:

            if self.dont_drop == True:
                # Don't drop the onion
                logging.info('Got onion because its the next task: just do nothing')
                # TODO: unrealistic to do nothing, but the alternatives are hard to code
                motion_goals = state.players[self.agent_index].pos_and_or
                temp_dont_drop = True
                motion_goals = self.sometimes_make_wrong_decision(motion_goals, state, counter_objects, am,
                                                                  current_goal='hold_onion',
                                                                  pot_states_dict_temp=pot_states_dict)
            else:
                # Drop onion
                logging.info('Onion not needed: drop it')
                motion_goals = am.place_obj_on_counter_actions(state)
                motion_goals = self.sometimes_make_wrong_decision(motion_goals, state, counter_objects, am,
                                                                  current_goal='drop_onion',
                                                                  pot_states_dict_temp=pot_states_dict)
        elif soups_need_onions == 1:
            # If closer than the other player, then deliver the onion

            own_min_cost, others_min_cost = self.find_min_cost_of_achieving_goal(player, default_motion_goals,
                                                                                 other_player, state)
            if other_player.has_object() and (other_player.get_object().name == 'onion') \
                    and (others_min_cost < own_min_cost):

                if number_of_pots == 2:
                    # Two pots but only 1 needs onion: this means a dish is needed!
                    logging.info('Two pots but only 1 needs onion: this means a dish is needed!')
                    motion_goals = am.place_obj_on_counter_actions(state)
                    motion_goals = self.sometimes_make_wrong_decision(motion_goals, state, counter_objects, am,
                                                                      current_goal='drop_onion_alt',
                                                                      pot_states_dict_temp=pot_states_dict)
                elif number_of_pots == 1:
                    logging.info('Only one pot, which needs an onion, but other is closer. So do the next task')
                    # Do the next task
                    tasks = self.OLD_tasks_to_do(state)

                    if tasks[0][1] == 'fetch_onion':  # tasks[0][1] is the next task (when there's one pot)
                        logging.info('Next task: deliver onion')
                        motion_goals = default_motion_goals
                        motion_goals = self.sometimes_make_wrong_decision(motion_goals, state, counter_objects,
                                                                          am, current_goal='use_onion')

                    elif tasks[0][1] == 'fetch_dish':
                        logging.info('Next task: dish needed')
                        motion_goals = am.place_obj_on_counter_actions(state)
                        motion_goals = self.sometimes_make_wrong_decision(motion_goals, state, counter_objects,
                                                                          am, current_goal='drop_onion_alt',
                                                                          pot_states_dict_temp=pot_states_dict)
                else:
                    raise ValueError('More pots than expected')

            else:
                # Deliver onion
                motion_goals = default_motion_goals
                motion_goals = self.sometimes_make_wrong_decision(motion_goals, state, counter_objects, am,
                                                                  current_goal='use_onion')
        elif soups_need_onions == 2:
            # Deliver onion regardless of other player
            motion_goals = am.put_onion_in_pot_actions(pot_states_dict)
            motion_goals = self.sometimes_make_wrong_decision(motion_goals, state, counter_objects, am,
                                                              current_goal='use_onion')

        return motion_goals, temp_dont_drop

    def revise_pickup_soup_goal_considering_other_player_and_noise(self,
                                                                   count_soups_nearly_ready, number_of_pots,
                                                                   temp_dont_drop, state, counter_objects, am,
                                                                   pot_states_dict, player, default_motion_goals,
                                                                   other_player):
        """Revise what the best goal is, depending on whether the other player is in a better/worse position to
                achieve the default goal. Noise refers to making wrong decisions with a given prob."""

        if count_soups_nearly_ready == 0:

            if (number_of_pots == 1) and (self.dont_drop == True):
                logging.info('Got dish because its the next task: just do nothing')
                temp_dont_drop = True  # Don't drop the dish
                motion_goals = state.players[self.agent_index].pos_and_or
                motion_goals = self.sometimes_make_wrong_decision(motion_goals, state, counter_objects, am,
                                                                  current_goal='hold_dish',
                                                                  pot_states_dict_temp=pot_states_dict)
            else:
                logging.info('Got dish but no soups to fetch')
                motion_goals = am.place_obj_on_counter_actions(state)
                motion_goals = self.sometimes_make_wrong_decision(motion_goals, state, counter_objects, am,
                                                                  current_goal='drop_dish',
                                                                  pot_states_dict_temp=pot_states_dict)

        elif count_soups_nearly_ready == 1:
            # If closer than the other player, then fetch the soup

            own_min_cost, others_min_cost = self.find_min_cost_of_achieving_goal(player, default_motion_goals,
                                                                                 other_player, state)

            if other_player.has_object() and (other_player.get_object().name == 'dish') \
                    and (others_min_cost < own_min_cost):

                logging.info('Other player has dish and is closer to a nearly ready soup')
                motion_goals = am.place_obj_on_counter_actions(state)
                motion_goals = self.sometimes_make_wrong_decision(motion_goals, state, counter_objects, am,
                                                                  current_goal='drop_dish_alt',
                                                                  pot_states_dict_temp=pot_states_dict)

            else:
                # Pickup soup with dish:
                motion_goals = default_motion_goals
                motion_goals = self.sometimes_make_wrong_decision(motion_goals, state, counter_objects, am,
                                                                  current_goal='use_dish')

        elif count_soups_nearly_ready == 2:
            # Collect soup regardless
            motion_goals = default_motion_goals
            motion_goals = self.sometimes_make_wrong_decision(motion_goals, state, counter_objects, am,
                                                              current_goal='use_dish')

        else:
            raise ValueError()

        return motion_goals, temp_dont_drop

    # === Depreciated sections of ACM === #

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



#============ DEPRECIATED =========================#

#
# class SimpleComplementaryModel(Agent):
#     """
#     Bulit on greedy human model
#     - added a simple heuristic that's used to factor in other player's expected moves
#     - motion goals can be retained rather than re-finding goals every timestep
#
#     """
#
#     def __init__(self, mlp, player_index, perseverance=0.5, teamwork=1, retain_goals=0.5):
#         self.mlp = mlp
#         self.agent_index = player_index
#         self.mdp = self.mlp.mdp
#         self.prev_state = None
#         self.timesteps_stuck = 0  # Count how many times there's a clash with the other player
#         self.dont_drop = False
#         self.current_motion_goal = None
#
#         # "Personality" parameters (within 0 to 1)
#         self.perseverance = perseverance  # perseverance = 1 means the agent always tries to go where it want to do, even when it's stuck
#         self.teamwork = teamwork  # teamwork = 0 should make this agent v similar to GreedyHuman
#         self.retain_goals = retain_goals  # Prob of keeping the previous goal each timestep (rather than re-calculating)
#
#
#     def action(self, state):
#
#         logging.info('Player: {}'.format(self.agent_index))
#         # Get motion_goals if i) self.current_motion_goal == None; ii) with prob = 1 - self.retain_goals;
#         # iii) if stuck for 2 steps (??); iv) if reached goal
#         # TODO: modify the getting stuck condition, e.g. include the goal to move out the way as a (different type of)
#         #  goal to be saved (see notes)
#         rand = random.random()
#         if self.current_motion_goal == None or rand > self.retain_goals \
#                 or self.timesteps_stuck > 0 or self.current_motion_goal == state.players_pos_and_or[self.agent_index]:
#
#             logging.info('Getting new goal:')
#             # logging.info('rand: {}, retain_goals: {}, stuck: {}, current goal: {}, current pos or: {}'
#             #       .format(rand, self.retain_goals, self.timesteps_stuck,
#             #               self.current_motion_goal, state.players_pos_and_or[self.agent_index]))
#
#             motion_goals = self.ml_action(state)
#
#             # Once we have identified the motion goals for the medium
#             # level action we want to perform, select the one with lowest cost
#             start_pos_and_or = state.players_pos_and_or[self.agent_index]
#             min_cost = np.Inf
#             best_action = None
#             for goal in motion_goals:
#                 action_plan, _, plan_cost = self.mlp.mp.get_plan(start_pos_and_or, goal)
#                 if plan_cost < min_cost:
#                     best_action = action_plan[0]
#                     min_cost = plan_cost
#                     best_goal = goal
#             # Save motion goal:
#             self.current_motion_goal = best_goal
#             #TODO: Doesn't this need to be [best_goal]?? Seems to work though?
#
#         else:
#
#             logging.info('Keeping old goal:')
#             # logging.info('rand: {}, retain_goals: {}, stuck: {}, current goal: {}, current pos or: {}'
#             #       .format(rand, self.retain_goals, self.timesteps_stuck,
#             #               self.current_motion_goal, state.players_pos_and_or[self.agent_index]))
#
#             # Use previous goal
#             motion_goals = self.current_motion_goal
#             # Find action for this goal:
#             start_pos_and_or = state.players_pos_and_or[self.agent_index]
#             action_plan, _, _ = self.mlp.mp.get_plan(start_pos_and_or, motion_goals)
#             best_action = action_plan[0]
#
#
#         """If the agent is stuck, then take an alternative action with a probability based on the time stuck and the
#         agent's "perseverance". Note: We consider an agent stuck if their whole state is unchanged (but this misses the
#         case when they try to move and do change direction but can't move <-- here the state changes & they're stuck).
#         Also, there exist cases when the state doesn't change but they're not stuck, they just can't complete their action
#         (e.g. on unident they could try to use an onion but the other player has already filled the pot)"""
#         if self.prev_state is not None and state.players[self.agent_index] == self.prev_state.players[self.agent_index]:
#             self.timesteps_stuck += 1
#             take_alternative = self.take_alternative_action()
#             # logging.info('Player {} timesteps stuck: {}'.format(self.agent_index, self.timesteps_stuck))
#             if take_alternative:
#                  # logging.info('Taking alternative action!')
#                  # Select an action at random that would change the player positions if the other player were not to move
#                  if self.agent_index == 0:
#                      joint_actions = list(itertools.product(Action.ALL_ACTIONS, [Action.STAY]))
#                  elif self.agent_index == 1:
#                      joint_actions = list(itertools.product([Action.STAY], Action.ALL_ACTIONS))
#                  else:
#                      raise ValueError("Player index not recognized")
#
#                  unblocking_joint_actions = []
#                  for j_a in joint_actions:
#                      new_state, _, _ = self.mlp.mdp.get_state_transition(state, j_a)
#                      if new_state.player_positions != self.prev_state.player_positions:
#                          unblocking_joint_actions.append(j_a)
#
#                  """Prefer adjacent actions if available:"""
#                  # Adjacent actions only exist if best_action is N, S, E, W
#                  if best_action in Direction.ALL_DIRECTIONS:
#
#                      # Find the adjacent actions:
#                      if self.agent_index == 0:
#                          joint_adjacent_actions = list(itertools.product(Direction.get_adjacent_directions(best_action),
#                                                                          [Action.STAY]))
#                      elif self.agent_index == 1:
#                          joint_adjacent_actions = list(itertools.product([Action.STAY],
#                                                                          Direction.get_adjacent_directions(best_action)))
#                      else:
#                          raise ValueError("Player index not recognized")
#
#                      # If at least one of the adjacent actions is in the set of unblocking_joint_actions, then select these:
#                      if (joint_adjacent_actions[0] in unblocking_joint_actions
#                              or joint_adjacent_actions[1] in unblocking_joint_actions):
#                          preferred_unblocking_joint_actions = []
#                          # There are only ever two adjacent actions:
#                          if joint_adjacent_actions[0] in unblocking_joint_actions:
#                              preferred_unblocking_joint_actions.append(joint_adjacent_actions[0])
#                          if joint_adjacent_actions[1] in unblocking_joint_actions:
#                              preferred_unblocking_joint_actions.append(joint_adjacent_actions[1])
#                      elif (joint_adjacent_actions[0] not in unblocking_joint_actions
#                            and joint_adjacent_actions[1] not in unblocking_joint_actions):
#                          # No adjacent actions in the set of unblocking_joint_actions, so keep these actions
#                          preferred_unblocking_joint_actions = unblocking_joint_actions
#                      else:
#                          raise ValueError("Binary truth value is neither true nor false")
#
#                  # If adjacent actions don't exist then keep unblocking_joint_actions as it is
#                  else:
#                      preferred_unblocking_joint_actions = unblocking_joint_actions
#
#                  best_action = preferred_unblocking_joint_actions[
#                      np.random.choice(len(preferred_unblocking_joint_actions))][self.agent_index]
#                  # Note: np.random isn't actually random!
#
#         else:
#             self.timesteps_stuck = 0  # Reset to zero if prev & current player states aren't the same (they're not stuck)
#
#         # NOTE: Assumes that calls to action are sequential
#         self.prev_state = state
#         return best_action
#
#
#     def ml_action(self, state):
#         """Selects a medium level action for the current state"""
#         player = state.players[self.agent_index]
#         other_player = state.players[1 - self.agent_index]
#         am = self.mlp.ml_action_manager
#
#         counter_objects = self.mlp.mdp.get_counter_objects_dict(state, list(self.mlp.mdp.terrain_pos_dict['X']))
#         pot_states_dict = self.mlp.mdp.get_pot_states(state)
#
#         #next_order = state.order_list[0]  # Assuming soup only for now
#
#         # (removed tomato)
#         ready_soups = pot_states_dict['onion']['ready']
#         cooking_soups = pot_states_dict['onion']['cooking']
#
#         soup_nearly_ready = len(ready_soups) > 0 or len(cooking_soups) > 0
#         count_soups_nearly_ready = len(ready_soups) + len(cooking_soups)
#         other_has_dish = other_player.has_object() and other_player.get_object().name == 'dish'
#         other_has_onion = other_player.has_object() and other_player.get_object().name == 'onion'
#
#         number_of_pots = self.mlp.mdp.get_pot_locations().__len__()
#
#         temp_dont_drop = False
#
#         if not player.has_object():
#
#             # Only get the dish if the soup is nearly ready
#             if soup_nearly_ready:
#
#                 """Get the dish... unless other player is better placed"""
#
#                 default_motion_goals = am.pickup_dish_actions(counter_objects)
#
#                 # Remove invalid goals:
#                 own_motion_goals = list(filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, goal), default_motion_goals))
#                 others_motion_goals = list(filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(other_player.pos_and_or, goal), default_motion_goals))
#
#                 own_min_cost = self.find_min_plan_cost(own_motion_goals, state, self.agent_index)
#                 others_min_cost = self.find_min_plan_cost(others_motion_goals, state, 1 - self.agent_index)
#
#                 logging.info('Player {}'.format(self.agent_index))
#                 logging.info('own min cost: {}'.format(own_min_cost))
#                 logging.info('other\'s min cost: {}'.format(others_min_cost))
#
#                 # Get the dish if the other player doesn't have a dish or they do have a dish but are further away
#                 if (not other_has_dish) and (others_min_cost > own_min_cost):  #TODO: consider having >=
#
#                     logging.info('Getting the dish: assume in better position that other player')
#                     motion_goals = default_motion_goals
#
#                 elif other_has_dish or ((not other_has_dish) and (others_min_cost <= own_min_cost)):
#                     # Now find the *next* task that needs to be done, and do that
#                     # Simple way of doing this: IF only one nearly_ready pot at the top of the list THEN get onion
#                     # IF both lists start with pot, THEN get a dish (worry about where to put it later!)
#
#                     logging.info('Assume other player is in better position, so do the next task on the list')
#                     tasks = self.tasks_to_do(state)
#
#                     # ASSUMING TWO POTS MAX
#                     number_of_pots = self.mlp.mdp.get_pot_locations().__len__()
#                     if number_of_pots == 1:
#                         # Next task will always be onion
#                         logging.info('Next task: onion')
#                         motion_goals = am.pickup_onion_actions(counter_objects)
#
#                         logging.info('Onion shouldnt be dropped, so set temp_dont_drop = True')
#                         temp_dont_drop = True
#
#                     elif number_of_pots == 2:
#
#                         if (tasks[0][0] == 'fetch_onion') or (tasks[1][0] == 'fetch_onion'):
#
#                             logging.info('Next task: onion')
#                             # Next task will be onion
#                             motion_goals = am.pickup_onion_actions(counter_objects)
#
#                         elif (tasks[0][0] == 'fetch_dish') and (tasks[1][0] == 'fetch_dish'):
#
#                             logging.info('Next task: dish')
#                             # Next task will be dish
#                             motion_goals = am.pickup_dish_actions(counter_objects)
#
#                         else:
#                             raise ValueError('Failed logic')
#                             #TODO: is there a special type of logical error??
#
#                 else:
#                     raise ValueError('Failed logic')
#
#                 if random.random() > self.teamwork:
#                     # Overwrite consideration of other player
#                     logging.info('Overwriting motion_goal with the greed option')
#                     motion_goals = default_motion_goals
#
#
#             elif not soup_nearly_ready:
#
#                 """If there are two pots, then we can only get inside this 'if' when both pots have onion first on the list
#                 of tasks, so we can always get an onion! For 1 pot then just look what's next on the list"""
#
#                 default_motion_goals = am.pickup_onion_actions(counter_objects)
#
#                 # ASSUMING TWO POTS MAX THROUGHOUT
#                 if number_of_pots == 1:
#
#                     # Remove invalid goals of getting an onion:
#                     own_motion_goals = list(
#                         filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, goal),
#                                default_motion_goals))
#                     others_motion_goals = list(
#                         filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(other_player.pos_and_or, goal),
#                                default_motion_goals))
#
#                     own_min_cost = self.find_min_plan_cost(own_motion_goals, state, self.agent_index)
#                     others_min_cost = self.find_min_plan_cost(others_motion_goals, state, 1 - self.agent_index)
#
#                     logging.info('Player {}'.format(self.agent_index))
#                     logging.info('own min cost: {}'.format(own_min_cost))
#                     logging.info('other\'s min cost: {}'.format(others_min_cost))
#
#                     # Get the onion if the other player doesn't have an onion or they do have an onion but are further away
#                     if (not other_has_onion) and (others_min_cost >= own_min_cost):  # TODO: make it random if others_min_cost = own_min_cost ?
#
#                         logging.info('Getting the onion: assume in better position that other player')
#                         motion_goals = default_motion_goals
#
#                     elif other_has_onion or ((not other_has_onion) and (others_min_cost <= own_min_cost)):
#                         # Now find the *next* task that needs to be done, and do that
#
#                         logging.info('Assume other player is in better position, so do the next task on the list')
#                         tasks = self.tasks_to_do(state)
#
#                         if tasks[0][1] == 'fetch_onion':  # tasks[0][1] is the next task (when there's one pot)
#                             logging.info('Next task: onion')
#                             motion_goals = am.pickup_onion_actions(counter_objects)
#
#                         elif tasks[0][1] == 'fetch_dish':
#                             logging.info('Next task: dish')
#                             motion_goals = am.pickup_onion_actions(counter_objects)
#
#                             logging.info('Dish shouldnt be dropped, so set temp_dont_drop = True')
#                             temp_dont_drop = True
#
#                 elif number_of_pots == 2:
#                     # Get an onion regardless: there's always a pot needing an onion (because 'not soup_nearly_ready')
#                     motion_goals = default_motion_goals
#
#                 else:
#                     raise ValueError('Assuming 1 or 2 pots, but there are more??')
#
#                 if random.random() > self.teamwork:
#                     # Overwrite consideration of other player
#                     logging.info('Overwriting motion_goal with the greed option')
#                     motion_goals = default_motion_goals
#
#             else:
#                 raise ValueError('Failed logic')
#
#             # If there's a soup on the counter, then override other goals and get the soup
#             # UNLESS other player (strictly) is closer
#             if 'soup' in counter_objects:
#
#                 default_motion_goals = am.pickup_counter_soup_actions(counter_objects)
#
#                 # Remove invalid goals of getting an onion:
#                 own_motion_goals = list(
#                     filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, goal),
#                            default_motion_goals))
#                 others_motion_goals = list(
#                     filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(other_player.pos_and_or, goal),
#                            default_motion_goals))
#
#                 own_min_cost = self.find_min_plan_cost(own_motion_goals, state, self.agent_index)
#                 others_min_cost = self.find_min_plan_cost(others_motion_goals, state, 1 - self.agent_index)
#
#                 if others_min_cost > own_min_cost:
#
#                     logging.info('Soup on counter and other player is futher away: get soup')
#                     motion_goals = default_motion_goals
#
#                 elif others_min_cost <= own_min_cost:
#
#                     logging.info('Soup on counter but other closer')
#                     logging.info('Own cost = {}'.format(own_min_cost))
#
#
#
#         elif player.has_object():
#
#             # Determine if any soups need onions
#             pots_empty = len(pot_states_dict['empty'])
#             pots_partially_full = len(pot_states_dict['onion']['partially_full'])
#             soups_need_onions = pots_empty + pots_partially_full
#
#             player_obj = player.get_object()
#
#             if player_obj.name == 'onion':
#
#                 default_motion_goals = am.put_onion_in_pot_actions(pot_states_dict)
#
#                 if soups_need_onions == 0:
#
#                     if self.dont_drop == True:
#                         # Don't drop the onion
#                         logging.info('Got onion because its the next task: just do nothing')
#                         #TODO: unrealistic to do nothing, but the alternatives are hard to code
#                         motion_goals = state.players[self.agent_index].pos_and_or
#                         temp_dont_drop = True
#                     else:
#                         # Drop onion
#                         logging.info('Onion not needed: drop it')
#                         motion_goals = am.place_obj_on_counter_actions(state)
#
#                 elif soups_need_onions == 1:
#                     # If closer than the other player, then deliver the onion
#
#                     # Remove invalid goals of delivering an onion:
#                     own_motion_goals = list(
#                         filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, goal),
#                                default_motion_goals))
#                     others_motion_goals = list(
#                         filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(other_player.pos_and_or, goal),
#                                default_motion_goals))
#
#                     own_min_cost = self.find_min_plan_cost(own_motion_goals, state, self.agent_index)
#                     others_min_cost = self.find_min_plan_cost(others_motion_goals, state, 1 - self.agent_index)
#
#                     if other_player.has_object() and (other_player.get_object().name == 'onion') and (others_min_cost < own_min_cost):
#
#                         if number_of_pots == 2:
#                             # Two pots but only 1 needs onion: this means a dish is needed!
#                             logging.info('Two pots but only 1 needs onion: this means a dish is needed!')
#                             motion_goals = am.place_obj_on_counter_actions(state)
#
#                         elif number_of_pots == 1:
#                             logging.info('Only one pot, which needs an onion, but other is closer. So do the next task')
#                             # Do the next task
#                             tasks = self.tasks_to_do(state)
#
#                             if tasks[0][1] == 'fetch_onion':  # tasks[0][1] is the next task (when there's one pot)
#                                 logging.info('Next task: deliver onion')
#                                 motion_goals = default_motion_goals
#
#                             elif tasks[0][1] == 'fetch_dish':
#                                 logging.info('Next task: dish needed')
#                                 motion_goals = am.place_obj_on_counter_actions(state)
#
#                         else:
#                             raise ValueError('More pots than expected')
#
#                     else:
#                         # Deliver onion
#                         motion_goals = default_motion_goals
#
#
#                 elif soups_need_onions == 2:
#                     # Deliver onion regardless of other player
#                     motion_goals = am.put_onion_in_pot_actions(pot_states_dict)
#
#                 if random.random() > self.teamwork:
#                     # Overwrite consideration of other player
#                     logging.info('Overwriting motion_goal with the greed option')
#                     if soups_need_onions == 0:
#                         # Player has an onion but there are no soups to put it in, then drop the onion
#                         motion_goals = am.place_obj_on_counter_actions(state)
#                     else:
#                         motion_goals = am.put_onion_in_pot_actions(pot_states_dict)
#
#
#             elif player_obj.name == 'dish':
#
#                 default_motion_goals = am.pickup_soup_with_dish_actions(pot_states_dict, only_nearly_ready=True)
#
#                 if count_soups_nearly_ready == 0:
#
#                     if (number_of_pots == 1) and (self.dont_drop == True):
#                             # Don't drop the dish
#                             logging.info('Got dish because its the next task: just do nothing')
#                             #TODO: unrealistic to do nothing, but the alternatives are hard to code
#                             motion_goals = state.players[self.agent_index].pos_and_or
#                             temp_dont_drop = True
#                     else:
#                         # Got dish but no soups to fetch
#                         logging.info('Got dish but no soups to fetch')
#                         motion_goals = am.place_obj_on_counter_actions(state)
#
#                 elif count_soups_nearly_ready == 1:
#                     # If closer than the other player, then fetch the soup
#
#                     # Remove invalid goals:
#                     own_motion_goals = list(
#                         filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, goal),
#                                default_motion_goals))
#                     others_motion_goals = list(
#                         filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(other_player.pos_and_or, goal),
#                                default_motion_goals))
#
#                     own_min_cost = self.find_min_plan_cost(own_motion_goals, state, self.agent_index)
#                     others_min_cost = self.find_min_plan_cost(others_motion_goals, state, 1 - self.agent_index)
#
#                     if other_player.has_object() and (other_player.get_object().name == 'dish') and (others_min_cost < own_min_cost):
#
#                         logging.info('Other player has dish and is closer to a nearly ready soup')
#
#                         #TODO: this if loop not needed
#                         if number_of_pots == 2:
#                             # Two pots but only 1 needs dish: so get an onion!
#                             logging.info('Two pots but only 1 needs dish: this means an onion is needed!')
#                             motion_goals = am.place_obj_on_counter_actions(state)
#
#                         elif number_of_pots == 1:
#                             logging.info('Only one pot, which needs a dish, but other is closer. Next task is always onion')
#                             motion_goals = am.place_obj_on_counter_actions(state)
#
#                         else:
#                             raise ValueError('More pots than expected')
#
#                     else:
#                         motion_goals = default_motion_goals
#
#                 elif count_soups_nearly_ready == 2:
#                     # Collect soup regardless
#                     motion_goals = default_motion_goals
#
#                 else:
#                     raise ValueError()
#
#                 if random.random() > self.teamwork:
#                     # Overwrite consideration of other player
#                     logging.info('Overwriting motion_goal with the greed option')
#                     if not soup_nearly_ready:
#                         # Player has a dish but there are no longer any "nearly ready" soups: drop the dish
#                         motion_goals = am.place_obj_on_counter_actions(state)
#                     elif soup_nearly_ready:
#                         motion_goals = am.pickup_soup_with_dish_actions(pot_states_dict, only_nearly_ready=True)
#
#
#             elif player_obj.name == 'soup':
#                 # Deliver soup whatever the other player is doing
#                 motion_goals = am.deliver_soup_actions()
#
#             else:
#                 raise ValueError()
#         else:
#             raise ValueError('Player has AND does not have an object!')
#
#
#
#         # OLd code:
#         #
#         # # Determine if any soups are "nearly ready", meaning that they are cooking or ready
#         # if next_order == 'any':
#         #     ready_soups = pot_states_dict['onion']['ready'] + pot_states_dict['tomato']['ready']
#         #     cooking_soups = pot_states_dict['onion']['cooking'] + pot_states_dict['tomato']['cooking']
#         # else:
#         #     ready_soups = pot_states_dict[next_order]['ready']
#         #     cooking_soups = pot_states_dict[next_order]['cooking']
#         #
#         # soup_nearly_ready = len(ready_soups) > 0 or len(cooking_soups) > 0
#         # #other_has_dish = other_player.has_object() and other_player.get_object().name == 'dish'  <-- no longer used
#
#         # if not player.has_object():
#         #
#         #     if soup_nearly_ready:  # PK removed "and not other_has_dish"
#         #         motion_goals = am.pickup_dish_actions(counter_objects)
#         #     else:
#         #         next_order = None
#         #         if len(state.order_list) > 1:
#         #             next_order = state.order_list[1]
#         #
#         #         if next_order == 'onion':
#         #             motion_goals = am.pickup_onion_actions(counter_objects)
#         #         elif next_order == 'tomato':
#         #             motion_goals = am.pickup_tomato_actions(counter_objects)
#         #         elif next_order is None or next_order == 'any':
#         #             motion_goals = am.pickup_onion_actions(counter_objects) + am.pickup_tomato_actions(counter_objects)
#
#             # # If there's a soup on the counter, then override other goals and get the soup
#             # if 'soup' in counter_objects:
#             #     motion_goals = am.pickup_counter_soup_actions(counter_objects)
#
#         # else:
#         #
#         #     # Determine if any soups need onions
#         #     pots_empty = len(pot_states_dict['empty'])
#         #     pots_partially_full = len(pot_states_dict['onion']['partially_full'])
#         #     soup_needs_onions = pots_empty > 0 or pots_partially_full > 0
#         #
#         #     player_obj = player.get_object()
#         #
#         #     if player_obj.name == 'onion':
#         #         if not soup_needs_onions:
#         #             # If player has an onion but there are no soups to put it in, then drop the onion!
#         #             motion_goals = am.place_obj_on_counter_actions(state)
#         #         else:
#         #             motion_goals = am.put_onion_in_pot_actions(pot_states_dict)
#         #
#         #     elif player_obj.name == 'tomato':
#         #         motion_goals = am.put_tomato_in_pot_actions(pot_states_dict)
#         #
#         #     elif player_obj.name == 'dish':
#         #         # If player has a dish but there are no longer any "nearly ready" soups, then drop the dish!
#         #         if not soup_nearly_ready:
#         #             motion_goals = am.place_obj_on_counter_actions(state)
#         #         elif soup_nearly_ready:
#         #             motion_goals = am.pickup_soup_with_dish_actions(pot_states_dict, only_nearly_ready=True)
#         #
#         #     elif player_obj.name == 'soup':
#         #         motion_goals = am.deliver_soup_actions()
#         #
#         #     else:
#         #         raise ValueError()
#
#         # Remove invalid goals:
#         motion_goals = list(filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, goal), motion_goals))
#
#         # If no goals, then just go to nearest feature
#         if len(motion_goals) == 0:
#             motion_goals = am.go_to_closest_feature_actions(player)
#             motion_goals = list(filter(lambda goal: self.mlp.mp.is_valid_motion_start_goal_pair(player.pos_and_or, goal), motion_goals))
#             assert len(motion_goals) != 0
#
#         if temp_dont_drop == True:
#             self.dont_drop = True
#         else:
#             self.dont_drop = False
#
#         return motion_goals
#
#     def take_alternative_action(self):
#         """This first gives Prob(taking alternative action)=1 if perseverance=0 and Prob=0 if perseverance=1. Otherwise,
#         e.g. perseverance=_, num_items, _ = state.get_object(self.mlp.mdp.get_pot_locations()[0]).state
#             #current_list = full_list[num_items:number_of_tasks]
#
#         # if number_of_pots == 2:
#             #0.5, for the first timestep Prob~0.5, then Prob-->1 quickly as time increases. Then using this prob
#         it randomly determines if the player should take an alternative action."""
#
#         Prob_taking_alternative = 1 - (1 / (np.exp(self.timesteps_stuck*10))) ** (1 - self.perseverance ** (1 / 10))
#         rand = random.random()
#         take_alternative = rand < Prob_taking_alternative
#
#         # logging.info('time stuck: {}, perserv: {}, Prob_taking_alt: {}, Taking alt: {}'.format(self.timesteps_stuck, self.perseverance, Prob_taking_alternative, take_alternative))
#
#         return take_alternative
#
#     def tasks_to_do(self, state):
#         """
#         :return: tasks = a list of list of tasks to be done, where tasks[i] is a list of what need doing for pot i (where
#         pot i is the pot at location get_pot_locations()[i])
#         """
#
#         # OLD
#         # if number_of_pots == 1:
#         #     order_length = state.order_list.__len__()
#         #     full_list = ['fetch_onion','fetch_onion','fetch_onion','fetch_dish']*order_length
#         #     pot_pos = self.mlp.mdp.get_pot_locations()[0]
#         #     if not state.has_object(pot_pos):
#         #         num_items = 0
#         #     else:
#         #         _, num_items, _ = state.get_object(self.mlp.mdp.get_pot_locations()[0]).state
#         #     current_list = full_list[num_items:num_items+number_of_tasks]
#
#
#         number_of_pots = self.mlp.mdp.get_pot_locations().__len__()
#         order_length = state.order_list.__len__()
#         initial_list_for_each_pot = ['fetch_onion', 'fetch_onion', 'fetch_onion', 'fetch_dish'] * order_length
#
#         # Find how many items in each pot, then make a new list of lists of what actually needs doing for each pot
#         tasks = []
#         for pot in range(number_of_pots):
#             # Currently state.get_object is false if the pot doesn't have any onions! #TODO: Rectify this??
#             pot_pos = self.mlp.mdp.get_pot_locations()[pot]
#             if not state.has_object(pot_pos):
#                 num_items = 0
#             else:
#                 _, num_items, _ = state.get_object(pot_pos).state
#
#             tasks.append(initial_list_for_each_pot[num_items:])
#
#         return tasks
#
#     def find_min_plan_cost(self, motion_goals, state, temp_player_index):
#         """
#         Given some motion goals, find the cost for the lowest cost goal
#         :param motion_goals:
#         :param state:
#         :param temp_player_index:
#         :return:
#         """
#
#         start_pos_and_or = state.players_pos_and_or[temp_player_index]
#
#         if len(motion_goals) == 0:
#             min_cost = np.Inf
#         else:
#             min_cost = np.Inf
#             # best_action = None
#             for goal in motion_goals:
#                 _, _, plan_cost = self.mlp.mp.get_plan(start_pos_and_or, goal)
#                 if plan_cost < min_cost:
#                     # best_action = action_plan[0]
#                     min_cost = plan_cost
#                     # best_goal = goal
#
#         return min_cost

# class GreedyHeuristicAgent(Agent):
#     """
#     (DEPRECIATED)

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
