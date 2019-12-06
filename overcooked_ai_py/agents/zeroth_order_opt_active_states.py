import time
from argparse import ArgumentParser
from human_aware_rl.human.process_dataframes import get_trajs_from_data
from human_aware_rl.pbt.pbt_hms import ToMAgent
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import GreedyHumanModel_pk
from overcooked_ai_py.planning.planners import MediumLevelPlanner
import logging, pickle
import numpy as np
from collections import Counter
# np.seterr(divide='ignore', invalid='ignore')  # Suppress error about diving by zero

"""
Here we only optimise the HM parameter on states for which the data acts. So if the data acts, we force the HM to 
also act, then compare their actions.
"""

# Helper functions:

def choose_hm_actions(expert_trajs, hm_agent, num_ep_to_use):
    """
    Take a human model with given parameters, then use this to choose one action for every state in the data.
    Correction: now we only find one action for each state in which the data acts!
    :return: hm_actions, a list of lists of actions chosen by the HM
    """

    hm_actions = []
    actions_from_data = expert_trajs['ep_actions']

    # For each episode we want to use
    for i in range(num_ep_to_use):

        hm_actions_this_ep = []

        # Reset histories in the HM:
        hm_agent.agent_index =  expert_trajs['ep_agent_idxs'][i]
        hm_agent.prev_state = None
        hm_agent.timesteps_stuck = 0  # Count how many times there's a clash with the other player
        hm_agent.dont_drop = False
        hm_agent.prev_motion_goal = None
        hm_agent.prev_best_action = None
        hm_agent.GHM = GreedyHumanModel_pk(hm_agent.mlp, player_index=1 - hm_agent.agent_index)

        # For each state in the episode trajectory:
        for j in range(actions_from_data[i].__len__()):

            # Only let the HM take an action when the data also acts. Otherwise force HM action to (0,0) also
            if actions_from_data[i][j] != (0,0):

                current_state = expert_trajs['ep_observations'][i][j]
                # The state seems to be missing an order list. Manually add the start_order_list:
                current_state.order_list = hm_agent.mdp.start_order_list
                #TODO: Fix this properly

                # # Print the state to view the game:
                # overcooked_env = OvercookedEnv(hm_agent.mdp)
                # logging.warning('i = {}, j = {}'.format(i, j))
                # overcooked_env.state = current_state
                # logging.warning(overcooked_env)

                # Force the agent to take an action
                temp_prob_pausing = hm_agent.prob_pausing
                hm_agent.prob_pausing = 0

                # Choose HM action from state
                hm_action = hm_agent.action(current_state)
                # This also automatically updates hm_agent.timesteps_stuck, hm_agent.dont_drop,
                # hm_agent.prev_motion_goal, hm_agent.prev_state

                # Then reset the prob_pausing:
                hm_agent.prob_pausing = temp_prob_pausing

                # Set the prev action from the data, but only if there's already a motion goal
                if hm_agent.prev_motion_goal != None:
                    hm_agent.prev_best_action = actions_from_data[i][j]

                # Print everything after:
                logging.warning('Action from HM: {}; Action from data: {}'.format(hm_action, actions_from_data[i][j]))
                logging.warning('HM prev_motion_goal: {}; HM dont_drop: {}'.format(hm_agent.prev_motion_goal,
                                                                                   hm_agent.dont_drop))
                logging.warning('HM time stuck: {}'.format(hm_agent.timesteps_stuck))

            else:
                hm_action = (0,0)

            hm_actions_this_ep.append(hm_action)

        hm_actions.append(hm_actions_this_ep)

    return hm_actions

def find_hm_probs_action_in_state(multi_hm_agent, actions_from_data, num_ep_to_use, expert_trajs):
    """
    Find the prob that hm takes action a in state s. BUT only for the actions the data actually takes (we don't need
    the probs for the other actions, as the loss is zero for these.
    :param multi_hm_agent:
    :param actions_from_data:
    :param num_ep_to_use:
    :param expert_trajs:
    :return:
    """

    multi_hm_actions = []

    # For each agent
    for hm_agent in multi_hm_agent:

        # Find all actions by this agent:
        hm_actions = choose_hm_actions(expert_trajs, hm_agent, num_ep_to_use)
        multi_hm_actions.append(hm_actions)

    # Now have all actions for all agents

    # List of lists of zeros:
    hm_probs_action_in_state = [[0] * actions_from_data[i].__len__() for i in range(num_ep_to_use)]

    # For each episode we want to use
    for i in range(num_ep_to_use):
        # For each state in the episode trajectory:
        for j in range(actions_from_data[i].__len__()):

            # For each agent
            for hm_actions in multi_hm_actions:
            #TODO: Should really swap this "for" with the "if" below

                # Only work out the probs for states where the data acts:
                if actions_from_data[i][j] != (0,0):

                    # If agent chooses the same action as the data, then count this
                    if actions_from_data[i][j] == hm_actions[i][j]:

                        # 1 / number of agents. Therefore if all agents act correctly then prob will be 1
                        hm_probs_action_in_state[i][j] += 1/multi_hm_agent.__len__()

                # data gives (0,0):
                else:
                    # Set to -1 to signal that we're not using this probability
                    hm_probs_action_in_state[i][j] = -1
                    # Both the data and the agent should give (0,0):
                    assert actions_from_data[i][j] == hm_actions[i][j]

            # Force prob to be 0.01 minimum (otherwise we get infinities in the cross entropy):
            if hm_probs_action_in_state[i][j] == 0:
                hm_probs_action_in_state[i][j] = 0.01

    return hm_probs_action_in_state

def find_prob_not_acting(actions_from_data, num_ep_to_use):

    count_number_not_acting = 0
    count_total_states = 0

    # For each episode we want to use
    for i in range(num_ep_to_use):
        # For each state in the episode trajectory:
        for j in range(actions_from_data[i].__len__()):

            count_total_states += 1

            if actions_from_data[i][j] == (0,0):
                # Count for how many of the states the data doesn't act:
                count_number_not_acting += 1

    prob_data_doesnt_act = count_number_not_acting / count_total_states
    number_states_with_acting = count_total_states - count_number_not_acting

    return prob_data_doesnt_act, number_states_with_acting

def find_cross_entropy_loss(actions_from_data, expert_trajs, multi_hm_agent, num_ep_to_use):
    """
    ...?
    :param expert_trajs:
    :param multi_hm_agent:
    :param num_ep_to_use:
    :return:
    """
    # Find Prob_HM(action|state) for all actions chosen by the data
    hm_probs_action_in_state = find_hm_probs_action_in_state(multi_hm_agent, actions_from_data, num_ep_to_use,
                                                          expert_trajs)

    loss = 0

    # For each episode we want to use
    for i in range(num_ep_to_use):
        # For each state in the episode trajectory:
        for j in range(actions_from_data[i].__len__()):

            # Only add to the loss if it's a state for which the data acts:
            if actions_from_data[i][j] != (0,0):
                loss += np.log(hm_probs_action_in_state[i][j])/np.log(0.01)  # Normalise by 0.01, so that loss_ij=1 max
            else:
                assert hm_probs_action_in_state[i][j] == -1

    return loss

def two_most_frequent(List):
    occurence_count = Counter(List)
    second_action = None
    if len(occurence_count) > 1:
        second_action = occurence_count.most_common(2)[1][0]
    return occurence_count.most_common(1)[0][0], second_action

def find_top_12_accuracy(actions_from_data, expert_trajs, multi_hm_agent, num_ep_to_use, number_states_with_acting):
    """
    Find top-1 (top-2) accuracy, which is the proportion of states in which the HM's most likely (2nd most likely)
    action equals the action from the data
    """

    multi_hm_actions = []
    # For each agent
    for hm_agent in multi_hm_agent:
        # Find all actions by this agent:
        hm_actions = choose_hm_actions(expert_trajs, hm_agent, num_ep_to_use)
        multi_hm_actions.append(hm_actions)
    # Now have all actions for all agents

    count_top_1 = 0
    count_top_2 = 0

    # For each state:
    for i in range(num_ep_to_use):
        for j in range(actions_from_data[i].__len__()):

            # Only consider states where the data takes an action:
            if actions_from_data[i][j] != (0, 0):

                # Change format of multi_hm_actions:
                list_actions = []
                for k in range(len(multi_hm_actions)):
                    list_actions.append(multi_hm_actions[k][i][j])

                #TODO: Here we're ignoring draws between two actions:
                top_action, second_action = two_most_frequent(list_actions)

                assert top_action != second_action

                # If the top/2nd action is the same as the data, count it:
                if top_action == actions_from_data[i][j]:
                    count_top_1 += 1
                    count_top_2 += 1
                elif second_action == actions_from_data[i][j]:
                    count_top_2 += 1

    top_1_acc = count_top_1 / number_states_with_acting
    top_2_acc = count_top_2 / number_states_with_acting

    assert top_2_acc >= top_1_acc

    return top_1_acc, top_2_acc

def shift_by_epsilon(params, epsilon):
    """
    Shift the PERSON_PARAMS_HM by epsilon, except RATIONALITY_COEFF, which isn't between 0 and 1 so is shifted further
    :return: PERSON_PARAMS_HMeps that've been shifted
    """

    PERSON_PARAMS_HM = params['PERSON_PARAMS_HM']
    PERSON_PARAMS_HMeps = params['PERSON_PARAMS_HMeps']
    PERSON_PARAMS_FIXED = params['PERSON_PARAMS_FIXED']

    for i, pparam in enumerate(PERSON_PARAMS_HM):

        if pparam in PERSON_PARAMS_FIXED:
            pass  # Don't change this parameter
        else:
            PERSON_PARAMS_HMeps[pparam+'eps'] = PERSON_PARAMS_HM[pparam] + epsilon[i]
        # print('param before: {}; param shifted raw: {}'.format(PERSON_PARAMS_HM[pparam], PERSON_PARAMS_HMeps[pparam+'eps']))

        # Ensure all params between 0 and 1; except RATIONALITY_COEFF which should be >= 0
        # RATIONALITY_COEFF can reasonably range from 0 to e.g. 10, so we need to shift it further!
        if pparam is 'RATIONALITY_COEFF_HM':
            PERSON_PARAMS_HMeps[pparam+'eps'] = PERSON_PARAMS_HM[pparam] + epsilon[i] * 10
            if PERSON_PARAMS_HMeps[pparam+'eps'] < 0:
                PERSON_PARAMS_HMeps[pparam+'eps'] = 0
            else: pass
        else:
            if PERSON_PARAMS_HMeps[pparam+'eps'] < 0:
                PERSON_PARAMS_HMeps[pparam+'eps'] = 0
            elif PERSON_PARAMS_HMeps[pparam+'eps'] > 1:
                PERSON_PARAMS_HMeps[pparam+'eps'] = 1
            else: pass

        # print('param before: {}; param shifted final: {}'.
        #       format(PERSON_PARAMS_HM[pparam], PERSON_PARAMS_HMeps[pparam+'eps']))

    return PERSON_PARAMS_HMeps

def shift_by_gradient(params, epsilon, delta_loss, lr):
    """
    Shift PERSON_PARAMS_HM in the direction of negative gradient, scaled by lr
    """

    PERSON_PARAMS_HM = params['PERSON_PARAMS_HM']
    PERSON_PARAMS_FIXED = params['PERSON_PARAMS_FIXED']

    for i, pparam in enumerate(PERSON_PARAMS_HM):
        # print('param before: {}'.format(PERSON_PARAMS_HM[pparam]))
        if pparam in PERSON_PARAMS_FIXED:
            # Don't change this parameter
            pass
        else:
            PERSON_PARAMS_HM[pparam] = PERSON_PARAMS_HM[pparam] + epsilon[i]*delta_loss*lr
        # print('delta_loss: {}; lr: {}; this eps: {}'.format(delta_loss, lr, epsilon[i]))
        # print('param shifted raw: {}'.format(PERSON_PARAMS_HM[pparam]))

        # Ensure all params between 0 and 1; except RATIONALITY_COEFF which should be >= 0
        # RATIONALITY_COEFF gets shifted by 10* compared to the others
        if pparam is 'RATIONALITY_COEFF_HM':
            PERSON_PARAMS_HM[pparam] = PERSON_PARAMS_HM[pparam] + epsilon[i]*delta_loss*lr*10
            if PERSON_PARAMS_HM[pparam] < 0:
                PERSON_PARAMS_HM[pparam] = 0
            else: pass
        else:
            if PERSON_PARAMS_HM[pparam] < 0:
                PERSON_PARAMS_HM[pparam] = 0
            elif PERSON_PARAMS_HM[pparam] > 1:
                PERSON_PARAMS_HM[pparam] = 1
            else: pass

        # print('param shifted final: {}'.format(PERSON_PARAMS_HM[pparam]))
    # return PERSON_PARAMS_HM

def find_gradient_and_step_multi_hm(params, mlp, expert_trajs, num_ep_to_use, lr, epsilon_sd,
                                    start_time, step_number, total_number_steps):
    """
    Same as find_gradient_and_step_single_hm except here we have multiple hms taking multiple actions, so we find
    prob(action|state) and use cross entropy loss
    :return: loss
    """

    actions_from_data = expert_trajs['ep_actions']

    hm_number = ''
    # Make multiple hm agents:
    multi_hm_agent = ToMAgent(params, hm_number).get_multi_agent(mlp)

    loss = find_cross_entropy_loss(actions_from_data, expert_trajs, multi_hm_agent, num_ep_to_use)

    # Choose random epsilon (eps) from normal dist, sd=epsilon_sd
    epsilon = np.random.normal(scale=epsilon_sd, size=params['PERSON_PARAMS_HM'].__len__())

    # Turn into random unit vector:
    if params["ensure_random_direction"]:
        print('NOTE: This only works for sd=0.01 and 8 pparams. AND I did this bit quickly so should be checked!')
        #TODO: There might be an issue that we have 8 pparams but only 6 are usually allowed to vary... this is like
        # taking a random direction then projecting onto the 6-dim subspace, which should be fine (??)

        print('ep initial length = {}'.format(np.sqrt(sum(epsilon**2))))

        # Make random vector from Gaussian sd=1
        random_gauss_vect = np.random.normal(scale=1, size=8)
        length = np.sqrt(sum(random_gauss_vect**2))
        unit_vect = random_gauss_vect / length
        # print('Unit length = {}'.format(np.sqrt(sum(unit_vect**2))))

        # Now scale by 0.023, which is the average length of an 8-dim vector from Gaussians with sd=0.01
        epsilon = 0.027*unit_vect
        print('ep final length = {}'.format(np.sqrt(sum(epsilon**2))))

    # Make PERSON_PARAMS_HM + epsilon. For rationality_coefficient, do eps*10 for now. shift_by_epsilon also ensures all
    # params are between 0 and 1:
    shift_by_epsilon(params, epsilon)

    # Find loss for new params
    hm_number = 'eps'
    multi_hm_agent_eps = ToMAgent(params, hm_number).get_multi_agent(mlp)
    loss_eps = find_cross_entropy_loss(actions_from_data, expert_trajs, multi_hm_agent_eps, num_ep_to_use)
    delta_loss = loss - loss_eps

    # Set new personality params by shifting in the direction of downhill
    shift_by_gradient(params, epsilon, delta_loss, lr)

    # What's the loss after this grad step:
    # hm_number = ''
    # multi_hm_agent = ToMAgent(params, hm_number).get_multi_agent(mlp)
    # loss_final = find_cross_entropy_loss(actions_from_data, expert_trajs, multi_hm_agent, num_ep_to_use)
    # return loss_final

    if (step_number % (total_number_steps / 1000)) == 0:
        print('Completed {}% in time {} mins; Loss before grad step: {}'.format(100 * step_number / total_number_steps,
                                                     round((time.time() - start_time)/60), loss))
    if (step_number % (total_number_steps / 100)) == 0:
        print(params["PERSON_PARAMS_HM"])

#------------- main -----------------#

if __name__ == "__main__":
    """

    """
    parser = ArgumentParser()
    parser.add_argument("-l", "--layout",
                        help="Layout, (Choose from: simple, scenario1_s, schelling_s, unident_s, random1)",
                        required=True)
    parser.add_argument("-p", "--params", help="Starting params (all params get this value). OR set to 9 to get "
                                               "random values for the starting params", required=False,
                        default=None, type=float)
    parser.add_argument("-ne", "--num_ep", help="Number of episodes to use when training (up to 16?)",
                        required=False, default=16, type=int)
    parser.add_argument("-lr", "--base_lr", help="Base learning rate. E.g. 0.1", required=False, default=0.1,
                        type=float)
    parser.add_argument("-sd", "--epsilon_sd", type=float,
                        help="Standard deviation of dist picking epison from. Initial runs suggest sd=0.02 is good",
                        required=False, default=0.02)
    parser.add_argument("-nh", "--num_hms", help="Number of human models to use for approximating P(action|state)",
                        required=False, default=3, type=int)
    parser.add_argument("-ns", "--num_grad_steps",  help="Number of gradient decent steps", required=False,
                        default=1e4, type=int)
    parser.add_argument("-acc", "--check_accuracy_only",
                        help="Set to true to just check the top-1 and top-2 accuracy (instead of optimising params)",
                        required=False, default=False, type=bool)
    # This gives problems with the BOOL for some reason:
    # parser.add_argument("-r", "--ensure_random_direction",
    #                     help="Should make extra sure that the random search direction is not biased towards corners "
    #                          "of the hypercube.", required=False, default=False, type=bool)

    args = parser.parse_args()
    layout = args.layout
    starting_params = args.params

    # -----------------------------#
    # Settings for the zeroth order optimisation:
    num_ep_to_use = args.num_ep  # How many episodes to use for the fitting
    base_learning_rate = args.base_lr  # Quick test suggests loss for a single episode can be up to around 10. So if
    # base learning rate is 1/5, we shift by roughly epsilon. NEED TO TUNE THIS!
    epsilon_sd = args.epsilon_sd  # standard deviation of the dist to pick epsilon from
    ensure_random_direction = False
    number_hms = args.num_hms
    total_number_steps = args.num_grad_steps  # Number of steps to do in gradient decent
    check_accuracy_only = args.check_accuracy_only
    # -----------------------------#

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)  # pk: Note sure why I need this line too

    # Load human data as state-action pairs:
    # TODO: Do other layouts
    train_mdps = [layout]
    ordered_trajs = True
    human_ai_trajs = False
    data_path = "human_aware_rl/data/human/anonymized/clean_{}_trials.pkl".format('train')
    expert_trajs = get_trajs_from_data(data_path, train_mdps, ordered_trajs, human_ai_trajs)
    # Load (file I saved using pickle) instead FOR SIMPLE ONLY???: pickle_in = open('expert_trajs.pkl',
    # 'rb'); expert_trajs = pickle.load(pickle_in)

    # Starting personality params
    if starting_params == None:
        PERSON_PARAMS_HM = {
            "PERSEVERANCE_HM": 0.5,
            "TEAMWORK_HM": 0.5,
            "RETAIN_GOALS_HM": 0.5,
            "WRONG_DECISIONS_HM": 0.5,
            "THINKING_PROB_HM": 1,  # NOTE: This is the probability of moving on, not of waiting to think!
            "PATH_TEAMWORK_HM": 0.5,
            "RATIONALITY_COEFF_HM": 2,
            "PROB_PAUSING_HM": 99  # This will be modified within the code. If not setting to 99 gives an error
        }
    elif starting_params == 9:
        # Random initialisation:
        PERSON_PARAMS_HM = {
            "PERSEVERANCE_HM": np.random.rand(),
            "TEAMWORK_HM": np.random.rand(),
            "RETAIN_GOALS_HM": np.random.rand(),
            "WRONG_DECISIONS_HM": np.random.rand(),
            "THINKING_PROB_HM": 1,  # NOTE: This is the probability of moving on, not of waiting to think!
            "PATH_TEAMWORK_HM": np.random.rand(),
            "RATIONALITY_COEFF_HM": np.random.randint(0, 20),
            "PROB_PAUSING_HM": 99  # This will be modified within the code. If not setting to 99 gives an error
        }
    else:
        PERSON_PARAMS_HM = {
            "PERSEVERANCE_HM": starting_params,
            "TEAMWORK_HM": starting_params,
            "RETAIN_GOALS_HM": starting_params,
            "WRONG_DECISIONS_HM": starting_params,
            "THINKING_PROB_HM": 1,  # NOTE: This is the probability of moving on, not of waiting to think!
            "PATH_TEAMWORK_HM": starting_params,
            "RATIONALITY_COEFF_HM": 20*starting_params,
            "PROB_PAUSING_HM": 99  # This will be modified within the code. If not setting to 99 gives an error
        }

    # Irrelevant what values these start as:
    PERSON_PARAMS_HMeps = {
        "PERSEVERANCE_HMeps": 9,
        "TEAMWORK_HMeps": 9,
        "RETAIN_GOALS_HMeps": 9,
        "WRONG_DECISIONS_HMeps": 9,
        "THINKING_PROB_HMeps": 9,
        "PATH_TEAMWORK_HMeps": 9,
        "RATIONALITY_COEFF_HMeps": 9,
        "PROB_PAUSING_HMeps": 9
    }

    # Keep some of the person params fixed. E.g. put {"PROB_PAUSING_HM"}
    PERSON_PARAMS_FIXED = {"PROB_PAUSING_HM"}

    # Need some params to create HM agent:
    LAYOUT_NAME = train_mdps[0]
    START_ORDER_LIST = ["any"] * 20
    REW_SHAPING_PARAMS = {
        "PLACEMENT_IN_POT_REW": 3,
        "DISH_PICKUP_REWARD": 3,
        "SOUP_PICKUP_REWARD": 5,
        "DISH_DISP_DISTANCE_REW": 0.015,
        "POT_DISTANCE_REW": 0.03,
        "SOUP_DISTANCE_REW": 0.1,
    }
    MDP_PARAMS = {"layout_name": LAYOUT_NAME,
                  "start_order_list": START_ORDER_LIST,
                  "rew_shaping_params": REW_SHAPING_PARAMS}
    # NO_COUNTER_PARAMS:
    START_ORIENTATIONS = False
    WAIT_ALLOWED = False
    COUNTER_PICKUP = []
    SAME_MOTION_GOALS = True

    params = {
        "MDP_PARAMS": MDP_PARAMS,
        "PERSON_PARAMS_HM": PERSON_PARAMS_HM,
        "PERSON_PARAMS_HMeps": PERSON_PARAMS_HMeps,
        "PERSON_PARAMS_FIXED": PERSON_PARAMS_FIXED,
        "START_ORIENTATIONS": START_ORIENTATIONS,
        "WAIT_ALLOWED": WAIT_ALLOWED,
        "COUNTER_PICKUP": COUNTER_PICKUP,
        "SAME_MOTION_GOALS": SAME_MOTION_GOALS,
        "ensure_random_direction": ensure_random_direction,
        "PERSON_PARAMS_HMcheck": None
    }  # Using same format as pbt_hms_v2

    mdp = OvercookedGridworld.from_layout_name(**params["MDP_PARAMS"])

    # Make the mlp:
    NO_COUNTERS_PARAMS = {
        'start_orientations': START_ORIENTATIONS,
        'wait_allowed': WAIT_ALLOWED,
        'counter_goals': mdp.get_counter_locations(),
        'counter_drop': mdp.get_counter_locations(),
        'counter_pickup': COUNTER_PICKUP,
        'same_motion_goals': params["SAME_MOTION_GOALS"]
    }  # This means that all counter locations are allowed to have objects dropped on them AND be "goals" (I think!)
    mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS, force_compute=False)

    #-----------------------------#
    lr = base_learning_rate / num_ep_to_use  # learning rate: the more episodes we use the more the loss will be,
    # so we need to scale it down by num_ep_to_use
    params["sim_threads"] = number_hms  # Needed when using ToMAgent
    actions_from_data = expert_trajs['ep_actions']
    # First find the probability of the data not acting:
    prob_data_doesnt_act, number_states_with_acting = find_prob_not_acting(actions_from_data, num_ep_to_use)
    print('Prob of data-agent taking ZERO action, (0,0): {}; Number states when data-agent acts: {}'.format(
        prob_data_doesnt_act, number_states_with_acting))

    if not check_accuracy_only:
        # Optimise the params to fit the data:
        start_time = time.time()
        # For each gradient decent step, find the gradient and step:
        for step_number in range(np.int(total_number_steps)):
            find_gradient_and_step_multi_hm(params, mlp, expert_trajs, num_ep_to_use, lr, epsilon_sd,
                                            start_time, step_number, total_number_steps)

    elif check_accuracy_only:
        # Just find the top-1 and top-2 accuracy:

        PERSON_PARAMS_HMcheck = {"PERSEVERANCE_HMcheck": 0.8, "TEAMWORK_HMcheck": 0.7,
            "RETAIN_GOALS_HMcheck": 0.6, "WRONG_DECISIONS_HMcheck": 0.1, "THINKING_PROB_HMcheck": 0.5,
            "PATH_TEAMWORK_HMcheck": 0.4, "RATIONALITY_COEFF_HMcheck": 3, "PROB_PAUSING_HMcheck": 0.2}
        params["PERSON_PARAMS_HMcheck"] = PERSON_PARAMS_HMcheck
        hm_number = 'check'
        multi_hm_agent = ToMAgent(params, hm_number).get_multi_agent(mlp)
        start_time = time.time()
        top_1_acc, top_2_acc = find_top_12_accuracy(actions_from_data, expert_trajs, multi_hm_agent, num_ep_to_use,
                                                    number_states_with_acting)

        print('\nTop-1 accuracy: {}; Top-2 accuracy: {}; Finished acc calc in time {} secs'.format(
                                                                top_1_acc, top_2_acc, round(time.time() - start_time)))

    print('\nend')

