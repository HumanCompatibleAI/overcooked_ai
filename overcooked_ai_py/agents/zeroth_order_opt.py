from human_aware_rl.human.process_dataframes import get_trajs_from_data
from human_aware_rl.pbt.pbt_hms import HMAgent
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import GreedyHumanModelv2
import logging, pickle
import numpy as np
# np.seterr(divide='ignore', invalid='ignore')  # Suppress error about diving by zero



# Helper functions:

def choose_hm_actions(expert_trajs, hm_agent, num_ep_to_use=1):
    """
    Take a human model with given parameters, then use this to choose one action for every state in the data.
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
        hm_agent.GHM = GreedyHumanModelv2(hm_agent.mlp, player_index=1-hm_agent.agent_index)

        # For each state in the episode trajectory:
        for j in range(actions_from_data[i].__len__()):

            current_state = expert_trajs['ep_observations'][i][j]
            # The state seems to be missing an order list. Manually add the start_order_list:
            current_state.order_list = hm_agent.mdp.start_order_list
            # #TODO: Fix this properly

            # Print the state to view the game:
            overcooked_env = OvercookedEnv(hm_agent.mdp)
            logging.warning('i = {}, j = {}'.format(i, j))
            overcooked_env.state = current_state
            logging.warning(overcooked_env)

            # Choose HM action from state
            hm_action = hm_agent.action(current_state)
            # This also automatically updates hm_agent.timesteps_stuck, hm_agent.dont_drop, hm_agent.prev_motion_goal

            # Set the prev state and prev action from the data
            hm_agent.prev_state = expert_trajs['ep_observations'][i][j]
            if hm_agent.prev_motion_goal != None:  # Only get prev action from data if there's already a motion goal
                hm_agent.prev_best_action = actions_from_data[i][j]

            hm_actions_this_ep.append(hm_action)

            # Print everything after:
            logging.warning('Action from HM: {}; Action from data: {}'.format(hm_action, actions_from_data[i][j]))
            logging.warning('HM prev_motion_goal: {}; HM dont_drop: {}'.format(hm_agent.prev_motion_goal,
                                                                               hm_agent.dont_drop))
            logging.warning('HM time stuck: {}'.format(hm_agent.timesteps_stuck))

        hm_actions.append(hm_actions_this_ep)

    return hm_actions

def deterministic_loss(hm_actions, actions_from_data, num_ep_to_use=1):
    """
    Compute the deterministic loss. I.e. if the HM and data's actions are the same for a given state, then loss=0,
    otherwise loss=1
    :param hm_actions: actions chosen by the human model
    :param actions_from_data: actions chosen by the real human in the data
    :return: loss
    """
    loss = 0
    # For each episode we want to use
    for i in range(num_ep_to_use):

        # For each state in the episode trajectory:
        for j in range(actions_from_data[i].__len__()):

            if actions_from_data[i][j] == hm_actions[i][j]:
                loss += 0
            else:
                loss += 1

    return loss


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
    count = [[0] * actions_from_data[i].__len__() for i in range(num_ep_to_use)]

    # For each episode we want to use
    for i in range(num_ep_to_use):
        # For each state in the episode trajectory:
        for j in range(actions_from_data[i].__len__()):

            # For each agent
            for hm_actions in multi_hm_actions:

                # If agent chooses the same action as the data, then count this
                if actions_from_data[i][j] == hm_actions[i][j]:
                    count[i][j] += 1
            #TODO: put the count->prob and the "force prob to to 0.01" codes here

    # To get probs from count, divide each element by the number of hms:
    hm_probs_action_in_state = [[count[i][j] / multi_hm_agent.__len__() for j in range(actions_from_data[i].__len__())]
                                for i in range(num_ep_to_use)]

    # Force prob to be 0.01 minimum (otherwise we get infinities in the cross entropy):
    for i in range(num_ep_to_use):
        for j in range(actions_from_data[i].__len__()):
            if hm_probs_action_in_state[i][j] == 0:
                hm_probs_action_in_state[i][j] = 0.01

    return hm_probs_action_in_state

def find_cross_entropy_loss(expert_trajs, multi_hm_agent, num_ep_to_use):
    """
    ...?
    :param expert_trajs:
    :param multi_hm_agent:
    :param num_ep_to_use:
    :return:
    """
    actions_from_data = expert_trajs['ep_actions']
    # Find Prob_HM(action|state) for all actions chosen by the data
    hm_probs_action_in_state = find_hm_probs_action_in_state(multi_hm_agent, actions_from_data, num_ep_to_use,
                                                          expert_trajs)

    loss = 0

    # For each episode we want to use
    for i in range(num_ep_to_use):
        # For each state in the episode trajectory:
        for j in range(actions_from_data[i].__len__()):

            loss += np.log(hm_probs_action_in_state[i][j])/np.log(0.01)  # Normalise by 0.01, so that loss_ij=1 max

    return loss


def shift_by_epsilon(PERSON_PARAMS_HMeps, PERSON_PARAMS_HM, PERSON_PARAMS_FIXED, epsilon):
    """
    Shift the PERSON_PARAMS_HM by epsilon, except RATIONALITY_COEFF, which isn't between 0 and 1 so is shifted further
    :return: PERSON_PARAMS_HMeps that've been shifted
    """

    for i, pparam in enumerate(PERSON_PARAMS_HM):

        if pparam in PERSON_PARAMS_FIXED:
            # Don't change this parameter
            pass
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


def shift_by_gradient(PERSON_PARAMS_HM, PERSON_PARAMS_FIXED, epsilon, delta_loss, lr):
    """
    Shift PERSON_PARAMS_HM in the direction of negative gradient, scaled by lr
    :return: PERSON_PARAMS_HM after they've been shifted
    """

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

    return PERSON_PARAMS_HM

# DEPRECIATED (for now)

# def find_gradient_and_step_single_hm(params, mdp, expert_trajs, num_ep_to_use, actions_from_data, lr, step_number,
#                            epsilon_sd):
#     """
#     :return: loss, PERSON_PARAMS_HM
#     """
#
#     hm_number = ''
#     player_index = 99  # We will change player index later (setting to 99 in case we forget!)
#     hm_agent = HMAgent(params, mdp, hm_number, player_index).get_agent()
#
#     # Find the actions from the HM for all states:
#     hm_actions = choose_hm_actions(expert_trajs, hm_agent, num_ep_to_use=num_ep_to_use)
#
#     # Work out initial loss
#     loss = deterministic_loss(hm_actions, actions_from_data)
#
#     if step_number == 0:
#         print('Initial loss: {}'.format(loss))
#
#     # Choose random epsilon (eps) from normal dist, sd=epsilon_sd
#     epsilon = np.random.normal(scale=epsilon_sd, size=params['PERSON_PARAMS_HM'].__len__())
#
#     # Make PERSON_PARAMS_HM + epsilon. For rationality_coefficient, do eps*10 for now. This func also ensures all
#     # params are between 0 and 1
#     # TODO: just use params in shift_by_epsilon instead of PERSON_PARAMS_HMeps and PERSON_PARAMS_HM separately
#     PERSON_PARAMS_HM = params['PERSON_PARAMS_HM']
#     PERSON_PARAMS_HMeps = params['PERSON_PARAMS_HMeps']
#     PERSON_PARAMS_FIXED = params['PERSON_PARAMS_FIXED']
#     PERSON_PARAMS_HMeps = shift_by_epsilon(PERSON_PARAMS_HMeps, PERSON_PARAMS_HM, PERSON_PARAMS_FIXED, epsilon)
#
#     # Find loss for new params
#     hm_number = 'eps'
#     hm_agent_eps = HMAgent(params, mdp, hm_number).get_agent()
#     hm_actions_eps = choose_hm_actions(expert_trajs, hm_agent_eps, num_ep_to_use=num_ep_to_use)
#     loss_eps = deterministic_loss(hm_actions_eps, actions_from_data)
#     delta_loss = loss - loss_eps
#
#     # Find new pparams by shifting in the direction of downhill
#     PERSON_PARAMS_HM = shift_by_gradient(PERSON_PARAMS_HM, PERSON_PARAMS_FIXED, epsilon, delta_loss, lr)
#
#     # What's the loss after this grad step:
#     hm_number = ''
#     hm_agent = HMAgent(params, mdp, hm_number).get_agent()
#     hm_actions = choose_hm_actions(expert_trajs, hm_agent, num_ep_to_use=num_ep_to_use)
#     loss_final = deterministic_loss(hm_actions, actions_from_data)
#
#     return loss_final, PERSON_PARAMS_HM

def find_gradient_and_step_multi_hm(params, mdp, expert_trajs, num_ep_to_use, actions_from_data, lr, step_number,
                           epsilon_sd):
    """
    Same as find_gradient_and_step_single_hm except here we have multiple hms taking multiple actions, so we find
    prob(action|state) and use cross entropy loss
    :return: loss, PERSON_PARAMS_HM
    """

    hm_number = ''
    # Make multiple hm agents:
    multi_hm_agent = HMAgent(params, mdp, hm_number).get_multi_agent()

    loss = find_cross_entropy_loss(expert_trajs, multi_hm_agent, num_ep_to_use)

    if step_number == 0:
        print('Initial loss: {}'.format(loss))

    # Choose random epsilon (eps) from normal dist, sd=epsilon_sd
    epsilon = np.random.normal(scale=epsilon_sd, size=params['PERSON_PARAMS_HM'].__len__())

    # Make PERSON_PARAMS_HM + eps
    # ilon. For rationality_coefficient, do eps*10 for now. This func also ensures all
    # params are between 0 and 1
    # TODO: just use params in shift_by_epsilon instead of PERSON_PARAMS_HMeps and PERSON_PARAMS_HM and
    #  PERSON_PARAMS_FIXED separately
    PERSON_PARAMS_HM = params['PERSON_PARAMS_HM']
    PERSON_PARAMS_HMeps = params['PERSON_PARAMS_HMeps']
    shift_by_epsilon(PERSON_PARAMS_HMeps, PERSON_PARAMS_HM, PERSON_PARAMS_FIXED, epsilon)

    # Find loss for new params
    hm_number = 'eps'
    multi_hm_agent_eps = HMAgent(params, mdp, hm_number).get_multi_agent()
    loss_eps = find_cross_entropy_loss(expert_trajs, multi_hm_agent_eps, num_ep_to_use)
    delta_loss = loss - loss_eps

    # Find new pparams by shifting in the direction of downhill
    PERSON_PARAMS_HM = shift_by_gradient(PERSON_PARAMS_HM, PERSON_PARAMS_FIXED, epsilon, delta_loss, lr)

    # What's the loss after this grad step:
    hm_number = ''
    multi_hm_agent = HMAgent(params, mdp, hm_number).get_multi_agent()
    loss_final = find_cross_entropy_loss(expert_trajs, multi_hm_agent, num_ep_to_use)

    # Also get deterministic loss for comparison
    # hm_agent = HMAgent(params, mdp, hm_number).get_agent()
    # hm_actions = choose_hm_actions(expert_trajs, hm_agent, num_ep_to_use=num_ep_to_use)
    # loss_final_deterministic = deterministic_loss(hm_actions, actions_from_data)
    loss_final_deterministic = np.Inf  # Could be any value!

    return loss_final, loss_final_deterministic, PERSON_PARAMS_HM


#------------- main -----------------#

if __name__ == "__main__":
    """

    """

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)  # pk: Note sure why I need this line too

    # Load human data as state-action pairs:
    data_path = "human_aware_rl/data/human/anonymized/clean_{}_trials.pkl".format('train')
    # TODO: Do other layouts
    train_mdps = ["simple"]
    ordered_trajs = True
    human_ai_trajs = False

    # expert_trajs = get_trajs_from_data(data_path, train_mdps, ordered_trajs, human_ai_trajs)
    # Load (file I saved using pickle) instead:
    pickle_in = open('expert_trajs.pkl','rb')
    expert_trajs = pickle.load(pickle_in)

    # Start with personality params taken from HM0:
    # Setting some to 0 and 1, to see if my code works to stop <0 or >1
    PERSON_PARAMS_HM = {
        "PERSEVERANCE_HM": 0.5,
        "TEAMWORK_HM": 0.5,
        "RETAIN_GOALS_HM": 0.5,
        "WRONG_DECISIONS_HM": 0.5,
        "THINKING_PROB_HM": 1,  # NOTE: This is the probability of moving on, not of waiting to think!
        "PATH_TEAMWORK_HM": 0.5,
        "RATIONALITY_COEFF_HM": 2,
        "PROB_PAUSING_HM": 0
    }

    # Irrelevant what values these start as:
    PERSON_PARAMS_HMeps = {
        "PERSEVERANCE_HMeps": 0.3,
        "TEAMWORK_HMeps": 0.8,
        "RETAIN_GOALS_HMeps": 0.8,
        "WRONG_DECISIONS_HMeps": 0.02,
        "THINKING_PROB_HMeps": 0.6,
        "PATH_TEAMWORK_HMeps": 0.8,
        "RATIONALITY_COEFF_HMeps": 3,
        "PROB_PAUSING_HMeps": 0
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
        "SAME_MOTION_GOALS": SAME_MOTION_GOALS
    }  # Using same format as pbt_hms_v2

    mdp = OvercookedGridworld.from_layout_name(**params["MDP_PARAMS"])

    #-----------------------------#
    # Settings for the zeroth order optimisation:
    num_ep_to_use = 1  # How many episodes to use for the fitting
    base_learning_rate = 0.2  # Quick test suggests loss for a single episode can be up to around 10. So if base
    # learning rate is 1/5, we shift by roughly epsilon. NEED TO TUNE THIS!
    epsilon_sd = 0.01  # standard deviation of the dist to pick epsilon from
    actions_from_data = expert_trajs['ep_actions']
    total_number_steps = 1000  # Number of steps to do in gradient decent
    lr = base_learning_rate / num_ep_to_use  # learning rate: the more episodes we use the more the loss will be,
    # so we need to scale it down by num_ep_to_use
    number_hms = 3
    params["sim_threads"] = number_hms  # Needed when using HMAgent

    #-----------------------------#
    # For each gradient decent step, find the gradient and step:
    for step_number in range(total_number_steps):
        # if number_hms == 1:
        #     loss_final, PERSON_PARAMS_HM = find_gradient_and_step_single_hm(params, mdp, expert_trajs, num_ep_to_use,
        #                                                     actions_from_data, lr, step_number, epsilon_sd)
        #     print(params["PERSON_PARAMS_HM"])
        # elif number_hms > 1:
        loss_final, loss_final_deterministic, PERSON_PARAMS_HM = find_gradient_and_step_multi_hm(params, mdp,
                                                            expert_trajs, num_ep_to_use,
                                                            actions_from_data, lr, step_number, epsilon_sd)
        print(params["PERSON_PARAMS_HM"])

        print('Final loss: {}'.format(loss_final))
        # print('Fin d-loss: {}'.format(loss_final_deterministic))


    print('...')

    # TODO: Neaten up into more helper functions

    print('end')


