
"""
Quick code to view an agent acting

"""

import pickle
import time
import unittest
import numpy as np
import random
import logging

from argparse import ArgumentParser
from overcooked_ai_py.agents.agent import Agent, AgentPair, FixedPlanAgent, CoupledPlanningAgent, StayAgent, \
    RandomAgent, GreedyHumanModel, GreedyHumanModel_pk, ToMModel
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, PlayerState, ObjectState
from overcooked_ai_py.mdp.actions import Direction, Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelPlanner
from overcooked_ai_py.agents.benchmarking import AgentEvaluator


def make_agent_pair(mlp):
    # Make agents:
    # perseverance0 = random.random()
    perseverance0 = 0.5
    # teamwork0 = random.random()
    teamwork0 = 0.5
    # retain_goals0 = random.random()
    retain_goals0 = 0.5
    # wrong_decisions0 = random.random()**30
    wrong_decisions0 = 0.1
    # thinking_prob0 = 1 - random.random()**30
    thinking_prob0 = 0.5
    # path_teamwork0 = 1 - random.random()**2
    path_teamwork0 = 0.5
    # rat_coeff0 = 2+random.random()*8
    rat_coeff0 = 3
    prob_pausing0 = 0
    a0 = ToMModel(mlp, player_index=0, perseverance=perseverance0, teamwork=teamwork0,
                  retain_goals=retain_goals0, wrong_decisions=wrong_decisions0,
                  thinking_prob=thinking_prob0, prob_pausing=prob_pausing0,
                  path_teamwork=path_teamwork0, rationality_coefficient=rat_coeff0)
    a1 = ToMModel(mlp, player_index=1, perseverance=perseverance0, teamwork=teamwork0,
                  retain_goals=retain_goals0, wrong_decisions=wrong_decisions0,
                  thinking_prob=thinking_prob0, prob_pausing=prob_pausing0,
                  path_teamwork=path_teamwork0, rationality_coefficient=rat_coeff0)
    # a0 = ToMModel(mlp, player_index=0, perseverance=0.9, teamwork=1, retain_goals=0.9,
    #                                  wrong_decisions=0.02, thinking_prob=1, path_teamwork=1, rationality_coefficient=2)
    # a1 = GreedyHumanModel_pk(mlp, player_index=0, perseverance=0.8)
    # a1 = RandomAgent(mlp)
    print(perseverance0, teamwork0, retain_goals0, wrong_decisions0, thinking_prob0, prob_pausing0, path_teamwork0,
          rat_coeff0)
    # print('Player 0: teamwork: {:.1f}, retain: {:.1f}, wrong dec: {:.1f}'.format(teamwork0, retain_goals0, wrong_decisions0))
    return AgentPair(a0, a1)


if __name__ == "__main__" :
    """
    
    """
    parser = ArgumentParser()
    # parser.add_argument("-l", "--fixed_mdp", dest="layout",
    #                     help="name of the layout to be played as found in data/layouts",
    #                     required=True)
    parser.add_argument("-l", "--layout",
                        help="layout, (Choose from: sim, sc1, sch, uni, ran1, ran0)", required=True)
    print("\n****************************************\nNOTE: To watch play in debug, put breakpoint in "
          "overcooked_env.OvercookedEnv.run_agents\n*****************************************\n")
    args = parser.parse_args()
    layout = args.layout

    np.random.seed(41)

    # To print the agent's decisions:
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.ERROR)

    n, s = Direction.NORTH, Direction.SOUTH
    e, w = Direction.EAST, Direction.WEST
    stay, interact = Action.STAY, Action.INTERACT
    P, Obj = PlayerState, ObjectState

    DISPLAY = True
    start_order_list = ["any"]*20
    horizon = 2000
    explosion_time = 500
    r_shaping = 0

    no_counters_params = {
        'start_orientations': False,
        'wait_allowed': False,
        'counter_goals': [],
        'counter_drop': [],
        'counter_pickup': [],
        'same_motion_goals': True
    }

    if layout == 'uni':
        start_state = OvercookedState([P((2, 2), n), P((5, 2), n)], {}, order_list=start_order_list)
    elif layout == 'sc1':
        start_state = OvercookedState([P((1, 2), n), P((1, 1), n)], {}, order_list=start_order_list)
    elif layout == 'ran1':
        start_state = OvercookedState([P((1, 2), n), P((1, 1), n)], {}, order_list=start_order_list)
    elif layout == 'ran0':
        start_state = OvercookedState([P((1, 2), n), P((3, 2), n)], {}, order_list=start_order_list)
    elif layout == 'ran3':
        start_state = OvercookedState([P((1, 2), n), P((6, 2), n)], {}, order_list=start_order_list)
    else:
        start_state = OvercookedState([P((2, 2), n), P((2, 1), n)], {}, order_list=start_order_list)

    if layout == 'sc1':
        # Set up mdp and mlp:
        # am_filename = "data/planners/LAYOUT_am.pkl"
        cook_time=5
        mdp = OvercookedGridworld.from_layout_name('scenario1_s', start_order_list=start_order_list,
                                                   cook_time=cook_time, rew_shaping_params=None)
        # pk: When can I set explosion_time?
        # pk: Location of layout, if needed: "data/layouts/LAYOUT.layout"

        # Doing this means that all counter locations are allowed to have objects dropped on them AND be "goals" (I think!)
        no_counters_params['counter_drop'] = mdp.get_counter_locations()
        no_counters_params['counter_goals'] = mdp.get_counter_locations()

        mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, no_counters_params, force_compute=False)

        # Added since restructuring changes:
        mdp_params = {"layout_name": "scenario1_s", "start_order_list": start_order_list, "cook_time": cook_time}
        env_params = {"start_state_fn": lambda: start_state, "horizon": horizon}
        mlp_params = no_counters_params
        # one_counter_params = { 'start_orientations': False, 'wait_allowed': False, 'counter_goals': valid_counters,
        #     'counter_drop': valid_counters, 'counter_pickup': [], 'same_motion_goals': True }
        # mlp_params=one_counter_params

        # Make and evaluate agents:
        ap = make_agent_pair(mlp)
        a_eval = AgentEvaluator(mdp_params=mdp_params, env_params=env_params, mlp_params=mlp_params)
        a_eval.evaluate_agent_pair(ap, display=True)

    elif layout == 'uni':

        # Set up mdp and mlp:
        # am_filename = "data/planners/LAYOUT_am.pkl"
        cook_time = 5

        mdp = OvercookedGridworld.from_layout_name('unident_s', start_order_list=start_order_list,
                                                   cook_time=cook_time, rew_shaping_params=None)
        # pk: When can I set explosion_time?
        # pk: Location of layout, if needed: "data/layouts/LAYOUT.layout"

        # Doing this means that all counter locations are allowed to have objects dropped on them AND be "goals" (I think!)
        no_counters_params['counter_drop'] = mdp.get_counter_locations()
        no_counters_params['counter_goals'] = mdp.get_counter_locations()

        mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, no_counters_params, force_compute=False)

        # Added since restructuring changes:
        mdp_params = {"layout_name": "unident_s", "start_order_list": start_order_list, "cook_time": cook_time}
        env_params = {"start_state_fn": lambda: start_state, "horizon": horizon}
        mlp_params = no_counters_params
        # one_counter_params = { 'start_orientations': False, 'wait_allowed': False, 'counter_goals': valid_counters,
        #     'counter_drop': valid_counters, 'counter_pickup': [], 'same_motion_goals': True }
        # mlp_params=one_counter_params

        # Make and evaluate agents:
        ap = make_agent_pair(mlp)
        a_eval = AgentEvaluator(mdp_params=mdp_params, env_params=env_params, mlp_params=mlp_params)
        a_eval.evaluate_agent_pair(ap, display=True)

    elif layout == 'sim':

        # Set up mdp and mlp:
        # am_filename = "data/planners/LAYOUT_am.pkl"
        cook_time = 5
        mdp = OvercookedGridworld.from_layout_name('simple', start_order_list=start_order_list,
                                                   cook_time=cook_time, rew_shaping_params=None)
        # pk: When can I set explosion_time?
        # pk: Location of layout, if needed: "data/layouts/LAYOUT.layout"

        # Doing this means that all counter locations are allowed to have objects dropped on them AND be "goals" (I think!)
        no_counters_params['counter_drop'] = mdp.get_counter_locations()
        no_counters_params['counter_goals'] = mdp.get_counter_locations()

        mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, no_counters_params, force_compute=False)

        # Added since restructuring changes:
        mdp_params = {"layout_name": "simple", "start_order_list": start_order_list, "cook_time": cook_time}
        env_params = {"start_state_fn": lambda: start_state, "horizon": horizon}
        mlp_params = no_counters_params
        # one_counter_params = { 'start_orientations': False, 'wait_allowed': False, 'counter_goals': valid_counters,
        #     'counter_drop': valid_counters, 'counter_pickup': [], 'same_motion_goals': True }
        # mlp_params=one_counter_params

        # Make and evaluate agents:
        ap = make_agent_pair(mlp)
        a_eval = AgentEvaluator(mdp_params=mdp_params, env_params=env_params, mlp_params=mlp_params)
        a_eval.evaluate_agent_pair(ap, display=True)

    elif layout == 'ran1':

        # Set up mdp and mlp:
        # am_filename = "data/planners/LAYOUT_am.pkl"
        cook_time = 5
        mdp = OvercookedGridworld.from_layout_name('random1', start_order_list=start_order_list,
                                                   cook_time=cook_time, rew_shaping_params=None)
        # pk: When can I set explosion_time?
        # pk: Location of layout, if needed: "data/layouts/LAYOUT.layout"

        # Doing this means that all counter locations are allowed to have objects dropped on them AND be "goals" (I think!)
        no_counters_params['counter_drop'] = mdp.get_counter_locations()
        no_counters_params['counter_goals'] = mdp.get_counter_locations()

        mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, no_counters_params, force_compute=False)

        # Added since restructuring changes:
        mdp_params = {"layout_name": "random1", "start_order_list": start_order_list, "cook_time": cook_time}
        env_params = {"start_state_fn": lambda: start_state, "horizon": horizon}
        mlp_params = no_counters_params
        # one_counter_params = { 'start_orientations': False, 'wait_allowed': False, 'counter_goals': valid_counters,
        #     'counter_drop': valid_counters, 'counter_pickup': [], 'same_motion_goals': True }
        # mlp_params=one_counter_params

        # Make and evaluate agents:
        ap = make_agent_pair(mlp)
        a_eval = AgentEvaluator(mdp_params=mdp_params, env_params=env_params, mlp_params=mlp_params)
        a_eval.evaluate_agent_pair(ap, display=True)

    elif layout == 'ran0':

        # Set up mdp and mlp:
        # am_filename = "data/planners/LAYOUT_am.pkl"
        cook_time = 5
        mdp = OvercookedGridworld.from_layout_name('random0', start_order_list=start_order_list,
                                                   cook_time=cook_time, rew_shaping_params=None)
        # pk: When can I set explosion_time?
        # pk: Location of layout, if needed: "data/layouts/LAYOUT.layout"

        # Doing this means that all counter locations are allowed to have objects dropped on them AND be "goals" (I think!)
        no_counters_params['counter_drop'] = mdp.get_counter_locations()
        no_counters_params['counter_goals'] = mdp.get_counter_locations()

        mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, no_counters_params, force_compute=False)

        # Added since restructuring changes:
        mdp_params = {"layout_name": "random0", "start_order_list": start_order_list, "cook_time": cook_time}
        env_params = {"start_state_fn": lambda: start_state, "horizon": horizon}
        mlp_params = no_counters_params
        # one_counter_params = { 'start_orientations': False, 'wait_allowed': False, 'counter_goals': valid_counters,
        #     'counter_drop': valid_counters, 'counter_pickup': [], 'same_motion_goals': True }
        # mlp_params=one_counter_params

        # Make and evaluate agents:
        ap = make_agent_pair(mlp)
        a_eval = AgentEvaluator(mdp_params=mdp_params, env_params=env_params, mlp_params=mlp_params)
        a_eval.evaluate_agent_pair(ap, display=True)

    elif layout == 'ran3':

        # Set up mdp and mlp:
        # am_filename = "data/planners/LAYOUT_am.pkl"
        cook_time = 5
        mdp = OvercookedGridworld.from_layout_name('random3', start_order_list=start_order_list,
                                                   cook_time=cook_time, rew_shaping_params=None)
        # pk: When can I set explosion_time?
        # pk: Location of layout, if needed: "data/layouts/LAYOUT.layout"

        # Doing this means that all counter locations are allowed to have objects dropped on them AND be "goals" (I think!)
        no_counters_params['counter_drop'] = mdp.get_counter_locations()
        no_counters_params['counter_goals'] = mdp.get_counter_locations()

        mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, no_counters_params, force_compute=False)

        # Added since restructuring changes:
        mdp_params = {"layout_name": "random3", "start_order_list": start_order_list, "cook_time": cook_time}
        env_params = {"start_state_fn": lambda: start_state, "horizon": horizon}
        mlp_params = no_counters_params
        # one_counter_params = { 'start_orientations': False, 'wait_allowed': False, 'counter_goals': valid_counters,
        #     'counter_drop': valid_counters, 'counter_pickup': [], 'same_motion_goals': True }
        # mlp_params=one_counter_params

        # Make and evaluate agents:
        ap = make_agent_pair(mlp)
        a_eval = AgentEvaluator(mdp_params=mdp_params, env_params=env_params, mlp_params=mlp_params)
        a_eval.evaluate_agent_pair(ap, display=True)

    elif layout == 'sch':

        # Set up mdp and mlp:
        # am_filename = "data/planners/LAYOUT_am.pkl"
        cook_time = 5
        mdp = OvercookedGridworld.from_layout_name('schelling_s', start_order_list=start_order_list,
                                                   cook_time=cook_time, rew_shaping_params=None)
        # pk: When can I set explosion_time?
        # pk: Location of layout, if needed: "data/layouts/LAYOUT.layout"

        # Doing this means that all counter locations are allowed to have objects dropped on them AND be "goals" (I think!)
        no_counters_params['counter_drop'] = mdp.get_counter_locations()
        no_counters_params['counter_goals'] = mdp.get_counter_locations()

        mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, no_counters_params, force_compute=False)

        # Added since restructuring changes:
        mdp_params = {"layout_name": "schelling_s", "start_order_list": start_order_list, "cook_time": cook_time}
        env_params = {"start_state_fn": lambda: start_state, "horizon": horizon}
        mlp_params = no_counters_params
        # one_counter_params = { 'start_orientations': False, 'wait_allowed': False, 'counter_goals': valid_counters,
        #     'counter_drop': valid_counters, 'counter_pickup': [], 'same_motion_goals': True }
        # mlp_params=one_counter_params

        # Make and evaluate agents:
        ap = make_agent_pair(mlp)
        a_eval = AgentEvaluator(mdp_params=mdp_params, env_params=env_params, mlp_params=mlp_params)
        a_eval.evaluate_agent_pair(ap, display=True)

    # elif layout == 'randomly_generated':
    #
    #     # Set up mdp and mlp:
    #     am_filename = "data/planners/randomly_generated_am.pkl"
    #
    #     PADDED_MDP_SHAPE = (11, 7)
    #     MDP_SHAPE_FN = ([5, 11], [5, 7])
    #     PROP_EMPTY_FN = [0.6, 1]
    #     PROP_FEATS_FN = [0, 0.6]
    #
    #     REW_SHAPING_PARAMS = {
    #         "PLACEMENT_IN_POT_REW": 3,
    #         "DISH_PICKUP_REWARD": 3,
    #         "SOUP_PICKUP_REWARD": 5,
    #         "DISH_DISP_DISTANCE_REW": 0.015,
    #         "POT_DISTANCE_REW": 0.03,
    #         "SOUP_DISTANCE_REW": 0.1,
    #     }
    #
    #     layout_generator = LayoutGenerator(outer_shape=PADDED_MDP_SHAPE, start_order_list=start_order_list, explosion_time=500, rew_shaping_params=REW_SHAPING_PARAMS)
    #     mdp = layout_generator.make_disjoint_sets_layout(
    #         inner_shape=[rnd_int_uniform(*dim) for dim in MDP_SHAPE_FN],
    #         prop_empty=rnd_uniform(*PROP_EMPTY_FN),
    #         prop_features=rnd_uniform(*PROP_FEATS_FN),
    #         display=True
    #     )
    #
    #     # mdp = OvercookedGridworld.from_file("data/layouts/randomly_generated.layout", start_order_list=start_order_list, explosion_time=500, rew_shaping_params=None)
    #
    #     # Doing this means that all counter locations are allowed to have objects dropped on them AND be "goals" (I think!)
    #     no_counters_params['counter_drop'] = mdp.get_counter_locations()
    #     no_counters_params['counter_goals'] = mdp.get_counter_locations()
    #
    #     mlp = MediumLevelPlanner.from_pickle_or_compute(am_filename, mdp, no_counters_params, force_compute=True)
    #
    #     # Make and evaluate agents:
    #     ap = make_agent_pair(mlp)
    #     a_eval = AgentEvaluator(layout_name="randomly_generated", horizon=100, start_state=start_state)
    #     a_eval.evaluate_agent_pair(ap)

    else:
        raise ValueError('layout not recognised')

