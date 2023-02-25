import argparse
import copy
import json
import os
import pickle

import pandas as pd

from human_aware_rl.human.data_processing_utils import (
    json_joint_action_to_python_action,
)
from overcooked_ai_py.mdp.overcooked_mdp import (
    OvercookedGridworld,
    OvercookedState,
)
from overcooked_ai_py.planning.planners import (
    NO_COUNTERS_PARAMS,
    MediumLevelActionManager,
)

# IMPORTANT FLAG: PLEASE READ BEFORE PROCEEDING
# This flag is meant to correct the fact that in the new dynamics, an additional INTERACT action is required
# to get soups to start cooking after the last onion is dropped in
# (previously, when the last onion is dropped in, the soup automatically start cooking).

JSON_FIXES = [
    ("'", '"'),
    ("False", "false"),
    ("True", "true"),
    ("INTERACT", "interact"),
]

"""
First parse raw pickle file indicated using the first command
line argument, to generate state action pairs.  Then featurize
all collected states for downstream behavior cloning usage.
"""


def insert_cooking_interact(state_dict):
    """
    Arguments:
        state_dict (dict): dictionary with player and object information for the state that needs insertion before it
    """
    # initialize the actions_insert to be returned
    actions_insert = [(0, 0), (0, 0)]
    # making a deep copy because we need to modify the attributes of the soup
    state_dict_insert = copy.deepcopy(state_dict)
    players = state_dict_insert["players"]
    # get the reach of each players
    players_reach = [
        (
            player["position"][0] + player["orientation"][0],
            player["position"][1] + player["orientation"][1],
        )
        for player in players
    ]
    objects = state_dict_insert["objects"]
    for o in objects:
        if o["name"] == "soup" and o["_cooking_tick"] == 1:
            for i, player_reach in enumerate(players_reach):
                if player_reach == o["position"]:
                    actions_insert[i] = "interact"
            # we need to rewind some of the attribut momentarily
            o["_cooking_tick"] = -1
            o["cooking_tick"] = -1  # duplicate tag
            o["cook_time"] = -1
            o["is_idle"] = True
            o["is_cooking"] = False
    assert (
        "interact" in actions_insert
    ), "was supposed to insert interact but did not find a player_reach to insert"
    return state_dict_insert, actions_insert


def is_insertion_needed(state):
    """
    Arguments:
        state (dict): OvercookedState dictionary. Must be under new dynamics schema; call `forward_port_state*` to update schema

    Returns:
        insertion_needed (bool): Whether a soup immediately starting cooking this timestep, indicating an 'interact' action is required under new dynamics
    """
    if "objects" not in state:
        raise ValueError(
            "Corrupted data detected, state missing 'objects' key"
        )

    soups = [obj for obj in state["objects"] if obj["name"] == "soup"]
    if not soups:
        return False

    insertion_needed = False
    for soup in soups:
        if not "cooking_tick" in soup:
            raise ValueError(
                "Legacy schema detected! Please ensure you are using updated state schema"
            )

        # this is the flag to signal if the soup just started cooking (and an additional frame with interact is required)
        insertion_needed = insertion_needed or soup["cooking_tick"] == 1
    return insertion_needed


def forward_port_state_dict(state):
    """
    Update state schema. If new shema state passed in, this is an identity function

    Arguments:
        state (dict): Serialized OvercookedState encoding under legacy schema

    Returns:
        state (dict): Serialized OvercookedState encoding under current schema
    """
    assert type(state) == dict, "Expected Dict input"

    if "players" in state:
        for player in state["players"]:
            if not "held_object" in player:
                player["held_object"] = None
            player
    if "objects" in state and type(state["objects"]) == dict:
        state["objects"] = list(state["objects"].values())

    # Convert all position and orientation lists to tuples
    return OvercookedState.from_dict(state).to_dict()


def forward_port_state_json(state):
    """
    Update state schema. If new schema JSON passed in, this is an identity function

    Arguments:
        state (str): Valid JSON encoding of legacy state

    Returns:
        state (str): Valid JSON encoding of state under current schema
    """
    assert type(state) == str, "Expected JSON string input"

    state_dict = json.loads(state)
    state_dict = forward_port_state_dict(state_dict)

    return json.dumps(state_dict)


def process_state(state_json, forward_port=False, fix_json=False):
    """
    Arguments:
        state_json (str): Valid JSON encoding of an (optionally legacy) OvercookedState
        forward_port (bool): Whether state encoding needs to be updated to current schema. Pass in
            forward_port=True if working with legacy encodings
        fix_json (bool): Whether legacy JSON fixes (such as converting single quotes to double quotes) are necessary.
            Probably not necessary, even if working with legacy schema

    Returns:
        state_dict (dict): New schema encoding of state
        insertion_needed (bool): Whether a soup began cooking at this timestep
    """
    if fix_json:
        for old, new in JSON_FIXES:
            state_json = state_json.replace(old, new)

    if forward_port:
        state_json = forward_port_state_json(state_json)

    state_dict = json.loads(state_json)

    # Perform housecleaning + necessary casting (i.e position lists get converted into tuples)
    state_dict = OvercookedState.to_dict(OvercookedState.from_dict(state_dict))
    return state_dict, is_insertion_needed(state_dict)


def process_actions(actions_json):
    """
    Arguments:
        actions_json (str): JSON encoding of joint agent action

    Returns:
        actions (Overcooked.Action): Current encoding compatible joint action

    Note: `json_joint_action_to_python_action` can handle both legacy and current schema, as well as any necessary JSON fixes under the hood
    """
    return json_joint_action_to_python_action(actions_json)


def display_state_dict_and_action(state_dict, actions):
    for item in state_dict.items():
        if item[0] == "objects":
            print("objects ------")
            for l in item[1]:
                print(l)
            print("--------------")
        else:
            print(item)
    print(actions)
    print()


def main(
    data_infile, data_outdir, insert_interacts, forward_port, fix_json, verbose
):
    """
    Arguments:
        data_infile (str): Full path to cleaned, pickled DataFrame of human data we wish to work with
        data_outdir (str): Directory in which we will save our results. Must exist already
        insert_interacts (bool): Whether to impute interact actions to be compatible with modern dynamics
        forward_port (bool): Whether states need to be converted from legacy to current schema
        fix_json (bool): Whether legacy JSON fixes (ie convert single to double quotes) need to be performed. Unless
            you are working with a very outdated version of our data, this is most likely not necessary
        verbose (bool): Whether to include debug logs

    Behavior:
        Converts data as specified by arguments, then saves 'state_action_pairs' dictionary in

                {data_outdir}/{data_infile_filename}._state_dict_and_action_{inserted|original}.pickle

        Where {data_infile_filename} is the base filename of the loaded datapath. For example, if data_infile='/foo/bar/baz.pickle', then
        data_infile_filename='baz'

        The structure of 'state_action_pairs' is as follows:

            state_action_pairs[layout] = [(state_1, joint_action_1), (state_2, joint_action_2), ...]
    """

    raw_data = pd.read_pickle(data_infile)
    N = len(raw_data)

    if verbose:
        print("Processing Raw Data")
    state_action_pairs = dict()

    for i, datapoint in raw_data.iterrows():
        if verbose:
            print(f"Processing {i}/{N}", end="\r")

        layout_name = datapoint.layout_name
        if layout_name == "random0":
            layout_name = "forced_coordination"
        elif layout_name == "random3":
            layout_name = "counter_circuit_o_1order"
        if layout_name not in state_action_pairs:
            state_action_pairs[layout_name] = []

        # Fix formatting issues then parse json state
        state = datapoint.state
        actions = datapoint.joint_action
        state_dict, insertion_needed = process_state(
            state, forward_port, fix_json
        )
        actions = process_actions(actions)

        # take care of insertion of interact
        if insert_interacts and insertion_needed:
            if verbose:
                print("INSERTING NEEDED, PERFORMING")
            state_dict_insert, actions_insert = insert_cooking_interact(
                state_dict
            )
            if verbose:
                display_state_dict_and_action(
                    state_dict_insert, actions_insert
                )
            state_action_pairs[layout_name].append(
                (state_dict_insert, actions_insert)
            )
        if verbose:
            display_state_dict_and_action(state_dict, actions)
        state_action_pairs[layout_name].append((state_dict, actions))
    if verbose:
        print("Done processing raw data!")

    # The tag to the file such that we know whether insertion has been performed
    filename = os.path.basename(data_infile).split(".")[0]
    tag = "inserted" if insert_interacts else "original"

    data_outfile = os.path.join(
        data_outdir, filename + "_state_dict_and_action_{}.pickle".format(tag)
    )
    with open(data_outfile, "wb") as f:
        pickle.dump(state_action_pairs, f)

    return data_outfile


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--data-infile", type=str, required=True)
    parser.add_argument("-o", "--data-outdir", type=str, required=True)
    parser.add_argument("-ii", "-insert-interacts", action="store_true")
    parser.add_argument("-j", "--fix-json", action="store_true")
    parser.add_argument("-fp", "--forward-port", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = vars(parser.parse_args())
    main(**args)


# def process_soup_not_held(soup_not_held):
#     # convert the soup not held by a player
#     assert soup_not_held['name'] == 'soup'
#     position = tuple(soup_not_held['position'])
#     new_soup_not_held = {
#         'name': 'soup',
#         'position': position,
#     }
#     type, num_onion_in_soup, cooking_tick = soup_not_held['state']
#     cooking_tick = min(20, cooking_tick)
#     assert type == "onion", "data is corrupted, because the type must be onion in old dynamics"
#     new_soup_not_held['_ingredients'] = [{'name': 'onion', 'position': position}] * num_onion_in_soup
#     new_soup_not_held['_cooking_tick'] = cooking_tick if cooking_tick > 0 else -1
#     new_soup_not_held['cooking_tick'] = new_soup_not_held['_cooking_tick'] # duplicate tag
#     new_soup_not_held['cook_time'] = 20 if cooking_tick > 0 else -1
#     new_soup_not_held['is_ready'] = cooking_tick == 20
#     new_soup_not_held['is_idle'] = cooking_tick == 0
#     new_soup_not_held['is_cooking'] = not new_soup_not_held['is_idle'] and not new_soup_not_held['is_ready']
#     # this is the flag to signal if the soup just started cooking (and an additional frame with interact is required)
#     insertion_needed_i = cooking_tick == 1
#     return new_soup_not_held, insertion_needed_i


# def process_held_object(held_object):
#     # convert held_object from old format to new format
#     position = tuple(held_object['position'])
#     new_held_object = {
#         'name': held_object['name'],
#         'position': position
#     }
#     # handling all the new tags for soup
#     if held_object['name'] == 'soup':
#         # only 3 onion soup is allowed in the old dynamics
#         new_held_object['_ingredients'] = [{'name': 'onion', 'position': position}] * 3
#         new_held_object['cooking_tick'] = 20
#         new_held_object['is_cooking'] = False
#         new_held_object['is_ready'] = True
#         new_held_object['is_idle'] = False
#         new_held_object['cook_time'] = 20
#         new_held_object['_cooking_tick'] = 20
#     return new_held_object

# def old_state_dict_to_new_state_dict(old_state_dict):
#     """
#     Arguments:
#         old_state_dict (python dictionary): state dict in the old dynamics
#     Return:
#         new_state_dict (python dictionary): state dict in the new dynamics
#         insertion_needed (bool): whether we need to insert an additional frame with interact to start soup cooking
#     """
#     # default insertion needed to false
#     insertion_needed = False

#     # players: tuple
#     players = old_state_dict["players"]

#     new_players = []
#     for player in players:
#         # convert position and orientation
#         new_player = {
#             'position': tuple(player['position']),
#             'orientation': tuple(player['orientation']),
#         }
#         if player.get('held_object', None):
#             new_held_object = process_held_object(player['held_object'])

#         else:
#             new_held_object = None
#         new_player['held_object'] = new_held_object

#         new_players.append(new_player)

#     objects = old_state_dict["objects"]
#     new_objects = []

#     for o in objects:
#         if o['name'] == 'soup':
#             processed_soup, insertion_needed_i = process_soup_not_held(o)
#             # update insertion
#             insertion_needed = insertion_needed or insertion_needed_i
#             new_objects.append(processed_soup)
#         else:
#             processed_object = {
#                 'name': o['name'],
#                 'position': tuple(o['position'])
#             }
#             new_objects.append(processed_object)

#     return {
#         "players": new_players,
#         "objects": new_objects,
#         "bonus_orders": [], # no bonus order in old dynamics
#         "all_orders": [{'ingredients': ('onion', 'onion', 'onion')}], # 3 onion soup only in old dynamics
#         "timestep": 0 # FIXME: This still needs to be fixed
#     }, insertion_needed
