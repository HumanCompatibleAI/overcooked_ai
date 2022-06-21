from arguments import get_arguments
from state_encodings import ENCODING_SCHEMES
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, OvercookedGridworld, Direction, Action
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS

from copy import deepcopy
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class OvercookedDataset(Dataset):
    def __init__(self, env, encoding_fn, args, add_subtask_info=True):
        self.env = env
        self.add_subtask_info = add_subtask_info
        self.encode_state_fn = encoding_fn
        self.data_path = args.base_dir / args.data_path / args.dataset
        self.main_trials = pd.read_pickle(self.data_path)
        print(f'Number of all trials: {len(self.main_trials)}')
        self.main_trials = self.main_trials[self.main_trials['layout_name'] == args.layout]
        print(f'Number of {args.layout} trials: {len(self.main_trials)}')
        # print(self.main_trials['layout_name'])

        self.action_ratios = {k: 0 for k in Action.ALL_ACTIONS}

        def str_to_actions(joint_action):
            """
            Convert df cell format of a joint action to a joint action as a tuple of indices.
            Used to convert pickle files which are stored as strings into np.arrays
            """
            try:
                joint_action = json.loads(joint_action)
            except json.decoder.JSONDecodeError:
                # Hacky fix taken from https://github.com/HumanCompatibleAI/human_aware_rl/blob/master/human_aware_rl/human/data_processing_utils.py#L29
                joint_action = eval(joint_action)
            for i in range(2):
                if type(joint_action[i]) is list:
                    joint_action[i] = tuple(joint_action[i])
                if type(joint_action[i]) is str:
                    joint_action[i] = joint_action[i].lower()
                assert joint_action[i] in Action.ALL_ACTIONS
                self.action_ratios[joint_action[i]] += 1
            return np.array([Action.ACTION_TO_INDEX[a] for a in joint_action])

        def str_to_obss(df):
            """
            Convert from a df cell format of a state to an Overcooked State
            Used to convert pickle files which are stored as strings into overcooked states
            """
            state = df['state']
            if type(state) is str:
                state = json.loads(state)
            state = OvercookedState.from_dict(state)
            visual_obs, agent_obs = self.encode_state_fn(env.mdp, state, args.horizon)
            df['state'] = state
            df['visual_obs'] = visual_obs
            df['agent_obs'] = agent_obs
            return df

        self.main_trials['joint_action'] = self.main_trials['joint_action'].apply(str_to_actions)
        self.main_trials = self.main_trials.apply(str_to_obss, axis=1)

        if add_subtask_info:
            self.subtasks = ['get_onion_from_dispenser', 'get_onion_from_counter', 'put_onion_in_pot',
                             'put_onion_closer',
                             'get_plate_from_dish_rack', 'get_plate_from_counter', 'put_plate_closer', 'get_soup',
                             'get_soup_from_counter', 'put_soup_closer', 'serve_soup', 'unknown']
            self.num_subtasks = len(self.subtasks)
            self.add_subtasks()

        # Remove all transitions where both players noop-ed
        self.main_trials = self.main_trials[self.main_trials['joint_action'] != '[[0, 0], [0, 0]]']
        print(f'Number of {args.layout} trials without double noops: {len(self.main_trials)}')

        # Calculate class weights for cross entropy
        self.action_weights = np.zeros(6)
        for action in Action.ALL_ACTIONS:
            self.action_weights[Action.ACTION_TO_INDEX[action]] = self.action_ratios[action]
        self.action_weights = 1.0 / self.action_weights
        self.action_weights = len(Action.ALL_ACTIONS) * self.action_weights / self.action_weights.sum()

    def get_action_weights(self):
        return self.action_weights

    def get_subtask_weights(self):
        return self.subtask_weights

    def __len__(self):
        return len(self.main_trials)

    def __getitem__(self, item):
        data_point = self.main_trials.iloc[item]
        a = {
            'visual_obs': data_point['visual_obs'],
            'agent_obs': data_point['agent_obs'],
            'joint_action': data_point['joint_action'],
            'subtasks': np.array( [data_point['p1_subtask'], data_point['p2_subtask']] )
        }
        return a

    def add_subtasks(self):
        prev_trials = []
        curr_trial = None
        curr_objs = None
        subtask_start_idx = [0, 0]
        interact_id = Action.ACTION_TO_INDEX[Action.INTERACT]
        subtasks_to_id = {s: i for i, s in enumerate(self.subtasks)}
        id_to_subtasks = {v: k for k, v in subtasks_to_id.items()}

        def facing(layout, player):
            '''Returns what object the player is facing'''
            x, y = np.array(player.position) + np.array(player.orientation)
            layout = [[t for t in row.strip("[]'")] for row in layout.split("', '")]
            return layout[y][x]

        self.main_trials['p1_subtask'] = None
        self.main_trials['p2_subtask'] = None
        for index, row in tqdm(self.main_trials.iterrows()):
            if row['trial_id'] != curr_trial:
                # Start of a new trial
                # if row['cur_gameloop'] != 0:
                #     print(row)
                # assert row['cur_gameloop'] == 0 # Ensure we are starting trial from the first timestep
                curr_trial = row['trial_id']
                curr_objs = [(p.held_object.name if p.held_object else None) for p in row['state'].players]
                subtask_start_idx = [index, index]

            # For each player
            for i in range(len(row['state'].players)):
                curr_objs = [(p.held_object.name if p.held_object else None) for p in row['state'].players]
                try:
                    next_row = self.main_trials.loc[index + 1]
                except KeyError:
                    subtask = 'unknown'
                    self.main_trials.loc[subtask_start_idx[i]:index, f'p{i + 1}_subtask'] = subtasks_to_id[subtask]
                    continue

                # All subtasks will start and end with an INTERACT action
                if row['joint_action'][i] == interact_id:
                    next_objs = [(p.held_object.name if p.held_object else None) for p in next_row['state'].players]
                    tile_in_front = facing(row['layout'], row['state'].players[i])

                    # Make sure the next row is part of the current
                    if next_row['trial_id'] != curr_trial:
                        subtask = 'unknown'
                    # Object held didn't change -- This interaction didn't actually transition to a new subtask
                    elif curr_objs[i] == next_objs[i]:
                        break
                    # Pick up an onion
                    elif curr_objs[i] is None and next_objs[i] == 'onion':
                        # Facing an onion dispenser
                        if tile_in_front == 'O':
                            subtask = 'get_onion_from_dispenser'
                        # Facing a counter
                        elif tile_in_front == 'X':
                            subtask = 'get_onion_from_counter'
                        else:
                            raise ValueError(
                                f'Unexpected transition. {curr_objs[i]} -> {next_objs[i]} while facing {tile_in_front}')
                    # Place an onion
                    elif curr_objs[i] == 'onion' and next_objs[i] is None:
                        # Facing a pot
                        if tile_in_front == 'P':
                            subtask = 'put_onion_in_pot'
                        # Facing a counter
                        elif tile_in_front == 'X':
                            subtask = 'put_onion_closer'
                        else:
                            raise ValueError(
                                f'Unexpected transition. {curr_objs[i]} -> {next_objs[i]} while facing {tile_in_front}')
                    # Pick up a dish
                    elif curr_objs[i] is None and next_objs[i] == 'dish':
                        # Facing a dish dispenser
                        if tile_in_front == 'D':
                            subtask = 'get_plate_from_dish_rack'
                        # Facing a counter
                        elif tile_in_front == 'X':
                            subtask = 'get_plate_from_counter'
                        else:
                            raise ValueError(
                                f'Unexpected transition. {curr_objs[i]} -> {next_objs[i]} while facing {tile_in_front}')
                    # Place a dish
                    elif curr_objs[i] == 'dish' and next_objs[i] is None:
                        # Facing a counter
                        if tile_in_front == 'X':
                            subtask = 'put_plate_closer'
                        else:
                            raise ValueError(
                                f'Unexpected transition. {curr_objs[i]} -> {next_objs[i]} while facing {tile_in_front}')
                    # Pick up soup from pot using plate
                    elif curr_objs[i] == 'dish' and next_objs[i] == 'soup':
                        # Facing a counter
                        if tile_in_front == 'P':
                            subtask = 'get_soup'
                        else:
                            raise ValueError(
                                f'Unexpected transition. {curr_objs[i]} -> {next_objs[i]} while facing {tile_in_front}')
                    # Pick up soup from counter
                    elif curr_objs[i] is None and next_objs[i] == 'soup':
                        # Facing a counter
                        if tile_in_front == 'X':
                            subtask = 'get_soup_from_counter'
                        else:
                            raise ValueError(
                                f'Unexpected transition. {curr_objs[i]} -> {next_objs[i]} while facing {tile_in_front}')
                    # Place soup
                    elif curr_objs[i] == 'soup' and next_objs[i] is None:
                        # Facing a service station
                        if tile_in_front == 'S':
                            subtask = 'serve_soup'
                        # Facing a counter
                        elif tile_in_front == 'X':
                            subtask = 'put_soup_closer'
                        else:
                            raise ValueError(
                                f'Unexpected transition. {curr_objs[i]} -> {next_objs[i]} while facing {tile_in_front}')
                    else:
                        raise ValueError(
                            f'Unexpected transition. {curr_objs[i]} -> {next_objs[i]}.')

                    self.main_trials.loc[subtask_start_idx[i]:index, f'p{i+1}_subtask'] = subtasks_to_id[subtask]
                    subtask_start_idx[i] = index + 1

        assert not (self.main_trials['p1_subtask'].isna().any())
        assert not (self.main_trials['p2_subtask'].isna().any())

        self.subtask_weights = np.zeros(len(self.subtasks))
        for i in range(2):
            counts = self.main_trials[f'p{i+1}_subtask'].value_counts().to_dict()
            print(f'Player {i+1} subtask splits')
            for k, v in counts.items():
                self.subtask_weights[k] += v
                print(f'{id_to_subtasks[k]}: {v}')
        self.subtask_weights = 1.0 / self.subtask_weights
        self.subtask_weights = len(self.subtasks) * self.subtask_weights / self.subtask_weights.sum()



def main():
    args = get_arguments()
    env = OvercookedEnv.from_mdp(OvercookedGridworld.from_layout_name(args.layout), horizon=400)
    encoding_fn = ENCODING_SCHEMES[args.encoding_fn]
    OD = OvercookedDataset(env, encoding_fn, args)

    dataloader = DataLoader(OD, batch_size=1, shuffle=True, num_workers=0)
    for batch in dataloader:
        print(batch)
        exit(0)


if __name__ == '__main__':
    main()
