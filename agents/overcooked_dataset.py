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


class OvercookedDataset(Dataset):
    def __init__(self, env, encoding_fn, args):
        self.env = env
        self.encode_state_fn = encoding_fn
        self.data_path = args.base_dir / args.data_path / args.dataset
        self.main_trials = pd.read_pickle(self.data_path)
        print(f'Number of all trials: {len(self.main_trials)}')
        self.main_trials = self.main_trials[self.main_trials['layout_name'] == args.layout]
        print(f'Number of {args.layout} trials: {len(self.main_trials)}')
        # Remove all transitions where both players noop-ed
        self.main_trials = self.main_trials[self.main_trials['joint_action'] != '[[0, 0], [0, 0]]']
        print(f'Number of {args.layout} trials without double noops: {len(self.main_trials)}')
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
            df['visual_obs'] = visual_obs
            df['agent_obs'] = agent_obs
            return df

        self.main_trials['joint_action'] = self.main_trials['joint_action'].apply(str_to_actions)
        self.main_trials = self.main_trials.apply(str_to_obss, axis=1)

        # Calculate class weights for cross entropy
        self.class_weights = np.zeros(6)
        for action in Action.ALL_ACTIONS:
            self.class_weights[Action.ACTION_TO_INDEX[action]] = self.action_ratios[action]
        self.class_weights = 1.0 / self.class_weights
        self.class_weights = len(Action.ALL_ACTIONS) * self.class_weights / self.class_weights.sum()

    def get_class_weights(self):
        return self.class_weights

    def __len__(self):
        return len(self.main_trials)

    def __getitem__(self, item):
        data_point = self.main_trials.iloc[item]
        a = {
            'visual_obs': data_point['visual_obs'],
            'agent_obs': data_point['agent_obs'],
            'joint_action': data_point['joint_action']
        }
        return a


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
