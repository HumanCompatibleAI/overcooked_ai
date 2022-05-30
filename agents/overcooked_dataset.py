from arguments import get_arguments
from state_encodings import ENCODING_SCHEMES
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, OvercookedGridworld, Direction, Action
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS


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
        self.main_trials = self.main_trials[self.main_trials['layout_name'] == args.layout]
        # print(self.main_trials['layout_name'])

        def str_to_actions(joint_action):
            """Convert df cell format of a joint action to a joint action as a tuple of indices"""
            joint_action = json.loads(joint_action)
            for i in range(2):
                if type(joint_action[i]) is list:
                    joint_action[i] = tuple(joint_action[i])
                if type(joint_action[i]) is str:
                    joint_action[i] = joint_action[i].lower()
                assert joint_action[i] in Action.ALL_ACTIONS
            return np.array([Action.ACTION_TO_INDEX[a] for a in joint_action])

        def str_to_obss(df):
            """Convert from a df cell format of a state to an Overcooked State"""
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
        # print(self.main_trials)

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