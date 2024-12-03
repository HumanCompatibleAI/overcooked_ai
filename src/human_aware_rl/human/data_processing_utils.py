import json
import time

import numpy as np

from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import (
    ObjectState,
    OvercookedGridworld,
    OvercookedState,
    PlayerState,
)

AI_ID = "I am robot!"


####################
# CONVERSION UTILS #
####################


def json_action_to_python_action(action):
    if type(action) is list:
        action = tuple(action)
    if type(action) is str:
        action = action.lower()
    assert action in Action.ALL_ACTIONS
    return action


def json_joint_action_to_python_action(json_joint_action):
    """Port format from javascript to python version of Overcooked"""
    if type(json_joint_action) is str:
        try:
            json_joint_action = json.loads(json_joint_action)
        except json.decoder.JSONDecodeError:
            # hacky fix to circumvent 'INTERACT' action being malformed json (because of single quotes)
            # Might need to find a more robust way around this in the future
            json_joint_action = eval(json_joint_action)
    return tuple(json_action_to_python_action(a) for a in json_joint_action)


def json_state_to_python_state(df_state):
    """Convert from a df cell format of a state to an Overcooked State"""
    if type(df_state) is str:
        df_state = json.loads(df_state)

    return OvercookedState.from_dict(df_state)


def is_interact(joint_action):
    joint_action = json_joint_action_to_python_action(joint_action)
    return np.array(
        [
            int(joint_action[0] == Action.INTERACT),
            int(joint_action[1] == Action.INTERACT),
        ]
    )


def is_button_press(joint_action):
    joint_action = json_joint_action_to_python_action(joint_action)
    return np.array(
        [
            int(joint_action[0] != Action.STAY),
            int(joint_action[1] != Action.STAY),
        ]
    )


def extract_df_for_worker_on_layout(main_trials, worker_id, layout_name):
    """
    WARNING: this function has been deprecated and is no longer compatible with current schema
    Extract trajectory for a specific layout and worker pair from main_trials df
    """
    worker_trajs_df = main_trials[main_trials["workerid_num"] == worker_id]
    worker_layout_traj_df = worker_trajs_df[
        worker_trajs_df["layout_name"] == layout_name
    ]
    return worker_layout_traj_df


def df_traj_to_python_joint_traj(
    traj_df, check_trajectories=True, silent=True, **kwargs
):
    if len(traj_df) == 0:
        return None

    datapoint = traj_df.iloc[0]
    layout_name = datapoint["layout_name"]
    agent_evaluator = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout_name},
        env_params={
            "horizon": 1250
        },  # Defining the horizon of the mdp of origin of the trajectories
    )
    mdp = agent_evaluator.env.mdp
    env = agent_evaluator.env

    overcooked_states = [json_state_to_python_state(s) for s in traj_df.state]
    overcooked_actions = [
        json_joint_action_to_python_action(joint_action)
        for joint_action in traj_df.joint_action
    ]
    overcooked_rewards = list(traj_df.reward)

    assert (
        sum(overcooked_rewards) == datapoint.score_total
    ), "Rewards didn't sum up to cumulative rewards. Probably trajectory df is corrupted / not complete"

    trajectories = {
        "ep_states": [overcooked_states],
        "ep_actions": [overcooked_actions],
        "ep_rewards": [overcooked_rewards],  # Individual (dense) reward values
        "ep_dones": [
            [False] * len(overcooked_states)
        ],  # Individual done values
        "ep_infos": [{}] * len(overcooked_states),
        "ep_returns": [
            sum(overcooked_rewards)
        ],  # Sum of dense rewards across each episode
        "ep_lengths": [len(overcooked_states)],  # Lengths of each episode
        "mdp_params": [mdp.mdp_params],
        "env_params": [env.env_params],
        "metadatas": {
            "player_0_id": [datapoint["player_0_id"]],
            "player_1_id": [datapoint["player_1_id"]],
            "env": [agent_evaluator.env],
        },
    }
    trajectories = {
        k: np.array(v) if k not in ["ep_actions", "metadatas"] else v
        for k, v in trajectories.items()
    }

    if check_trajectories:
        agent_evaluator.check_trajectories(trajectories, verbose=not silent)
    return trajectories


def convert_joint_df_trajs_to_overcooked_single(
    main_trials, layouts, silent=False, **kwargs
):
    """
    Takes in a dataframe `main_trials` containing joint trajectories, and extract trajectories of workers `worker_ids`
    on layouts `layouts`, with specific options.
    """

    single_agent_trajectories = {
        # With shape (n_episodes, game_len), where game_len might vary across games:
        "ep_states": [],
        "ep_actions": [],
        "ep_rewards": [],  # Individual reward values
        "ep_dones": [],  # Individual done values
        "ep_infos": [],
        # With shape (n_episodes, ):
        "ep_returns": [],  # Sum of rewards across each episode
        "ep_lengths": [],  # Lengths of each episode
        "mdp_params": [],
        "env_params": [],
        "metadatas": {"ep_agent_idxs": []},  # Agent index for current episode
    }

    human_indices = []
    num_trials_for_layout = {}
    for layout_name in layouts:
        trial_ids = np.unique(
            main_trials[main_trials["layout_name"] == layout_name]["trial_id"]
        )
        num_trials = len(trial_ids)
        num_trials_for_layout[layout_name] = num_trials

        if num_trials == 0:
            print(
                "WARNING: No trajectories found on {} layout!".format(
                    layout_name
                )
            )

        for trial_id in trial_ids:
            # Get an single game
            one_traj_df = main_trials[main_trials["trial_id"] == trial_id]

            # Get python trajectory data and information on which player(s) was/were human
            joint_traj_data = df_traj_to_python_joint_traj(
                one_traj_df, silent=silent, **kwargs
            )

            human_idx = get_human_player_index_for_df(one_traj_df)
            human_indices.append(human_idx)

            # Convert joint trajectories to single agent trajectories, appending recovered info to the `trajectories` dict
            joint_state_trajectory_to_single(
                single_agent_trajectories, joint_traj_data, human_idx, **kwargs
            )

    if not silent:
        print(
            "Number of trajectories processed for each layout: {}".format(
                num_trials_for_layout
            )
        )
    return single_agent_trajectories, human_indices


def get_human_player_index_for_df(one_traj_df):
    """Determines which player index had a human player"""
    human_player_indices = []
    assert len(one_traj_df["player_0_id"].unique()) == 1
    assert len(one_traj_df["player_1_id"].unique()) == 1
    datapoint = one_traj_df.iloc[0]
    if datapoint["player_0_is_human"]:
        human_player_indices.append(0)
    if datapoint["player_1_is_human"]:
        human_player_indices.append(1)

    return human_player_indices


def joint_state_trajectory_to_single(
    trajectories,
    joint_traj_data,
    player_indices_to_convert=None,
    featurize_states=True,
    silent=False,
    **kwargs
):
    """
    Take a joint trajectory and split it into two single-agent trajectories, adding data to the `trajectories` dictionary
    player_indices_to_convert: which player indexes' trajs we should return
    """

    env = joint_traj_data["metadatas"]["env"][0]

    assert (
        len(joint_traj_data["ep_states"]) == 1
    ), "This method only takes in one trajectory"
    states, joint_actions = (
        joint_traj_data["ep_states"][0],
        joint_traj_data["ep_actions"][0],
    )
    rewards, length = (
        joint_traj_data["ep_rewards"][0],
        joint_traj_data["ep_lengths"][0],
    )

    # Getting trajectory for each agent
    for agent_idx in player_indices_to_convert:
        ep_obs, ep_acts, ep_dones = [], [], []
        for i in range(len(states)):
            state, action = states[i], joint_actions[i][agent_idx]

            if featurize_states:
                action = np.array([Action.ACTION_TO_INDEX[action]]).astype(int)
                state = env.featurize_state_mdp(state)[agent_idx]

            ep_obs.append(state)
            ep_acts.append(action)
            ep_dones.append(False)

        ep_dones[-1] = True

        trajectories["ep_states"].append(ep_obs)
        trajectories["ep_actions"].append(ep_acts)
        trajectories["ep_rewards"].append(rewards)
        trajectories["ep_dones"].append(ep_dones)
        trajectories["ep_infos"].append([{}] * len(rewards))
        trajectories["ep_returns"].append(sum(rewards))
        trajectories["ep_lengths"].append(length)
        trajectories["mdp_params"].append(env.mdp.mdp_params)
        trajectories["env_params"].append({})
        trajectories["metadatas"]["ep_agent_idxs"].append(agent_idx)
