import copy
import json
import os
import random
from collections import defaultdict
from typing import DefaultDict

import numpy as np
import pandas as pd
from numpy.core.numeric import full

from human_aware_rl.human.data_processing_utils import (
    convert_joint_df_trajs_to_overcooked_single,
    df_traj_to_python_joint_traj,
    is_button_press,
    is_interact,
)
from human_aware_rl.static import *
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.overcooked_trajectory import append_trajectories
from overcooked_ai_py.utils import mean_and_std_err

######################
# HIGH LEVEL METHODS #
######################


def get_human_human_trajectories(
    layouts, dataset_type="train", data_path=None, **kwargs
):
    """
    Get human-human trajectories for a layout. Automatically

    Arguments:
        layouts (list): List of strings corresponding to layouts we wish to retrieve data for
        data_path (str): Full path to pickled DataFrame we wish to load. If not specified, default to CLEAN_{2019|2020}_HUMAN_DATA_{train|test|all}
        dataset_type (str): Either 'train', 'test', or 'all', determines which data to load if data_path=None

    Keyword Arguments:
        featurize_states (bool): Whether the states in returned trajectories should be OvercookedState objects (false) or vectorized np.Arrays (true)
        check_trajectories (bool): If True, we ensure the consistency of the MDP dynamics within the trajectory. This is slow and has lots of overhead
        silent (bool): If true, silence logging and print statements
    """
    if not set(layouts).issubset(LAYOUTS_WITH_DATA):
        # Note: doesn't necessarily mean we'll find data for this layout as this is a loose check
        # for example, if layouts=['cramped_room'] and the data path is CLEAN_HUMAN_DATA_{train|test|all}, no data will be found
        raise ValueError("Layout for which no data collected detected")

    if data_path and not os.path.exists(data_path):
        raise FileNotFoundError(
            "Tried to load human data from {} but file does not exist!".format(
                data_path
            )
        )

    data = {}

    # Determine which paths are needed for which layouts (according to hierarchical path resolution rules outlined in docstring)
    data_path_to_layouts = DefaultDict(list)
    for layout in layouts:
        curr_data_path = _get_data_path(layout, dataset_type, data_path)
        data_path_to_layouts[curr_data_path].append(layout)

    # For each data path, load data once and parse trajectories for all corresponding layouts
    for data_path in data_path_to_layouts:
        curr_data = get_trajs_from_data(
            curr_data_path, layouts=[layout], **kwargs
        )[0]
        data = append_trajectories(data, curr_data)

    # Return all accumulated data for desired layouts
    return data


def csv_to_df_pickle(
    csv_path,
    out_dir,
    out_file_prefix,
    button_presses_threshold=0.25,
    perform_train_test_split=True,
    silent=True,
    **kwargs
):
    """
    High level function that converts raw CSV data into well formatted and cleaned pickled pandas dataframes.

    Arguments:
        - csv_path (str): Full path to human csv data
        - out_dir(str): Full path to directory where cleaned data will be saved
        - out_file_prefix(str): common prefix for all saved files
        - button_presses_threshold (float): minimum button presses per timestep over rollout required to
            keep entire game
        - perform_train_test_split (bool): Whether to partition dataset into training and testing portions
        - kwargs (dict): keyword args to pass to all helper functions

    After running, the following files are created

        if traintest_split:
            /{out_dir}
                - {out_file_prefix}_all.pickle
                - {out_file_prefix}_train.pickle
                - {out_file_prefix}_test.pickle
        else:
            /{out_dir}
                - {out_file_prefix}_all.pickle

    Returns:
        if perform_train_test_split:
            - tuple(pd.DataFrame, pd.DateFrame): tuple of train data, test data
        else:
            - clean_trials (pd.DataFrame): Dataframe containing _all_ cleaned and formatted transitions
    """
    if not silent:
        print("Loading raw data from", csv_path)
    all_trials = pd.read_csv(csv_path)
    if not silent:
        print("Success")

    if not silent:
        print("Raw data columns:", all_trials.columns)

    if not silent:
        print("Formatting...")
    all_trials = format_trials_df(all_trials, silent=silent, **kwargs)
    if not silent:
        print("Success!")

    def filter_func(row):
        return row["button_presses_per_timstep"] >= button_presses_threshold

    if not silent:
        print("Filtering...")
    clean_trials = filter_trials(all_trials, filter_func, **kwargs)
    if not silent:
        print("Success!")

    full_outfile_prefix = os.path.join(out_dir, out_file_prefix)
    if not silent:
        print("Saving processed pickle data with prefix", full_outfile_prefix)
    clean_trials.to_pickle(full_outfile_prefix + "_all.pickle")
    if not silent:
        print("Success!")

    if perform_train_test_split:
        if not silent:
            print("Performing train/test split...")
        cleaned_trials_dict = train_test_split(clean_trials, **kwargs)
        layouts = np.unique(clean_trials["layout_name"])
        train_trials = pd.concat(
            [cleaned_trials_dict[layout]["train"] for layout in layouts]
        )
        test_trials = pd.concat(
            [cleaned_trials_dict[layout]["test"] for layout in layouts]
        )
        clean_trials = pd.concat([train_trials, test_trials])
        train_trials.to_pickle(full_outfile_prefix + "_train.pickle")
        test_trials.to_pickle(full_outfile_prefix + "_test.pickle")
        if not silent:
            print("Success!")

    return clean_trials


#############################
# DATAFRAME TO TRAJECTORIES #
#############################


def get_trajs_from_data(data_path, layouts, silent=True, **kwargs):
    """
    Converts and returns trajectories from dataframe at `data_path` to overcooked trajectories.
    """
    if not silent:
        print("Loading data from {}".format(data_path))

    main_trials = pd.read_pickle(data_path)

    trajs, info = convert_joint_df_trajs_to_overcooked_single(
        main_trials, layouts, silent=silent, **kwargs
    )

    return trajs, info


############################
# DATAFRAME PRE-PROCESSING #
############################


def format_trials_df(trials, clip_400=False, silent=False, **kwargs):
    """Get trials for layouts in standard format for data exploration, cumulative reward and length information + interactivity metrics"""
    layouts = np.unique(trials["layout_name"])
    if not silent:
        print("Layouts found", layouts)

    if clip_400:
        trials = trials[trials["cur_gameloop"] <= 400]

    # Add game length for each round
    trials = trials.join(
        trials.groupby(["trial_id"])["cur_gameloop"].count(),
        on=["trial_id"],
        rsuffix="_total",
    )

    # Calculate total reward for each round
    trials = trials.join(
        trials.groupby(["trial_id"])["score"].max(),
        on=["trial_id"],
        rsuffix="_total",
    )

    # Add interactivity metadata
    trials = _add_interactivity_metrics(trials)
    trials["button_presses_per_timstep"] = (
        trials["button_press_total"] / trials["cur_gameloop_total"]
    )

    return trials


def filter_trials(trials, filter, **kwargs):
    """
    Prune games based on user-defined fileter function

    Note: 'filter' must accept a single row as input and whether the entire trial should be kept
    based on its first row
    """
    trial_ids = np.unique(trials["trial_id"])

    cleaned_trial_dfs = []
    for trial_id in trial_ids:
        curr_trial = trials[trials["trial_id"] == trial_id]

        # Discard entire trials based on filter function applied to first row
        element = curr_trial.iloc[0]
        keep = filter(element)
        if keep:
            cleaned_trial_dfs.append(curr_trial)

    return pd.concat(cleaned_trial_dfs)


def filter_transitions(trials, filter):
    """
    Prune games based on user-defined fileter function

    Note: 'filter' must accept a pandas Series as input and return a Series of booleans
    where the ith boolean is True if the ith entry should be kept
    """
    trial_ids = np.unique(trials["trial_id"])

    cleaned_trial_dfs = []
    for trial_id in trial_ids:
        curr_trial = trials[trials["trial_id"] == trial_id]

        # Discard entire trials based on filter function applied to first row
        keep = filter(curr_trial)
        curr_trial_kept = curr_trial[keep]
        cleaned_trial_dfs.append(curr_trial_kept)

    return pd.concat(cleaned_trial_dfs)


def train_test_split(trials, train_size=0.7, print_stats=False):
    cleaned_trials_dict = defaultdict(dict)

    layouts = np.unique(trials["layout_name"])
    for layout in layouts:
        # Gettings trials for curr layout
        curr_layout_trials = trials[trials["layout_name"] == layout]

        # Get all trial ids for the layout
        curr_trial_ids = np.unique(curr_layout_trials["trial_id"])

        # Split trials into train and test sets
        random.shuffle(curr_trial_ids)
        mid_idx = int(np.ceil(len(curr_trial_ids) * train_size))
        train_trials, test_trials = (
            curr_trial_ids[:mid_idx],
            curr_trial_ids[mid_idx:],
        )
        assert (
            len(train_trials) > 0 and len(test_trials) > 0
        ), "Cannot have empty split"

        # Get corresponding trials
        layout_train = curr_layout_trials[
            curr_layout_trials["trial_id"].isin(train_trials)
        ]
        layout_test = curr_layout_trials[
            curr_layout_trials["trial_id"].isin(test_trials)
        ]

        train_dset_avg_rew = int(np.mean(layout_train["score_total"]))
        test_dset_avg_rew = int(np.mean(layout_test["score_total"]))

        if print_stats:
            print(
                "Layout: {}\nNum Train Trajs: {}\nTrain Traj Average Rew: {}\nNum Test Trajs: {}\nTest Traj Average Rew: {}".format(
                    layout,
                    len(train_trials),
                    train_dset_avg_rew,
                    len(test_trials),
                    test_dset_avg_rew,
                )
            )

        cleaned_trials_dict[layout]["train"] = layout_train
        cleaned_trials_dict[layout]["test"] = layout_test
    return cleaned_trials_dict


def get_trials_scenario_and_worker_rews(trials):
    scenario_rews = defaultdict(list)
    worker_rews = defaultdict(list)
    for _, trial in trials.groupby("trial_id"):
        datapoint = trial.iloc[0]
        layout = datapoint["layout_name"]
        player_0, player_1 = datapoint["player_0_id"], datapoint["player_1_id"]
        tot_rew = datapoint["score_total"]
        scenario_rews[layout].append(tot_rew)
        worker_rews[player_0].append(tot_rew)
        worker_rews[player_1].append(tot_rew)
    return dict(scenario_rews), dict(worker_rews)


#####################
# Lower-level Utils #
#####################


def remove_worker(trials, worker_id):
    return trials[
        trials["player_0_id"] != worker_id & trials["player_1_id"] != worker_id
    ]


def remove_worker_on_map(trials, workerid_num, layout):
    to_remove = (
        (trials["player_0_id"] == workerid_num)
        | (trials["player_1_id"] == workerid_num)
    ) & (trials["layout_name"] == layout)
    to_keep = ~to_remove
    assert to_remove.sum() > 0
    return trials[to_keep]


def _add_interactivity_metrics(trials):
    # this method is non-destructive
    trials = trials.copy()

    # whether any human INTERACT actions were performed
    is_interact_row = lambda row: int(
        np.sum(
            np.array([row["player_0_is_human"], row["player_1_is_human"]])
            * is_interact(row["joint_action"])
        )
        > 0
    )
    # Whehter any human keyboard stroked were performed
    is_button_press_row = lambda row: int(
        np.sum(
            np.array([row["player_0_is_human"], row["player_1_is_human"]])
            * is_button_press(row["joint_action"])
        )
        > 0
    )

    # temp column to split trajectories on INTERACTs
    trials["interact"] = trials.apply(is_interact_row, axis=1).cumsum()
    trials["dummy"] = 1

    # Temp column indicating whether current timestep required a keyboard press
    trials["button_press"] = trials.apply(is_button_press_row, axis=1)

    # Add 'button_press_total' column to each game indicating total number of keyboard strokes
    trials = trials.join(
        trials.groupby(["trial_id"])["button_press"].sum(),
        on=["trial_id"],
        rsuffix="_total",
    )

    # Count number of timesteps elapsed since last human INTERACT action
    trials["timesteps_since_interact"] = (
        trials.groupby(["interact"])["dummy"].cumsum() - 1
    )

    # Drop temp columns
    trials = trials.drop(columns=["interact", "dummy"])

    return trials


def _get_data_path(layout, dataset_type, data_path):
    if data_path:
        return data_path
    if dataset_type == "train":
        return (
            CLEAN_2019_HUMAN_DATA_TRAIN
            if layout in LAYOUTS_WITH_DATA_2019
            else CLEAN_2020_HUMAN_DATA_TRAIN
        )
    if dataset_type == "test":
        return (
            CLEAN_2019_HUMAN_DATA_TEST
            if layout in LAYOUTS_WITH_DATA_2019
            else CLEAN_2020_HUMAN_DATA_TEST
        )
    if dataset_type == "all":
        return (
            CLEAN_2019_HUMAN_DATA_ALL
            if layout in LAYOUTS_WITH_DATA_2019
            else CLEAN_2020_HUMAN_DATA_ALL
        )


##############
# DEPRECATED #
##############


def trial_type_by_unique_id_dict(trial_questions_df):
    trial_type_dict = {}
    unique_ids = trial_questions_df["workerid"].unique()
    for unique_id in unique_ids:
        person_data = trial_questions_df[
            trial_questions_df["workerid"] == unique_id
        ]
        model_type, player_index = (
            person_data["MODEL_TYPE"].iloc[0],
            int(person_data["PLAYER_INDEX"].iloc[0]),
        )
        trial_type_dict[unique_id] = (model_type, player_index)
    return trial_type_dict


def add_means_and_stds_from_df(data, main_trials, algo_name):
    """Calculate means and SEs for each layout, and add them to the data dictionary under algo name `algo`"""
    layouts = [
        "asymmetric_advantages",
        "coordination_ring",
        "cramped_room",
        "random0",
        "random3",
    ]
    for layout in layouts:
        layout_trials = main_trials[main_trials["layout_name"] == layout]

        idx_1_workers = []
        idx_0_workers = []
        for worker_id in layout_trials["player_0_id"].unique():
            if layout_trials[layout_trials["player_0_id"] == worker_id][
                "player_0_is_human"
            ][0]:
                idx_0_workers.append(worker_id)

        for worker_id in layout_trials["player_1_id"].unique():
            if layout_trials[layout_trials["player_1_id"] == worker_id][
                "player_1_is_human"
            ][0]:
                idx_1_workers.append(worker_id)

        idx_0_trials = layout_trials[
            layout_trials["player_0_id"].isin(idx_0_workers)
        ]
        data[layout][algo_name + "_0"] = mean_and_std_err(
            idx_0_trials.groupby("player_0_id")["score_total"].mean()
        )

        idx_1_trials = layout_trials[
            layout_trials["plaer_1_id"].isin(idx_1_workers)
        ]
        data[layout][algo_name + "_1"] = mean_and_std_err(
            idx_1_trials.groupby("plaer_1_id")["score_total"].mean()
        )


def interactive_from_traj_df(df_traj):
    python_traj = df_traj_to_python_joint_traj(df_traj)
    AgentEvaluator.interactive_from_traj(python_traj, traj_idx=0)


def display_interactive_by_workerid(main_trials, worker_id, limit=None):
    print("Displaying main trials for worker", worker_id)
    worker_trials = main_trials[
        main_trials["player_0_id"]
        == worker_id | main_trials["player_1_id"]
        == worker_id
    ]
    count = 0
    for _, rtrials in worker_trials.groupby(["trial_id"]):
        interactive_from_traj_df(rtrials)
        count += 1
        if limit is not None and count >= limit:
            return


def display_interactive_by_layout(main_trials, layout_name, limit=None):
    print("Displaying main trials for layout", layout_name)
    layout_trials = main_trials[main_trials["layout_name"] == layout_name]
    count = 0
    for wid, wtrials in layout_trials.groupby("player_0_id"):
        print("Worker: ", wid)
        for _, rtrials in wtrials.groupby(["trial_id"]):
            interactive_from_traj_df(rtrials)
            count += 1
            if limit is not None and count >= limit:
                return
    for wid, wtrials in layout_trials.groupby("player_1_id"):
        print("Worker: ", wid)
        for _, rtrials in wtrials.groupby(["trial_id"]):
            interactive_from_traj_df(rtrials)
            count += 1
            if limit is not None and count >= limit:
                return
