import argparse
import os

import numpy as np
import pandas as pd

from human_aware_rl.human.data_processing_utils import AI_ID
from human_aware_rl.static import NEW_SCHEMA, OLD_SCHEMA

"""
Script for converting legacy-schema human data to current schema.

Note: This script, and working with the raw CSV files in general, should only be done by advanced users.
It is recommended that most users work with the pre-processed pickle files in /human_aware_rl/data/cleaned.
See docs for more info
"""


def write_csv(data, output_file_path):
    if os.path.exists(output_file_path):
        raise FileExistsError(
            "File {} already exists, aborting to avoid overwriting".format(
                output_file_path
            )
        )
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data.to_csv(output_file_path, index=False)


def main(input_file, output_file, is_human_ai=False):
    ### Load in data ###
    print("Loading data from {}...".format(input_file))
    data = pd.read_csv(input_file, header=0)
    print("Success!")

    ### Update schema ###
    print("Updating schema...")

    # Ensure proper legacy schema
    assert set(data.columns) == OLD_SCHEMA, "Input data has unexected schema"

    # add unique trial_id to each game. A game is defined as a single trajectory on a single layout.
    # This only works because the data is stored in chronological order
    data["trial_id"] = (
        data["layout_name"] != data["layout_name"].shift(1)
    ).astype(int).cumsum() - 1

    # Unique for each human-human pairing. Note, one pairing will play multiple games
    data["pairing_id"] = (
        (data["workerid_num"] != data["workerid_num"].shift(1))
        .astype(int)
        .cumsum()
    )

    # Drop redundant games
    # Note: this is necessary due to how data was collected on the backend. If player A and B are paired, the game is recorded twice.
    # once with player A as player 0 and once with player B as player 0
    data = data[data["is_leader"]]

    if not is_human_ai:
        data["player_0_is_human"] = True
        data["player_1_is_human"] = True
        data["player_0_id"] = (data["pairing_id"] * 2).astype(str)
        data["player_1_id"] = (data["pairing_id"] * 2 + 1).astype(str)
    else:
        data["player_0_is_human"] = True
        data["player_1_is_human"] = False
        data["player_0_id"] = data["pairing_id"].astype(str)
        data["player_1_id"] = AI_ID

    columns_to_drop = (OLD_SCHEMA - NEW_SCHEMA).union(set(["pairing_id"]))
    data = data.drop(columns=columns_to_drop)

    assert set(data.columns == NEW_SCHEMA), "Output data has misformed schema"
    print("Success!")

    ### Write out data ###
    print("Writing data to {}...".format(output_file))
    write_csv(data, output_file)
    print("Success!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        "-i",
        type=str,
        required=True,
        help="path to old-schema data",
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        required=True,
        help="path to save new-schema data",
    )
    parser.add_argument(
        "--is_human_ai",
        "-ai",
        action="store_true",
        help="Provide this flag if data from human-AI games",
    )

    args = vars(parser.parse_args())
    main(**args)
