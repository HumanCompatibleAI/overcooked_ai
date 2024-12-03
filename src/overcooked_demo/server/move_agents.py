import argparse
import json
import os
import shutil
import sys


def main():
    with open("config.json", "r") as f:
        config = json.load(f)
    # the agents dir
    agent_dir = config["AGENT_DIR"]
    parser = argparse.ArgumentParser(
        prog="move_agent",
        description="Create a directory for agent to be loaded into the game",
    )
    parser.add_argument(
        "checkpoint",
        help="The path to the checkpoint directory, e.g. ~/ray_results/run_xyz/checkpoint_000500",
    )
    parser.add_argument(
        "agent_name",
        help="The name you want for this agent; remember to follow the naming conventions: the name must start with 'Rllib'",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        default=False,
        help="Whether to overwrite existing agent if one with the same name already exists",
    )
    parser.add_argument(
        "-b",
        "--bc",
        default=None,
        help="If the agent was trained with BC agent, provide the path to the saved bc model directory",
    )

    args = parser.parse_args()
    checkpoint, agent_name, overwrite, bc_model = (
        args.checkpoint,
        args.agent_name,
        args.overwrite == "True",
        args.bc,
    )

    if agent_name.lower()[:5] != "rllib":
        sys.exit("Incampatible agent name")
    elif agent_name in os.listdir(agent_dir) and not overwrite:
        sys.exit("agent name already exists")

    # make a new directory for the agent
    new_agent_dir = os.path.join(agent_dir, agent_name, "agent")
    if os.path.exists(new_agent_dir):
        parent_dir = os.path.dirname(new_agent_dir)
        shutil.rmtree(parent_dir)
    # copy over files
    shutil.copytree(checkpoint, new_agent_dir)

    # copy over the config.pickle file
    run_dir = os.path.dirname(checkpoint)
    new_dir = os.path.dirname(new_agent_dir)
    shutil.copy(
        os.path.join(run_dir, "config.pkl"),
        os.path.join(new_dir, "config.pkl"),
    )

    # if bc_model is provided
    if bc_model:
        bc_params = os.path.join(new_dir, "bc_params")
        if not os.path.exists(bc_model):
            sys.exit("bc_model dir doesn't exist")
        shutil.copytree(bc_model, bc_params)

    sys.exit("Copy succeeded")


if __name__ == "__main__":
    main()
