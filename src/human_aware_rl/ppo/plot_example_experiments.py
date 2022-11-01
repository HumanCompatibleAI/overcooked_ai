import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from human_aware_rl.utils import *
from human_aware_rl.utils import set_style

envs = [
    "cramped_room",
    "forced_coordination",
    "counter_circuit_o_1",
    "coordination_ring",
    "asymmetric_advantages",
]


def get_list_experiments(path):
    result = {}
    subdirs = [
        name
        for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
    ]
    for env in envs:
        result[env] = {
            "files": [path + "/" + x for x in subdirs if re.search(env, x)]
        }
    return result


def get_statistics(dict):
    for env in dict:
        rewards = [
            get_last_episode_rewards(file + "/result.json")[
                "sparse_reward_mean"
            ]
            for file in dict[env]["files"]
        ]
        dict[env]["rewards"] = rewards
        dict[env]["std"] = np.std(rewards)
        dict[env]["mean"] = np.mean(rewards)
    return dict


def plot_statistics(dict):
    names = []
    stds = []
    means = []
    for env in dict:
        names.append(env)
        stds.append(dict[env]["std"])
        means.append(dict[env]["mean"])

    x_pos = np.arange(len(names))
    matplotlib.rc("xtick", labelsize=7)
    fig, ax = plt.subplots()
    ax.bar(
        x_pos,
        means,
        yerr=stds,
        align="center",
        alpha=0.5,
        ecolor="black",
        capsize=10,
    )
    ax.set_ylabel("Average reward per episode")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names)
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig("example_rewards.png")
    plt.show()


if __name__ == "__main__":
    experiments = get_list_experiments("results")
    experiments_results = get_statistics(experiments)
    plot_statistics(experiments_results)
