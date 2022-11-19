import itertools
import json
import os
import random
import re
import shutil

import git
import numpy as np
import tensorflow as tf

WANDB_PROJECT = "Overcooked AI"


def delete_dir_if_exists(dir_path, verbose=False):
    if os.path.exists(dir_path):
        if verbose:
            print("Deleting old dir", dir_path)
        shutil.rmtree(dir_path)


def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def reset_tf():
    """Clean up tensorflow graph and session.
    NOTE: this also resets the tensorflow seed"""
    tf.reset_default_graph()
    if tf.get_default_session() is not None:
        tf.get_default_session().close()


def num_tf_params():
    """Prints number of trainable parameters defined"""
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
            total_parameters += variable_parameters
    print(total_parameters)


def get_current_commit_hash():
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha


def get_trailing_number(s):
    """
    Get the trailing number from a string,
    i.e. 'file123' -> '123'
    """
    m = re.search(r"\d+$", s)
    return int(m.group()) if m else None


def get_max_iter(agent_folder):
    """Return biggest PBT iteration that has been run"""
    saved_iters = []
    for folder_s in os.listdir(agent_folder):
        folder_iter = get_trailing_number(folder_s)
        if folder_iter is not None:
            saved_iters.append(folder_iter)
    if len(saved_iters) == 0:
        raise ValueError(
            "Agent folder {} seemed to not have any pbt_iter subfolders".format(
                agent_folder
            )
        )
    return max(saved_iters)


def cross_entropy(action_probs, y, eps=1e-4):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
        Note that y is not one-hot encoded vector.
        It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = y.shape[0]
    # We use multidimensional array indexing to extract
    # softmax probability of the correct label for each sample.
    probs_for_correct = action_probs[range(m), y]

    # NOTE: eps was added to correct for some actions being deterministically removed from
    # the human model when it would get stuck. It was chosen empirically as to be about an order of
    # magnitude less than the smallest probability assigned to any event by the model
    probs_for_correct = np.array(
        [p if p > eps else eps for p in probs_for_correct]
    ).astype(float)

    log_likelihood = -np.log(probs_for_correct)
    cross_entropy_loss = np.sum(log_likelihood) / m
    return cross_entropy_loss


def accuracy(action_probs, y):
    return np.sum(np.argmax(action_probs, axis=1) == y) / len(y)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def prepare_nested_default_dict_for_pickle(nested_defaultdict):
    """Need to make all nested defaultdicts into normal dicts to pickle"""
    for k, v in nested_defaultdict.items():
        nested_defaultdict[k] = dict(v)
    pickleable_dict = dict(nested_defaultdict)
    return pickleable_dict


def set_style(font_scale=1.6):
    import matplotlib
    import seaborn

    seaborn.set(font="serif", font_scale=font_scale)
    # Make the background white, and specify the specific font family
    seaborn.set_style(
        "white",
        {
            "font.family": "serif",
            "font.weight": "normal",
            "font.serif": ["Times", "Palatino", "serif"],
            "axes.facecolor": "white",
            "lines.markeredgewidth": 1,
        },
    )
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rc("font", family="serif", serif=["Palatino"])


def common_keys_equal(dict_a, dict_b):
    common_keys = set(dict_a.keys()).intersection(set(dict_b.keys()))
    for k in common_keys:
        if dict_a[k] != dict_b[k]:
            return False
    return True


class Node(object):
    def __init__(self, agent_name, params, parent=None):
        self.agent_name = agent_name
        self.params = params
        self.parent = parent


def get_flattened_keys(dictionary):
    if type(dictionary) != dict:
        return []
    return list(dictionary.keys()) + list(
        itertools.chain(
            *[get_flattened_keys(dictionary[key]) for key in dictionary]
        )
    )


def recursive_dict_update(map, key, value):
    if type(map) != dict:
        return False
    if key in map:
        map[key] = value
        return True
    return any(
        [recursive_dict_update(child, key, value) for child in map.values()]
    )


def equal_dicts(d1, d2, ignore_keys):
    ignored = set(ignore_keys)
    for k1, v1 in d1.items():
        if k1 not in ignored and (k1 not in d2 or d2[k1] != v1):
            if k1 not in d2:
                print("d2 missing", k1)
            else:
                if k1 == "objects":
                    print("object difference")
                    for o1 in d1[k1]:
                        print(o1)
                    print("----")
                    for o2 in d2[k1]:
                        print(o2)
                else:
                    print(
                        "different at ", k1, "one is ", d2[k1], "one is ", v1
                    )
            return False
    for k2, v2 in d2.items():
        if k2 not in ignored and k2 not in d1:
            print("d1 missing", k2)
            return False
    return True


def get_dict_stats(d):
    new_d = d.copy()
    for k, v in d.items():
        new_d[k] = {
            "mean": np.mean(v),
            "standard_error": np.std(v) / np.sqrt(len(v)),
            "max": np.max(v),
            "n": len(v),
        }
    return new_d


def get_last_episode_rewards(filename):
    with open(filename) as f:
        j = json.loads(f.readlines()[-1])
        result = {
            "episode_reward_mean": j["episode_reward_mean"],
            "sparse_reward_mean": j["custom_metrics"]["sparse_reward_mean"],
        }
        return result
