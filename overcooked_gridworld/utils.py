import os
import io
import re
import git
import json
import time
import shutil
import pickle
import pstats
import random
import cProfile
import numpy as np
import tensorflow as tf
import stable_baselines

def profile(fnc):
    """A decorator that uses cProfile to profile a function (from https://osf.io/upav8/)"""
    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval
    return inner

def manhattan_distance(pos1, pos2):
    """Returns manhattan distance between two points in (x, y) format"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def save_as_json(filename, data):
    with open(filename, 'w') as outfile:
	    json.dump(data, outfile)

def save_dict_to_file(dic, filename):
    dic = dict(dic)
    f = open(filename + '.txt','w')
    f.write(str(dic))
    f.close()

def load_dict_from_file(filepath):
    f = open(filepath,'r')
    data=f.read()
    f.close()
    return eval(data)

def load_dict_from_txt(filename):
    return load_dict_from_file(filename + ".txt")

def delete_dir_if_exists(dir_path, verbose=False):
    if os.path.exists(dir_path):
        if verbose:
            print("Deleting old dir", dir_path)
        shutil.rmtree(dir_path)

def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
	    os.makedirs(dir_path)

def save_pickle(data, filename):
    with open(filename + '.pickle', 'wb') as f:
	    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(filename):
    with open(filename + '.pickle', 'rb') as f:
	    return pickle.load(f)

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

def rnd_uniform(low, high):
    if low == high:
        return low
    return np.random.uniform(low, high)

def rnd_int_uniform(low, high):
    if low == high:
        return low
    return np.random.choice(range(low, high + 1))

def get_current_commit_hash():
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha

def get_trailing_number(s):
    """
    Get the trailing number from a string,
    i.e. 'file123' -> '123'
    """
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None

def get_max_iter(agent_folder):
    """Return biggest PBT iteration that has been run"""
    saved_iters = []
    for folder_s in os.listdir(agent_folder):
        folder_iter = get_trailing_number(folder_s) 
        if folder_iter is not None:
            saved_iters.append(folder_iter)
    if len(saved_iters) == 0:
        raise ValueError("Agent folder {} seemed to not have any pbt_iter subfolders".format(agent_folder))
    return max(saved_iters)

def pos_distance(pos0, pos1):
    return tuple(np.array(pos0) - np.array(pos1))

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
    probs_for_correct = [p if p > 0 else eps for p in probs_for_correct]
    
    log_likelihood = -np.log(probs_for_correct)
    cross_entropy_loss = np.sum(log_likelihood) / m
    return cross_entropy_loss

def accuracy(action_probs, y):
    return np.sum(np.argmax(action_probs, axis=1) == y) / len(y)

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_random_seed(seed)
    stable_baselines.common.set_global_seeds(seed)

def cumulative_rewards_from_rew_list(rews):
    return [sum(rews[:t]) for t in range(len(rews))]

def mean_and_std_err(lst):
    "Mean and standard error"
    mu, sd = np.mean(lst), np.std(lst)
    n = len(lst)
    se = sd / np.sqrt(n)
    return mu, se

def prepare_nested_default_dict_for_pickle(nested_defaultdict):
    """Need to make all nested defaultdicts into normal dicts to pickle"""
    for k,v in nested_defaultdict.items():
        nested_defaultdict[k] = dict(v)
    pickleable_dict = dict(nested_defaultdict)
    return pickleable_dict

def set_style(font_scale=1.6):
    import seaborn, matplotlib
    seaborn.set(font='serif', font_scale=font_scale)
    # Make the background white, and specify the specific font family
    seaborn.set_style("white", {
        "font.family": "serif",
        "font.weight": "normal",
        "font.serif": ["Times", "Palatino", "serif"],
        'axes.facecolor': 'white',
        'lines.markeredgewidth': 1})
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rc('font',family='serif', serif=['Palatino'])

class Node(object):
    def __init__(self, agent_name, params, parent=None):
        self.agent_name = agent_name
        self.params = params
        self.parent = parent