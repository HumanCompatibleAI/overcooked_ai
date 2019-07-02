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
import stable_baselines

# I/O

def save_pickle(data, filename):
    with open(filename + '.pickle', 'wb') as f:
	    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(filename):
    with open(filename + '.pickle', 'rb') as f:
	    return pickle.load(f)

def load_dict_from_file(filepath):
    f = open(filepath, 'r')
    data = f.read()
    f.close()
    return eval(data)

def save_dict_to_file(dic, filename):
    dic = dict(dic)
    f = open(filename + '.txt','w')
    f.write(str(dic))
    f.close()

def load_dict_from_txt(filename):
    return load_dict_from_file(filename + ".txt")

# MDP

def cumulative_rewards_from_rew_list(rews):
    return [sum(rews[:t]) for t in range(len(rews))]

# Gridworld

def manhattan_distance(pos1, pos2):
    """Returns manhattan distance between two points in (x, y) format"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def pos_distance(pos0, pos1):
    return tuple(np.array(pos0) - np.array(pos1))

# Randomness

def rnd_uniform(low, high):
    if low == high:
        return low
    return np.random.uniform(low, high)

def rnd_int_uniform(low, high):
    if low == high:
        return low
    return np.random.choice(range(low, high + 1))

# Statistics

def mean_and_std_err(lst):
    "Mean and standard error"
    mu, sd = np.mean(lst), np.std(lst)
    n = len(lst)
    se = sd / np.sqrt(n)
    return mu, se

# Utils

def profile(fnc):
    """A decorator that uses cProfile to profile a function (from https://osf.io/upav8/)"""
    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        print(s.getvalue())
        return retval
    return inner