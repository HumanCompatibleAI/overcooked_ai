import io, json, pickle, pstats, cProfile
import numpy as np
from pathlib import Path

# I/O

def save_pickle(data, filename):
    with open(fix_filetype(filename, ".pickle"), "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(filename):
    with open(fix_filetype(filename, ".pickle"), "rb") as f:
        return pickle.load(f)

def load_dict_from_file(filepath):
    with open(filepath, "r") as f:
        return eval(f.read())

def save_dict_to_file(dic, filename):
    dic = dict(dic)
    with open(fix_filetype(filename, ".txt"),"w") as f:
        f.write(str(dic))

def load_dict_from_txt(filename):
    return load_dict_from_file(fix_filetype(filename, ".txt"))

def save_as_json(filename, data):
    with open(fix_filetype(filename, ".json"), "w") as outfile:
        json.dump(data, outfile)

def load_from_json(filename):
    with open(fix_filetype(filename, ".json"), "r") as json_file:
        return json.load(json_file)

def iterate_over_files_in_dir(dir_path):
    pathlist = Path(dir_path).glob("*.json")
    return [str(path) for path in pathlist]

def fix_filetype(path, filetype):
    if path[-len(filetype):] == filetype:
        return path
    else:
        return path + filetype

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
    mu = np.mean(lst)
    return mu, std_err(lst)

def std_err(lst):
    sd = np.std(lst)
    n = len(lst)
    return sd / np.sqrt(n)

# Utils

def profile(fnc):
    """A decorator that uses cProfile to profile a function (from https://osf.io/upav8/)"""
    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats()
        print(s.getvalue())
        return retval
    return inner