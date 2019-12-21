import os
from gym.envs.registration import register
from overcooked_ai_py.utils import load_dict_from_file

register(
    id='Overcooked-v0',
    entry_point='overcooked_ai_py.mdp.overcooked_env:Overcooked',
)

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + "/data/"
LAYOUTS_DIR = DATA_DIR + "layouts/"

def read_layout_dict(layout_name):
    return load_dict_from_file(os.path.join(LAYOUTS_DIR, layout_name + ".layout"))