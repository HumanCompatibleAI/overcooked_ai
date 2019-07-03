import os
from overcooked_ai_py.utils import load_dict_from_file

LAYOUTS_DIR = os.path.dirname(os.path.abspath(__file__))

def read_layout_dict(layout_name):
    return load_dict_from_file(os.path.join(LAYOUTS_DIR, layout_name + ".layout"))
