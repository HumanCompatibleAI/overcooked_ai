import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_current_dir, "data")
HUMAN_DATA_DIR = os.path.join(DATA_DIR, "human_data")
PLANNERS_DIR = os.path.join(DATA_DIR, "planners")
LAYOUTS_DIR = os.path.join(DATA_DIR, "layouts")