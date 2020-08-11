import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_current_dir, "data")
COMMON_TESTS_DIR = os.path.join(_current_dir, os.pardir, "common_tests")
HUMAN_DATA_DIR = os.path.join(DATA_DIR, "human_data")
LAYOUTS_DIR = os.path.join(DATA_DIR, "layouts")
TESTING_DATA_DIR = os.path.join(DATA_DIR, "testing")