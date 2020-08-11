import os
from common_test_generator import generate_serialized_trajectory

_current_dir = os.path.dirname(os.path.abspath(__file__))
TESTING_DATA_DIR = os.path.join(_current_dir, os.pardir, 'src', 'overcooked_ai_py', 'data', 'testing')