#!/usr/bin/env bash
export RUN_ENV=local

# Create a dummy data_dir.py if the file does not already exist
[ ! -f data_dir.py ] && echo "import os; DATA_DIR = os.path.abspath('.')" >> data_dir.py

# Human data tests
python -m unittest human.tests
# BC tests, skipping the LSTM tests by default
python -m unittest imitation.behavior_cloning_tf2_test.TestBCTraining
# rllib tests
python -m unittest rllib.tests
# PPO tests
python -m unittest ppo.ppo_rllib_test


