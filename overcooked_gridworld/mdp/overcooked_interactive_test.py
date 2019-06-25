import unittest
from overcooked_gridworld.utils import reset_tf
from overcooked_gridworld.mdp.overcooked_interactive import setup_game

class TestInteractiveSetup(unittest.TestCase):

    def setUp(self):
        reset_tf()
    
    def test_bc_setup(self):
        run_type = "bc"
        run_dir = "data/bc_runs/bc_run"
        cfg_dir = "2019_03_17-18_52_13_undefined_name"
        setup_game(run_type, run_dir=run_dir, cfg_run_dir=cfg_dir, run_seed=0, agent_num=0, player_idx=0)

if __name__ == '__main__':
    unittest.main()