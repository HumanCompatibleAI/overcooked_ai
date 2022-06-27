import pygame
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Direction, Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import StayAgent, RandomAgent, AgentFromPolicy, GreedyHumanModel
from overcooked_ai_py.planning.planners import MediumLevelPlanner
from overcooked_ai_py.mdp.layout_generator import LayoutGenerator
from overcooked_ai_py.utils import load_dict_from_file

from human_aware_rl.baselines_utils import load_baselines_model, get_agent_from_saved_model
from human_aware_rl.ppo.ppo import get_ppo_agent
from human_aware_rl.imitation.behavioural_cloning import get_bc_agent_from_saved

UP = 273
RIGHT = 275
DOWN = 274
LEFT = 276
SPACEBAR = 32

no_counters_params = {
    'start_orientations': False,
    'wait_allowed': False,
    'counter_goals': [],
    'counter_drop': [],
    'counter_pickup': [],
    'same_motion_goals': True
}

valid_counters = [(5, 3)]
one_counter_params = {
    'start_orientations': False,
    'wait_allowed': False,
    'counter_goals': valid_counters,
    'counter_drop': valid_counters,
    'counter_pickup': [],
    'same_motion_goals': True
}


class App:
    """Class to run an Overcooked Gridworld game, leaving one of the players as fixed.
    Useful for debugging. Most of the code from http://pygametutorials.wikidot.com/tutorials-basic."""

    def __init__(self, env, agent, player_idx, slow_time):
        self._running = True
        self._display_surf = None
        self.env = env
        self.agent = agent
        self.agent_idx = player_idx
        self.slow_time = slow_time
        print("Human player index:", player_idx)

    def on_init(self):
        pygame.init()

        # Adding pre-trained agent as teammate
        self.agent.set_agent_index(self.agent_idx)
        self.agent.set_mdp(self.env.mdp)

        print(self.env)
        self._running = True

    def on_event(self, event):
        done = False

        if event.type == pygame.KEYDOWN:
            pressed_key = event.dict['key']
            action = None

            if pressed_key == UP:
                action = Direction.NORTH
            elif pressed_key == RIGHT:
                action = Direction.EAST
            elif pressed_key == DOWN:
                action = Direction.SOUTH
            elif pressed_key == LEFT:
                action = Direction.WEST
            elif pressed_key == SPACEBAR:
                action = Action.INTERACT

            if action in Action.ALL_ACTIONS:

                done = self.step_env(action)

                if self.slow_time and not done:
                    for _ in range(2):
                        action = Action.STAY
                        done = self.step_env(action)
                        if done:
                            break

        if event.type == pygame.QUIT or done:
            print("TOT rew", self.env.cumulative_sparse_rewards)
            self._running = False

    def step_env(self, my_action):
        agent_action = self.agent.action(self.env.state)

        if self.agent_idx == 0:
            joint_action = (agent_action, my_action)
        else:
            joint_action = (my_action, agent_action)

        s_t, r_t, done, info = self.env.step(joint_action)

        print(self.env)
        print("Curr reward: (sparse)", r_t, "\t(dense)", info["shaped_r"])
        print(self.env.t)
        return done

    def on_loop(self):
        pass

    def on_render(self):
        pass

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        while (self._running):
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
        self.on_cleanup()


def setup_game(run_type, run_dir, run_seed, agent_num, player_idx):
    if run_type == "ppo":
        print("Seed", run_seed)
        agent, config = get_ppo_agent(run_dir, run_seed, best=True)
    elif run_type == "pbt":
        run_path = "data/" + run_type + "_runs/" + run_dir + "/seed_{}".format(run_seed)
        config = load_dict_from_file(run_path + "/config.txt")

        agent_path = run_path + '/agent' + str(agent_num) + "/best"
        agent = get_agent_from_saved_model(agent_path, config["sim_threads"])
    elif run_type == "bc":
        agent, config = get_bc_agent_from_saved(run_dir)
    else:
        raise ValueError("Unrecognized run type")

    env = OvercookedEnv(OvercookedGridworld.from_layout_name(**config["mdp_params"]), **config["env_params"])
    return env, agent, player_idx


if __name__ == "__main__":
    """
    Sample commands
    -> pbt
    python overcooked_interactive.py -t pbt -r pbt_simple -a 0 -s 8015
    ->
    python overcooked_interactive.py -t ppo -r ppo_sp_simple -s 386
    -> BC
    python overcooked_interactive.py -t bc -r simple_bc_test_seed4
    """
    parser = ArgumentParser()
    parser.add_argument("-t", "--type", dest="type",
                        help="type of run, (i.e. pbt, bc, ppo, etc)", required=True)
    parser.add_argument("-r", "--run_dir", dest="run",
                        help="name of run dir in data/*_runs/", required=True)
    parser.add_argument("-no_slowed", "--no_slowed_down", dest="slow",
                        help="Slow down time for human to simulate actual test time", action='store_false')
    parser.add_argument("-s", "--seed", dest="seed", required=False, default=0)
    parser.add_argument("-a", "--agent_num", dest="agent_num", default=0)
    parser.add_argument("-i", "--idx", dest="idx", default=0)

    args = parser.parse_args()
    run_type, run_dir, slow_time, run_seed, agent_num, player_idx = args.type, args.run, bool(args.slow), int(
        args.seed), int(args.agent_num), int(args.idx)

    env, agent, player_idx = setup_game(run_type, run_dir, run_seed, agent_num, player_idx)

    theApp = App(env, agent, player_idx, slow_time)
    print("Slowed time:", slow_time)
    theApp.on_execute()