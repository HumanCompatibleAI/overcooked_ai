import pygame
from pygame import K_UP, K_LEFT, K_RIGHT, K_DOWN, K_SPACE
from pygame.locals import HWSURFACE, DOUBLEBUF, RESIZABLE, QUIT, VIDEORESIZE
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Direction, Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import StayAgent, RandomAgent, AgentFromPolicy, GreedyHumanModel
# from overcooked_ai_py.planning.planners import MediumLevelPlanner
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.mdp.layout_generator import LayoutGenerator
from overcooked_ai_py.utils import load_dict_from_file


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
        surface = StateVisualizer().render_state(self.env.state, grid=self.env.mdp.terrain_mtx)
        self.window = pygame.display.set_mode(surface.get_size(), HWSURFACE | DOUBLEBUF | RESIZABLE)
        self.window.blit(surface, (0, 0))
        pygame.display.flip()
        # pygame.display.set_mode((100, 100))

        # Adding pre-trained agent as teammate
        #self.agent.set_agent_index(self.agent_idx)
        #self.agent.set_mdp(self.env.mdp)

        print(self.env)
        self._running = True

    def on_event(self, event):
        done = False

        if event.type == pygame.KEYDOWN:
            print(event)
            pressed_key = event.dict['key']
            action = None

            if pressed_key == K_UP:
                action = Direction.NORTH
            elif pressed_key == K_RIGHT:
                action = Direction.EAST
            elif pressed_key == K_DOWN:
                action = Direction.SOUTH
            elif pressed_key == K_LEFT:
                action = Direction.WEST
            elif pressed_key == K_SPACE:
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
        agent_action = Direction.NORTH #self.agent.action(self.env.state)

        print(my_action, agent_action)
        print("PRIOR:", self.env.state)
        if self.agent_idx == 0:
            joint_action = (agent_action, my_action)
        else:
            joint_action = (my_action, agent_action)

        s_t, r_t, done, info = self.env.step(joint_action)
        print("POST :", self.env.state)
        self.on_render()
        # print(info)

        # print(self.env)
        # print("Curr reward: (sparse)", r_t, "\t(dense)")#, info["shaped_r"])
        # print(self.env.t)
        return done

    def on_loop(self):
        pass

    def on_render(self):
        print(self.env)
        # print("AT R :", self.env.state)
        print(self.env.mdp.terrain_mtx)
        surface = StateVisualizer().render_state(self.env.state, grid=self.env.mdp.terrain_mtx)
        self.window = pygame.display.set_mode(surface.get_size(), HWSURFACE | DOUBLEBUF | RESIZABLE)
        self.window.blit(surface, (0, 0))
        pygame.display.flip()


    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        while (self._running):
            # print("1")
            for event in pygame.event.get():
                # print(event)
                # pressed = pygame.key.get_pressed()
                # if event.type == pygame.KEYDOWN:
                #     print("--->", pressed)
                # keys_pressed = pygame.key.get_pressed()
                # if any in (keys_pressed):
                #     print(keys_pressed)
                # print("2")
                self.on_event(event)
            self.on_loop()
            # self.on_render()
            # pygame.event.pump()
        self.on_cleanup()


def setup_game(env_name, player_idx):
    # if run_type == "ppo":
    #     print("Seed", run_seed)
    #     agent, config = get_ppo_agent(run_dir, run_seed, best=True)
    # elif run_type == "pbt":
    #     run_path = "data/" + run_type + "_runs/" + run_dir + "/seed_{}".format(run_seed)
    #     config = load_dict_from_file(run_path + "/config.txt")
    #
    #     agent_path = run_path + '/agent' + str(agent_num) + "/best"
    #     agent = get_agent_from_saved_model(agent_path, config["sim_threads"])
    # elif run_type == "bc":
    #     agent, config = get_bc_agent_from_saved(run_dir)
    # else:
    #     raise ValueError("Unrecognized run type")
    agent = None

    # base_mdp = OvercookedGridworld.from_layout_name("cramped_room")
    # self.mlam = MediumLevelActionManager.from_pickle_or_compute(self.base_mdp, NO_COUNTERS_PARAMS, force_compute=True)
    # self.env = OvercookedEnv.from_mdp(self.base_mdp, **DEFAULT_ENV_PARAMS, info_level=0)
    # self.greedy_human_model_pair = AgentPair(GreedyHumanModel(self.mlam), GreedyHumanModel(self.mlam))
    # np.random.seed(0)
    env = OvercookedEnv.from_mdp(OvercookedGridworld.from_layout_name(env_name), horizon=400)
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
    # parser.add_argument("-t", "--type", dest="type",
    #                     help="type of run, (i.e. pbt, bc, ppo, etc)", required=True)
    # parser.add_argument("-r", "--run_dir", dest="run",
    #                     help="name of run dir in data/*_runs/", required=True)
    parser.add_argument("-no_slowed", "--no_slowed_down", dest="slow",
                        help="Slow down time for human to simulate actual test time", action='store_false')
    parser.add_argument("-s", "--seed", dest="seed", required=False, default=0)
    parser.add_argument("-a", "--agent_num", dest="agent_num", default=0)
    parser.add_argument("-i", "--idx", dest="idx", default=0)

    args = parser.parse_args()
    # run_type, run_dir, slow_time, run_seed, agent_num, player_idx = args.type, args.run, bool(args.slow), int(
    #     args.seed), int(args.agent_num), int(args.idx)
    slow_time = False

    env, agent, player_idx = setup_game('tf_test_4', player_idx=0)

    theApp = App(env, agent, player_idx, slow_time)
    print("Slowed time:", slow_time)
    theApp.on_execute()