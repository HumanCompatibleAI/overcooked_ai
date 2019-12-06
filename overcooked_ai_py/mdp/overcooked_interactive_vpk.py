import pygame
import random
from argparse import ArgumentParser

from overcooked_ai_py.agents.agent import StayAgent, RandomAgent, AgentFromPolicy, GreedyHumanModel_pk, ToMModel
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Direction, Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelPlanner
from overcooked_ai_py.utils import load_dict_from_file #, get_max_iter

from human_aware_rl.utils import get_max_iter

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
    def __init__(self, env, agent, player_idx):
        self._running = True
        self._display_surf = None
        self.env = env
        self.agent = agent
        self.agent_idx = player_idx
        self.size = self.weight, self.height = 1, 1

    def on_init(self):
        pygame.init()

        # Adding pre-trained agent as teammate
        self.agent.set_agent_index(self.agent_idx)
        self.agent.set_mdp(self.env.mdp)

        # if layout_name == "scenario2":
        #     # Set to standard coordination failure scenario
        #     P, Obj = PlayerState, ObjectState
        #     n, s = Direction.NORTH, Direction.SOUTH
        #     self.env.state = OvercookedState(
        #         [P((7, 2), n),
        #         P((5, 2), n)],
        #         {(6, 3): Obj('soup', (6, 3), ('onion', 2, 0))},
        #         order_list=['onion'])
    
        print(self.env)
        self._display_surf = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self._running = True
 
    def on_event(self, event):
        done = False

        if event.type == pygame.KEYDOWN:
            pressed_key = event.dict['key']
            action = None

            if pressed_key == UP or pressed_key == ord('w'):
                action = Direction.NORTH
            elif pressed_key == RIGHT or pressed_key == ord('d'):
                action = Direction.EAST
            elif pressed_key == DOWN or pressed_key == ord('s'):
                action = Direction.SOUTH
            elif pressed_key == LEFT or pressed_key == ord('a'):
                action = Direction.WEST
            elif pressed_key == SPACEBAR:
                action = Action.INTERACT
                
            if action in Action.ALL_ACTIONS:
                done = self.step_env(action)

        if event.type == pygame.QUIT or done:
            self._running = False


    def step_env(self, my_action):
        agent_action = self.agent.action(self.env.state)
        # other_agent_action = Direction.STAY

        if self.agent_idx == 0:
            joint_action = (agent_action, my_action)
        else:
            joint_action = (my_action, agent_action)

        s_t, r_t, done, info = self.env.step(joint_action)

        print(self.env)
        # Changed from this: print("Curr reward: (sparse)", r_t, "\t(dense)", info["dense_r"])
        print("Curr reward: (sparse)", r_t, "\t(dense)", info["shaped_r"])
        print("Time: {}".format(self.env.t))
        # process_observations([next_state], self.env.mdp, 0, debug=True)
        # print("Pos", next_state.player_positions[1])
        # print("Heuristic: ", self.hlp.hard_heuristic_fn(next_state, 1))
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
 
        while( self._running ):
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
        self.on_cleanup()
 
def setup_game(run_type, run_dir, cfg_run_dir, run_seed, agent_num, player_idx):
    if run_type in ["pbt", "ppo"]:
        # TODO: Add testing for this
        run_path = "data/" + run_type + "_runs/" + run_dir + "/seed_{}".format(run_seed)
        # TODO: use get_config_from_pbt_dir if will be split up for the two cases
        config = load_dict_from_file(run_path + "/config")

        agent_folder = run_path + '/agent' + str(agent_num)
        agent_to_load_path = agent_folder + "/pbt_iter" + str(get_max_iter(agent_folder))
        agent = get_agent_from_saved_model(agent_to_load_path, config["SIM_THREADS"])

        if config["FIXED_MDP"]:
            layout_name = config["FIXED_MDP"]
            layout_filepath = "data/layouts/{}.layout".format(layout_name)
            mdp = OvercookedGridworld.from_file(layout_filepath, config["ORDER_GOAL"], config["EXPLOSION_TIME"], rew_shaping_params=None)
            env = OvercookedEnv(mdp)
        else:
            env = setup_mdp_env(display=False, **config)

    elif run_type == "bc":
        config = get_config_from_pbt_dir(cfg_run_dir)

        # Modifications from original pbt config
        config["ENV_HORIZON"] = 1000

        gym_env, _ = get_env_and_policy_fn(config)
        env = gym_env.base_env
    
        model_path = run_dir #'data/bc_runs/test_BC'
        agent = get_agent_from_saved_BC(cfg_run_dir, model_path, stochastic=True)

    elif run_type == "hardcoded":

        cook_time = 5

        if layout == 'sc1':

            # Setup mdp
            mdp = OvercookedGridworld.from_layout_name('scenario1_s', start_order_list=["any", "any", "any"],
                                                       cook_time=cook_time, rew_shaping_params=None)

        elif layout == 'uni':

            # start_state = OvercookedState([P((2, 2), n), P((5, 2), n)], {}, order_list=start_order_list)
            # Setup mdp
            mdp = OvercookedGridworld.from_layout_name('unident_s', start_order_list=["any", "any", "any"],
                                                       cook_time=cook_time, rew_shaping_params=None)

        elif layout == 'sim':

            # Setup mdp
            mdp = OvercookedGridworld.from_layout_name('simple', start_order_list=["any", "any", "any"],
                                                       cook_time=cook_time, rew_shaping_params=None)

        elif layout == 'ran':

            # Setup mdp
            mdp = OvercookedGridworld.from_layout_name('random1', start_order_list=["any", "any", "any"],
                                                       cook_time=cook_time, rew_shaping_params=None)

        elif layout == 'sch':

            # Setup mdp
            mdp = OvercookedGridworld.from_layout_name('schelling_s', start_order_list=["any", "any", "any"],
                                                       cook_time=cook_time, rew_shaping_params=None)

        else:
            raise ValueError('layout not recognised')

        env = OvercookedEnv(mdp)
        # Doing this means that all counter locations are allowed to have objects dropped on them AND be "goals" (I think!)
        no_counters_params['counter_drop'] = mdp.get_counter_locations()
        no_counters_params['counter_goals'] = mdp.get_counter_locations()
        mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, no_counters_params, force_compute=False)

        # perseverance0 = random.random()
        # teamwork0 = random.random()
        # retain_goals0 = random.random()
        # wrong_decisions0 = random.random() ** 5
        # thinking_prob0 = 1 - random.random() **5
        # path_teamwork0 = 1 - random.random() **2
        # rat_coeff0 = 1+random.random()*3

        perseverance0 = 0.3
        teamwork0 = 0.5
        retain_goals0 = 0.8
        wrong_decisions0 = 0.05
        thinking_prob0 = 0.5
        path_teamwork0 = 0.3
        rat_coeff0 = 1.5
        prob_pausing0 = 0.3


        agent = ToMModel(mlp, player_index=0, perseverance=perseverance0, teamwork=teamwork0,
                         retain_goals=retain_goals0, wrong_decisions=wrong_decisions0,
                         thinking_prob=thinking_prob0, path_teamwork=path_teamwork0,
                         rationality_coefficient=rat_coeff0, prob_pausing=prob_pausing0)

        print('Ply 1: tw: {:.1f}, retain: {:.1f}, wrong dec: {:.1f}, think: {:.1f}, path_tw: {:.1f}, rat: {:.1f}'.
              format(teamwork0, retain_goals0, wrong_decisions0, thinking_prob0, path_teamwork0,
                     rat_coeff0))

    else:
        raise ValueError("Unrecognized run type")

    return env, agent, player_idx

if __name__ == "__main__" :
    """
    Sample commands
    -> hardcoded
    python mdp/overcooked_interactive_vpk.py -t hardcoded -l sim
    """
    parser = ArgumentParser()
    # parser.add_argument("-l", "--fixed_mdp", dest="layout",
    #                     help="name of the layout to be played as found in data/layouts",
    #                     required=True)
    parser.add_argument("-t", "--type", dest="type",
                        help="type of run, (i.e. pbt, bc, ppo, hardcoded, etc)", required=True)
    parser.add_argument("-r", "--run_dir", dest="run",
                        help="name of run dir in data/*_runs/", required=True)
    parser.add_argument("-c", "--config_run_dir", dest="cfg",
                        help="name of run dir in data/*_runs/", required=False)
    parser.add_argument("-s", "--seed", dest="seed", default=0)
    parser.add_argument("-a", "--agent_num", dest="agent_num", default=0)
    parser.add_argument("-i", "--idx", dest="idx", default=0)
    parser.add_argument("-l", "--layout", default='sim')

    args = parser.parse_args()
    run_type, run_dir, cfg_run_dir, run_seed, agent_num, player_idx, layout = args.type, args.run, args.cfg, \
                                                                             int(args.seed), int(args.agent_num), \
                                                                              int(args.idx), args.layout

    env, agent, player_idx = setup_game(run_type, run_dir, cfg_run_dir, run_seed, agent_num, player_idx)
    
    theApp = App(env, agent, player_idx)
    theApp.on_execute()