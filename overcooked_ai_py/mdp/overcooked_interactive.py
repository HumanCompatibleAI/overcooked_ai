import pygame
from argparse import ArgumentParser

# TODO: Think about how to deal with this
from hr_coordination.pbt.pbt_utils import load_baselines_model, get_agent_from_saved_model, setup_mdp_env, get_config_from_pbt_dir
from hr_coordination.ppo.ppo import get_ppo_agent
from hr_coordination.imitation.behavioural_cloning import get_bc_agent_from_saved

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Direction, Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import StayAgent, RandomAgent, AgentFromPolicy, GreedyHumanModel
from overcooked_ai_py.planning.planners import MediumLevelPlanner
from overcooked_ai_py.mdp.layout_generator import LayoutGenerator
from overcooked_ai_py.utils import load_dict_from_file, get_max_iter

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
        print(player_idx)
 
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
                if not done:
                    for _ in range(2):
                        action = Direction.STAY
                        done = self.step_env(action)
                        if done:
                            break

        if event.type == pygame.QUIT or done:
            print("TOT rew", self.env.cumulative_rewards)
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
        print("Curr reward: (sparse)", r_t, "\t(dense)", info["dense_r"])
        print(self.env.t)
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
    if run_type == "ppo":
        print("Seed", run_seed)
        agent, evaluator = get_ppo_agent(run_dir, run_seed, best=True)
        env = evaluator.env
    elif run_type in ["pbt", "ppo"]:
        # TODO: Add testing for this
        run_path = "data/" + run_type + "_runs/" + run_dir + "/seed_{}".format(run_seed)
        # TODO: use get_config_from_pbt_dir if will be split up for the two cases
        config = load_dict_from_file(run_path + "/config")

        agent_folder = run_path + '/agent' + str(agent_num)
        agent_to_load_path = agent_folder + "/pbt_iter" + str(get_max_iter(agent_folder))
        agent = get_agent_from_saved_model(agent_to_load_path, config["SIM_THREADS"])

        if config["FIXED_MDP"]:
            layout_name = config["FIXED_MDP"]
            config["ORDER_GOAL"] = config["ORDER_GOAL"] * 3
            mdp = OvercookedGridworld.from_file(layout_name, config["ORDER_GOAL"], config["EXPLOSION_TIME"], rew_shaping_params=None)
            env = OvercookedEnv(mdp, horizon=604, random_start_objs=False, random_start_pos=False)
        else:
            env = setup_mdp_env(display=False, **config)

    elif run_type == "bc":
        # config = get_config_from_pbt_dir(cfg_run_dir)

        # # Modifications from original pbt config
        # config["ENV_HORIZON"] = 1000

        # gym_env, _ = get_env_and_policy_fn(config)
        # env = gym_env.base_env
    
        # model_path = 'data/bc_runs/' + run_dir #test_BC'
        agent, env_params, data_params = get_bc_agent_from_saved(run_dir)
        env = OvercookedEnv.from_config(env_params)
    else:
        raise ValueError("Unrecognized run type")

    env.horizon = 604
    return env, agent, player_idx

if __name__ == "__main__" :
    """
    Sample commands
    -> pbt
    python mdp/overcooked_interactive.py -t pbt -r scenario2_best -a 0 
    ->
    python mdp/overcooked_interactive.py -t ppo -r 2019_05_04-10_32_37_rand1_rew_shape -s 1
    -> BC
    python mdp/overcooked_interactive.py -t bc -r data/bc_runs/bc_run -c 2019_03_17-18_52_13_undefined_name  -a 0   
    """
    parser = ArgumentParser()
    # parser.add_argument("-l", "--fixed_mdp", dest="layout",
    #                     help="name of the layout to be played as found in data/layouts",
    #                     required=True)
    parser.add_argument("-t", "--type", dest="type",
                        help="type of run, (i.e. pbt, bc, ppo, etc)", required=True)
    parser.add_argument("-r", "--run_dir", dest="run",
                        help="name of run dir in data/*_runs/", required=True)
    parser.add_argument("-c", "--config_run_dir", dest="cfg",
                        help="name of run dir in data/*_runs/", required=False)
    parser.add_argument("-s", "--seed", dest="seed", default=0)
    parser.add_argument("-a", "--agent_num", dest="agent_num", default=0)
    parser.add_argument("-i", "--idx", dest="idx", default=0)

    args = parser.parse_args()
    run_type, run_dir, cfg_run_dir, run_seed, agent_num, player_idx = args.type, args.run, args.cfg, int(args.seed), int(args.agent_num), int(args.idx)
    
    env, agent, player_idx = setup_game(run_type, run_dir, cfg_run_dir, run_seed, agent_num, player_idx)
    
    theApp = App(env, agent, player_idx)
    theApp.on_execute()