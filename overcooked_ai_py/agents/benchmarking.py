import json, copy
import numpy as np

from overcooked_ai_py.utils import save_pickle, load_pickle, cumulative_rewards_from_rew_list, save_as_json, load_from_json
from overcooked_ai_py.planning.planners import NO_COUNTERS_PARAMS, MediumLevelPlanner
from overcooked_ai_py.mdp.layout_generator import LayoutGenerator
from overcooked_ai_py.agents.agent import AgentPair, CoupledPlanningAgent, RandomAgent, GreedyHumanModel
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Action, OvercookedState
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv


DEFAULT_TRAJ_KEYS = ["ep_observations", "ep_actions", "ep_rewards", "ep_dones", "ep_returns", "ep_returns_sparse", "ep_lengths", "mdp_params", "env_params"]


class AgentEvaluator(object):
    """
    Class used to get rollouts and evaluate performance of various types of agents.
    """

    def __init__(self, mdp_params, env_params={}, mdp_fn_params=None, force_compute=False, mlp_params=NO_COUNTERS_PARAMS, debug=False):
        """
        mdp_params (dict): params for creation of an OvercookedGridworld instance through the `from_layout_name` method
        env_params (dict): params for creation of an OvercookedEnv
        mdp_fn_params (dict): params to setup random MDP generation
        force_compute (bool): whether should re-compute MediumLevelPlanner although matching file is found
        mlp_params (dict): params for MediumLevelPlanner
        """
        assert type(mdp_params) is dict, "mdp_params must be a dictionary"

        if mdp_fn_params is None:
            self.variable_mdp = False
            self.mdp_fn = lambda: OvercookedGridworld.from_layout_name(**mdp_params)
        else:
            self.variable_mdp = True
            self.mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_params, **mdp_fn_params)
            
        self.env = OvercookedEnv(self.mdp_fn, **env_params)
        self.force_compute = force_compute
        self.debug = debug
        self.mlp_params = mlp_params
        self._mlp = None

    @property
    def mlp(self):
        assert not self.variable_mdp, "Variable mdp is not currently supported for planning"
        if self._mlp is None: 
            if self.debug: print("Computing Planner")
            self._mlp = MediumLevelPlanner.from_pickle_or_compute(self.env.mdp, self.mlp_params, force_compute=self.force_compute)
        return self._mlp

    def evaluate_human_model_pair(self, display=True):
        a0 = GreedyHumanModel(self.mlp)
        a1 = GreedyHumanModel(self.mlp)
        agent_pair = AgentPair(a0, a1)
        return self.evaluate_agent_pair(agent_pair, display=display)

    def evaluate_optimal_pair(self, display=True, delivery_horizon=2):
        a0 = CoupledPlanningAgent(self.mlp, delivery_horizon=delivery_horizon)
        a1 = CoupledPlanningAgent(self.mlp, delivery_horizon=delivery_horizon)
        a0.mlp.env = self.env
        a1.mlp.env = self.env
        agent_pair = AgentPair(a0, a1)
        return self.evaluate_agent_pair(agent_pair, display=display)

    def evaluate_one_optimal_one_random(self, display=True):
        a0 = CoupledPlanningAgent(self.mlp)
        a1 = RandomAgent()
        agent_pair = AgentPair(a0, a1)
        return self.evaluate_agent_pair(agent_pair, display=display)

    def evaluate_one_optimal_one_greedy_human(self, h_idx=0, display=True):
        h = GreedyHumanModel(self.mlp)
        r = CoupledPlanningAgent(self.mlp)
        agent_pair = AgentPair(h, r) if h_idx == 0 else AgentPair(r, h)
        return self.evaluate_agent_pair(agent_pair, display=display)

    def evaluate_agent_pair(self, agent_pair, num_games=1, display=False, info=True):
        return self.env.get_rollouts(agent_pair, num_games, display=display, info=info)

    @staticmethod
    def check_trajectories(trajectories):
        """
        Checks that of trajectories are in standard format and are consistent with dynamics of mdp.
        """
        AgentEvaluator._check_standard_traj_keys(set(trajectories.keys()))
        AgentEvaluator._check_right_types(trajectories)
        AgentEvaluator._check_trajectories_dynamics(trajectories)

    @staticmethod
    def _check_standard_traj_keys(traj_keys_set):
        assert traj_keys_set == set(DEFAULT_TRAJ_KEYS), "Keys of traj dict did not match standard form.\nMissing keys: {}\nAdditional keys: {}".format(
            [k for k in DEFAULT_TRAJ_KEYS if k not in traj_keys_set], [k for k in traj_keys_set if k not in DEFAULT_TRAJ_KEYS]
        )
    
    @staticmethod
    def _check_right_types(trajectories):
        for idx in range(len(trajectories["ep_observations"])):
            states, actions, rewards = trajectories["ep_observations"][idx], trajectories["ep_actions"][idx], trajectories["ep_rewards"][idx]
            mdp_params, env_params = trajectories["mdp_params"][idx], trajectories["env_params"][idx]
            assert all(type(a) is tuple for a in actions)
            assert all(type(s) is OvercookedState for s in states)
            assert type(mdp_params) is dict
            assert type(env_params) is dict

    @staticmethod
    def _check_trajectories_dynamics(trajectories):
        for idx in range(len(trajectories["ep_observations"])):
            states, actions, rewards = trajectories["ep_observations"][idx], trajectories["ep_actions"][idx], trajectories["ep_rewards"][idx]
            mdp_params, env_params = trajectories["mdp_params"][idx], trajectories["env_params"][idx]

            assert len(states) == len(actions) == len(rewards), "# states {}\t# actions {}\t# rewards {}".format(
                len(states), len(actions), len(rewards)
            )

            # Checking that actions would give rise to same behaviour in current MDP
            simulation_env = OvercookedEnv(OvercookedGridworld.from_layout_name(**mdp_params), **env_params)
            for i in range(len(states) - 1):
                curr_state = states[i]
                simulation_env.state = curr_state

                next_state, reward, done, info = simulation_env.step(actions[i])

                assert states[i + 1] == next_state, "States differed (expected vs actual): {}".format(
                    simulation_env.display_states(states[i + 1], next_state)
                )
                assert rewards[i] == reward, "{} \t {}".format(rewards[i], reward)

    ### I/O METHODS ###

    @staticmethod
    def save_trajectory(trajectory, filename):
        AgentEvaluator.check_trajectories(trajectory)
        save_pickle(trajectory, filename)

    @staticmethod
    def load_trajectory(filename):
        traj = load_pickle(filename)
        AgentEvaluator.check_trajectories(traj)
        return traj

    @staticmethod
    def save_traj_in_stable_baselines_format(rollout_trajs, filename):
        # Converting episode dones to episode starts
        eps_starts = [np.zeros(len(traj)) for traj in rollout_trajs["ep_dones"]]
        for ep_starts in eps_starts:
            ep_starts[0] = 1
        eps_starts = [ep_starts.astype(np.bool) for ep_starts in eps_starts]

        stable_baselines_trajs_dict = {
            'actions': np.concatenate(rollout_trajs["ep_actions"]),
            'obs': np.concatenate(rollout_trajs["ep_observations"]),
            'rewards': np.concatenate(rollout_trajs["ep_rewards"]),
            'episode_starts': np.concatenate(eps_starts),
            'episode_returns': rollout_trajs["ep_returns"]
        }
        stable_baselines_trajs_dict = { k:np.array(v) for k, v in stable_baselines_trajs_dict.items() }
        np.savez(filename, **stable_baselines_trajs_dict)

    @staticmethod
    def save_traj_as_json(trajectory, filename):
        """Saves the `idx`th trajectory as a list of state action pairs"""
        assert set(DEFAULT_TRAJ_KEYS) == set(trajectory.keys()), "{} vs\n{}".format(DEFAULT_TRAJ_KEYS, trajectory.keys())

        dict_traj = copy.deepcopy(trajectory)
        dict_traj["ep_observations"] = [[ob.to_dict() for ob in one_ep_obs] for one_ep_obs in trajectory["ep_observations"]]

        save_as_json(filename, dict_traj)

    @staticmethod
    def load_traj_from_json(filename):
        traj_dict = load_from_json(filename)
        traj_dict["ep_observations"] = [[OvercookedState.from_dict(ob) for ob in curr_ep_obs] for curr_ep_obs in traj_dict["ep_observations"]]
        traj_dict["ep_actions"] = [[tuple(tuple(a) if type(a) is list else a for a in j_a) for j_a in ep_acts] for ep_acts in traj_dict["ep_actions"]]
        return traj_dict


    ### VIZUALIZATION METHODS ###

    @staticmethod
    def interactive_from_traj(trajectories, traj_idx=0):
        """
        Displays ith trajectory of trajectories (in standard format) 
        interactively in a Jupyter notebook.
        """
        from ipywidgets import widgets, interactive_output

        states = trajectories["ep_observations"][traj_idx]
        joint_actions = trajectories["ep_actions"][traj_idx]
        cumulative_rewards = cumulative_rewards_from_rew_list(trajectories["ep_rewards"][traj_idx])
        mdp_params = trajectories["mdp_params"][traj_idx]
        env_params = trajectories["env_params"][traj_idx]
        env = AgentEvaluator(mdp_params, env_params=env_params).env

        def update(t = 1.0):
            env.state = states[int(t)]
            joint_action = joint_actions[int(t - 1)] if t > 0 else (Action.STAY, Action.STAY)
            print(env)
            print("Joint Action: {} \t Score: {}".format(Action.joint_action_to_char(joint_action), cumulative_rewards[t]))
            
            
        t = widgets.IntSlider(min=0, max=len(states) - 1, step=1, value=0)
        out = interactive_output(update, {'t': t})
        display(out, t)