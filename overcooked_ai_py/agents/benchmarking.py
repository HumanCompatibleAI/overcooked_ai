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

    def __init__(self, mdp_params, env_params={}, mdp_fn_params=None, force_compute=False, mlp_params=None, debug=False):
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
            mlp_params = self.mlp_params if self.mlp_params is not None else NO_COUNTERS_PARAMS
            if self.debug: print("Computing Planner")
            self._mlp = MediumLevelPlanner.from_pickle_or_compute(self.env.mdp, mlp_params, force_compute=self.force_compute)
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
        h, r = GreedyHumanModel, CoupledPlanningAgent
        if h_idx == 0:
            a0, a1 = h(self.mlp), r(self.mlp)
        elif h_idx == 1:
            a0, a1 = r(self.mlp), h(self.mlp)
        agent_pair = AgentPair(a0, a1)
        return self.evaluate_agent_pair(agent_pair, display=display)

    def evaluate_agent_pair(self, agent_pair, num_games=1, display=False):
        return self.env.get_rollouts(agent_pair, num_games, display=display)

    @staticmethod
    def cumulative_rewards_from_trajectory(trajectory):
        cumulative_rew = 0
        for trajectory_item in trajectory:
            r_t = trajectory_item[2]
            cumulative_rew += r_t
        return cumulative_rew

    @staticmethod
    def check_trajectories(trajectories):
        """
        Checks consistency of trajectories in standard format with dynamics of mdp.
        
        NOTE: does not check dones positions, lengths consistency, order lists reducing if not None

        TODO: Should also check signature!!!
        """
        for idx in range(len(trajectories["ep_observations"])):
            states, actions, rewards = trajectories["ep_observations"][idx], trajectories["ep_actions"][idx], trajectories["ep_rewards"][idx]
            mdp_params, env_params = trajectories["mdp_params"][idx], trajectories["env_params"][idx]

            assert len(states) == len(actions)

            # Checking that actions would give rise to same behaviour in current MDP
            simulation_env = OvercookedEnv(OvercookedGridworld.from_layout_name(**mdp_params), **env_params)
            for i in range(len(states)):
                curr_state = states[i]
                simulation_env.state = curr_state

                if i + 1 < len(states):
                    next_state, reward, done, info = simulation_env.step(actions[i])

                    assert states[i + 1] == next_state, "States differed (expected vs actual): {}".format(
                        simulation_env.display_states(states[i + 1], next_state)
                    )
                    assert rewards[i] == reward, "{} \t {}".format(rewards[i], reward)


    ### I/O METHODS ###

    def save_trajectory(self, trajectory, filename):
        trajectory_dict_standard_signature = [
            "ep_actions", "ep_observations", "ep_rewards", "ep_dones", "ep_returns", "ep_lengths"
        ]
        assert set(trajectory.keys()) == set(trajectory_dict_standard_signature)
        self.check_trajectories(trajectory)
        save_pickle(trajectory, filename)

    def load_trajectory(self, filename):
        traj = load_pickle(filename)
        self.check_trajectories(traj)
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