import os
import warnings

import numpy as np

from human_aware_rl.imitation.behavior_cloning_tf2 import (
    BehaviorCloningPolicy,
    _get_base_ae,
    evaluate_bc_model,
    load_bc_model,
)
from human_aware_rl.rllib.rllib import (
    AgentPair,
    RlLibAgent,
    evaluate,
    get_agent_from_trainer,
    load_agent,
    load_agent_pair,
)
from overcooked_ai_py.agents.benchmarking import AgentEvaluator

# Ignore all warnings
warnings.filterwarnings("ignore")

# Customized evaluation functions


def evaluate_hp_bc(bc_model_path, hp_model_path, layout, order=0):
    """
    This function evaluates the performance between a BC model (trained with the human training data) and a human proxy model (trained with the human testing data)
    The order parameter determines the placement of the agents
    """
    bc_model, bc_params = load_bc_model(bc_model_path)
    bc_policy = BehaviorCloningPolicy.from_model(
        bc_model, bc_params, stochastic=True
    )

    hp_model, hp_params = load_bc_model(hp_model_path)
    hp_policy = BehaviorCloningPolicy.from_model(
        hp_model, hp_params, stochastic=True
    )

    base_ae = _get_base_ae(bc_params)
    base_env = base_ae.env

    bc_agent = RlLibAgent(bc_policy, 0, base_env.featurize_state_mdp)
    hp_agent = RlLibAgent(hp_policy, 1, base_env.featurize_state_mdp)
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout, "old_dynamics": True},
        env_params={"horizon": 400},
    )
    if order == 0:
        ap = AgentPair(hp_agent, bc_agent)
    else:
        ap = AgentPair(bc_agent, hp_agent)
    result = ae.evaluate_agent_pair(ap, 1, 400)
    return result, result["ep_returns"]


def evaluate_ppo_bc(path, layout, order=0):
    """
    This function loads and evaluates a PPO agent and a BC agent that was trained together, thus stored in the same trainer
    Order determines the starting position of the agents
    """
    ae = AgentEvaluator.from_layout_name(
        {"layout_name": layout, "old_dynamics": True}, {"horizon": 400}
    )
    if order == 0:
        ap = load_agent_pair(path, "ppo", "bc")
    else:
        ap = load_agent_pair(path, "bc", "ppo")
    result = ae.evaluate_agent_pair(ap, 1, 400)
    return result, result["ep_returns"]


def evaluate_ppo(path, layout):
    """
    This function loads and evaluates the performance of 2 PPO self-play agents
    Order doesn't matter here since the agents are self-play
    """
    ae = AgentEvaluator.from_layout_name(
        {"layout_name": layout, "old_dynamics": True}, {"horizon": 400}
    )
    ap = load_agent_pair(path, "ppo", "ppo")
    result = ae.evaluate_agent_pair(ap, 1, 400)
    return result, result["ep_returns"]


def evaluate_hp_ppo(bc_model_path, trainer_path, layout, order=0):
    """
    This function evaluates the performance between a PPO agent and a human proxy model (trained with the human testing data)
    The order parameter determines the placement of the agents
    """
    bc_model, bc_params = load_bc_model(bc_model_path)
    bc_policy = BehaviorCloningPolicy.from_model(
        bc_model, bc_params, stochastic=True
    )
    base_ae = _get_base_ae(bc_params)
    base_env = base_ae.env
    bc_agent = RlLibAgent(bc_policy, 0, base_env.featurize_state_mdp)
    print(trainer_path)
    ppo_agent = load_agent(trainer_path, policy_id="ppo", agent_index=1)

    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout, "old_dynamics": True},
        env_params={"horizon": 400},
    )
    if order == 0:
        ap = AgentPair(ppo_agent, bc_agent)
    else:
        ap = AgentPair(bc_agent, ppo_agent)
    result = ae.evaluate_agent_pair(ap, 1, 400)
    return result, result["ep_returns"]


# the order of layouts we want to evaluate
layouts = [
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring",
    "forced_coordination",
    "counter_circuit_o_1order",
]

file_dir = os.path.dirname(os.path.abspath(__file__))
bc_path = os.path.join(file_dir, "../imitation/bc_runs")

# directories where the BC agents are stored
bc = [
    os.path.join(bc_path, "train/cramped_room"),
    os.path.join(bc_path, "train/asymmetric_advantages"),
    os.path.join(bc_path, "train/coordination_ring"),
    os.path.join(bc_path, "train/random0"),
    os.path.join(bc_path, "train/random3"),
]

# directories where the human proxy agents are stored
hp = [
    os.path.join(bc_path, "test/cramped_room"),
    os.path.join(bc_path, "test/asymmetric_advantages"),
    os.path.join(bc_path, "test/coordination_ring"),
    os.path.join(bc_path, "test/random0"),
    os.path.join(bc_path, "test/random3"),
]

# reproduced agents ppo agents trained with bc, change the comments to the path of your trained agents
# change this to one of the agents creatd after running run_ppo_bc_experiments.sh bash script
ppo_bc = [
    # ppo_bc_crammed_room,
    # ppo_bc_asymmetric_advantages,
    # ppo_bc_coordination_ring,
    # ppo_bc_forced_coordination,
    # ppo_bc_counter_circuit_o_1order,
]
# reproduced agents ppo agents trained with self-play, change the comments to the path of your trained agents
# change this to one of the agents creatd after running run_experiments.sh bash script
ppo_sp = [
    # ppo_sp_crammed_room,
    # ppo_sp_asymmetric_advantages,
    # ppo_sp_coordination_ring,
    # ppo_sp_forced_coordination,
    # ppo_sp_counter_circuit_o_1order,
]


def eval_models(order):
    hp_PBC = {}
    hp_PSP = {}
    bc_PBC = {}
    PSP_PSP = {}
    hp_BC = {}

    for i in range(5):
        # hp vs ppo_bc
        _, res = evaluate_hp_ppo(hp[i], ppo_bc[i], layouts[i], order)
        hp_PBC[layouts[i]] = (np.mean(res), np.std(res) / len(res) ** 0.5)
        # hp vs ppo_sp
        _, res = evaluate_hp_ppo(hp[i], ppo_sp[i], layouts[i], order)
        hp_PSP[layouts[i]] = (np.mean(res), np.std(res) / len(res) ** 0.5)
        # bc vs ppo_bc
        _, res = evaluate_ppo_bc(ppo_bc[i], layouts[i], order)
        bc_PBC[layouts[i]] = (np.mean(res), np.std(res) / len(res) ** 0.5)
        # ppo_sp vs ppo_sp
        _, res = evaluate_ppo(ppo_sp[i], layouts[i])
        PSP_PSP[layouts[i]] = (np.mean(res), np.std(res) / len(res) ** 0.5)
        # bc vs hp
        _, res = evaluate_hp_bc(bc[i], hp[i], layouts[i], order)
        hp_BC[layouts[i]] = (np.mean(res), np.std(res) / len(res) ** 0.5)
    return PSP_PSP, hp_PSP, hp_PBC, hp_BC, bc_PBC
