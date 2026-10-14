from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

# Load a classic layout (e.g., "cramped_room")
# Other layouts include: "asymmetric_advantages", "coordination_ring", "forced_coordination", "counter_circuit_o_1order"
mdp = OvercookedGridworld.from_layout_name("cramped_room")
env = OvercookedEnv.from_mdp(mdp, horizon=400)

# Reset the environment
env.reset()

# Print the text-based representation of the grid
print(env)