import os
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.mdp.actions import Action
import random

# 1. Setup the environment
layout_name = "cramped_room"
mdp = OvercookedGridworld.from_layout_name(layout_name)
env = OvercookedEnv.from_mdp(mdp, horizon=400)
env.reset()

# 2. Get a state (just stepping randomly for demonstration)
# Taking a random action for agent 0 and agent 1
action_0 = random.choice(Action.ALL_ACTIONS)
action_1 = random.choice(Action.ALL_ACTIONS)
state, reward, done, info = env.step((action_0, action_1))

print("actions taken:", action_0, action_1)
print("Current State after one random step:")
print(state)
print("Reward:", reward)
print("Done:", done)
print("Info:", info)

# 3. Create the visualizer
visualizer = StateVisualizer()

# 4. Render and Save
# You need the 'grid' from the mdp to tell the visualizer where the walls/counters are
img_dir = "Sybe/visualizations"
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

img_path = os.path.join(img_dir, "test_state.png")

print(f"Saving image to {img_path}...")
visualizer.display_rendered_state(
    state=state, 
    grid=mdp.terrain_mtx, 
    img_path=img_path
)
print("Done!")

StateVisualizer().display_rendered_trajectory(trajs2, ipython_display=True)