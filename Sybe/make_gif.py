import os
from overcooked_ai_py.agents.agent import AgentPair, RandomAgent
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

# --- 1. Setup the Agent Evaluator ---
# This object manages the environment and runs the games.

# Define the layout and game settings
layout_name = "cramped_room"
horizon = 200 # Max number of steps per game (shorter for faster GIF testing)

agent_evaluator = AgentEvaluator.from_layout_name(
    mdp_params={"layout_name": layout_name},
    env_params={"horizon": horizon}
)

# --- 2. Create Agent Pair (The "Players") ---
# Create two random agents for demonstration
agent_0 = RandomAgent()
agent_1 = RandomAgent()
agent_pair = AgentPair(agent_0, agent_1)

# --- 3. Generate Trajectory ---
# Run 1 game and store the results in the 'trajectory' dictionary
print("Running game and generating trajectory data...")
trajectories = agent_evaluator.evaluate_agent_pair(
    agent_pair=agent_pair,
    num_games=1
)

# --- 4. Visualize and Save to GIF ---
gif_dir = "Sybe/visualizations"
gif_path = os.path.join(gif_dir, "game1")

print(f"Generating GIF and saving to {gif_path}...")
StateVisualizer().display_rendered_trajectory(
    trajectories=trajectories,
    img_directory_path=gif_path, # Saves the GIF to this path
    # Setting ipython_display=False prevents it from trying to display in a notebook
    # ipython_display=True
)

StateVisualizer().save_animation(gif_dir, "game1.gif")

print(f"Visualization complete. Check your project directory for '{gif_path}'")