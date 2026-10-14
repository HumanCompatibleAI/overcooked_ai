import pygame
import time
from overcooked_ai_py.agents.agent import AgentPair, RandomAgent
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
import os
import imageio

# --- 1. Generate the Data ---
print("Simulating game...")
layout_name = "cramped_room"
horizon = 400
evaluator = AgentEvaluator.from_layout_name(
    mdp_params={"layout_name": layout_name},
    env_params={"horizon": horizon}
)
pair = AgentPair(RandomAgent(), RandomAgent())
trajectories = evaluator.evaluate_agent_pair(pair, num_games=1)

# Extract the list of states and the static grid
states = trajectories["ep_states"][0] # Get states from the first game
grid = evaluator.env.mdp.terrain_mtx


viz = StateVisualizer()
img_dir = "Sybe/visualizations"
img_path = os.path.join(img_dir, "game1")

print(f"Generating path images and saving to {img_path}...")
StateVisualizer().display_rendered_trajectory(
    trajectories=trajectories,
    img_directory_path=img_path, # Saves the GIF to this path
    # Setting ipython_display=False prevents it from trying to display in a notebook
    # ipython_display=True
)

# Stitch images into a GIF
print("Stitching GIF...")
images = []
filenames = sorted(
        [f for f in os.listdir(img_path) if f.endswith('.png')],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )

for filename in filenames:
        images.append(imageio.imread(os.path.join(img_path, filename)))
        
# Save the GIF
output_filename = os.path.join(img_path, "0interactive_replay.gif")
imageio.mimsave(output_filename, images, duration=0.1, loop=0)

# --- 2. Setup Pygame Viewer ---
pygame.init()

# Helper to get the image surface for a specific timestep
def get_surface(timestep):
    # render_state returns a pygame surface
    # Note: We rely on the internal render method here
    return viz.render_state(states[timestep], grid)

# Initialize display based on the size of the first frame
first_surface = get_surface(0)
width, height = first_surface.get_size()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption(f"Overcooked Replay - Step 0/{len(states)-1}")

# --- 3. Interactive Loop ---
running = True
timestep = 0
playing = False
clock = pygame.time.Clock()

print("\n--- CONTROLS ---")
print("Right Arrow : Next Step")
print("Left Arrow  : Previous Step")
print("Space       : Play / Pause")
print("----------------")

while running:
    # 1. Handle Input
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                timestep = min(timestep + 1, len(states) - 1)
                playing = False # Stop auto-play if manual control used
            elif event.key == pygame.K_LEFT:
                timestep = max(timestep - 1, 0)
                playing = False
            elif event.key == pygame.K_SPACE:
                playing = not playing

    # 2. Handle Auto-Play
    if playing:
        if timestep < len(states) - 1:
            timestep += 1
            time.sleep(0.1) # Control playback speed
        else:
            playing = False

    # 3. Draw
    surface = get_surface(timestep)
    screen.blit(surface, (0, 0))
    pygame.display.set_caption(f"Overcooked Replay - Step {timestep}/{len(states)-1}")
    pygame.display.flip()
    
    clock.tick(30) # Limit to 30 FPS loop

pygame.quit()