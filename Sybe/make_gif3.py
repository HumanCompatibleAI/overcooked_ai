import os
import time
import pygame
import imageio
from overcooked_ai_py.agents.agent import AgentPair, RandomAgent
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

# --- Configuration ---
LAYOUT_NAME = "cramped_room"
HORIZON = 400
FPS = 30
OUTPUT_DIR = os.path.join("Sybe", "visualizations", "game1")

# --- Utility Functions ---

def generate_game_data(layout_name, horizon):
    """Runs a single simulation with random agents and returns the history data."""
    print(f"Simulating game on '{layout_name}' for {horizon} steps...")
    
    # 1. Setup Evaluator
    agent_evaluator = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout_name},
        env_params={"horizon": horizon}
    )
    
    # 2. Run Game
    agent_pair = AgentPair(RandomAgent(), RandomAgent())
    # The full history is a dict of lists containing states, actions, rewards, etc.
    game_history = agent_evaluator.evaluate_agent_pair(agent_pair, num_games=1)
    
    # 3. Extract Core Data (assuming only one game was run)
    state_list = game_history["ep_states"][0]
    grid_matrix = agent_evaluator.env.mdp.terrain_mtx
    
    return game_history, state_list, grid_matrix, agent_evaluator

def stitch_gif(game_history, output_dir, gif_name="interactive_replay.gif", gif_fps=10):
    """Saves individual frames to a directory and stitches them into a GIF."""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Rendering frames to {output_dir}...")
    
    # 1. Render all frames to PNG files
    StateVisualizer().display_rendered_trajectory(
        trajectories=game_history,
        img_directory_path=output_dir,
        ipython_display=False
    )
    
    # 2. Stitch images into a GIF
    print("Stitching frames into GIF...")
    images = []
    
    # Get all PNG filenames and sort them numerically
    filenames = sorted(
        [f for f in os.listdir(output_dir) if f.endswith('.png')],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )
    
    for filename in filenames:
        file_path = os.path.join(output_dir, filename)
        images.append(imageio.imread(file_path))
        
    output_path = os.path.join(output_dir, gif_name)
    imageio.mimsave(output_path, images, duration=(1/gif_fps), loop=0)
    
    print(f"GIF successfully saved to: {output_path}")

def run_interactive_viewer(game_history, state_list, grid_matrix, mdp_env, screen_fps=30):
    """Opens a Pygame window to interactively replay the game, including HUD."""
    
    pygame.init()
    
    # --- KEY REPEAT SETUP ---
    # Argument 1: Delay before repeat starts (500ms = 0.5 seconds)
    # Argument 2: Interval between repeats (30ms = ~33 steps per second)
    pygame.key.set_repeat(500, 30) 

    viz = StateVisualizer()
    font = pygame.font.SysFont(None, 24) 
    
    rewards_list = game_history["ep_rewards"][0] 

    def get_game_surface(t):
        return viz.render_state(state_list[t], grid_matrix)

    def draw_hud(surface, timestep):
        total_score = int(game_history["ep_returns"][0])
        
        if timestep == 0:
            current_score = 0
        else:
            current_score = int(sum(rewards_list[:timestep]))

        max_time = mdp_env.horizon
        time_left = max(0, max_time - timestep)
        
        # Text Rendering
        text_time = font.render(f"TIME LEFT: {time_left}", True, (255, 255, 255))
        text_score = font.render(f"SCORE: {current_score} / {total_score}", True, (255, 255, 255))
        
        surface.blit(text_time, (20, 10))
        surface.blit(text_score, (20, 35))

    # Setup Window
    game_surface = get_game_surface(0)
    game_width, game_height = game_surface.get_size()
    
    HUD_HEIGHT = 60
    screen = pygame.display.set_mode((game_width, game_height + HUD_HEIGHT))
    clock = pygame.time.Clock()
    
    # State variables
    running = True
    current_timestep = 0
    playing = False
    
    BG_COLOR = (153, 102, 0) 

    print("\n--- INTERACTIVE CONTROLS ---")
    print("Tap/Hold Right : Step Forward")
    print("Tap/Hold Left  : Step Backward")
    print("Space          : Play / Pause")
    print("Esc            : Quit")
    print("----------------------------")

    while running:
        # 1. Event Handling (Handles Taps AND Holds via set_repeat)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_RIGHT:
                    current_timestep = min(current_timestep + 1, len(state_list) - 1)
                    playing = False # Stop auto-play on manual input
                elif event.key == pygame.K_LEFT:
                    current_timestep = max(current_timestep - 1, 0)
                    playing = False
                elif event.key == pygame.K_SPACE:
                    playing = not playing
        
        # 2. Auto-Play Logic
        if playing:
            if current_timestep < len(state_list) - 1:
                current_timestep += 1
                # We use clock.tick to control speed, so no sleep needed here
            else:
                playing = False

        # 3. Drawing
        screen.fill(BG_COLOR) 
        
        draw_hud(screen, current_timestep)
        
        game_surface = get_game_surface(current_timestep)
        screen.blit(game_surface, (0, HUD_HEIGHT)) 
        
        pygame.display.set_caption(f"Replay - Step {current_timestep}/{len(state_list)-1} {'[PLAYING]' if playing else '[PAUSED]'}")
        pygame.display.flip()
        
        clock.tick(screen_fps)

    pygame.quit()


# --- Main Execution Block ---
if __name__ == "__main__":
    
    # 1. Run Simulation
    game_history, state_list, grid_matrix, agent_evaluator = generate_game_data(
        layout_name=LAYOUT_NAME, 
        horizon=HORIZON
    )

    # 2. Save GIF (Uncomment this line to generate the GIF)
    # stitch_gif(game_history, output_dir=OUTPUT_DIR, gif_fps=10)

    # 3. Open Interactive Viewer 
    run_interactive_viewer(
        game_history=game_history, 
        state_list=state_list, 
        grid_matrix=grid_matrix,
        mdp_env=agent_evaluator.env 
    )