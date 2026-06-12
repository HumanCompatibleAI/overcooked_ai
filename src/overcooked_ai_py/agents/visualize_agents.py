"""Run an agent pairing for one episode and render it to a GIF.

Usage:
    python -m overcooked_ai_py.agents.visualize_agents [pairing] [--layout NAME] [--games N]

Pairings: loader_server | greedy_server | loader_greedy | greedy_greedy

With --games N > 1, plays N games and only renders the first one (reward
printed for every game). Frames go to /tmp/overcooked_frames_<pairing>/;
the animation goes to <repo>/visualizations/<pairing>_NNN.gif, where NNN
auto-increments so previous runs are never overwritten.
"""
import argparse
import glob
import os
import re

from PIL import Image

import overcooked_ai_py
from overcooked_ai_py.agents.agent import AgentPair, GreedyHumanModel
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.custom_agents import (
    OnionLoaderAgent,
    ServerAgent,
)
from overcooked_ai_py.planning.planners import (
    NO_COUNTERS_PARAMS,
    MediumLevelActionManager,
)
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

PAIRINGS = {
    "loader_server": (OnionLoaderAgent, ServerAgent),
    "greedy_server": (GreedyHumanModel, ServerAgent),
    "loader_greedy": (OnionLoaderAgent, GreedyHumanModel),
    "greedy_greedy": (GreedyHumanModel, GreedyHumanModel),
}

# <repo>/src/overcooked_ai_py/__init__.py -> <repo>/visualizations
REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(overcooked_ai_py.__file__)))
)
VIS_DIR = os.path.join(REPO_ROOT, "visualizations")


def next_gif_path(pairing):
    """Next non-clobbering path: <pairing>_001.gif, <pairing>_002.gif, ..."""
    os.makedirs(VIS_DIR, exist_ok=True)
    taken = [
        int(m.group(1))
        for p in glob.glob(os.path.join(VIS_DIR, f"{pairing}_*.gif"))
        if (m := re.fullmatch(rf"{re.escape(pairing)}_(\d+)\.gif", os.path.basename(p)))
    ]
    return os.path.join(VIS_DIR, f"{pairing}_{max(taken, default=0) + 1:03d}.gif")


def render_episode(pairing, layout="cramped_room", horizon=400, fps=10, num_games=1):
    ae = AgentEvaluator.from_layout_name(
        {"layout_name": layout}, {"horizon": horizon}
    )
    mlam = MediumLevelActionManager.from_pickle_or_compute(
        ae.env.mdp, NO_COUNTERS_PARAMS, force_compute=False
    )
    cls_a, cls_b = PAIRINGS[pairing]
    pair = AgentPair(cls_a(mlam), cls_b(mlam))
    trajs = ae.evaluate_agent_pair(pair, num_games=num_games)
    print("rewards per game:", trajs["ep_returns"])

    out_dir = f"/tmp/overcooked_frames_{pairing}"
    gif_path = next_gif_path(pairing)
    StateVisualizer().display_rendered_trajectory(
        trajs, img_directory_path=out_dir, ipython_display=False
    )
    frame_paths = sorted(
        glob.glob(os.path.join(out_dir, "*.png")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
    )
    frames = [Image.open(p) for p in frame_paths]
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // fps,
        loop=0,
    )
    print("wrote", gif_path, f"({len(frames)} frames)")
    return gif_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pairing", nargs="?", default="loader_server", choices=PAIRINGS
    )
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--games", type=int, default=1)
    args = parser.parse_args()
    render_episode(
        args.pairing,
        layout=args.layout,
        horizon=args.horizon,
        num_games=args.games,
    )
