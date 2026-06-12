# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Overcooked-AI: a cooperative multi-agent benchmark environment (HumanCompatibleAI, NeurIPS 2019). Two chefs in a grid kitchen take **simultaneous** actions to cook and deliver soups for a single shared reward. Python package rooted at `src/`, two main packages: `overcooked_ai_py` (environment, maintained) and `human_aware_rl` (DRL training stack, **deprecated** — see upstream issue #162).

## Setup and commands

Requires Python `>=3.10,<3.11` (hard constraint in pyproject.toml). Use uv:

```bash
uv venv && uv sync                # base environment
uv sync --extra harl              # only if touching human_aware_rl (old ray 2.2 + tf 2.19)
```

Tests (plain `unittest`, no pytest):

```bash
.venv/bin/python testing/overcooked_test.py                            # core MDP tests, fast
.venv/bin/python -m unittest discover -s testing/ -p "*_test.py"       # full suite, 5–10 min (planners)
.venv/bin/python -m unittest testing.agent_test.<TestClass>.<test>     # single test
```

CI (`.github/workflows/pythontests.yml`) runs the discovery command above on push/PR. The `human_aware_rl` tests are separate: `cd src/human_aware_rl && ./run_tests.sh` (must be run from that directory; needs the `harl` extra).

## Architecture

Three layers, each wrapping the previous:

1. **`mdp/overcooked_mdp.py` — `OvercookedGridworld`**: stateless game logic. `get_state_transition(state, joint_action)` is a pure function; all game state lives in `OvercookedState` objects (player positions/orientations, held objects, `SoupState` pot contents). Layouts load from `data/layouts/*.layout`; recipes are configured class-wide via `Recipe.configure()` (test pollution between MDP configs usually traces back to this).
2. **`mdp/overcooked_env.py` — `OvercookedEnv`**: adds trajectory rollout and a fixed horizon (default 400; episodes end on horizon only, there are no terminal states). `Overcooked` (same file) is the `gymnasium.Env` wrapper.
3. **`agents/` + `planning/`**: agents and the planning substrate they rely on.

### The Agent contract

Subclass `Agent` (`agents/agent.py`) and implement `action(state) -> (action, info_dict)` where action is one of the 6 constants in `Action.ALL_ACTIONS` (`mdp/actions.py`). The rollout harness (`AgentPair`, `AgentEvaluator`) calls `set_agent_index`/`set_mdp` for you. Anything placed in `info_dict` is persisted into trajectories (`ep_infos`) — use it to log goals/labels. Override `reset()` (and call `super().reset()`) only if the agent keeps cross-step state.

`AgentEvaluator` (`agents/benchmarking.py`) is the standard entry point for rollouts: `AgentEvaluator.from_layout_name(...)` → `evaluate_agent_pair(AgentPair(a, b), num_games=N)` returns a trajectories dict (format in `mdp/overcooked_trajectory.py`) including every intermediate state.

### Planning layer (`planning/planners.py`)

- `MotionPlanner` / `JointMotionPlanner`: exact A* shortest paths between (position, orientation) pairs.
- `MediumLevelActionManager` (MLAM): high-level affordances (`pickup_onion_actions`, `put_onion_in_pot_actions`, `start_cooking_actions`, `pickup_dish_actions`, `deliver_soup_actions`, …) that map decisions to concrete motion goals. Rule-based agents are typically: state predicates → pick an affordance → filter goals with `is_valid_motion_start_goal_pair` (unfiltered unreachable goals crash the planner) → take the first action of the cheapest `MotionPlanner.get_plan`.
- Planners are pickle-cached in `src/overcooked_ai_py/data/planners/` keyed by layout name only — pass `force_compute=True` after changing MDP/layout parameters or you'll get stale plans.

Known agent pitfalls: `GreedyHumanModel` asserts the only order is 3-onion soup and can fail on layouts requiring true coordination (e.g. `forced_coordination`); pairs of memoryless agents can deadlock face-to-face — copy the `auto_unstuck` block from `GreedyHumanModel.action` (`agents/agent.py:366`).

### Deprecated code — important context

Large commented-out blocks are *intentionally dead*, pending a port to the rewritten MDP:

- `agents/agent.py`: `CoupledPlanningAgent`, `EmbeddedPlanningAgent` (planning/ToM agents from the 2019 paper).
- `planning/planners.py`: `MediumLevelPlanner`, its A* methods, and `Heuristic`.

They reference a deleted state API (`state.order_list`, `state.num_delivered`, tuple-encoded object state). Don't uncomment them expecting them to run; the live `MediumLevelActionManager` still exposes the methods they call (`get_medium_level_actions`, `joint_ml_actions`), so they serve as porting blueprints. Original working versions: `neurips2019` branch of `HumanCompatibleAI/human_aware_rl`.

`human_aware_rl` itself (PPO self-play, PPO_BC, behavior cloning) is deprecated but runnable with the `harl` extra. The real human gameplay data in `src/human_aware_rl/static/human_data/` is still valuable. PBT from the paper is not in this repo at all.

### Custom experiment agents (not upstream)

`agents/custom_agents.py` holds project-specific rule-based agents — `OnionLoaderAgent` (only stocks pots with onions; memoryless) and `ServerAgent` (starts cooking, fetches dish, collects and delivers soup; has auto-unstuck). They form an exact division of labor: paired together they score a deterministic 200 on `cramped_room`, while mismatched pairings score lower with high variance (greedy+server can livelock with both agents stuck holding dishes).

Run and visualize any pairing (one episode → animated GIF):

```bash
.venv/bin/python -m overcooked_ai_py.agents.visualize_agents loader_server
.venv/bin/python -m overcooked_ai_py.agents.visualize_agents greedy_server --layout cramped_room --games 5
```

Pairings: `loader_server`, `greedy_server`, `loader_greedy`, `greedy_greedy`. Rewards are printed per game; the first game is rendered to `visualizations/<pairing>_NNN.gif` (auto-incrementing, never overwrites; the directory is git-excluded locally via `.git/info/exclude`). Frames land in `/tmp/overcooked_frames_<pairing>/`. GIF assembly needs Pillow, which is not a declared dependency: `uv pip install pillow`.

### Visualization

`visualization/state_visualizer.py` — `StateVisualizer().display_rendered_trajectory(trajs, img_directory_path=..., ipython_display=False)` renders one PNG per timestep (pygame, works headless); `ipython_display=True` gives a timestep slider in notebooks. `Overcooked Tutorial.ipynb` at the repo root demonstrates the env + visualization end to end.

### Demo server

`src/overcooked_demo/` is a Flask/Docker app for playing in-browser against agents (`overcooked-demo-up` entry point). Not needed for programmatic experiments.
