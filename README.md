# REPO_NAME

TODO: Change REPO_NAME

## Installation

When cloning the repo, make sure you also clone the submodules.

```
git clone --recurse-submodules -j8 REPO_GITPATH
```

It is useful to setup a conda environment with Python 3.7:

```
conda create -n REPO_NAME python=3.7
conda activate REPO_NAME
```

To complete the installation, run the following commands:

```
cd REPO_NAME
pip setup.py develop
pip install -r requirements.txt
```

## Verify Installation

To verify your installation, you can try running the following command from the inner `REPO_NAME` folder (with flag `-f` for a quick run, and without for a more comprehensive suite that will take approximately 5/10 minutes to run):

```
python run_tests.py -f
```

## Repo Structure Overview

`mdp/`:
- `overcooked_mdp.py`: main Overcooked game logic
- `overcooked_env.py`: environment classes built on top of the Overcooked mdp
- `overcooked_interactive.py`: script to play Overcooked in terminal against trained agents
- `layout_generator.py`: functions to generate random layouts programmatically

`agents/`:
- `agent.py`: where agent types are defined
- `benchmarking.py`: sample trajectories of agents (both trained and planners) and load various models

`planning`:
- `planners.py`: near-optimal agent planning logic
- `search.py`: A* search and shortest path logic

`run_tests.py`: script to run all tests
