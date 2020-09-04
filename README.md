![MDP python tests](https://github.com/HumanCompatibleAI/overcooked_ai/workflows/.github/workflows/pythontests.yml/badge.svg) ![overcooked-ai codecov](https://codecov.io/gh/HumanCompatibleAI/overcooked_ai/branch/master/graph/badge.svg)

# Overcooked-AI

<p align="center">
  <!-- <img src="overcooked_ai_js/images/screenshot.png" width="350"> -->
  <img src="./images/layouts.gif" width="100%"> 
  <i>5 of the available layouts. New layouts are easy to hardcode or generate programmatically.</i>
</p>

## Introduction

Overcooked-AI is a benchmark environment for fully cooperative multi-agent performance, based on the wildly popular video game [Overcooked](http://www.ghosttowngames.com/overcooked/). 

The goal of the game is to deliver soups as fast as possible. Each soup requires placing up to 3 ingredients in a pot, waiting for the soup to cook, and then having an agent pick up the soup and delivering it. The agents should split up tasks on the fly and coordinate effectively in order to achieve high reward.

You can **try out the game [here](https://humancompatibleai.github.io/overcooked-demo/)** (playing with some previously trained DRL agents). To play with your own trained agents using this interface, you can use [this repo](https://github.com/HumanCompatibleAI/overcooked-demo). To run human-AI experiments, check out [this repo](https://github.com/HumanCompatibleAI/overcooked-hAI-exp). You can find some human-human gameplay data already collected [here](https://github.com/HumanCompatibleAI/human_aware_rl/tree/master/human_aware_rl/data/human/anonymized).

Check out [this repo](https://github.com/HumanCompatibleAI/human_aware_rl) for the DRL implementations compatible with the environment and reproducible results to our paper: *[On the Utility of Learning about Humans for Human-AI Coordination](https://arxiv.org/abs/1910.05789)* (also see our [blog post](https://bair.berkeley.edu/blog/2019/10/21/coordination/)).

## Installation

### Installing from PyPI

You can install the pre-compiled wheel file using pip.
```
pip install overcooked-ai
```
Note that PyPI releases are stable but infrequent. For the most up-to-date development features, build from source


### Building from source

It is useful to setup a conda environment with Python 3.7 (virtualenv works too):

```
conda create -n overcooked_ai python=3.7
conda activate overcooked_ai
```

Clone the repo 
```
git clone https://github.com/HumanCompatibleAI/overcooked_ai.git
```
Finally, use python setup-tools to locally install

```
pip install -e overcooked_ai/
```


### Verifying Installation

When building from source, you can verify the installation by running the Overcooked unit test suite. The following commands should all be run from the `overcooked_ai` project root directory:

```
python testing/overcooked_test.py
```

If you're thinking of using the planning code extensively, you should run the full testing suite that verifies all of the Overcooked accessory tools (this can take 5-10 mins): 
```
python -m unittest discover -s testing/ -p "*_test.py"
```


## Code Structure Overview

`overcooked_ai_py` contains:

`mdp/`:
- `overcooked_mdp.py`: main Overcooked game logic
- `overcooked_env.py`: environment classes built on top of the Overcooked mdp
- `layout_generator.py`: functions to generate random layouts programmatically

`agents/`:
- `agent.py`: location of agent classes
- `benchmarking.py`: sample trajectories of agents (both trained and planners) and load various models

`planning/`:
- `planners.py`: near-optimal agent planning logic
- `search.py`: A* search and shortest path logic


## Python Visualizations

One can adapt a version of [this file](https://github.com/HumanCompatibleAI/human_aware_rl/blob/master/human_aware_rl/overcooked_interactive.py) in order to be able to play games in terminal graphics with custom-defined agents.


## Further Issues and questions

If you have issues or questions, don't hesitate to contact either [Micah Carroll](https://micahcarroll.github.io) at mdc@berkeley.edu or [Nathan Miller](https://github.com/nathan-miller23) at nathan_miller23@berkeley.edu

