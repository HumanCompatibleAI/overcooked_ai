# Overcooked-AI
<p align="center">
  <!-- <img src="overcooked_ai_js/images/screenshot.png" width="350"> -->
  <img src="overcooked_ai_js/images/layouts.gif" width="100%"> 
  <i>5 of the available layouts. New layouts are easy to hardcode or generate programmatically.</i>
</p>

## Introduction

Overcooked-AI is a benchmark environment for fully cooperative multi-agent performance, based on the wildly popular video game [Overcooked](http://www.ghosttowngames.com/overcooked/). 

The goal of the game is to deliver soups as fast as possible. Each soup requires taking 3 items and placing them in a pot, waiting for the soup to cook, and then having an agent pick up the soup and delivering it. The agents should split up tasks on the fly and coordinate effectively in order to achieve high reward.

You can **try out the game [here](https://humancompatibleai.github.io/overcooked-demo/)** (playing with some previously trained DRL agents). To play with your own trained agents using this interface, you can use [this repo](https://github.com/HumanCompatibleAI/overcooked-demo)). To run human-AI experiments, check out [this repo](https://github.com/HumanCompatibleAI/overcooked-hAI-exp).

For DRL implementations compatible with the environment and reproducible results to our paper, *[On the Utility of Learning about Humans for Human-AI Coordination](https://arxiv.org/abs/1910.05789)* ([blog post](https://bair.berkeley.edu/blog/2019/10/21/coordination/)), check out [this repo](https://github.com/HumanCompatibleAI/human_aware_rl).

## Installation

It is useful to setup a conda environment with Python 3.7:

```
conda create -n overcooked_ai python=3.7
conda activate overcooked_ai
```

To complete the installation after cloning the repo, run the following commands:

```
cd overcooked_ai
python setup.py develop
```

In `overcooked_ai_js` there is a javascript implementation of the Overcooked MDP and game visualizer.

To install it, cd into `overcooked_ai_js` and set up the package with `npm install`.
For development, you will also need to install browserify: `npm install -g browserify`


### Verifying Installation

#### Python Code

To verify your python installation, you can try running the following command from the inner `overcooked_ai_py` folder:

```
python run_tests_fast.py
```

or alternatively (this can take 5-10 mins): `python run_tests_full.py`

#### Javascript Code

Run tests with `npm run test`. Testing scripts use `jest`, which exposes a `window` object, and so
`npm run build-window` should be run before running modified tests.

`overcooked-window.js` is used for the demo and testing.

## Python Code Structure Overview

`overcooked_ai_py` contains:

`mdp/`:
- `overcooked_mdp.py`: main Overcooked game logic
- `overcooked_env.py`: environment classes built on top of the Overcooked mdp
- `layout_generator.py`: functions to generate random layouts programmatically

`agents/`:
- `agent.py`: location of agent classes
- `benchmarking.py`: sample trajectories of agents (both trained and planners) and load various models

`planning`:
- `planners.py`: near-optimal agent planning logic
- `search.py`: A* search and shortest path logic

`run_tests.py`: script to run all tests

## Python Visualizations

One can adapt a version of [this file](https://github.com/HumanCompatibleAI/human_aware_rl/blob/master/human_aware_rl/overcooked_interactive.py) in order to be able to play games in terminal graphics with custom-defined agents.

## Javascript Visualizations

To run a simple demo that plays a trajectory demonstrating the
transitions in the game (requires having npm installed):

```
$ npm run demo
```

## Further Issues and questions

If you have issues or questions, don't hesitate to contact [Micah Carroll](https://micahcarroll.github.io) at mdc@berkeley.edu.

