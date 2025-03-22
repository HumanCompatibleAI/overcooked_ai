![MDP python tests](https://github.com/HumanCompatibleAI/overcooked_ai/workflows/.github/workflows/pythontests.yml/badge.svg) ![overcooked-ai codecov](https://codecov.io/gh/HumanCompatibleAI/overcooked_ai/branch/master/graph/badge.svg) [![PyPI version](https://badge.fury.io/py/overcooked-ai.svg)](https://badge.fury.io/py/overcooked-ai) [!["Open Issues"](https://img.shields.io/github/issues-raw/HumanCompatibleAI/overcooked_ai.svg)](https://github.com/HumanCompatibleAI/minerl/overcooked_ai) [![GitHub issues by-label](https://img.shields.io/github/issues-raw/HumanCompatibleAI/overcooked_ai/bug.svg?color=red)](https://github.com/HumanCompatibleAI/overcooked_ai/issues?utf8=%E2%9C%93&q=is%3Aissue+is%3Aopen+label%3Abug) [![Downloads](https://pepy.tech/badge/overcooked-ai)](https://pepy.tech/project/overcooked-ai)
[![arXiv](https://img.shields.io/badge/arXiv-1910.05789-bbbbbb.svg)](https://arxiv.org/abs/1910.05789)

# Overcooked-AI 🧑‍🍳🤖

<p align="center">
  <!-- <img src="overcooked_ai_js/images/screenshot.png" width="350"> -->
  <img src="./images/layouts.gif" width="100%"> 
  <i>5 of the available layouts. New layouts are easy to hardcode or generate programmatically.</i>
</p>

## Introduction 🥘

Overcooked-AI is a benchmark environment for fully cooperative human-AI task performance, based on the wildly popular video game [Overcooked](http://www.ghosttowngames.com/overcooked/).

The goal of the game is to deliver soups as fast as possible. Each soup requires placing up to 3 ingredients in a pot, waiting for the soup to cook, and then having an agent pick up the soup and delivering it. The agents should split up tasks on the fly and coordinate effectively in order to achieve high reward.

You can **try out the game [here](https://humancompatibleai.github.io/overcooked-demo/)** (playing with some previously trained DRL agents). To play with your own trained agents using this interface, or to collect more human-AI or human-human data, you can use the code [here](https://github.com/HumanCompatibleAI/overcooked_ai/tree/master/src/overcooked_demo). You can find some human-human and human-AI gameplay data already collected [here](https://github.com/HumanCompatibleAI/overcooked_ai/tree/master/src/human_aware_rl/static/human_data).

**NOTE + LOOKING FOR CONTRIBUTORS:** DRL and BC implementations are now deprecated. We used to include code for training BC and PPO agents in the `human_aware_rl` directory. See [this issue](https://github.com/HumanCompatibleAI/overcooked_ai/issues/162) for more details.

This benchmark was build in the context of a 2019 paper: *[On the Utility of Learning about Humans for Human-AI Coordination](https://arxiv.org/abs/1910.05789)*. Also see our [blog post](https://bair.berkeley.edu/blog/2019/10/21/coordination/).

## Research Papers using Overcooked-AI 📑


- Carroll, Micah, Rohin Shah, Mark K. Ho, Thomas L. Griffiths, Sanjit A. Seshia, Pieter Abbeel, and Anca Dragan. ["On the utility of learning about humans for human-ai coordination."](https://arxiv.org/abs/1910.05789) NeurIPS 2019.
- Charakorn, Rujikorn, Poramate Manoonpong, and Nat Dilokthanakul. [“Investigating Partner Diversification Methods in Cooperative Multi-Agent Deep Reinforcement Learning.”](https://www.rujikorn.com/files/papers/diversity_ICONIP2020.pdf) Neural Information Processing. ICONIP 2020.
- Knott, Paul, Micah Carroll, Sam Devlin, Kamil Ciosek, Katja Hofmann, Anca D. Dragan, and Rohin Shah. ["Evaluating the Robustness of Collaborative Agents."](https://arxiv.org/abs/2101.05507) AAMAS 2021.
- Nalepka, Patrick, Jordan P. Gregory-Dunsmore, James Simpson, Gaurav Patil, and Michael J. Richardson. ["Interaction Flexibility in Artificial Agents Teaming with Humans."](https://www.researchgate.net/publication/351533529_Interaction_Flexibility_in_Artificial_Agents_Teaming_with_Humans) Cogsci 2021.
- Fontaine, Matthew C., Ya-Chuan Hsu, Yulun Zhang, Bryon Tjanaka, and Stefanos Nikolaidis. [“On the Importance of Environments in Human-Robot Coordination”](http://arxiv.org/abs/2106.10853) RSS 2021.
- Zhao, Rui, Jinming Song, Hu Haifeng, Yang Gao, Yi Wu, Zhongqian Sun, Yang Wei. ["Maximum Entropy Population Based Training for Zero-Shot Human-AI Coordination"](https://arxiv.org/abs/2112.11701). NeurIPS Cooperative AI Workshop, 2021.
- Sarkar, Bidipta, Aditi Talati, Andy Shih, and Dorsa Sadigh. [“PantheonRL: A MARL Library for Dynamic Training Interactions”](https://iliad.stanford.edu/pdfs/publications/sarkar2022pantheonrl.pdf). AAAI 2022.
- Ribeiro, João G., Cassandro Martinho, Alberto Sardinha, Francisco S. Melo. ["Assisting Unknown Teammates in Unknown Tasks: Ad Hoc Teamwork under Partial Observability"](https://arxiv.org/abs/2201.03538).
- Xihuai Wang, Shao Zhang, Wenhao Zhang, Wentao Dong, Jingxiao Chen, Ying Wen and Weinan Zhang. NeurIPS 2024. [“ZSC-Eval: An Evaluation Toolkit and Benchmark for Multi-agent Zero-shot Coordination”](https://arxiv.org/abs/2310.05208v2).


## Installation ☑️

### Installing from PyPI 🗜

You can install the pre-compiled wheel file using pip.
```
pip install overcooked-ai
```
Note that PyPI releases are stable but infrequent. For the most up-to-date development features, build from source. We recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/) to install the package, so that you can use the provided lockfile to ensure no minimal package version issues.


### Building from source 🔧

Clone the repo 
```
git clone https://github.com/HumanCompatibleAI/overcooked_ai.git
```

Using uv (recommended):
```
uv venv
uv sync
```


### Verifying Installation 📈

When building from source, you can verify the installation by running the Overcooked unit test suite. The following commands should all be run from the `overcooked_ai` project root directory:

```
python testing/overcooked_test.py
```




If you're thinking of using the planning code extensively, you should run the full testing suite that verifies all of the Overcooked accessory tools (this can take 5-10 mins): 
```
python -m unittest discover -s testing/ -p "*_test.py"
```

See this [notebook](Overcooked%20Tutorial.ipynb) for a quick guide on getting started using the environment.

## Code Structure Overview 🗺

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

`overcooked_demo` contains:

`server/`:
- `app.py`: The Flask app 
- `game.py`: The main logic of the game. State transitions are handled by overcooked.Gridworld object embedded in the game environment
- `move_agents.py`: A script that simplifies copying checkpoints to [agents](src/overcooked_demo/server/static/assets/agents/) directory. Instruction of how to use can be found inside the file or by running `python move_agents.py -h`

`up.sh`: Shell script to spin up the Docker server that hosts the game 

`human_aware_rl` contains (NOTE: this is not supported anymore, see bottom of the README for more info):

`ppo/`:
- `ppo_rllib.py`: Primary module where code for training a PPO agent resides. This includes an rllib compatible wrapper on `OvercookedEnv`, utilities for converting rllib `Policy` classes to Overcooked `Agent`s, as well as utility functions and callbacks
- `ppo_rllib_client.py` Driver code for configuing and launching the training of an agent. More details about usage below
- `ppo_rllib_from_params_client.py`: train one agent with PPO in Overcooked with variable-MDPs 
- `ppo_rllib_test.py` Reproducibility tests for local sanity checks
- `run_experiments.sh` Script for training agents on 5 classical layouts
- `trained_example/` Pretrained model for testing purposes

`rllib/`:
- `rllib.py`: rllib agent and training utils that utilize Overcooked APIs
- `utils.py`: utils for the above
- `tests.py`: preliminary tests for the above

`imitation/`:
- `behavior_cloning_tf2.py`:  Module for training, saving, and loading a BC model
- `behavior_cloning_tf2_test.py`: Contains basic reproducibility tests as well as unit tests for the various components of the bc module.

`human/`:
- `process_data.py` script to process human data in specific formats to be used by DRL algorithms
- `data_processing_utils.py` utils for the above

`utils.py`: utils for the repo


## Raw Data :ledger:

The raw data used during BC training is >100 MB, which makes it inconvenient to distribute via git. The code uses pickled dataframes for training and testing, but in case one needs to original data it can be found [here](https://drive.google.com/drive/folders/1aGV8eqWeOG5BMFdUcVoP2NHU_GFPqi57?usp=share_link) 

## Deprecated: Behavior Cloning and Reinforcement Learning 





## Further Issues and questions ❓

If you have issues or questions, you can contact [Micah Carroll](https://micahcarroll.github.io) at mdc@berkeley.edu.