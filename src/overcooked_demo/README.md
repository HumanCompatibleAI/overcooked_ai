# Overcooked Demo
<p align="center">
<img src="./server/static/images/browser_view.png" >
</p>

A web application where humans can play Overcooked with trained AI agents.

* [Installation](#installation)
* [Usage](#usage)
* [Dependencies](#dependencies)
* [Using Pre-trained Agents](#using-pre-trained-agents)
* [Updating](#updating)
* [Configuration](#configuration)
* [Legacy Code](#legacy-code)

## Installation

Building the server image requires [Docker](https://docs.docker.com/get-docker/)

## Usage

The server can be deployed locally using the driver script included in the repo. To run the production server, use the command
```bash
./up.sh production
```

In order to build and run the development server, which includes a deterministic scheduler and helpful debugging logs, run
```bash
./up.sh
```

After running one of the above commands, navigate to http://localhost

In order to kill the production server, run
```bash
./down.sh
```

You can also start the server via the command line. After installing the `overcooked_ai` via pip, you can start the server by typing

```
overcooked-demo-up
```

in the terminal. The same arguments still apply

## Dependencies

The Overcooked-Demo server relies on the [overcooked-ai](https://github.com/HumanCompatibleAI/overcooked_ai) repo, specifically the submodules overcooked_ai_py and human_aware_rl. Changes made in these modules will be reflected in the overcooked_demo server.

One thing to note is that local changes to the game logic will not be present by default in the demo (it is pulled directly from the GitHub version). Only changes made in the overcooked_demo folder locally will be reflected in the demo.

## Using Pre-trained Agents

Overcooked-Demo can dynamically load pre-trained agents provided by the user. In order to use a pre-trained agent, a pickle file should be added to the `agents` directory. The final structure will look like `static/assets/agents/<agent_name>/agent.pickle`. Note, to use the pre-defined rllib loading routine, the agent directory name must start with 'rllib', and contain the appropriate rllib checkpoint, config, and metadata files. Details can be found in the [README](server/static/assets/agents/README.md) under the agent directory. We also provide a [move_agents.py](server/move_agents.py) file that can help copy over checkpoint files.

If a more complex or custom loading routing is necessary, one can subclass the `OvercookedGame` class and override the `get_policy` method, as done in [DummyOvercookedGame](server/game.py#L420). Make sure the subclass is properly imported [here](server/app.py#L5)

## Use the human vs. human game mode.

With the Overcooked demo, you can test the interaction between two human players. To do this, you need to deploy this code on the server (https://docs.docker.com/language/python/deploy/). 
After successful deployment, the first user should open http://[server_ip_address]/, select the human keyboard input for both players and click on "Create game". If everything has been successful, he will receive a message: "Waiting for game to start".
Another user should open a page at http://[server_ip_address]/psiturk to start the game.  

If you want to run a test on a local computer, you should use "localhost" instead of "server_ip_address" and open the corresponding links in different tabs.

## Updating
Changes to the JSON state representation of the game will require updating the JS graphics. At the highest level, a graphics implementation must implement the functions `graphics_start`, called at the start of each game, `graphics_end`, called at the end of each game, and `drawState`, called at every timestep tick. See [dummy_graphcis.js](server/graphics/dummy_graphics.js) for a barebones example.

The graphics file is dynamically loaded into the docker container and served to the client. Which file is loaded is determined by the `GRAPHICS` environment variable. For example, to server `dummy_graphics.js` one would run
```bash
GRAPHICS=dummy_graphics.js ./up.sh
```
The default graphics file is currently `overcooked_graphics_v2.1.js`


## Configuration

Basic game settings can be configured by changing the values in [config.json](server/config.json)

## Legacy Code

For legacy code compatible with the Neurips2019 submission please see [this](https://github.com/HumanCompatibleAI/overcooked-demo/tree/legacy) branch of this repo. 
