# Example Directory for Pre-trained Rllib Agent

## Structure

* `agent`  
  * `.is_checkpoint`
  * `.tune_metadata`
  * `checkpoint-#`

Note that the file names must match EXACTLY the names and relative structure as above. The agent name is only determined and reflected in the parent directory name.

In the future, the naming convenction will probably be more flexible, but for now it is very strict to simplify serialization logic

## Training the Agent

In general, a directory with the structure above can be trained with the human_aware_rl module in the [overcooked_ai](https://github.com/HumanCompatibleAI/overcooked_ai) repo. Specific installation instruction is available. 

## Moving the Agent into Demo

You can move this agent into Overcooked-Demo by running
```
cd  <Overcooked-Demo-Root>/server/static/assets/agents/
mkdir RllibMyAgent
```
Note that all rllib agent directory names must match the regex pattern `/rllib.*/i`

Now copy over the appropriate files

```
cp -r ~/ray_results/MyAgent_0_2020-09-24_01-24-43m6jg7brh/checkpoint_<i> ./RllibMyAgent/agent
cp -r ~/ray_results/MyAgent_0_2020-09-24_01-24-43m6jg7brh/config.pkl ./RllibMyAgent/config.pkl
```

If the PPO agent is trained with a BC agent, you also need to supply the trained BC_model

```
cd  <Overcooked-Demo-Root>/server/static/assets/agents/RllibMyAgent
mkdir bc_params
```

The `train_bc_model` function `human_aware_rl.imitation.behavior_cloning_tf2` takes in a `model_dir` parameter. Find the saved bc_model at that location, and move everything to the bc_params directory we just created.

For example, if the `model_dir="...../bc_results/model1"`, 
```
cp -r ...../bc_results/model1 <Overcooked-Demo-Root>/server/static/assets/agents/RllibMyAgent/bc_params
```

Another option is to use the [move_agents.py](../../../move_agents.py) file and supplies the path to the checkpoint directory and an agent name 

```
python move_agents.py ~/ray_results/MyAgent_0_2020-09-24_01-24-43m6jg7brh/checkpoint_<i> RllibAgentX
```

It also takes in optional arguments to overwrite existing agent and to copy bc_model 

You can also directly call the function via CLI by typing

```
overcooked-demo-move path/to/checkpoint RllibAgentX
```

The same optional arguments apply

Relaunching the Overcooked-Demo server, you should now see `RllibMyAgent` in the dropdown of available agents

## Layout Compatibility

Please ensure that agents trained on layout `<X>` are only run on layout `<X>`. Attempting to run such an agent on layout `<Y>` will either result in very poor performance, or, if observation dimensions are mismatched, the thread executing the agent's actions will fail silently and the agent will remain stationary for the entire duration of the game! We recommend including the compatible layout names in the agent name, for example `RllibPPOSelfPlay_CounterCircuit` is a name of one of our trained agents. 
