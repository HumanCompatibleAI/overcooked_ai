import collections

from human_aware_rl.rllib.rllib import load_agent
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

path = "/Users/jacksonyan/Desktop/project with Micah/overcooked-demo/server/static/assets/agents/RllibCrampedRoomPPO_SP_temp/agent"
agent = load_agent(path)
agent.reset()
mdp = OvercookedGridworld.from_layout_name("cramped_room")
start_state = mdp.get_standard_start_state()
count = collections.defaultdict(int)
action, prob = agent.action(start_state)
print(action)
