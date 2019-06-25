"""
Each Player has a set of visible cards / knowledge about their own cards in the state
"""

import random
import numpy as np

COLORS = ['white', 'yellow', 'green', 'blue', 'red']
COLORS_TO_INDEX = { COLORS[i]: i for i in range(len(COLORS)) }
NUMBERS = list(range(1, 6))
NUMBER_COUNT = { 1: 3, 2: 2, 3: 2, 4: 2, 5: 1 }
CARDS_IN_HAND = 4
NUM_INFORMATION_TOKENS = 8
NUM_FUSE_TOKENS = 3

class Card(object):
    """
    Card objects are either in the deck
    """

    def __init__(self, color, number, idx, color_knowledge=None, number_knowledge=None):
        assert color in COLORS_TO_INDEX.keys()
        assert number in NUMBER_COUNT.keys()
        self.color = color
        self.number = number
        self.idx = idx # For preventing hash conflicts between multiple cards

        # Possible numbers and colors the card could have
        # according to the holder
        self.color_knowledge = color_knowledge if color_knowledge is not None else COLORS
        self.number_knowledge = number_knowledge if number_knowledge is not None else NUMBERS

    def to_card_idx(self):
        return AbstractHanabiDeck.CARD_TO_INDEX[self]

    def knowledge_to_vec(self):
        knowledge_vec = []
        for col in COLORS:
            if col in self.color_knowledge:
                knowledge_vec.append(1)
            else:
                knowledge_vec.append(0)
        for num in NUMBERS:
            if num in self.number_knowledge:
                knowledge_vec.append(1)
            else:
                knowledge_vec.append(0)
        return knowledge_vec

    def __repr__(self):
        card_string = self.color
        if self.color_knowledge != COLORS:
            card_string += " {}".format(self.color_knowledge)
        card_string += " {}".format(self.number)
        if self.number_knowledge != NUMBERS:
            card_string += " {}".format(self.number_knowledge)
        return card_string

    def __hash__(self):
        # Does not take into account knowledge state
        return hash((self.color, self.number, self.idx))

    def __eq__(self, other):
        return self.color == other.color and self.number == other.number and self.idx == other.idx

    def deepcopy(self):
        return Card(
            self.color,
            self.number,
            self.idx,
            list(self.color_knowledge),
            list(self.number_knowledge)
        )

class AbstractHanabiDeck(object):
    BASE_DECK = []
    CARD_TO_INDEX = {}

    card_idx = 0
    for col in COLORS_TO_INDEX.keys():
        for num, count in NUMBER_COUNT.items():
            for i in range(count):
                card = Card(col, num, i)
                BASE_DECK.append(card)
                CARD_TO_INDEX[card] = card_idx
                card_idx += 1
    
    INDEX_TO_CARD = { v:k for k, v in CARD_TO_INDEX.items() }
    DECK_SIZE = len(BASE_DECK)

class HanabiDeck(object):

    def __init__(self, cards=None):
        if cards is None:
            pass
            self.reset()
        else:
            self.deck = cards

    @property
    def num_cards(self):
        return len(self.deck)

    def reset(self):
        self.deck = [c.deepcopy() for c in AbstractHanabiDeck.BASE_DECK]

    def draw_card(self):
        card_idx = np.random.choice(self.num_cards)
        return self.deck.pop(card_idx)

    def deepcopy(self):
        return HanabiDeck([c.deepcopy() for c in self.deck])

class PlayerState(object):
    
    def __init__(self, cards, idx):
        """
        Cards are the actual cards
        """
        self.cards = cards
        self.idx = idx

    def remove_card(self, card_idx):
        return self.cards.pop(card_idx)

    def draw_card(self, deck):
        self.cards.append(deck.draw_card())

    def receive_hint(self, hint_type):
        if type(hint_type) is int:
            for c in self.cards:
                if c.number == hint_type:
                    c.number_knowledge = [c.number]
                elif hint_type in c.number_knowledge:
                    c.number_knowledge.remove(hint_type)
        elif type(hint_type) is str:
            for c in self.cards:
                if c.color == hint_type:
                    c.color_knowledge = [c.color]
                elif hint_type in c.color_knowledge:
                    c.color_knowledge.remove(hint_type)
        else:
            raise ValueError("Unrecognized hint type")

    @property
    def num_cards(self):
        return len(self.cards)

    def deepcopy(self):
        return PlayerState([c.deepcopy() for c in self.cards], self.idx)

class HanabiState(object):

    def __init__(self, players, deck, tokens, table=None, discarded=None):
        """
        tokens: (num_info, num_fuse)
        """
        self.players = players
        self.deck = deck
        self.tokens = tokens
        self.table = table if table is not None else []
        self.discarded = discarded if discarded is not None else []

    @property
    def info_tokens(self):
        return self.tokens[0]

    @property
    def fuse_tokens(self):
        return self.tokens[1]

    def on_table_with_color(self, color):
        return [c for c in self.table if c.color == color]

    def largest_on_table_with_color(self, color):
        same_color_nums = [c.number for c in self.on_table_with_color(color)]
        if len(same_color_nums) == 0:
            return None
        return max(same_color_nums)

    def add_to_table(self, card):
        self.table.append(card)

    def consume_info_token(self):
        self.tokens = (self.tokens[0] - 1, self.tokens[1])

    def consume_fuse_token(self):
        self.tokens = (self.tokens[0], self.tokens[1] - 1)

    def discard_card(self, card):
        self.discarded.append(card)

    def deepcopy(self):
        return HanabiState(
            [p.deepcopy() for p in self.players], 
            self.deck.deepcopy(),
            tuple(self.tokens),
            [c.deepcopy() for c in self.table],
            [c.deepcopy() for c in self.discarded]
        )
    
    def __repr__(self):
        state_string = "\nDeckSize: {}\nCards on table: {}".format(self.deck.num_cards,self.table)
        for p in self.players:
            state_string += "\nPlayer {} cards:\n\t{}".format(p.idx, p.cards)
        state_string += "\nToken status: {}".format(self.tokens)
        if HanabiGame.is_terminal(self):
            state_string += "\nGAME OVER\n\n"
        return state_string

class HanabiGame(object):

    def __init__(self, num_players, cards_per_hand):
        self.num_players = num_players
        self.cards_per_hand = cards_per_hand
        self.action_manager = Action(self.num_players, self.cards_per_hand)

    def get_start_state(self):
        deck = HanabiDeck()
        players = []
        for i in range(self.num_players):
            first_hand = []
            for _ in range(self.cards_per_hand):
                first_hand.append(deck.draw_card())
            players.append(PlayerState(first_hand, i))
        start_tokens = (NUM_INFORMATION_TOKENS, NUM_FUSE_TOKENS)
        return HanabiState(players, deck, start_tokens)

    @staticmethod
    def get_actions(state, player_idx):
        # TODO: Currently unnecessary code overhead here
        # repeating logic from Action class
        player = state.players[player_idx]
        player_actions = []
        if state.info_tokens > 0:
            for i in range(len(state.players)):
                if i != player.idx:
                    # Iterating over players that can receive hint
                    for col in COLORS:
                        player_actions.append((Action.HINT, i, col))

                    for num in NUMBERS:
                        player_actions.append((Action.HINT, i, num))

        for i in range(player.num_cards):
            player_actions.append((Action.PLAY_CARD, i))
            player_actions.append((Action.DISCARD, i))
        return player_actions

    def get_transition_states_and_probs(self, state, player_idx, action):
        reward = 0

        valid_actions = self.get_actions(state, player_idx)
        if action not in valid_actions:
            print("Invalid action")
            reward -= 10
            action_idx = np.random.choice(range(len(valid_actions)))
            action = valid_actions[action_idx]

        new_state = state.deepcopy()
        player = new_state.players[player_idx]

        action_type = action[0]
        if action_type == Action.PLAY_CARD:
            assert len(action) == 2
            card_idx = action[1]
            reward += self.play_card(new_state, player, card_idx)
            player.draw_card(new_state.deck)

        elif action_type == Action.DISCARD:
            assert len(action) == 2
            card_idx = action[1]
            self.discard_card(new_state, player, card_idx)
            player.draw_card(new_state.deck)
        
        elif action_type == Action.HINT:
            assert len(action) == 3
            to_player_idx, hint_type = action[1], action[2]
            assert to_player_idx != player_idx
            self.give_hint(new_state, to_player_idx, hint_type)
            
        else:
            raise ValueError("Unrecognized action")

        return [(new_state, 1.0)], reward

    @staticmethod
    def is_terminal(state):
        is_defeat = state.deck.num_cards == 0 or state.fuse_tokens == 0
        is_victory = sum([1 for c in COLORS if state.largest_on_table_with_color(c) == 5]) == len(COLORS)
        return is_defeat or is_victory
            
    def give_hint(self, state, to_player_idx, hint_type):
        state.players[to_player_idx].receive_hint(hint_type)
        state.consume_info_token()

    def discard_card(self, state, player, card_idx):
        card_to_discard = player.remove_card(card_idx)
        state.discard_card(card_to_discard)
    
    def play_card(self, state, player, card_idx):
        card_to_play = player.remove_card(card_idx)
        buildoff_num = state.largest_on_table_with_color(card_to_play.color)
        reward = 0
        if buildoff_num is None and card_to_play.number == 1:
            state.add_to_table(card_to_play)
            reward = 1
        elif buildoff_num == card_to_play.number - 1:
            state.add_to_table(card_to_play)
            reward = 1
        else:
            state.consume_fuse_token()
            state.discard_card(card_to_play)
        return reward

    @staticmethod
    def preprocess_observation(hanabi_state, mdp, primary_agent_idx):
        """
        Preprocessing observations:
        Observation is the state of all other cards on the board (with respective players), hints about one's own cards, fuses, information tokens

        50 cards, 
        50 - 50 - 50 boolean of what people have in their hands

        4 cards in hand.
        4 * [ ( 5 number bools ), ( 5 color bools ) ]

        discarded_cards

        [num_info_tokens, num_fuse_tokens, num_cards_in_deck]
        """
        observation = []

        for p in hanabi_state.players:
            if p.idx != primary_agent_idx:
                player_obs = np.zeros(AbstractHanabiDeck.DECK_SIZE)
                card_idxs = [c.to_card_idx() for c in p.cards]
                for idx in card_idxs:
                    player_obs[idx] = 1
                observation.append(("player{}_cards".format(p.idx), player_obs))
        
        curr_player = hanabi_state.players[p.idx]

        player_hand = []
        for card in curr_player.cards:
            player_hand.extend(card.knowledge_to_vec())

        for _ in range(curr_player.num_cards, mdp.cards_per_hand):
            player_hand.extend(np.zeros(len(COLORS) + len(NUMBERS)))
        
        observation.append(("player{}_knowledge".format(primary_agent_idx), player_hand))
        
        discarded_cards = np.zeros(AbstractHanabiDeck.DECK_SIZE)
        card_idxs = [c.to_card_idx() for c in hanabi_state.discarded]
        for idx in card_idxs:
            discarded_cards[idx] = 1
        observation.append(("discarded_cards", discarded_cards))
            
        observation.append(("info/fuse tokens", [hanabi_state.info_tokens, hanabi_state.fuse_tokens]))

        actual_obs = []
        for item in observation:
            actual_obs.extend(item[1])
        return np.array(actual_obs).astype(np.int8)

    def _check_valid_state(self, state):
        pass
        # Sum all cards == 50
        # fuse tokens > 0
        # cards on table are all in increasing order


class Action(object):
    HINT = "hint"
    DISCARD = "discard"
    PLAY_CARD = "play"

    def __init__(self, num_players, cards_per_hand):
        self.num_players = num_players
        self.cards_per_hand = cards_per_hand
        self.ACTION_TO_INDEX = { i: self.get_all_possible_hint_actions(i) for i in range(self.num_players)}
        self.INDEX_TO_ACTION = { i: { v:k for k, v in self.ACTION_TO_INDEX[i].items() } for i in range(self.num_players)}
        # hacky, asserting all actions spaces are same size
        assert len(self.INDEX_TO_ACTION[0]) == len(self.INDEX_TO_ACTION[1])
        self.num_actions = len(self.INDEX_TO_ACTION[0])

    def get_all_possible_hint_actions(self, player_idx):
        count = 0
        possible_actions = {}
        for i in range(self.num_players):
            if i != player_idx:
                # Iterating over players that can receive hint
                for col in COLORS:
                    possible_actions[(Action.HINT, i, col)] = count
                    count += 1
                for num in NUMBERS:
                    possible_actions[(Action.HINT, i, num)] = count
                    count += 1
        for i in range(self.cards_per_hand):
            possible_actions[(Action.PLAY_CARD, i)] = count
            count += 1
            possible_actions[(Action.DISCARD, i)] = count
            count += 1
        return possible_actions 

class Env(object):

    def __init__(self, mdp, start_state=None, horizon=float('inf')):
        """
        start_state (OvercookedState): what the environemt resets to when calling reset
        horizon (float): number of steps before the environment returns True to .is_done()
        """
        self.mdp = mdp
        self.start_state = start_state if start_state is not None else self.mdp.get_start_state()
        self.horizon = horizon
        self.cumulative_rewards = 0
        self.reset()

    def __repr__(self):
        return "Current player up {}".format(self.player_idx) + str(self.state)

    def get_current_state(self):
        return self.state

    def step(self, action):
        """Performs a joint action, updating the state and providing a reward."""
        assert not self.is_done()
        state = self.get_current_state()
        next_state, reward, prob = self.get_random_next_state(state, action)
        self.cumulative_rewards += reward
        self.state = next_state
        self.t += 1
        self.player_idx = (self.player_idx + 1) % self.mdp.num_players
        done = self.is_done()
        info = {"prob": prob}
        if done:
            info['episode'] = {
                'r': self.cumulative_rewards, 
                'l': self.t, 
            }
        return (next_state, reward, done, info)

    def get_random_next_state(self, state, action):
        """Chooses the next state according to T(state, action)."""
        results, reward = self.mdp.get_transition_states_and_probs(state, self.player_idx, action)

        # If deterministic, don't generate a random number
        if len(results) == 1:
            return (results[0][0], reward, 1.0)

        rand = random.random()
        sum = 0.0
        for next_state, prob in results:
            sum += prob
            if sum > 1.0:
                raise ValueError('Total transition probability more than one.')
            if rand < sum:
                return (next_state, reward, prob)
        raise ValueError('Total transition probability less than one.')

    def reset(self):
        """Resets the environment. Does NOT reset the agent."""
        self.state = self.start_state
        self.cumulative_rewards = 0
        self.t = 0
        self.player_idx = 0

    def is_done(self):
        """Returns True if the episode is over and the agent cannot act."""
        return self.t >= self.horizon or self.mdp.is_terminal(self.state)

    def run_agents(self, agent_list, display=False, displayEnd=False, final_state=True):
        """
        Trajectory returned will a list of state-action pairs (s_t, a_t, r_t), in which
        the last element will be the last state visited and a None joint action.
        Therefore, there will be t + 1 tuples in the trajectory list.
        NOTE: Does not reset environment
        """
        trajectory = []
        done = False

        if display: print(self)
        while not done:
            s_t = self.state
            a_t = agent_list[self.player_idx].action(s_t)

            # Break if either agent is out of actions
            if any([a is None for a in a_t]):
                break

            s_tp1, r_t, done, _ = self.step(a_t)
            trajectory.append((s_t, a_t, r_t))
            
            if display or (done and displayEnd): 
                print("\nTimestep: {}\nJoint action: {}\n{}".format(self.t, a_t, self))

        # Add final state
        # TODO: Clean up
        if final_state:
            trajectory.append((s_tp1, (None, None), 0))
            assert len(trajectory) == self.t + 1, "{} vs {}".format(len(trajectory), self.t)
        else:
            assert len(trajectory) == self.t, "{} vs {}".format(len(trajectory), self.t)
        return trajectory, self.t #, self.cumulative_rewards

    @staticmethod
    def print_state(mdp, s):
        e = Env(mdp, s)
        print(e)

class RandomAgent(object):

    def __init__(self, player_idx):
        self.player_idx = player_idx

    def action(self, state):
        valid_actions = HanabiGame.get_actions(state, self.player_idx)
        act_idx = np.random.choice(len(valid_actions))
        return valid_actions[act_idx]


import gym
from gym import spaces
class Hanabi(gym.Env):
    """Wrapper for the Env class above that is compatible with gym API
    
    The convention is that all processed observations returned are ordered in such a way
    to be standard input for the main agent. One can use the switch_player function to change
    the observation arrays as to be fed as input to the secondary agent."""

    def __init__(self, env_config=None):
        # TODO: clean this. Used for Rllib
        if env_config is not None:
            self.custom_init(**env_config)

    def custom_init(self, base_env, **kwargs):
        self.base_env = base_env
        self.mdp = self.base_env.mdp
        self.action_manager = self.mdp.action_manager
        
        dummy_state = self.mdp.get_start_state()
        observation_shape = self.mdp.preprocess_observation(dummy_state, self.mdp, 0).shape
        high = np.ones(observation_shape)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Discrete(self.action_manager.num_actions)
        self.reset()

    def step(self, action):
        """
        action: a tuple with the action of the primary and secondary agents in index format
        """
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        agent_action = self.mdp.action_manager.INDEX_TO_ACTION[self.base_env.player_idx][action]
        next_state, reward, done, info = self.base_env.step(agent_action)
        observation = self.mdp.preprocess_observation(next_state, self.mdp, primary_agent_idx=self.base_env.player_idx)
        assert len(np.argwhere(np.isnan(observation))) == 0, "There was a NaN among the observations"
        return observation, reward, done, info

    def reset(self):
        self.base_env.reset()
        self.agent_idx = np.random.choice(range(self.mdp.num_players))
        observation = self.mdp.preprocess_observation(self.base_env.state, self.mdp, primary_agent_idx=self.base_env.player_idx)
        return observation

    def render(self, mode='human', close=False):
        pass

for _ in range(100):
    a_l = [RandomAgent(0), RandomAgent(1), RandomAgent(2), RandomAgent(3)]
    g = HanabiGame(3, CARDS_IN_HAND)
    e = Env(g)
    _=e.run_agents(a_l, display=False)
    e.reset()
