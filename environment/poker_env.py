# environment/poker_env.py

import gym
import torch
from gym import spaces
import numpy as np
from typing import List
from classes.player import Player
from classes.game_state import GameState
from utils.helper_functions import encode_card, prepare_features
from utils.helper_functions import default_strategy, map_action_decision_to_action, get_legal_bets, update_gameplay_data
from actions.player_actions import update_game_state
from config import logger, device
from treys import Card as TreysCard
from treys import Evaluator

class FullyObservablePokerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, players: List[Player], device: torch.device):
        super(FullyObservablePokerEnv, self).__init__()
        self.players = players
        self.num_players = len(players)
        self.current_player_idx = 0
        self.game_state = GameState(players)
        self.device = device

        feature_length = 6 + (6 * self.num_players) + 10 + 1  # 6 fixed + 6 per player + 10 community + 1 pot size
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(feature_length,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # Actions: fold, call/check, raise/bet

    def reset(self):
        self.game_state.reset_for_new_hand()
        self.game_state.deal_cards()
        self.game_state.post_blinds()
        self.current_player_idx = (self.game_state.button_position + 3) % len(self.game_state.players) if len(self.game_state.players) > 0 else 0
        return self._get_obs()

    def step(self, action):
        player = self.players[self.current_player_idx]
        action_decision, amount = map_action_decision_to_action(action, self.game_state, player)
        update_game_state(self.game_state, player, action_decision, amount)

        gameplay_data = {
            'player_id': player.player_id,
            'features': prepare_features(self.game_state, player),
            'action': action_decision,
            'amount': amount
        }

        done = self.game_state.is_betting_round_over()

        self.current_player_idx = (self.current_player_idx + 1) % self.num_players

        reward = calculate_reward(player)

        if self.game_state.round_stage == 3 or len(self.game_state.active_players) == 1:
            done = True
            winners, pot_share = self.game_state.determine_winners_and_pot()
            self.game_state.assign_rewards(winners, pot_share, self.players)  # Call assign_rewards via GameState
            update_gameplay_data(winners, pot_share, gameplay_data, self.players)
            reward = rewards.get(player.player_id, 0)

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        obs = []
        for player in self.players:
            for card in player.hand:
                obs.extend(encode_card(card))
            while len(player.hand) < 2:
                obs.extend([0.0, 0.0, 0.0])  # Pad with zeros if hand has less than 2 cards
        for card in self.game_state.community_cards:
            obs.extend(encode_card(card))
        while len(self.game_state.community_cards) < 5:
            obs.extend([0.0, 0.0])  # Pad with zeros if less than 5 community cards
        obs.append(self.game_state.pot_size / 1000)
        for player in self.players:
            obs.append(player.stack_size / 1000)
        obs.append(self.game_state.current_bet / 1000)
        return np.array(obs, dtype=np.float32)

def calculate_reward(player: Player) -> float:
    return -0.01  # Placeholder for actual reward logic

# Ensure that map_action_decision_to_action is correctly defined
# and that determine_winners_and_pot is accessed via game_state

