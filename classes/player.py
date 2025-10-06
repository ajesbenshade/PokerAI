# classes/player.py

import torch
from torch import nn
from typing import List, Optional
from classes.card import Card
from models.player_model import PlayerModel
from config import Config, device, player_models, logger, used_player_ids
from utils.helper_functions import generate_unique_player_id

TOTAL_PLAYERS = Config.TOTAL_PLAYERS

class Player:
    def __init__(self, player_id: Optional[int] = None, player_model: Optional[nn.Module] = None, num_players=TOTAL_PLAYERS):
        self.player_id = player_id if player_id is not None else generate_unique_player_id()
        if player_model is not None:
            self.model = player_model.to(device)
        else:
            self.model = PlayerModel(num_players=num_players).to(device)
        self.model.eval()
        self.stack_size = 1000
        self.hand: List[Card] = []
        self.folded = False
        self.acted = False
        self.position = 0
        self.current_bet = 0
        self.pref_features: List[int] = []
        self.postf_features: List[int] = []

    def is_model_fitted(self):
        return self.model is not None

    def __hash__(self):
        return hash(self.player_id)
    
    def __eq__(self, other):
        return isinstance(other, Player) and self.player_id == other.player_id

    def __repr__(self):
        return f"Player(id={self.player_id}, stack_size={self.stack_size})"
    
    def __str__(self):
        return f"Player {self.player_id}: Stack={self.stack_size}, Folded={self.folded}"
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state['model'] = None  # Exclude the model from pickling
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        if state['model'] is not None:
            pass  # Handle if model is provided
        else:
            self.model = PlayerModel(num_players=TOTAL_PLAYERS).to(device)
            logger.error("The parameter configuration is missing after deserialization.")
