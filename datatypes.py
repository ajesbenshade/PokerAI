from __future__ import annotations
from typing import List, Any
from config import Config, Action
import logging

logger = logging.getLogger(__name__)

class Player:
    def __init__(self, player_id: int, agent=None) -> None:
        self.player_id = player_id
        self.stack = Config.INITIAL_STACK
        self.hand: List[Card] = []
        self.folded = False
        self.current_bet = 0
        self.all_in = False
        self.hand_contribution = 0  # Total contribution to the current hand
        # The agent is now passed during initialization.
        self.agent = agent

    def copy(self, sim: bool = False) -> Player:
        """
        Creates a copy of the player.
        
        CHANGE: This method is now much simpler. For simulations, it creates a
        lightweight copy of the player's state (stack, hand, etc.) but shares
        the reference to the original agent for decision-making. This avoids
        creating numerous copies of the neural network models, which was a
        major source of memory overhead.
        """
        new_player = Player(self.player_id, self.agent)
        new_player.stack = self.stack
        new_player.hand = self.hand.copy() # Shallow copy is fine for cards
        new_player.folded = self.folded
        new_player.current_bet = self.current_bet
        new_player.all_in = self.all_in
        new_player.hand_contribution = getattr(self, "hand_contribution", 0)
        return new_player

    def choose_action(self, *args, **kwargs):
        """
        Delegates the choose_action call to the assigned agent.
        This simplifies the logic and avoids re-creating temporary agents.
        """
        if self.agent:
            return self.agent.choose_action(*args, **kwargs)
        # Fallback for bots or players without a full agent
        # Return 5 values to match ActorCriticAgent interface
        return Action.FOLD.value, None, Action.FOLD.value, -10.0, 0.0


class GameState:
    def __init__(self, players: List[Player]) -> None:
        self.players = players
        self.pot_size = 0
        self.community_cards: List[Card] = []
        self.betting_history: List[Any] = []
        self.deck: List[Card] = []
        self.last_raise_size = Config.BIG_BLIND  # Initialize with big blind as first raise size
        # CFR-specific attributes
        self.current_player_idx = 0
        self.call_amount = 0
        self.min_raise = Config.BIG_BLIND
        self.max_raise = max(p.stack for p in players) if players else 0
        self.can_raise = True

class Card:
    def __init__(self, value: int, suit):
        self.value = value
        self.suit = suit

    def __repr__(self):
        value_map = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T'}
        return f"{value_map.get(self.value, self.value)}{self.suit.name[0]}"

    def __eq__(self, other):
        if isinstance(other, Card):
            return self.value == other.value and self.suit == other.suit
        return False

    def __hash__(self):
        return hash((self.value, self.suit))
