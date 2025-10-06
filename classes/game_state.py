# classes/game_state.py

from typing import List, Dict, Optional, Tuple
from classes.player import Player
from classes.deck import Deck
from utils.helper_functions import pref
from treys import Card as TreysCard
from treys import Evaluator
import logging

logger = logging.getLogger(__name__)

class GameState:
    def __init__(self, players: List[Player]):
        self.players = players
        self.deck = Deck()
        self.pot_size = 0
        self.community_cards: List[TreysCard] = []
        self.current_bet = 0
        self.last_raiser: Optional[Player] = None
        self.round_stage = 0
        self.button_position = 0
        self.small_blind = 1
        self.big_blind = 2
        self.bet_history = []
        self.active_players = self.players.copy()
        self.aggression = 0
        self.max_raises_per_round = 3
        self.current_raises = 0
        self.evaluator = Evaluator()

    def reset_for_new_hand(self):
        self.deck = Deck()
        self.deck.shuffle()
        self.community_cards = []
        self.pot_size = 0
        self.current_bet = 0
        self.last_raiser = None
        self.round_stage = 0
        self.bet_history = []
        self.button_position = (self.button_position + 1) % len(self.players) if len(self.players) > 0 else 0
        self.active_players = self.players.copy()
        self.current_raises = 0
        for player in self.players:
            player.hand = []
            player.folded = False
            player.acted = False
            player.current_bet = 0

    def deal_cards(self):
        self.deck.shuffle()
        for player in self.players:
            player.hand = [self.deck.deal(), self.deck.deal()]
            player.pref_features = pref(player.hand)
            player.postf_features = []
            player.folded = False
            player.acted = False
            player.current_bet = 0

    def post_blinds(self):
        if len(self.players) < 2:
            logger.warning("Not enough players to post blinds.")
            return
        sb_position = (self.button_position + 1) % len(self.players)
        bb_position = (self.button_position + 2) % len(self.players)
        sb_player = self.players[sb_position]
        bb_player = self.players[bb_position]
        sb_bet = min(self.small_blind, sb_player.stack_size)
        sb_player.stack_size -= sb_bet
        sb_player.current_bet += sb_bet
        self.pot_size += sb_bet
        bb_bet = min(self.big_blind, bb_player.stack_size)
        bb_player.stack_size -= bb_bet
        bb_player.current_bet += bb_bet
        self.pot_size += bb_bet
        self.current_bet = bb_bet
        self.last_raiser = bb_player
        self.bet_history.append({'player': sb_player.player_id, 'action': 'post_small_blind', 'amount': sb_bet})
        self.bet_history.append({'player': bb_player.player_id, 'action': 'post_big_blind', 'amount': bb_bet})

    def deal_community_cards(self):
        if self.round_stage == 1:
            self.community_cards.extend([self.deck.deal(), self.deck.deal(), self.deck.deal()])
        elif self.round_stage == 2:
            self.community_cards.append(self.deck.deal())
        elif self.round_stage == 3:
            self.community_cards.append(self.deck.deal())
            for player in self.active_players:
                full_hand = player.hand + self.community_cards
                player.postf_features = postf(full_hand)

    def next_betting_round(self):
        self.current_bet = 0
        for player in self.players:
            player.current_bet = 0
            player.acted = False
        self.round_stage += 1
        if self.round_stage <= 3:
            self.deal_community_cards()
        self.last_raiser = None
        self.current_raises = 0

    def remove_folded_players(self):
        self.active_players = [p for p in self.active_players if not p.folded]

    def is_betting_round_over(self):
        if not self.active_players:
            return True
        active_bets = [p.current_bet for p in self.active_players]
        all_acted = all(p.acted for p in self.active_players)
        all_bets_equal = len(set(active_bets)) == 1
        return all_acted and all_bets_equal

    def collect_bets(self):
        for player in self.players:
            self.pot_size += player.current_bet
            player.current_bet = 0

    def determine_winners_and_pot(self) -> Tuple[List[Player], float]:
        """
        Determines the winners of the current hand and calculates their share of the pot.

        Returns:
            Tuple containing a list of winning players and the pot share per winner.
        """
        if len(self.active_players) == 1:
            winners = [self.active_players[0]]
        else:
            hands = {
                player: [TreysCard.new(card.to_treys_str()) for card in player.hand + self.community_cards]
                for player in self.active_players
            }
            winners = self.compare_hands(hands)

        if winners:
            pot_share = self.pot_size / len(winners)
        else:
            pot_share = 0

        return winners, pot_share

    def compare_hands(self, hands: Dict[Player, List[str]]) -> List[Player]:
        """
        Compares the hands of active players to determine the winner(s).

        Args:
            hands (dict): A dictionary mapping players to their full hands.

        Returns:
            List[Player]: A list of winning players.
        """
        evaluated_hands = {}
        for player, cards in hands.items():
            try:
                score = self.evaluator.evaluate([], cards)
                evaluated_hands[player] = score
            except Exception as e:
                logger.error(f"Error evaluating hand for Player {player.player_id}: {e}")
                evaluated_hands[player] = self.evaluator.evaluate([], [])  # Assign worst possible hand

        if not evaluated_hands:
            return []

        best_score = min(evaluated_hands.values())
        winners = [player for player, score in evaluated_hands.items() if score == best_score]
        return winners

    def assign_rewards(self, winners: List[Player], pot_share: float, players: List[Player]):
        """
        Assigns rewards to players based on winners and pot share.

        Args:
            winners (List[Player]): List of winning players.
            pot_share (float): The share of the pot each winner receives.
            players (List[Player]): List of all players.
        """
        for player in players:
            if player in winners:
                player.stack_size += pot_share
            # No need to handle losers here as their stack_size is already deducted when they bet/folded
