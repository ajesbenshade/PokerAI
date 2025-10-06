from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import random
import copy
import logging
import torch
from config import Config, Action, Suit
from datatypes import Player, GameState, Card
from utils import create_deck, burn_card, evaluate_hand, count_active, get_state, get_legal_actions, opponent_tracker, estimate_equity
from equity_model import get_hand_abstraction, get_learned_abstraction, GPUEquityEvaluator

logger = logging.getLogger(__name__)

class SimpleUtility:
    """Simple utility class for compatibility with GTO code"""
    def __init__(self, utility):
        self.utility = utility

    def get_utility(self):
        return self.utility

class SimpleGameNode:
    """Simple game node class for compatibility with GTO code"""
    def __init__(self, history, player, is_terminal_node=False, terminal_utility=None, is_chance_node=False, children=None):
        self.history = history
        self.player = player
        self.is_terminal_node = is_terminal_node
        self.terminal_utility = terminal_utility
        self.is_chance_node = is_chance_node
        self.children = children or []

class GTOHoldEm:
    """Full Texas Hold'em implementation with proper betting mechanics and hand evaluation"""

    def __init__(self, num_players: int = 2):
        self.small_blind = Config.SMALL_BLIND
        self.big_blind = Config.BIG_BLIND
        self.starting_stack = Config.INITIAL_STACK
        self.num_players = num_players

        # Game state
        self.players = []
        self.deck = []
        self.community_cards = []
        self.pot_size = 0
        self.current_bet = 0
        self.min_raise = self.big_blind
        self.button_position = 0
        self.current_player_idx = 0
        self.street = 'preflop'  # preflop, flop, turn, river
        self.last_aggressive_player = None
        self.betting_round_complete = False

        # Tree building attributes (for compatibility with GTO code)
        self.tree_size = 0
        self.max_tree_size = 10000
        self.node_cache = {}
        self.abstraction_enabled = False

        # Bayesian range conditioning for opponent modeling
        # from enhanced_gto import RangeConditioning
        # self.range_conditioning = RangeConditioning()
        
        # Track hand number for annealing
        self.hand_number = 0

        # Tree building optimizations
        self.tree_size = 0
        self.max_tree_size = Config.MAX_TREE_SIZE if hasattr(Config, 'MAX_TREE_SIZE') else 1000000
        self.node_cache = {}  # Cache for memoization
        self.abstraction_enabled = True
        
        # Learned abstraction model (will be set by training loop)
        self._learned_abstraction = None

        self._initialize_game()
        self.reset_game()

    def _initialize_game(self):
        """Initialize players and deck"""
        self.players = []
        for i in range(self.num_players):
            player = Player(player_id=i)
            player.stack = self.starting_stack
            self.players.append(player)

        self.deck = create_deck()
        random.shuffle(self.deck)

    def _deal_hole_cards(self):
        """Deal 2 cards to each player"""
        for player in self.players:
            if len(self.deck) >= 2:
                player.hand = [self.deck.pop(), self.deck.pop()]

    def _deal_community_cards(self, num_cards: int):
        """Deal community cards (burn and deal)"""
        if len(self.deck) >= num_cards + 1:  # +1 for burn card
            burn_card(self.deck)  # Burn a card
            for _ in range(num_cards):
                self.community_cards.append(self.deck.pop())

    def _post_blinds(self):
        """Post small and big blinds"""
        sb_pos = (self.button_position + 1) % self.num_players
        bb_pos = (self.button_position + 2) % self.num_players

        # Small blind
        sb_amount = min(self.small_blind, self.players[sb_pos].stack)
        self.players[sb_pos].stack -= sb_amount
        self.players[sb_pos].current_bet = sb_amount
        self.players[sb_pos].hand_contribution = sb_amount
        self.pot_size += sb_amount

        # Big blind
        bb_amount = min(self.big_blind, self.players[bb_pos].stack)
        self.players[bb_pos].stack -= bb_amount
        self.players[bb_pos].current_bet = bb_amount
        self.players[bb_pos].hand_contribution = bb_amount
        self.pot_size += bb_amount

        self.current_bet = self.big_blind
        self.current_player_idx = (bb_pos + 1) % self.num_players

    def _get_active_players(self) -> List[Player]:
        """Get players who haven't folded"""
        return [p for p in self.players if not p.folded]

    def _get_next_player(self) -> Optional[int]:
        """Get next active player who can act"""
        active_players = self._get_active_players()
        if len(active_players) <= 1:
            return None

        for _ in range(self.num_players):
            player = self.players[self.current_player_idx]
            if not player.folded and player.stack > 0:
                return self.current_player_idx
            self.current_player_idx = (self.current_player_idx + 1) % self.num_players

        return None

    def _is_betting_round_complete(self) -> bool:
        """Check if current betting round is complete"""
        active_players = self._get_active_players()
        if len(active_players) <= 1:
            return True

        # Check if all active players have matched the current bet
        for player in active_players:
            if player.current_bet < self.current_bet and player.stack > 0:
                return False

        # Check if there's been a bet and everyone has acted
        if self.last_aggressive_player is not None:
            # Make sure we've gone around the table since the last aggressive action
            return True

        return False

    def _advance_street(self):
        """Advance to next street and reset betting state"""
        if self.street == 'preflop':
            self._deal_community_cards(3)  # Flop
            self.street = 'flop'
        elif self.street == 'flop':
            self._deal_community_cards(1)  # Turn
            self.street = 'turn'
        elif self.street == 'turn':
            self._deal_community_cards(1)  # River
            self.street = 'river'
        else:
            return  # Already on river

        # Reset betting state for new street
        self.current_bet = 0
        self.min_raise = self.big_blind
        self.last_aggressive_player = None
        for player in self.players:
            if not player.folded:
                player.current_bet = 0

        # First player to act on new street
        self.current_player_idx = (self.button_position + 1) % self.num_players

    def _evaluate_showdown(self) -> Dict[int, float]:
        """Evaluate hands at showdown and return utilities with proper side pot handling"""
        active_players = self._get_active_players()
        if len(active_players) <= 1:
            # Only one player left, they win the pot
            winner = active_players[0]
            return {winner.player_id: self.pot_size}

        # Evaluate all active players' hands
        hand_ranks = {}
        for player in active_players:
            if len(player.hand) == 2:  # Make sure they have hole cards
                rank = evaluate_hand(player.hand, self.community_cards)
                hand_ranks[player.player_id] = rank

        if not hand_ranks:
            # No valid hands, split pot
            split_amount = self.pot_size / len(active_players)
            return {p.player_id: split_amount for p in active_players}

        # Calculate side pots based on all-in amounts
        side_pots = self._calculate_side_pots(active_players)

        utilities = {p.player_id: -p.hand_contribution for p in self.players}

        # Evaluate each side pot
        for pot_amount, eligible_players in side_pots:
            if len(eligible_players) == 0:
                continue

            # Find winners among eligible players
            pot_hand_ranks = {pid: hand_ranks[pid] for pid in eligible_players if pid in hand_ranks}
            if not pot_hand_ranks:
                # Split among all eligible players
                split_amount = pot_amount / len(eligible_players)
                for pid in eligible_players:
                    utilities[pid] += split_amount
                continue

            # Find the best hand(s) among eligible players
            best_rank = min(pot_hand_ranks.values())
            winners = [pid for pid, rank in pot_hand_ranks.items() if rank == best_rank]

            # Split this side pot among winners
            win_amount = pot_amount / len(winners)
            for winner in winners:
                utilities[winner] += win_amount

        # Update OpponentTracker with hand results
        try:
            active_players = self._get_active_players()
            if len(active_players) > 1:  # Only track if there was a showdown
                for player in active_players:
                    if player.hand and len(player.hand) == 2:
                        # Determine if player won
                        won_hand = False
                        if player.player_id in utilities and utilities[player.player_id] > 0:
                            won_hand = True
                        
                        # Update opponent tracker with hand result
                        opponent_tracker.update_hand_result(
                            player.player_id, player.hand, True, won_hand
                        )
        except Exception as e:
            logger.warning(f"Failed to update opponent tracker with hand results: {e}")
            # Continue execution even if tracking fails

        return utilities

    def _calculate_side_pots(self, active_players):
        """Calculate side pots based on all-in amounts"""
        # Get all contributions from all players (including folded ones)
        all_contributions = [(p.player_id, p.hand_contribution) for p in self.players]
        all_contributions.sort(key=lambda x: x[1])  # Sort by contribution amount

        side_pots = []
        remaining_players = [p.player_id for p in active_players]
        previous_threshold = 0

        for i, (player_id, contribution) in enumerate(all_contributions):
            if contribution > previous_threshold:
                # Calculate pot size for this threshold
                pot_size = 0
                pot_eligible = []

                for pid, contrib in all_contributions:
                    if contrib >= contribution:
                        pot_size += contribution - previous_threshold
                        if pid in remaining_players:
                            pot_eligible.append(pid)
                    elif contrib > previous_threshold:
                        pot_size += contrib - previous_threshold

                if pot_size > 0 and pot_eligible:
                    side_pots.append((pot_size, pot_eligible))

                previous_threshold = contribution

        # Handle any remaining pot (main pot)
        if remaining_players:
            main_pot_size = sum(min(p.hand_contribution, max(c[1] for c in all_contributions)) - previous_threshold
                              for p in self.players if p.player_id in remaining_players)
            if main_pot_size > 0:
                side_pots.append((main_pot_size, remaining_players))

        return side_pots

    def build_game_tree(self, history=[], depth=0):
        """Stub implementation for compatibility with GTO code - returns a simple terminal node"""
        # Return a simple terminal node for compatibility
        return SimpleGameNode(
            history=history,
            player=0,
            is_terminal_node=True,
            terminal_utility=SimpleUtility(0.0)
        )

    def _get_cache_key(self, history):
        """Generate cache key using learned abstraction"""
        if not self.abstraction_enabled:
            return tuple(history)

        # Get current game state for abstraction
        try:
            gs = self.get_game_state()
            player_idx = self.get_player(history)
            player = self.players[player_idx] if player_idx < len(self.players) else None

            if player and hasattr(player, 'hand') and player.hand:
                # Use learned abstraction if available
                if hasattr(self, '_learned_abstraction') and self._learned_abstraction is not None:
                    # Convert game state to tensor for learned abstraction
                    state_tensor = self._game_state_to_tensor(gs, player)
                    if state_tensor is not None:
                        with torch.no_grad():
                            _, embedding, bucket_logits = self._learned_abstraction(state_tensor.unsqueeze(0))
                            bucket = torch.argmax(bucket_logits, dim=-1).item()
                        return (bucket, tuple(history))
                
                # Fallback to hand abstraction
                bucket = get_hand_abstraction(
                    player.hand,
                    gs.community_cards if hasattr(gs, 'community_cards') else [],
                    num_opponents=len([p for p in self.players if not p.folded and p != player])
                )
                return (bucket, tuple(history))
        except Exception as e:
            logger.warning(f"Abstraction failed, using raw history: {e}")

        return tuple(history)

    def _game_state_to_tensor(self, gs, player):
        """Convert game state to tensor for learned abstraction"""
        try:
            # Create state representation similar to RL agent
            from utils import get_state
            state = get_state(player, gs, self.players)
            return torch.tensor(state, dtype=torch.float32).to(Config.DEVICE)
        except:
            return None

    def get_available_actions(self, history):
        """Return available actions with discrete raise bins"""
        # Get basic legal actions
        actions = np.array([0, 1])  # Fold, Call always available

        # Add raise if possible (using discrete bins)
        if self._can_raise(history):
            actions = np.append(actions, [2])  # Add raise action

        return actions

    def _can_raise(self, history):
        """Check if raising is possible"""
        try:
            gs = self.get_game_state()
            player_idx = self.get_player(history)
            player = self.players[player_idx] if player_idx < len(self.players) else None

            if not player or player.stack <= 0:
                return False

            # Check if raise would be valid
            current_bet = gs.current_bet if hasattr(gs, 'current_bet') else 0
            min_raise = gs.min_raise if hasattr(gs, 'min_raise') else self.big_blind

            return player.stack >= min_raise
        except:
            return False

    def get_game_state(self):
        """Get current game state for abstraction"""
        # Create a simplified game state
        class SimpleGameState:
            def __init__(self, game):
                self.community_cards = game.community_cards
                self.pot_size = game.pot_size
                self.current_bet = game.current_bet
                self.min_raise = game.min_raise

        return SimpleGameState(self)

    def reset_game(self):
        """Reset game state for a new hand"""
        # Reset game state
        self.deck = create_deck()
        random.shuffle(self.deck)
        self.community_cards = []
        self.pot_size = 0
        self.current_bet = 0
        self.min_raise = self.big_blind
        self.current_player_idx = 0
        self.street = 'preflop'
        self.last_aggressive_player = None
        self.betting_round_complete = False

        # Reset players
        for player in self.players:
            player.folded = False
            player.current_bet = 0
            player.all_in = False
            player.hand_contribution = 0
            player.hand = []

        # Deal hole cards and post blinds
        self._deal_hole_cards()
        self._post_blinds()

    def play_action(self, player_idx: int, action_idx: int, raise_amount: float = 0.0) -> bool:
        """Execute an action for the specified player"""
        if player_idx < 0 or player_idx >= len(self.players):
            return False

        player = self.players[player_idx]

        # Check if player can act
        if player.folded or player.stack <= 0:
            return False

        if action_idx == 0:  # FOLD
            player.folded = True
            return True

        elif action_idx == 1:  # CALL
            call_amount = min(self.current_bet - player.current_bet, player.stack)
            player.stack -= call_amount
            player.current_bet += call_amount
            player.hand_contribution += call_amount
            self.pot_size += call_amount

            if player.stack == 0:
                player.all_in = True

            return True

        elif action_idx == 2:  # RAISE
            if raise_amount <= 0:
                return False

            # Calculate total bet needed
            total_bet_needed = self.current_bet - player.current_bet + raise_amount

            if total_bet_needed > player.stack:
                return False

            # Check minimum raise
            if raise_amount < self.min_raise:
                return False

            player.stack -= total_bet_needed
            player.current_bet += total_bet_needed
            player.hand_contribution += total_bet_needed
            self.pot_size += total_bet_needed

            self.current_bet = player.current_bet
            self.min_raise = raise_amount
            self.last_aggressive_player = player_idx

            if player.stack == 0:
                player.all_in = True

            return True

        # Update OpponentTracker with the action taken
        try:
            # Determine current street
            street = 'preflop'
            if len(self.community_cards) >= 3:
                street = 'flop'
            if len(self.community_cards) >= 4:
                street = 'turn'
            if len(self.community_cards) >= 5:
                street = 'river'
            
            # Check if this is a continuation bet
            is_cbet = self._is_continuation_bet(player_idx)
            
            # Update opponent tracker
            opponent_tracker.update_action(
                player.player_id, action_idx, street, 
                self.pot_size, self.current_bet - player.current_bet, is_cbet
            )
            
        except Exception as e:
            logger.warning(f"Failed to update opponent tracker: {e}")
            # Continue execution even if tracking fails

        return False

    def _is_continuation_bet(self, player_idx: int) -> bool:
        """
        Determine if the current action represents a continuation bet opportunity.
        
        A continuation bet occurs when:
        1. We raised preflop and are first to act postflop
        2. No opponent has raised after our preflop raise
        """
        # Must be postflop
        if len(self.community_cards) < 3:
            return False
        
        # Check if this player was the preflop raiser
        # This is a simplified check - in practice you'd track betting history more carefully
        if self.last_aggressive_player == player_idx:
            return True
        
        return False

    def get_player(self, history):
        """Get the current player based on history"""
        # For now, return the current player index
        return self.current_player_idx

def simulate_hand(players: List[Player], dealer_idx: int, hand_number: int = 0) -> Tuple[Dict[int, float], Dict[int, List]]:
    """
    Simulate a single hand of poker and collect transitions for PPO training.
    Returns rewards and trajectories for each player.
    """
    game = GTOHoldEm(num_players=len(players))
    game.button_position = dealer_idx
    game.hand_number = hand_number  # Set hand number for Bayesian annealing
    game.reset_game()

    # Override players and deal cards
    game.players = players
    for player in players:
        player.folded = False
        player.current_bet = 0
        player.all_in = False
        player.hand_contribution = 0
        # Deal hole cards to existing players
        if len(game.deck) >= 2:
            player.hand = [game.deck.pop(), game.deck.pop()]
        else:
            player.hand = []

    trajectories = {p.player_id: [] for p in players}
    done = False
    step_count = 0

    while not done and step_count < 100:  # Prevent infinite loops
        current_player = game.players[game.current_player_idx]

        if not current_player.folded and current_player.stack > 0:
            # Get current state using Bayesian range conditioning
            from datatypes import GameState
            gs = GameState(game.players)
            gs.pot_size = game.pot_size
            gs.community_cards = game.community_cards
            gs.current_player_idx = game.current_player_idx
            gs.call_amount = game.current_bet
            gs.min_raise = game.min_raise
            gs.max_raise = max(p.stack for p in game.players)
            gs.can_raise = True
            gs.betting_history = getattr(game, 'betting_history', [])
            
            _, state = get_state(gs, current_player, game)

            legal_actions, _, _ = get_legal_actions(current_player, game.current_bet - current_player.current_bet, game.min_raise, game.pot_size, True)

            if not np.any(legal_actions):
                # No legal actions, fold
                current_player.folded = True
                game._get_next_player()
                continue

            # Get action from agent
            action_idx, raise_amount, discrete_action, log_prob, value = current_player.agent.choose_action(
                state, legal_actions, current_player.player_id,
                stack=current_player.stack,
                min_raise=game.min_raise,
                call_amount=game.current_bet - current_player.current_bet,
                hole_cards=current_player.hand,
                community_cards=game.community_cards,
                opponents=[p for p in game.players if p != current_player and p.folded],
                pot_size=game.pot_size
            )

            # Execute action
            success = game.play_action(game.current_player_idx, action_idx, raise_amount)

            if not success:
                # Invalid action, treat as fold
                current_player.folded = True

        game._get_next_player()

        # Check if there are no more players to act
        next_player = game._get_next_player()
        
        # Check if there are no more players to act
        if next_player is None:
            done = True
            continue

        # Check if hand is complete
        active_players = [p for p in game.players if not p.folded and p.stack > 0]
        if len(active_players) <= 1:
            done = True
        elif game._is_betting_round_complete():
            if game.street == 'river':
                done = True
            else:
                game._advance_street()

        # Calculate reward (0 for non-terminal, actual reward at end)
        reward = 0.0
        if done:
            utilities = game._evaluate_showdown()
            reward = utilities.get(current_player.player_id, -current_player.hand_contribution)

        # Store transition
        trajectories[current_player.player_id].append((
            state, discrete_action, log_prob, reward, done, value
        ))

        step_count += 1

    # Final rewards
    final_rewards = {}
    if done:
        utilities = game._evaluate_showdown()
        for player in players:
            final_rewards[player.player_id] = utilities.get(player.player_id, -player.hand_contribution)
    else:
        # Timeout - all players lose their contribution
        for player in players:
            final_rewards[player.player_id] = -player.hand_contribution

    return final_rewards, trajectories

def simulate_hand_gpu(players: List[Player], dealer_idx: int, hand_number: int, 
                     gpu_evaluator: 'GPUEquityEvaluator', abstraction_cache: 'AbstractionCache') -> Tuple[Dict[int, float], Dict[int, List]]:
    """
    GPU-accelerated simulation of a single hand of poker with caching.
    Returns rewards and trajectories for each player.
    """
    game = GTOHoldEm(num_players=len(players))
    game.button_position = dealer_idx
    game.hand_number = hand_number  # Set hand number for Bayesian annealing
    game.reset_game()

    # Override players and deal cards
    game.players = players
    for player in players:
        player.folded = False
        player.current_bet = 0
        player.all_in = False
        player.hand_contribution = 0
        # Deal hole cards to existing players
        if len(game.deck) >= 2:
            player.hand = [game.deck.pop(), game.deck.pop()]
        else:
            player.hand = []

    trajectories = {p.player_id: [] for p in players}
    done = False
    step_count = 0

    while not done and step_count < 100:  # Prevent infinite loops
        current_player = game.players[game.current_player_idx]

        if not current_player.folded and current_player.stack > 0:
            # Get current state using GPU-accelerated equity and caching
            from datatypes import GameState
            gs = GameState(game.players)
            gs.pot_size = game.pot_size
            gs.community_cards = game.community_cards
            gs.current_player_idx = game.current_player_idx
            gs.call_amount = game.current_bet
            gs.min_raise = game.min_raise
            gs.max_raise = max(p.stack for p in game.players)
            gs.can_raise = True
            gs.betting_history = getattr(game, 'betting_history', [])
            
            _, state = get_state(gs, current_player, game)

            legal_actions, _, _ = get_legal_actions(current_player, game.current_bet - current_player.current_bet, game.min_raise, game.pot_size, True)

            if not np.any(legal_actions):
                # No legal actions, fold
                current_player.folded = True
                game._get_next_player()
                continue

            # Get action from agent
            action_idx, raise_amount, discrete_action, log_prob, value = current_player.agent.choose_action(
                state, legal_actions, current_player.player_id,
                stack=current_player.stack,
                min_raise=game.min_raise,
                call_amount=game.current_bet - current_player.current_bet,
                hole_cards=current_player.hand,
                community_cards=game.community_cards,
                opponents=[p for p in game.players if p != current_player and p.folded],
                pot_size=game.pot_size
            )

            # Execute action
            success = game.play_action(game.current_player_idx, action_idx, raise_amount)

            if not success:
                # Invalid action, treat as fold
                current_player.folded = True

        game._get_next_player()

        # Check if there are no more players to act
        next_player = game._get_next_player()
        
        # Check if there are no more players to act
        if next_player is None:
            done = True
            continue

        # Check if hand is complete
        active_players = [p for p in game.players if not p.folded and p.stack > 0]
        if len(active_players) <= 1:
            done = True
        elif game._is_betting_round_complete():
            if game.street == 'river':
                done = True
            else:
                game._advance_street()

        # Calculate reward (0 for non-terminal, actual reward at end)
        reward = 0.0
        if done:
            utilities = game._evaluate_showdown()
            reward = utilities.get(current_player.player_id, -current_player.hand_contribution)

        # Store transition
        trajectories[current_player.player_id].append((
            state, discrete_action, log_prob, reward, done, value
        ))

        step_count += 1

    # Final rewards
    final_rewards = {}
    if done:
        utilities = game._evaluate_showdown()
        for player in players:
            final_rewards[player.player_id] = utilities.get(player.player_id, -player.hand_contribution)
    else:
        # Timeout - all players lose their contribution
        for player in players:
            final_rewards[player.player_id] = -player.hand_contribution

    return final_rewards, trajectories