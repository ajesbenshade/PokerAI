from __future__ import annotations
from typing import List, Tuple, Union, Any, Dict, Optional
import copy
import torch
import numpy as np
import random
import threading
from collections import Counter
import itertools
from treys import Evaluator, Card as TreysCard
from collections import defaultdict

from config import Suit, Action, Config
from datatypes import Player, GameState, Card

import concurrent.futures
import logging
from equity_model import GPUEquityEvaluator

logger = logging.getLogger(__name__)

CARD_RANKS = 13  # 2-A
CARD_SUITS = 4

class CustomBeta:
    def __init__(self, alpha: Union[torch.Tensor, float], beta: Union[torch.Tensor, float]):
        self.alpha = torch.as_tensor(alpha, device=Config.DEVICE, dtype=torch.float32) if not torch.is_tensor(alpha) else alpha
        self.beta = torch.as_tensor(beta, device=Config.DEVICE, dtype=torch.float32) if not torch.is_tensor(beta) else beta
        self.alpha = torch.clamp(self.alpha, min=1.01, max=100.0)
        self.beta = torch.clamp(self.beta, min=1.01, max=100.0)

    def sample(self) -> torch.Tensor:
        try:
            x = torch.distributions.Gamma(self.alpha, 1.0).sample()
            y = torch.distributions.Gamma(self.beta, 1.0).sample()
            return torch.clamp(x / (x + y + 1e-8), min=0.01, max=0.99)
        except Exception:
            mean = self.alpha / (self.alpha + self.beta)
            return torch.clamp(mean, min=0.01, max=0.99)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        safe_value = torch.clamp(value, min=1e-6, max=1.0 - 1e-6)
        log_value = torch.log(safe_value)
        log_one_minus_value = torch.log(1.0 - safe_value)
        log_beta_func = torch.lgamma(self.alpha) + torch.lgamma(self.beta) - torch.lgamma(self.alpha + self.beta)
        log_prob = (self.alpha - 1) * log_value + (self.beta - 1) * log_one_minus_value - log_beta_func
        return torch.nan_to_num(log_prob, nan=-10.0, posinf=-10.0, neginf=-10.0)

evaluator = Evaluator()

class AbstractionCache:
    """RAM cache for hand abstractions with LRU eviction"""
    
    def __init__(self, max_size: int = Config.ABSTRACTION_CACHE_SIZE):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
        self.lock = threading.Lock()
        
    def get(self, key: str) -> Optional[int]:
        """Get cached abstraction bucket"""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
        return None
    
    def put(self, key: str, value: int):
        """Cache abstraction bucket"""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.access_order.remove(key)
            elif len(self.cache) >= self.max_size:
                # Evict least recently used
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]
            
            self.cache[key] = value
            self.access_order.append(key)
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()

# Global abstraction cache
abstraction_cache = AbstractionCache()

def card_to_treys(card: Card) -> int:
    value_map = {2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
    suit_map = {Suit.HEARTS: 'h', Suit.DIAMONDS: 'd', Suit.CLUBS: 'c', Suit.SPADES: 's'}
    
    # Handle different suit representations
    if hasattr(card.suit, 'name'):
        suit_char = suit_map.get(card.suit, 'h')  # Default to hearts if unknown
    else:
        # If suit is stored as int, map it
        suit_int_map = {0: 'h', 1: 'd', 2: 'c', 3: 's'}
        suit_char = suit_int_map.get(card.suit, 'h')
    
    card_str = value_map.get(card.value, '2') + suit_char
    return TreysCard.new(card_str)

def evaluate_hand(hole_cards: List[Card], community_cards: List[Card]) -> int:
    hole_treys = [card_to_treys(c) for c in hole_cards]
    comm_treys = [card_to_treys(c) for c in community_cards]
    if len(hole_treys + comm_treys) < 5:
        return 7462  # Worst rank
    return evaluator.evaluate(comm_treys, hole_treys)

def card_embedding(cards: List[Card]) -> np.ndarray:
    emb = np.zeros((len(cards), CARD_RANKS + CARD_SUITS))
    for i, c in enumerate(cards):
        emb[i, c.value - 2] = 1  # Rank one-hot
        emb[i, CARD_RANKS + c.suit.value] = 1  # Suit one-hot
    return emb.flatten()

def estimate_equity(hole_cards: List[Card], community_cards: List[Card], num_opponents: int) -> float:
    if num_opponents < 1:
        return 0.5
    
    # Convert card tuples to Card objects if necessary
    from datatypes import Card
    from config import Suit
    
    hole_card_objects = []
    for card in hole_cards:
        if isinstance(card, tuple) and len(card) == 2:
            suit = Suit(card[0])
            hole_card_objects.append(Card(card[1], suit))
        elif isinstance(card, Card):
            hole_card_objects.append(card)
        else:
            # Invalid card format, return default equity
            return 0.5
    
    community_card_objects = []
    for card in community_cards:
        if isinstance(card, tuple) and len(card) == 2:
            suit = Suit(card[0])
            community_card_objects.append(Card(card[1], suit))
        elif isinstance(card, Card):
            community_card_objects.append(card)
        else:
            # Invalid card format, return default equity
            return 0.5
    
    # Use converted Card objects for the rest of the function
    hole_cards = hole_card_objects
    community_cards = community_card_objects
    
    # Existing NN equity - use lazy CPU model for multiprocessing safety
    card_emb = np.concatenate([card_embedding(hole_cards), card_embedding(community_cards)])
    card_emb = np.pad(card_emb, (0, max(0, 7 * (CARD_RANKS + CARD_SUITS) - len(card_emb))))[:7 * (CARD_RANKS + CARD_SUITS)]
    emb_tensor = torch.tensor(card_emb, dtype=torch.float32).cpu()  # Keep on CPU for multiprocessing safety
    with torch.no_grad():
        nn_equity = Config.get_equity_model()(emb_tensor).item()
    if len(community_cards) >= 5:
        return nn_equity
    # Quick MC for pre-flop/flop/turn
    deck = [c for c in create_deck() if c not in hole_cards and c not in community_cards]
    wins = ties = 0
    num_sims = max(20, 100 // (num_opponents + 1))  # Reduced for speed
    for _ in range(num_sims):
        random.shuffle(deck)
        needed = 5 - len(community_cards)
        sim_comm = community_cards + deck[:needed]
        my_rank = evaluate_hand(hole_cards, sim_comm)
        # Sample opponent hands without replacement within each trial
        opp_hands = []
        card_idx = needed
        for _ in range(num_opponents):
            opp_hand = [deck[card_idx], deck[card_idx + 1]]
            opp_hands.append(opp_hand)
            card_idx += 2
        opp_ranks = [evaluate_hand(opp_hand, sim_comm) for opp_hand in opp_hands]
        min_opp = min(opp_ranks)
        if my_rank < min_opp:
            wins += 1
        elif my_rank == min_opp:
            ties += 1
    mc_equity = (wins + ties / 2) / num_sims
    return 0.6 * nn_equity + 0.4 * mc_equity  # Weighted blend

# GPU-accelerated equity evaluation
_gpu_equity_evaluator = None

def get_gpu_equity_evaluator():
    """Lazy initialization of GPU equity evaluator"""
    global _gpu_equity_evaluator
    if _gpu_equity_evaluator is None:
        from equity_model import GPUEquityEvaluator
        _gpu_equity_evaluator = GPUEquityEvaluator()
    return _gpu_equity_evaluator

def estimate_equity_gpu(hole_cards: List[Card], community_cards: List[Card], num_opponents: int) -> float:
    """GPU-accelerated equity estimation with caching"""
    evaluator = get_gpu_equity_evaluator()
    result = evaluator.estimate_equity_batch([hole_cards], [community_cards], [num_opponents])
    return result.item()

def estimate_equity_batch_gpu(hole_cards_batch: List[List[Card]], 
                            community_cards_batch: List[List[Card]], 
                            num_opponents_batch: List[int]) -> List[float]:
    """Batch GPU equity estimation"""
    evaluator = get_gpu_equity_evaluator()
    results = evaluator.estimate_equity_batch(hole_cards_batch, community_cards_batch, num_opponents_batch)
    return results.cpu().tolist()

def quick_simulate(hole_cards: List[Card], community_cards: List[Card], num_opponents: int, 
                  pot_size: float, stack: float, call_amount: float, min_raise: float, 
                  action_idx: int, raise_amount: Optional[float] = None) -> float:
    """Quick simulation for action value estimation.
    Returns expected reward for taking the given action.
    """
    # Simplified simulation - in practice this would run Monte Carlo simulations
    base_reward = 0.0

    if action_idx == Action.FOLD.value:
        # Folding loses current contribution
        return -call_amount
    elif action_idx == Action.CALL.value:
        # Calling - simplified equity-based calculation
        equity = estimate_equity(hole_cards, community_cards, num_opponents)
        win_prob = equity
        expected_value = win_prob * pot_size - (1 - win_prob) * call_amount
        return expected_value
    elif action_idx == Action.RAISE.value:
        # Raising - more aggressive, higher variance
        equity = estimate_equity(hole_cards, community_cards, num_opponents)
        # Raising increases pot but also risk
        pot_multiplier = 1.5 if raise_amount else 1.2
        new_pot = pot_size * pot_multiplier
        win_prob = min(equity * 1.1, 0.95)  # Slight boost for aggression
        expected_value = win_prob * new_pot - (1 - win_prob) * (call_amount + (raise_amount or min_raise))
        return expected_value

    return base_reward

def quick_simulate_gpu(hole_cards: List[Card], community_cards: List[Card], num_opponents: int, 
                      pot_size: float, stack: float, call_amount: float, min_raise: float, 
                      action_idx: int, raise_amount: Optional[float] = None) -> float:
    """GPU-accelerated quick simulation"""
    evaluator = get_gpu_equity_evaluator()
    result = evaluator.quick_simulate_batch(
        [hole_cards], [community_cards], [num_opponents], 
        [pot_size], [stack], [call_amount], [min_raise], [action_idx], [raise_amount]
    )
    return result.item()

def quick_simulate_batch_gpu(hole_cards_batch: List[List[Card]], community_cards_batch: List[List[Card]], 
                           num_opponents_batch: List[int], pot_size_batch: List[float], 
                           stack_batch: List[float], call_amount_batch: List[float], 
                           min_raise_batch: List[float], action_idx_batch: List[int],
                           raise_amount_batch: List[float]) -> List[float]:
    """Batch GPU quick simulation"""
    evaluator = get_gpu_equity_evaluator()
    results = evaluator.quick_simulate_batch(
        hole_cards_batch, community_cards_batch, num_opponents_batch,
        pot_size_batch, stack_batch, call_amount_batch, min_raise_batch, 
        action_idx_batch, raise_amount_batch
    )
    return results.cpu().tolist()

def create_deck() -> List[Card]:
    """Create a standard 52-card deck"""
    deck = []
    for suit in [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]:
        for value in range(2, 15):  # 2 through Ace (14)
            deck.append(Card(value, suit))
    return deck

def burn_card(deck: List[Card]) -> Optional[Card]:
    """Burn (remove and return) the top card from the deck"""
    if deck:
        return deck.pop(0)
    return None

def count_active(players: List["Player"]) -> int:
    return sum(1 for p in players if not p.folded and p.stack > 0)

def get_legal_actions(player: "Player", call_amount: int, min_raise_size: int, max_raise_size: int, legal_raise: bool) -> Tuple[np.ndarray, int, bool]:
    legal = np.zeros(Config.ACTION_SIZE)
    can_reopen_raise = False
    if player.stack <= 0:
        legal[Action.FOLD.value] = 1
        return legal, min_raise_size, can_reopen_raise
    legal[Action.FOLD.value] = 1
    if player.stack >= call_amount:
        legal[Action.CALL.value] = 1
        # Only allow raises if it's legally possible to raise
        if legal_raise:
            # Check if player can make a legal raise (meets min_raise_size)
            if player.stack >= call_amount + min_raise_size:
                # Enable all raise bins that are legal
                for i, frac in enumerate(Config.RAISE_BIN_FRACTIONS):
                    if frac == 'all_in':
                        raise_amount = player.stack - call_amount
                    else:
                        raise_amount = int(frac * (call_amount + min_raise_size))  # Pot-relative
                    
                    if raise_amount >= min_raise_size and raise_amount <= player.stack - call_amount:
                        legal[2 + i] = 1  # Raise bins start at index 2
                        can_reopen_raise = True
            # All-in raises smaller than min raise are also legal (don't reopen action)
            elif player.stack > call_amount:
                # Only enable all-in bin if it's the only option
                legal[2 + len(Config.RAISE_BIN_FRACTIONS) - 1] = 1
                can_reopen_raise = False
    return legal, min_raise_size, can_reopen_raise

def get_state(gs: GameState, player: Player, game=None) -> Tuple[str, np.ndarray]:
    """
    Get bucketed infoset key and range vector for consistent RL/CFR representation.
    Returns (infoset_key, state_vector) where state_vector includes range distribution.
    """
    num_opponents = count_active(gs.players) - 1
    
    # Create cache key for abstraction
    hole_str = ''.join([f'{c.value}{c.suit.value}' for c in player.hand]) if player.hand else ''
    comm_str = ''.join([f'{c.value}{c.suit.value}' for c in gs.community_cards]) if gs.community_cards else ''
    cache_key = f"{hole_str}_{comm_str}_{num_opponents}_{len(gs.community_cards)}"
    
    # Check cache first
    bucket = abstraction_cache.get(cache_key)
    if bucket is None:
        # Compute abstraction using learned model if available
        from equity_model import get_hand_abstraction
        hand_ab = get_hand_abstraction()
        bucket = hand_ab.get_bucket(player.hand, gs.community_cards, num_opponents)
        abstraction_cache.put(cache_key, bucket)
    
    # Create bucketed infoset key
    infoset_key = f"{bucket}_{len(gs.community_cards)}_{gs.current_player_idx}"

    # Get range vector using Bayesian conditioning (169-dim distribution over possible hole cards)
    if game and hasattr(game, 'range_conditioning'):
        range_vec = game.range_conditioning.get_range_vector(player.player_id)
    else:
        # Fallback to heuristic range if Bayesian conditioning not available
        range_vec = np.ones(169) / 169  # Uniform distribution

    # Add essential game state features (keep minimal for efficiency)
    max_pot = Config.INITIAL_STACK * len(gs.players)
    call_amount = max(gs.call_amount - player.current_bet, 0)
    pot_odds = call_amount / (gs.pot_size + call_amount + 1e-8) if call_amount > 0 else 0.0

    # Essential features only (avoid redundant point estimates)
    essential_features = np.array([
        gs.pot_size / max_pot,  # Normalized pot size
        player.stack / max_pot, # Normalized stack
        pot_odds,              # Pot odds
        len(gs.community_cards) / 5.0,  # Street progress (0-1)
        gs.current_player_idx / len(gs.players)  # Position
    ])

    # Concatenate range vector with essential features
    state_vec = np.concatenate([range_vec, essential_features])

    return infoset_key, state_vec

def get_state_gpu(gs: GameState, player: Player, game=None, 
                  gpu_evaluator: 'GPUEquityEvaluator' = None, abstraction_cache: 'AbstractionCache' = None) -> Tuple[str, np.ndarray]:
    """
    GPU-accelerated get_state with caching for improved performance.
    Returns bucketed infoset key and range vector for consistent RL/CFR representation.
    """
    num_opponents = count_active(gs.players) - 1
    
    # Create cache key for abstraction
    hole_str = ''.join([f'{c.value}{c.suit.value}' for c in player.hand]) if player.hand else ''
    comm_str = ''.join([f'{c.value}{c.suit.value}' for c in gs.community_cards]) if gs.community_cards else ''
    cache_key = f"{hole_str}_{comm_str}_{num_opponents}_{len(gs.community_cards)}"
    
    # Check cache first
    bucket = abstraction_cache.get(cache_key) if abstraction_cache else None
    if bucket is None:
        # Use GPU-accelerated equity evaluation for bucketing
        if gpu_evaluator and player.hand:
            equity = gpu_evaluator.estimate_equity_batch([player.hand], [gs.community_cards], [num_opponents]).item()
            bucket = min(int(equity * 20), 19)  # 20 buckets based on equity
        else:
            # Fallback to simple bucketing
            bucket = 10  # Default bucket
        
        if abstraction_cache:
            abstraction_cache.put(cache_key, bucket)
    
    # Create bucketed infoset key
    infoset_key = f"{bucket}_{len(gs.community_cards)}_{gs.current_player_idx}"

    # Get range vector using Bayesian conditioning (169-dim distribution over possible hole cards)
    if game and hasattr(game, 'range_conditioning'):
        range_vec = game.range_conditioning.get_range_vector(player.player_id)
    else:
        # Fallback to heuristic range if Bayesian conditioning not available
        range_vec = np.ones(169) / 169  # Uniform distribution

    # Add essential game state features (keep minimal for efficiency)
    max_pot = Config.INITIAL_STACK * len(gs.players)
    call_amount = max(gs.call_amount - player.current_bet, 0)
    pot_odds = call_amount / (gs.pot_size + call_amount + 1e-8) if call_amount > 0 else 0.0

    # Essential features only (avoid redundant point estimates)
    essential_features = np.array([
        gs.pot_size / max_pot,  # Normalized pot size
        player.stack / max_pot, # Normalized stack
        pot_odds,              # Pot odds
        len(gs.community_cards) / 5.0,  # Street progress (0-1)
        gs.current_player_idx / len(gs.players)  # Position
    ])

    # Concatenate range vector with essential features
    state_vec = np.concatenate([range_vec, essential_features])

    return infoset_key, state_vec

def regret_matching_adjustment(probs: np.ndarray, cached_vpip: float) -> np.ndarray:
    if not np.isfinite(probs).all() or (probs < 0).any():
        return np.ones_like(probs) / len(probs)
    adjusted = probs.copy()
    if cached_vpip > 0.5:
        adjusted[Action.RAISE.value] *= 0.8
        adjusted[Action.CALL.value] *= 1.2
    total = adjusted.sum()
    if total <= 0 or not np.isfinite(total):
        return np.ones_like(adjusted) / len(adjusted)
    return adjusted / total

def adjust_action_by_opponent(
    probs: np.ndarray, cached_vpip: float, cached_aggression: float
) -> np.ndarray:
    if not np.isfinite(probs).all() or (probs < 0).any():
        return np.ones_like(probs) / len(probs)
    adjusted = probs.copy()
    if cached_vpip > 0.6:
        adjusted[Action.RAISE.value] *= 0.8
        adjusted[Action.CALL.value] *= 1.2
    elif cached_vpip < 0.2:
        adjusted[Action.RAISE.value] *= 1.2
        adjusted[Action.CALL.value] *= 0.8
    if cached_aggression > 0.5:
        adjusted[Action.CALL.value] *= 1.2
        adjusted[Action.RAISE.value] *= 0.8
    total = adjusted.sum()
    if total <= 0 or not np.isfinite(total):
        return np.ones_like(adjusted) / len(adjusted)
    return adjusted / total

def use_preflop_chart(
    hand: List[Card], position: int, stack_size: float, player_id: int,
    rl_agent=None, training_step: int = 0
) -> Tuple[int, float] | None:
    """Enhanced preflop chart using RL-guided CFR instead of hardcoded ranges"""
    if len(hand) < 2:
        return None

    if not Config.PREFLOP_USE_HYBRID_CFR:
        # Fall back to original hardcoded chart
        return _original_preflop_chart(hand, position, stack_size, player_id)

    # Use RL-guided CFR system
    try:
        # from enhanced_gto import RLGuidedCFRPreflop, create_enhanced_gto_trainer

        # Create or get cached CFR preflop solver
        # if not hasattr(use_preflop_chart, '_cfr_solver'):
        #     game = create_enhanced_gto_trainer(num_players=2)
        #     use_preflop_chart._cfr_solver = RLGuidedCFRPreflop(game, rl_agent)

        # action_idx, raise_amount = use_preflop_chart._cfr_solver.get_preflop_action(
        #     hand, position, stack_size, player_id, training_step
        # )

        # return action_idx, raise_amount
        raise Exception("Enhanced GTO disabled")  # Force fallback

    except Exception as e:
        # print(f"RL-guided CFR failed, falling back to original chart: {e}")
        return _original_preflop_chart(hand, position, stack_size, player_id)

def _original_preflop_chart(
    hand: List[Card], position: int, stack_size: float, player_id: int
) -> Tuple[int, float] | None:
    """Original hardcoded preflop chart (kept for fallback)"""
    if len(hand) < 2:
        return None
    card_values = sorted([c.value for c in hand], reverse=True)
    suited = hand[0].suit == hand[1].suit
    value_map = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T', 9: '9', 8: '8', 7: '7', 6: '6', 5: '5', 4: '4', 3: '3', 2: '2'}
    hand_str = f"{value_map[card_values[0]]}{value_map[card_values[1]]}{'s' if suited else 'o'}"
    num_players = 6
    pos_str = ["UTG", "MP", "CO", "BTN", "SB", "BB"][min(position, 5)]
    preflop_ranges = {
        "UTG": {"AA", "KK", "QQ", "JJ", "AKs", "AKo", "AQs"},
        "MP": {"AA", "KK", "QQ", "JJ", "TT", "AKs", "AQs", "AJs", "KQs", "AKo", "AQo", "ATs"},
        "CO": {"AA", "KK", "QQ", "JJ", "TT", "99", "88", "AKs", "AQs", "AJs", "ATs", "KQs", "KJs", "QJs", "AKo", "AQo", "KQo", "A9s"},
        "BTN": {"AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "AKs", "AQs", "AJs", "ATs", "A9s", "KQs", "KJs", "KTs", "QJs", "QTs", "JTs", "AKo", "AQo", "AJo", "KQo", "A8s"},
        "SB": {"AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "KQs", "KJs", "KTs", "QJs", "QTs", "JTs", "T9s", "AKo", "AQo", "AJo", "KQo", "A7s"},
        "BB": {"AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22", "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s", "KQs", "KJs", "KTs", "K9s", "QJs", "QTs", "Q9s", "JTs", "J9s", "T9s", "98s", "87s", "76s", "65s", "AKo", "AQo", "AJo", "ATo", "KQo", "KJo", "A5o"},
    }
    if hand_str in preflop_ranges.get(pos_str, set()):
        raise_size = Config.BIG_BLIND * (3 if stack_size > 50 * Config.BIG_BLIND else 2.5)
        return Action.RAISE.value, raise_size
    elif random.random() < 0.3:
        return Action.CALL.value, 0.0
    return None

def should_bluff(
    hand_strength: float,
    pot_size: int,
    opponent_vpip: float,
    opponent_aggression: float,
    player_id: int,
    fold_equity: float
) -> bool:
    if hand_strength > 0.5:
        return False
    bluff_chance = 0.2 + (1 - hand_strength) * 0.3 * fold_equity
    if opponent_vpip > 0.5:
        bluff_chance *= 0.8
    if opponent_aggression > 0.6:
        bluff_chance *= 0.7
    if opponent_tracker.get_stat(player_id, "fold_to_c_bet_count") > 0.5:
        bluff_chance *= 1.3
    if pot_size > 100 and hand_strength > 0.3:
        bluff_chance *= 1.2
    return random.random() < bluff_chance

def cfr_regret_adjust(probs: np.ndarray, regrets: np.ndarray) -> np.ndarray:
    """Simple CFR adjustment: normalize probs by cumulative regrets."""
    regrets = np.maximum(regrets, 0)
    total_regret = regrets.sum()
    if total_regret > 0:
        return regrets / total_regret
    return probs / probs.sum()

class OpponentModel:
    def __init__(self) -> None:
        self.stats: Dict[int, Dict[str, float]] = {}
        self.lock = threading.Lock()
        self.stats_template = {
            "vpip_count": 0, "vpip_total": 0,
            "aggression_count": 0, "aggression_total": 0,
            "fold_to_c_bet_count": 0, "fold_to_c_bet_total": 0,
            "pfr_count": 0, "pfr_total": 0,
            "regrets": np.zeros(Config.ACTION_SIZE),
        }

    def update_regret(self, player_id: int, action: int, counterfactual_value: float):
        """Update regrets based on counterfactual value (estimated from critic)."""
        with self.lock:
            if player_id not in self.stats:
                self.stats[player_id] = copy.deepcopy(self.stats_template)
            self.stats[player_id]["regrets"][action] += counterfactual_value

    def update(self, player_id: int, action: int, is_preflop: bool, is_cbet: bool) -> None:
        decay = Config.OPPONENT_DECAY
        with self.lock:
            if player_id not in self.stats:
                self.stats[player_id] = copy.deepcopy(self.stats_template)
            stats = self.stats[player_id]
            if action in [Action.CALL.value, Action.RAISE.value]:
                stats["vpip_count"] = decay * stats["vpip_count"] + 1
            stats["vpip_total"] = decay * stats["vpip_total"] + 1
            if action == Action.RAISE.value:
                stats["aggression_count"] = decay * stats["aggression_count"] + 1
                if is_preflop:
                    stats["pfr_count"] = decay * stats["pfr_count"] + 1
            if action in [Action.CALL.value, Action.RAISE.value]:
                stats["aggression_total"] = decay * stats["aggression_total"] + 1
            if is_cbet:
                if action == Action.FOLD.value:
                    stats["fold_to_c_bet_count"] = decay * stats["fold_to_c_bet_count"] + 1
                stats["fold_to_c_bet_total"] = decay * stats["fold_to_c_bet_total"] + 1

    def get_vpip(self, player_id: int) -> float:
        with self.lock:
            stats = self.stats.get(player_id, self.stats_template)
            return stats["vpip_count"] / stats["vpip_total"] if stats["vpip_total"] > 0 else 0.3

    def get_aggression(self, player_id: int) -> float:
        with self.lock:
            stats = self.stats.get(player_id, self.stats_template)
            return stats["aggression_count"] / stats["aggression_total"] if stats["aggression_total"] > 0 else 0.3

    def get_stat(self, player_id: int, stat_name: str) -> float:
        with self.lock:
            stats = self.stats.get(player_id, self.stats_template)
            total_key = stat_name + "_total"
            return stats.get(stat_name, 0.5) / stats.get(total_key, 1) if stats.get(total_key, 0) > 0 else 0.5

opponent_tracker = OpponentModel()

class PositionAbstraction:
    """Simple position abstraction for poker"""
    
    def get_abstract_position(self, position: int, num_players: int) -> int:
        """Get abstracted position"""
        if num_players <= 2:
            return 0
        elif position == 0:
            return 0  # Button/SB
        elif position == num_players - 1:
            return 2  # BB
        else:
            return 1  # Middle

class RangeConditioning:
    """Simple range conditioning for opponent modeling"""
    
    def __init__(self):
        self.range_beliefs = {}
    
    def get_range_vector(self, player_id: int) -> np.ndarray:
        """Get range vector for player"""
        if player_id not in self.range_beliefs:
            self.range_beliefs[player_id] = np.ones(169) / 169
        return self.range_beliefs[player_id].copy()

class HandAbstraction:
    """Enhanced hand abstraction using equity buckets with position and history"""

    def __init__(self, num_buckets: int = 20):  # Increased for finer GTO
        self.num_buckets = num_buckets
        self.equity_model = Config.get_equity_model()
        self.equity_model.eval()
        self.position_abs = PositionAbstraction()
        self.range_conditioning = RangeConditioning()

    def get_bucket(self, hole_cards, community_cards, num_opponents):
        """Get equity bucket for a hand"""
        if not hole_cards:
            return 0

        equity = estimate_equity(hole_cards, community_cards, num_opponents)
        bucket = min(int(equity * self.num_buckets), self.num_buckets - 1)
        return bucket

    def get_equity_range(self, bucket: int) -> Tuple[float, float]:
        """Get equity range for a bucket"""
        bucket_size = 1.0 / self.num_buckets
        min_equity = bucket * bucket_size
        max_equity = (bucket + 1) * bucket_size
        return min_equity, max_equity

    def get_infoset_bucket(self, gs: GameState, player: Player, num_opponents: int) -> str:
        """Get bucketed infoset key for GTOHoldEm compatibility"""
        if not player.hand:
            bucket = 0
        else:
            equity = estimate_equity(player.hand, gs.community_cards, num_opponents)
            bucket = min(int(equity * self.num_buckets), self.num_buckets - 1)

        position = self.position_abs.get_abstract_position(gs.current_player_idx, len(gs.players))

        # Create history key from betting history
        history_parts = []
        for action_tuple in gs.betting_history[-5:]:  # Last 5 actions
            if len(action_tuple) >= 2:
                action_idx, bet_size = action_tuple[0], action_tuple[1]
                history_parts.append(f"{action_idx}_{int(bet_size)}")

        history_key = "_".join(history_parts) if history_parts else "none"

        return f"bucket_{bucket}_pos_{position}_hist_{history_key}"

    def get_range_vector(self, player_id: int) -> np.ndarray:
        """Get range distribution vector (169-dim for 13x13 hole card combos)"""
        return self.range_conditioning.get_range_vector(player_id)

def get_vram_usage():
    """Get current VRAM usage in GB (allocated/reserved, ROCm-safe)."""
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            return max(alloc, reserved) / (1024 ** 3)
    except Exception:
        pass
    return 0.0

def interpret_discrete_action(discrete_action: int, pot_size: int, call_amount: int, min_raise: int, stack: int) -> Tuple[int, Optional[int]]:
    """
    Convert discrete action index to (action_type, raise_amount) tuple.
    Uses Beta distribution for continuous raise sizing.
    
    Args:
        discrete_action: Index from 0 to ACTION_SIZE-1
        pot_size: Current pot size
        call_amount: Amount needed to call
        min_raise: Minimum raise size
        stack: Player's remaining stack
        
    Returns:
        Tuple of (action_type, raise_amount) where action_type is Action enum value
    """
    if discrete_action == Action.FOLD.value:
        return Action.FOLD.value, None
    elif discrete_action == Action.CALL.value:
        return Action.CALL.value, None
    elif discrete_action >= 2:  # Raise action
        bin_idx = discrete_action - 2
        
        # Use Beta distribution for continuous raise sizing
        # Different bins correspond to different Beta parameters for different raise sizes
        beta_params = [
            (2.0, 5.0),   # Small raise - concentrated around 0.25-0.3
            (2.5, 4.0),   # Medium-small raise
            (3.0, 3.0),   # Medium raise
            (4.0, 2.5),   # Medium-large raise
            (5.0, 2.0),   # Large raise
            (6.0, 1.5),   # Very large raise
            (7.0, 1.2),   # Huge raise
            (8.0, 1.0),   # Massive raise
            (10.0, 0.8),  # All-in sized raise
            (15.0, 0.5),  # All-in
        ]
        
        if bin_idx < len(beta_params):
            alpha, beta = beta_params[bin_idx]
            beta_dist = CustomBeta(alpha, beta)
            
            # Sample raise fraction from Beta distribution
            raise_fraction = beta_dist.sample().item()
            
            # Convert fraction to actual raise amount
            max_raise = stack - call_amount
            if max_raise <= min_raise:
                raise_amount = stack  # All-in if can't make minimum raise
            else:
                # Scale the fraction to the available raise range
                raise_amount = min_raise + raise_fraction * (max_raise - min_raise)
                raise_amount = max(min_raise, min(raise_amount, stack))
            
            return Action.RAISE.value, int(raise_amount)
    
    # Invalid action
    return Action.FOLD.value, None

class ParallelMCTS:
    """Parallel MCTS implementation using CPU cores for enhanced rollouts"""
    
    def __init__(self, num_workers: int = Config.MCTS_CPU_WORKERS):
        self.num_workers = num_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        
    def improve_policies(self, agents, states: List[np.ndarray], 
                        num_rollouts: int = 100) -> Dict[int, Dict]:
        """Improve agent policies using parallel MCTS rollouts"""
        improvements = {}
        
        # Group states by agent for parallel processing
        agent_states = {}
        for i, agent in enumerate(agents):
            agent_states[i] = states
        
        # Perform parallel MCTS for each agent
        futures = []
        for pid, agent_states_list in agent_states.items():
            future = self.executor.submit(
                self._mcts_improve_single_agent,
                agents[pid], agent_states_list, num_rollouts
            )
            futures.append((pid, future))
        
        # Collect results
        for pid, future in futures:
            try:
                improvement = future.result(timeout=30)  # 30 second timeout
                if improvement:
                    improvements[pid] = improvement
                    logger.info(f"MCTS improvement for agent {pid}: {len(improvement.get('policy', {}))} actions, "
                              f"value: {improvement.get('value', 0):.3f}")
            except Exception as e:
                logger.warning(f"MCTS improvement failed for agent {pid}: {e}")
        
        logger.info(f"MCTS improvements completed for {len(improvements)}/{len(agents)} agents")
        return improvements
    
    def _mcts_improve_single_agent(self, agent, states: List[np.ndarray], 
                                  num_rollouts: int) -> Dict:
        """Perform MCTS improvement for a single agent"""
        total_visits = {}
        total_values = {}
        
        for state in states[:5]:  # Limit to 5 states per agent for efficiency
            # Simple MCTS implementation
            root = MCTSNode(state)
            
            for _ in range(num_rollouts):
                # Selection
                node = root
                path = []
                
                while node.children and not node.is_terminal():
                    node = self._select_child(node)
                    path.append(node)
                
                # Expansion
                if not node.is_terminal():
                    node = self._expand_node(node, agent)
                    path.append(node)
                
                # Simulation
                value = self._simulate_rollout(node, agent)
                
                # Backpropagation
                for node in reversed(path):
                    node.visits += 1
                    node.value += value
        
        # Extract improved policy
        if root.children:
            improved_policy = {}
            total_root_visits = sum(child.visits for child in root.children)
            
            for child in root.children:
                action = child.action_taken
                visit_count = child.visits
                improved_policy[action] = visit_count / total_root_visits if total_root_visits > 0 else 0
            
            return {
                'policy': improved_policy,
                'value': root.value / root.visits if root.visits > 0 else 0,
                'total_rollouts': num_rollouts,
                'states_processed': len(states[:5])
            }
        
        return None
    
    def _select_child(self, node: 'MCTSNode') -> 'MCTSNode':
        """Select child using UCB1 formula"""
        best_child = None
        best_score = -float('inf')
        
        for child in node.children:
            if child.visits == 0:
                return child
            
            exploitation = child.value / child.visits
            exploration = 2 * (node.visits ** 0.5) / child.visits
            score = exploitation + exploration
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def _expand_node(self, node: 'MCTSNode', agent) -> 'MCTSNode':
        """Expand node by adding children for untried actions"""
        try:
            # Get legal actions from agent
            legal_actions = agent.get_legal_actions(node.state)
            
            # Create children for untried actions
            for action_idx in range(len(legal_actions)):
                if legal_actions[action_idx] > 0:
                    child = MCTSNode(node.state, parent=node, action_taken=action_idx)
                    node.children.append(child)
            
            return node.children[0] if node.children else node
        except Exception as e:
            logger.warning(f"Failed to expand node: {e}")
            return node
    
    def _simulate_rollout(self, node: 'MCTSNode', agent) -> float:
        """Simulate a rollout from the current node"""
        try:
            # Get action probabilities from agent
            action_probs = agent.get_action_probabilities(node.state)
            
            # Sample action based on probabilities
            if len(action_probs) > 0 and np.sum(action_probs) > 0:
                action_probs = action_probs / np.sum(action_probs)  # Normalize
                chosen_action = np.random.choice(len(action_probs), p=action_probs)
                
                # Simple value estimation based on action type
                if chosen_action == 0:  # Fold
                    return -0.5  # Negative value for folding
                elif chosen_action == 1:  # Call
                    return 0.0   # Neutral value for calling
                else:  # Raise
                    return 0.2   # Positive value for raising (aggression)
            else:
                return 0.0  # Default neutral value
                
        except Exception as e:
            logger.warning(f"Failed to simulate rollout: {e}")
            return 0.0

class MCTSNode:
    """MCTS tree node"""
    
    def __init__(self, state: np.ndarray, parent: 'MCTSNode' = None, action_taken: int = None):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.children = []
        self.visits = 0
        self.value = 0.0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal node"""
        # Simple terminal check - can be enhanced
        return False  # For now, assume no terminal states in MCTS

class OpponentTracker:
    """Enhanced opponent tracking with neural range prediction and adaptive strategies"""
    
    def __init__(self):
        self.opponent_stats = {}  # player_id -> stats dict
        self.range_predictions = {}  # player_id -> predicted range vector
        self.action_history = {}  # player_id -> list of recent actions
        self.hand_history = {}  # player_id -> list of shown hands
        self.lock = threading.Lock()
        self.max_history = 100
        
        # Initialize stats template
        self.stats_template = {
            'vpip': 0.0, 'pfr': 0.0, 'aggression': 0.0,
            'fold_to_cbet': 0.0, 'call_freq': 0.0, 'raise_freq': 0.0,
            'hands_played': 0, 'total_actions': 0,
            'preflop_raises': 0, 'postflop_raises': 0,
            'bluff_frequency': 0.0, 'value_bet_frequency': 0.0
        }
    
    def update_action(self, player_id: int, action: int, street: str, 
                     pot_size: float, call_amount: float, is_cbet: bool = False):
        """Update opponent statistics based on observed action"""
        with self.lock:
            if player_id not in self.opponent_stats:
                self.opponent_stats[player_id] = self.stats_template.copy()
                self.action_history[player_id] = []
            
            stats = self.opponent_stats[player_id]
            stats['total_actions'] += 1
            
            # Update action frequencies
            if action == Action.FOLD.value:
                pass  # Fold doesn't update VPIP directly
            elif action == Action.CALL.value:
                stats['vpip'] = (stats['vpip'] * (stats['hands_played']) + 1) / (stats['hands_played'] + 1)
                stats['call_freq'] = (stats['call_freq'] * (stats['total_actions'] - 1) + 1) / stats['total_actions']
            elif action == Action.RAISE.value:
                stats['vpip'] = (stats['vpip'] * (stats['hands_played']) + 1) / (stats['hands_played'] + 1)
                stats['raise_freq'] = (stats['raise_freq'] * (stats['total_actions'] - 1) + 1) / stats['total_actions']
                stats['aggression'] = (stats['aggression'] * (stats['total_actions'] - 1) + 1) / stats['total_actions']
                
                if street == 'preflop':
                    stats['pfr'] = (stats['pfr'] * (stats['hands_played']) + 1) / (stats['hands_played'] + 1)
                    stats['preflop_raises'] += 1
                else:
                    stats['postflop_raises'] += 1
            
            # Update c-bet statistics
            if is_cbet and action == Action.FOLD.value:
                stats['fold_to_cbet'] = (stats['fold_to_cbet'] * (stats['total_actions'] - 1) + 1) / stats['total_actions']
            
            # Store action in history
            self.action_history[player_id].append({
                'action': action,
                'street': street,
                'pot_size': pot_size,
                'call_amount': call_amount,
                'is_cbet': is_cbet
            })
            
            # Limit history size
            if len(self.action_history[player_id]) > self.max_history:
                self.action_history[player_id].pop(0)
    
    def update_hand_result(self, player_id: int, hand_cards: List[Card], 
                          showed_down: bool, won_hand: bool):
        """Update statistics when a hand is shown down"""
        with self.lock:
            if player_id not in self.opponent_stats:
                self.opponent_stats[player_id] = self.stats_template.copy()
                self.hand_history[player_id] = []
            
            if player_id not in self.hand_history:
                self.hand_history[player_id] = []
            
            stats = self.opponent_stats[player_id]
            stats['hands_played'] += 1
            
            if showed_down:
                self.hand_history[player_id].append({
                    'cards': hand_cards,
                    'won': won_hand
                })
                
                # Limit hand history
                if len(self.hand_history[player_id]) > 20:
                    self.hand_history[player_id].pop(0)
    
    def get_opponent_profile(self, player_id: int) -> Dict[str, float]:
        """Get comprehensive opponent profile"""
        with self.lock:
            if player_id not in self.opponent_stats:
                return self.stats_template.copy()
            
            stats = self.opponent_stats[player_id].copy()
            
            # Calculate derived statistics
            if stats['hands_played'] > 0:
                stats['bluff_frequency'] = self._estimate_bluff_frequency(player_id)
                stats['value_bet_frequency'] = self._estimate_value_bet_frequency(player_id)
            
            return stats
    
    def predict_range(self, player_id: int, game_state) -> np.ndarray:
        """Predict opponent's current range based on game state and history"""
        # Use neural opponent model if available
        try:
            from opponent_model import OpponentRangePredictor
            predictor = OpponentRangePredictor()
            
            # Create features for prediction
            features = self._create_range_features(player_id, game_state)
            
            # Get predicted range
            predicted_range = predictor.predict_range(features)
            self.range_predictions[player_id] = predicted_range
            
            return predicted_range
            
        except (ImportError, AttributeError):
            # Fallback to heuristic range
            return self._heuristic_range_prediction(player_id, game_state)
    
    def get_adaptive_strategy(self, player_id: int, hand_strength: float, 
                            pot_odds: float) -> Dict[str, float]:
        """Get adaptive strategy adjustments based on opponent profile"""
        profile = self.get_opponent_profile(player_id)
        
        adjustments = {
            'fold_threshold': 0.0,
            'call_threshold': 0.0, 
            'raise_frequency': 1.0,
            'bluff_multiplier': 1.0
        }
        
        # Adjust based on opponent tendencies
        if profile['vpip'] > 0.6:  # Loose opponent
            adjustments['fold_threshold'] -= 0.1
            adjustments['raise_frequency'] *= 1.2
        elif profile['vpip'] < 0.2:  # Tight opponent
            adjustments['fold_threshold'] += 0.1
            adjustments['bluff_multiplier'] *= 1.3
        
        if profile['aggression'] > 0.5:  # Aggressive opponent
            adjustments['call_threshold'] -= 0.1
            adjustments['raise_frequency'] *= 0.8
        elif profile['aggression'] < 0.2:  # Passive opponent
            adjustments['raise_frequency'] *= 1.3
        
        if profile['fold_to_cbet'] > 0.6:  # Folds to c-bets frequently
            adjustments['bluff_multiplier'] *= 1.4
        
        return adjustments
    
    def _estimate_bluff_frequency(self, player_id: int) -> float:
        """Estimate how often opponent bluffs based on hand history"""
        if player_id not in self.hand_history:
            return 0.0
        
        hands = self.hand_history[player_id]
        if not hands:
            return 0.0
        
        bluff_hands = 0
        total_hands = len(hands)
        
        for hand in hands:
            if hand['won'] and self._is_bluff_hand(hand['cards']):
                bluff_hands += 1
        
        return bluff_hands / total_hands if total_hands > 0 else 0.0
    
    def _estimate_value_bet_frequency(self, player_id: int) -> float:
        """Estimate how often opponent value bets"""
        if player_id not in self.action_history:
            return 0.0
        
        actions = self.action_history[player_id]
        if not actions:
            return 0.0
        
        value_bets = 0
        total_bets = 0
        
        for action in actions:
            if action['action'] == Action.RAISE.value and action['street'] != 'preflop':
                total_bets += 1
                # Consider it a value bet if pot is large (likely strong hand)
                if action['pot_size'] > 50:
                    value_bets += 1
        
        return value_bets / total_bets if total_bets > 0 else 0.0
    
    def _is_bluff_hand(self, cards: List[Card]) -> bool:
        """Simple heuristic to detect potential bluff hands"""
        if len(cards) < 2:
            return False
        
        # Consider weak holdings as potential bluffs
        values = sorted([c.value for c in cards])
        suited = cards[0].suit == cards[1].suit
        
        # Low unsuited connectors or weak aces
        if not suited and values[1] - values[0] <= 3 and values[0] <= 10:
            return True
        if values[0] == 14 and values[1] <= 8 and not suited:  # Weak aces
            return True
        
        return False
    
    def _create_range_features(self, player_id: int, game_state) -> np.ndarray:
        """Create feature vector for range prediction"""
        profile = self.get_opponent_profile(player_id)
        
        if game_state is None:
            # Default features when no game state is available
            features = np.array([
                profile['vpip'],
                profile['pfr'], 
                profile['aggression'],
                profile['fold_to_cbet'],
                0.0,  # Street (preflop)
                0.0,  # Normalized pot
                0.0,  # Position
            ])
        else:
            features = np.array([
                profile['vpip'],
                profile['pfr'], 
                profile['aggression'],
                profile['fold_to_cbet'],
                len(game_state.community_cards),  # Street
                game_state.pot_size / 100.0,     # Normalized pot
                game_state.current_player_idx / len(game_state.players),  # Position
            ])
        
        return features
    
    def _heuristic_range_prediction(self, player_id: int, game_state: 'GameState') -> np.ndarray:
        """Fallback heuristic range prediction"""
        profile = self.get_opponent_profile(player_id)
        
        # Base uniform range
        range_vec = np.ones(169) / 169
        
        # Adjust based on opponent profile
        if profile['vpip'] > 0.5:  # Loose opponent
            range_vec *= 1.5  # Broader range
        elif profile['vpip'] < 0.3:  # Tight opponent
            range_vec *= 0.7  # Narrower range
        
        if profile['aggression'] > 0.5:  # Aggressive
            # Boost premium hands
            premium_indices = self._get_premium_hand_indices()
            range_vec[premium_indices] *= 1.3
        
        # Normalize
        range_vec = range_vec / range_vec.sum()
        
        return range_vec
    
    def _get_premium_hand_indices(self) -> List[int]:
        """Get indices of premium hands in 169-dim range vector"""
        # This would map hole card combinations to indices
        # Simplified implementation
        return list(range(0, 20))  # First 20 combinations (approximate premium hands)

# Global opponent tracker instance
opponent_tracker = OpponentTracker()
