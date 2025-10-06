"""
GTO Evaluation Module for PokerAI
Implements exploitability evaluation using best response calculations
"""

from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import time

from config import Config
from game import GTOHoldEm
from utils import estimate_equity, evaluate_hand
from datatypes import Player, GameState, Card

device = torch.device(Config.DEVICE)

class StrategyProfile:
    """Represents a strategy profile from CFR training"""

    def __init__(self, infosets: Dict[str, Dict[int, float]]):
        self.infosets = infosets
        self.expected_value_cache = {}

    def get_action_prob(self, infoset_key: str, action: int) -> float:
        """Get probability of taking action in infoset"""
        if infoset_key not in self.infosets:
            return 1.0 / len(self.infosets.get(infoset_key, {0: 1.0}))  # Uniform if unknown

        actions = self.infosets[infoset_key]
        return actions.get(action, 0.0)

    def get_mixed_prob(self, infoset_key: str, action: int) -> float:
        """Get mixed probability of taking action in infoset"""
        if infoset_key not in self.infosets:
            # Uniform fallback if infoset unknown
            return 1.0 / len(self.infosets.get(infoset_key, {0: 1.0}))
        
        actions = self.infosets[infoset_key]
        total = sum(actions.values())
        if total > 0:
            return actions.get(action, 0.0) / total
        else:
            # Uniform if no actions recorded
            return 1.0 / len(actions)

    def expected_value(self) -> float:
        """Compute expected value of the strategy profile using tree traversal"""
        if 'expected_value' in self.expected_value_cache:
            return self.expected_value_cache['expected_value']

        # Build game tree and compute EV
        game = GTOHoldEm()
        root = game.build_game_tree()
        
        memo_cache = {}
        max_cache_size = Config.EXPLOITABILITY_MEMO_CACHE_SIZE

        def compute_ev(node, reach_prob: float = 1.0, depth: int = 0) -> float:
            """Compute expected value for the strategy profile"""
            # Check depth limit
            if depth > Config.EXPLOITABILITY_MAX_TREE_DEPTH:
                return 0.0
            
            cache_key = (id(node), reach_prob)
            if cache_key in memo_cache:
                return memo_cache[cache_key]

            if node.is_terminal_node:
                utility = node.terminal_utility.get_utility()
                if isinstance(utility, dict):
                    utility = utility.get(0, 0)  # Player 0's utility
                result = utility * reach_prob
                if len(memo_cache) < max_cache_size:
                    memo_cache[cache_key] = result
                return result

            if node.is_chance_node:
                ev = 0.0
                if hasattr(node, 'next_nodes') and node.next_nodes:
                    if hasattr(node, 'chance_probs') and node.chance_probs:
                        for child, prob in zip(node.next_nodes, node.chance_probs):
                            ev += prob * compute_ev(child, reach_prob, depth + 1)
                    else:
                        # Uniform chance if no probs specified
                        prob = 1.0 / len(node.next_nodes)
                        for child in node.next_nodes:
                            ev += prob * compute_ev(child, reach_prob, depth + 1)
                result = ev
                if len(memo_cache) < max_cache_size:
                    memo_cache[cache_key] = result
                return result

            # Player node - use strategy profile
            infoset_key = game.get_infoset_key(node.history)
            ev = 0.0
            
            if hasattr(node, 'next_nodes') and node.next_nodes:
                for action_idx, child in enumerate(node.next_nodes):
                    action_prob = self.get_mixed_prob(infoset_key, action_idx)
                    ev += action_prob * compute_ev(child, reach_prob * action_prob, depth + 1)

            result = ev
            if len(memo_cache) < max_cache_size:
                memo_cache[cache_key] = result
            return result

        ev = compute_ev(root, 1.0)
        self.expected_value_cache['expected_value'] = ev
        return ev

    def get_cf_reach(self, infoset_key: str, action: int, history: List) -> float:
        """
        Compute counterfactual reach probability for an action.
        CF reach is the probability of reaching this point ignoring the player's own actions.
        This is used for exact best response calculations.
        """
        cf_prob = 1.0

        # Multiply probabilities of all opponent actions in history
        for hist_action in history:
            if len(hist_action) >= 3:  # (player, action, infoset_key)
                hist_player, hist_action_idx, hist_infoset = hist_action
                if hist_player != 0:  # Opponent action
                    opp_prob = self.get_mixed_prob(hist_infoset, hist_action_idx)
                    cf_prob *= opp_prob

        # Multiply by the probability of the current action
        current_prob = self.get_mixed_prob(infoset_key, action)
        cf_prob *= current_prob

        return cf_prob

class BestResponseCalculator:
    """Calculates best response strategies and exploitability"""

    def __init__(self, game: GTOHoldEm):
        self.game = game
        self.device = device

    def compute_best_response(self, opponent_profile: StrategyProfile) -> Tuple[Dict[str, Dict[int, float]], float]:
        """
        Compute best response strategy against opponent profile
        Returns: (best_response_strategy, best_response_ev)
        """
        # Skip GPU implementation for now due to recursion issues
        # if Config.GPU_EXPLOITABILITY and torch.cuda.is_available():
        #     try:
        #         gpu_evaluator = GPUExploitabilityEvaluator(self.game)
        #         root = self.game.build_game_tree()
        #         return gpu_evaluator.gpu_tree_traversal(root, opponent_profile.infosets)
        #     except Exception as e:
        #         print(f"GPU BR failed, falling back to CPU: {e}")

        # Use CPU implementation
        br_strategy = {}
        br_ev = 0.0

        # Build game tree and compute BR bottom-up
        root = self.game.build_game_tree()

        # Memoization cache for RAM efficiency
        memo_cache = {}
        max_cache_size = Config.EXPLOITABILITY_MEMO_CACHE_SIZE

        def traverse_tree(node, player_reach: float = 1.0, opp_reach: float = 1.0,
                         cf_reach: float = 1.0, opponent_profile=None, depth: int = 0) -> float:
            """Recursive tree traversal for BR calculation with mixed strategies"""
            # Check depth limit
            if depth > Config.EXPLOITABILITY_MAX_TREE_DEPTH:
                return 0.0  # Approximate with 0 for deep trees
            
            # Create cache key
            cache_key = (id(node), player_reach, opp_reach, cf_reach)
            if cache_key in memo_cache:
                return memo_cache[cache_key]

            if node.is_terminal_node:
                utility = node.terminal_utility.get_utility()
                if isinstance(utility, dict):
                    utility = utility.get(0, 0)  # Player 0's utility
                else:
                    utility = utility
                # Use counterfactual reach for exact BR calculation
                result = utility * player_reach * cf_reach
                if len(memo_cache) < max_cache_size:
                    memo_cache[cache_key] = result
                return result

            if node.is_chance_node:
                ev = 0.0
                if hasattr(node, 'next_nodes') and node.next_nodes:
                    if hasattr(node, 'chance_probs') and node.chance_probs:
                        for child, prob in zip(node.next_nodes, node.chance_probs):
                            ev += prob * traverse_tree(child, player_reach, opp_reach, cf_reach, opponent_profile, depth + 1)
                    else:
                        # Uniform chance if no probs specified
                        prob = 1.0 / len(node.next_nodes)
                        for child in node.next_nodes:
                            ev += prob * traverse_tree(child, player_reach, opp_reach, cf_reach, opponent_profile, depth + 1)
                result = ev
                if len(memo_cache) < max_cache_size:
                    memo_cache[cache_key] = result
                return result

            # Player node
            infoset_key = self.game.get_infoset_key(node.history)
            player = node.player

            if not hasattr(node, 'next_nodes') or not node.next_nodes:
                result = 0.0
                if len(memo_cache) < max_cache_size:
                    memo_cache[cache_key] = result
                return result

            if player == 0:  # BR player: Maximize over actions
                best_ev = -float('inf')
                best_action = None
                action_evs = {}
                
                for action_idx, child in enumerate(node.next_nodes):
                    action_ev = traverse_tree(child, player_reach, opp_reach, cf_reach, opponent_profile, depth + 1)
                    action_evs[action_idx] = action_ev
                    if action_ev > best_ev:
                        best_ev = action_ev
                        best_action = action_idx

                # Store BR strategy (deterministic for best action)
                if hasattr(node, 'available_actions') and node.available_actions is not None:
                    # Handle numpy arrays
                    actions_list = node.available_actions.tolist() if hasattr(node.available_actions, 'tolist') else list(node.available_actions)
                    br_strategy[infoset_key] = {
                        action: 1.0 if idx == best_action else 0.0
                        for idx, action in enumerate(actions_list)
                    }
                else:
                    br_strategy[infoset_key] = {
                        idx: 1.0 if idx == best_action else 0.0
                        for idx in range(len(node.next_nodes))
                    }

                result = best_ev
                if len(memo_cache) < max_cache_size:
                    memo_cache[cache_key] = result
                return result

            else:  # Opponent: Average over mixed strategy with counterfactual reaches
                ev = 0.0
                total_prob = 0.0
                
                for action_idx, child in enumerate(node.next_nodes):
                    action_prob = opponent_profile.get_mixed_prob(infoset_key, action_idx)
                    if action_prob > 0:
                        # Compute counterfactual reach for this action path
                        child_cf = opponent_profile.get_cf_reach(infoset_key, action_idx, node.history)
                        child_ev = traverse_tree(child, player_reach, opp_reach * action_prob,
                                               cf_reach * child_cf, opponent_profile, depth + 1)
                        ev += action_prob * child_ev
                        total_prob += action_prob

                # Normalize if probabilities don't sum to 1
                if total_prob > 0:
                    ev /= total_prob

                result = ev
                if len(memo_cache) < max_cache_size:
                    memo_cache[cache_key] = result
                return result

        # Start traversal from root
        # Disable parallel processing for now to ensure br_strategy is populated
        # if Config.EXPLOITABILITY_CPU_PARALLEL and len(memo_cache) < max_cache_size // 2:
        #     br_ev = self._parallel_traverse_tree(root, opponent_profile)
        # else:
        br_ev = traverse_tree(root, 1.0, 1.0, 1.0, opponent_profile)

        return br_strategy, br_ev

    def compute_exploitability(self, strategy_profile: StrategyProfile) -> float:
        """
        Compute exploitability of a strategy profile
        Exploitability = max(BR_EV - Strategy_EV, 0)
        """
        # Compute BR against the strategy
        br_strategy, br_ev = self.compute_best_response(strategy_profile)

        # Get strategy's expected value
        strategy_ev = strategy_profile.expected_value()

        # Exploitability is the maximum advantage an opponent can get
        exploitability = max(br_ev - strategy_ev, 0)

        # Convert to BB/100 hands for poker context
        # Assuming 1 BB = 20 chips (from config), and we want per 100 hands
        bb_per_hand = exploitability / Config.BIG_BLIND
        exploitability_bb_per_100 = bb_per_hand * 100

        return exploitability_bb_per_100

    def _parallel_traverse_tree(self, root, opponent_profile: StrategyProfile) -> float:
        """Parallel tree traversal using ThreadPoolExecutor for large trees"""
        import concurrent.futures
        
        memo_cache = {}
        max_cache_size = Config.EXPLOITABILITY_MEMO_CACHE_SIZE
        
        def traverse_subtree(node, player_reach: float = 1.0, opp_reach: float = 1.0, depth: int = 0):
            """Worker function for parallel traversal"""
            return self._traverse_tree_worker(node, opponent_profile, memo_cache, 
                                            player_reach, opp_reach, depth, max_cache_size)
        
        # For chance nodes with many children, parallelize
        if root.is_chance_node and hasattr(root, 'next_nodes') and len(root.next_nodes) > 3:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(root.next_nodes), 8)) as executor:
                futures = []
                if hasattr(root, 'chance_probs') and root.chance_probs:
                    for child, prob in zip(root.next_nodes, root.chance_probs):
                        future = executor.submit(traverse_subtree, child, 1.0, 1.0, 1)
                        futures.append((future, prob))
                else:
                    prob = 1.0 / len(root.next_nodes)
                    for child in root.next_nodes:
                        future = executor.submit(traverse_subtree, child, 1.0, 1.0, 1)
                        futures.append((future, prob))
                
                ev = 0.0
                for future, prob in futures:
                    ev += prob * future.result()
                return ev
        else:
            return traverse_subtree(root, 1.0, 1.0, 0)

    def _traverse_tree_worker(self, node, opponent_profile: StrategyProfile, memo_cache: dict, 
                            player_reach: float, opp_reach: float, depth: int, max_cache_size: int) -> float:
        """Worker function for parallel tree traversal"""
        # Check depth limit
        if depth > Config.EXPLOITABILITY_MAX_TREE_DEPTH:
            return 0.0
        
        # Create cache key
        cache_key = (id(node), player_reach, opp_reach)
        if cache_key in memo_cache:
            return memo_cache[cache_key]

        if node.is_terminal_node:
            utility = node.terminal_utility.get_utility()
            if isinstance(utility, dict):
                utility = utility.get(0, 0)
            else:
                utility = utility
            result = utility * player_reach * opp_reach
            if len(memo_cache) < max_cache_size:
                memo_cache[cache_key] = result
            return result

        if node.is_chance_node:
            ev = 0.0
            if hasattr(node, 'next_nodes') and node.next_nodes:
                for child, prob in zip(node.next_nodes, node.chance_probs or [1.0/len(node.next_nodes)] * len(node.next_nodes)):
                    ev += prob * self._traverse_tree_worker(child, opponent_profile, memo_cache, 
                                                          player_reach, opp_reach, depth + 1, max_cache_size)
            result = ev
            if len(memo_cache) < max_cache_size:
                memo_cache[cache_key] = result
            return result

        # Player node
        infoset_key = self.game.get_infoset_key(node.history)
        player = node.player

        if not hasattr(node, 'next_nodes') or not node.next_nodes:
            result = 0.0
            if len(memo_cache) < max_cache_size:
                memo_cache[cache_key] = result
            return result

        if player == 0:  # BR player
            best_ev = -float('inf')
            for child in node.next_nodes:
                action_ev = self._traverse_tree_worker(child, opponent_profile, memo_cache, 
                                                     player_reach, opp_reach, depth + 1, max_cache_size)
                if action_ev > best_ev:
                    best_ev = action_ev
            result = best_ev
        else:  # Opponent
            ev = 0.0
            total_prob = 0.0
            for action_idx, child in enumerate(node.next_nodes):
                action_prob = opponent_profile.get_mixed_prob(infoset_key, action_idx)
                if action_prob > 0:
                    child_ev = self._traverse_tree_worker(child, opponent_profile, memo_cache, 
                                                        player_reach, opp_reach * action_prob, depth + 1, max_cache_size)
                    ev += action_prob * child_ev
                    total_prob += action_prob
            if total_prob > 0:
                ev /= total_prob
            result = ev

        if len(memo_cache) < max_cache_size:
            memo_cache[cache_key] = result
        return result

class GPUExploitabilityEvaluator:
    """GPU-accelerated exploitability evaluation using PyTorch"""

    def __init__(self, game: GTOHoldEm):
        self.game = game
        self.device = device

    def gpu_tree_traversal(self, root_node, opponent_strategy: Dict[str, Dict[int, float]]) -> Tuple[Dict[str, Dict[int, float]], float]:
        """
        GPU-accelerated tree traversal for best response calculation
        Currently falls back to CPU implementation due to recursion issues
        """
        # For now, fall back to CPU implementation to avoid recursion issues
        strategy_profile = StrategyProfile(opponent_strategy)
        br_calc = BestResponseCalculator(self.game)
        return br_calc.compute_best_response(strategy_profile)


class MCCFRSolver:
    """
    Monte Carlo Counterfactual Regret Minimization (MCCFR) for heads-up Texas Hold'em
    Uses outcome sampling for efficiency and vectorized regret updates on GPU
    """

    def __init__(self, game: GTOHoldEm, max_iterations: int = 1000):
        self.game = game
        self.max_iterations = max_iterations
        self.device = device

        # Strategy and regret storage
        self.regret_sum = defaultdict(lambda: torch.zeros(Config.ACTION_SIZE, device=self.device))
        self.strategy_sum = defaultdict(lambda: torch.zeros(Config.ACTION_SIZE, device=self.device))

        # Memoization cache for tree traversal
        self.tree_cache = {}
        # Bound cache size using Config to avoid excessive RAM growth
        self.max_cache_size = getattr(Config, 'TREE_MAX_SIZE_WITH_ABSTRACTIONS', 200_000)

        # Tree pruning parameters
        self.pruning_ratio = Config.TREE_PRUNING_RATIO
        self.min_prob_threshold = 0.01  # Prune nodes with probability < 1%

        # GPU batch processing
        self.batch_size = Config.GPU_CFR_BATCH_SIZE

        # CPU multiprocessing for tree building
        self.num_cpu_workers = min(12, torch.multiprocessing.cpu_count())  # Ryzen 9 7900X has 12 cores

    def solve(self, num_iterations: Optional[int] = None) -> StrategyProfile:
        """
        Run MCCFR for specified iterations
        Returns converged strategy profile
        """
        iterations = num_iterations or self.max_iterations

        print(f"Starting MCCFR with {iterations} iterations...")

        for t in range(iterations):
            if t % 100 == 0:
                print(f"MCCFR iteration {t}/{iterations}")

            # Alternate updating player 0 and player 1
            for player in [0, 1]:
                self._mccfr_iteration(player)

            # Prune low-probability nodes periodically
            if t % 500 == 0 and t > 0:
                self._prune_tree()

        # Convert to strategy profile
        strategy = self._get_average_strategy()
        return StrategyProfile(strategy)

    def _mccfr_iteration(self, player: int):
        """
        Single MCCFR iteration for given player using outcome sampling
        """
        # Sample a random hand
        hand = self._sample_random_hand()

        # Build game tree (CPU parallel)
        root = self._build_game_tree_parallel(hand)

        # Run outcome sampling traversal
        self._outcome_sampling_traversal(root, player, 1.0, 1.0)

    def _sample_random_hand(self) -> Tuple[List[Card], List[Card]]:
        """
        Sample random hole cards for both players
        """
        deck = [Card(suit, rank) for suit in range(4) for rank in range(13)]
        np.random.shuffle(deck)

        hole_cards_p0 = [deck.pop(), deck.pop()]
        hole_cards_p1 = [deck.pop(), deck.pop()]

        return hole_cards_p0, hole_cards_p1

    def _build_game_tree_parallel(self, hand: Tuple[List[Card], List[Card]]) -> Any:
        """
        Build game tree using CPU multiprocessing for parallelization
        """
        import multiprocessing as mp

        # Use multiprocessing Pool for CPU parallelization
        with mp.Pool(processes=self.num_cpu_workers) as pool:
            # Build preflop tree first (most important)
            preflop_tree = self._build_preflop_tree_parallel(hand, pool)

            # Build postflop trees as needed
            # For now, focus on preflop for efficiency

        return preflop_tree

    def _build_preflop_tree_parallel(self, hand: Tuple[List[Card], List[Card]], pool) -> Any:
        """
        Build preflop betting tree using parallel CPU processing
        """
        # Create root node
        root = GameNode(
            history=[],
            player=0,
            hole_cards=hand,
            community_cards=[],
            pot_size=Config.BIG_BLIND + Config.SMALL_BLIND,
            current_bet=Config.BIG_BLIND,
            is_terminal=False,
            is_chance=False
        )

        # Build tree using parallel processing for action branches
        self._expand_node_parallel(root, pool)

        return root

    def _expand_node_parallel(self, node, pool):
        """
        Expand game tree node using parallel processing
        """
        if node.is_terminal or len(node.history) > 10:  # Limit tree depth
            return

        # Generate possible actions
        actions = self._get_legal_actions(node)

        # For now, use sequential processing to avoid pickling issues
        # TODO: Implement proper parallel tree building with process-safe functions
        child_nodes = []
        for action_idx in range(len(actions)):
            child_nodes.append(self._create_child_node(node, action_idx))

        node.next_nodes = child_nodes
        node.available_actions = actions

        # Recursively expand children (limit depth for preflop)
        if len(node.history) < 8:  # Preflop depth limit
            for child in child_nodes:
                self._expand_node_parallel(child, pool)

    def _create_child_node(self, parent_node, action_idx: int):
        """
        Create child node for given action
        """
        new_history = parent_node.history + [(parent_node.player, action_idx)]
        new_player = 1 - parent_node.player

        # Update game state based on action
        new_pot = parent_node.pot_size
        new_bet = parent_node.current_bet

        if action_idx == 0:  # Fold
            # Terminal node
            return GameNode(
                history=new_history,
                player=new_player,
                hole_cards=parent_node.hole_cards,
                community_cards=parent_node.community_cards,
                pot_size=new_pot,
                current_bet=new_bet,
                is_terminal=True,
                is_chance=False,
                terminal_utility=self._compute_fold_utility(parent_node, new_player)
            )
        elif action_idx == 1:  # Call
            new_bet = max(new_bet, parent_node.current_bet)
            new_pot = parent_node.pot_size + (new_bet - parent_node.current_bet)

            # Check if we need to deal flop
            if len(new_history) >= 4:  # After preflop betting
                return self._create_flop_node(new_history, new_player, parent_node.hole_cards, new_pot, new_bet)
            else:
                return GameNode(
                    history=new_history,
                    player=new_player,
                    hole_cards=parent_node.hole_cards,
                    community_cards=parent_node.community_cards,
                    pot_size=new_pot,
                    current_bet=new_bet,
                    is_terminal=False,
                    is_chance=False
                )
        else:  # Raise
            raise_amount = self._get_raise_amount(action_idx, parent_node)
            new_bet = raise_amount
            new_pot = parent_node.pot_size + (raise_amount - parent_node.current_bet)

            return GameNode(
                history=new_history,
                player=new_player,
                hole_cards=parent_node.hole_cards,
                community_cards=parent_node.community_cards,
                pot_size=new_pot,
                current_bet=new_bet,
                is_terminal=False,
                is_chance=False
            )

    def _create_flop_node(self, history, player, hole_cards, pot_size, current_bet):
        """
        Create flop dealing node
        """
        # Deal flop cards
        deck = [Card(suit, rank) for suit in range(4) for rank in range(13)]
        # Remove hole cards from deck
        for card in hole_cards[0] + hole_cards[1]:
            if card in deck:
                deck.remove(card)

        np.random.shuffle(deck)
        community_cards = deck[:3]  # Flop

        return GameNode(
            history=history,
            player=player,
            hole_cards=hole_cards,
            community_cards=community_cards,
            pot_size=pot_size,
            current_bet=current_bet,
            is_terminal=False,
            is_chance=False
        )

    def _get_legal_actions(self, node) -> List[int]:
        """
        Get legal actions for current node
        """
        actions = []

        # Fold always available unless we're already all-in
        if node.current_bet > 0:
            actions.append(0)  # Fold

        # Call
        actions.append(1)  # Call

        # Raises (simplified for preflop)
        if len(node.history) < 6:  # Limit raise depth
            actions.extend([2, 3, 4])  # Different raise sizes

        return actions

    def _get_raise_amount(self, action_idx: int, node) -> float:
        """
        Get raise amount for action index
        """
        base_raise = node.current_bet * 2  # Min raise

        if action_idx == 2:
            return base_raise
        elif action_idx == 3:
            return base_raise * 1.5
        elif action_idx == 4:
            return base_raise * 3.0
        else:
            return base_raise

    def _compute_fold_utility(self, node, folding_player: int):
        """
        Compute utility when a player folds
        """
        if folding_player == 0:
            return -node.pot_size  # Player 0 loses pot
        else:
            return node.pot_size   # Player 0 wins pot

    def _outcome_sampling_traversal(self, node, player: int, p0_reach: float, p1_reach: float):
        """
        Outcome sampling traversal for MCCFR
        """
        if node.is_terminal:
            return self._get_terminal_utility(node, player)

        if node.is_chance:
            # Sample random outcome
            if hasattr(node, 'chance_probs') and node.chance_probs:
                outcome_idx = np.random.choice(len(node.next_nodes), p=node.chance_probs)
            else:
                outcome_idx = np.random.randint(len(node.next_nodes))

            child = node.next_nodes[outcome_idx]
            return self._outcome_sampling_traversal(child, player, p0_reach, p1_reach)

        # Check if node has no available actions (shouldn't happen but safety check)
        if not hasattr(node, 'next_nodes') or not node.next_nodes:
            return 0.0  # Return 0 utility for invalid nodes

        # Player node
        infoset_key = self._get_infoset_key(node, player)

        if node.player == player:
            # Update regrets for traversing player
            strategy = self._get_strategy(infoset_key)
            
            # Ensure strategy length matches available actions
            num_actions = len(node.next_nodes)
            if num_actions == 0:
                # Terminal node or no actions available
                return 0.0
            
            # Final safety check - if no actions, return 0
            if num_actions == 0:
                return 0.0
                
            if len(strategy) != num_actions:
                # Pad or truncate strategy to match available actions
                if len(strategy) < num_actions:
                    # Pad with uniform probability
                    padding = torch.ones(num_actions - len(strategy), device=strategy.device) / num_actions
                    strategy = torch.cat([strategy, padding])
                    strategy = strategy / strategy.sum()  # Renormalize
                else:
                    # Truncate
                    strategy = strategy[:num_actions]
                    strategy = strategy / strategy.sum()  # Renormalize
            
            # Final safety check - if strategy is still empty or invalid, use uniform
            if len(strategy) == 0 or torch.sum(strategy) == 0:
                strategy = torch.ones(num_actions, device=strategy.device) / num_actions

            # DEBUG: Check strategy values before sampling
            # print(f"DEBUG - Player {player} - Infoset: {infoset_key} - Strategy: {strategy.cpu().numpy()}")

            action_idx = np.random.choice(len(strategy), p=strategy.cpu().numpy())

            # Sample action and recurse
            if action_idx < len(node.next_nodes):
                child = node.next_nodes[action_idx]
                action_prob = strategy[action_idx].item()

                if node.player == 0:
                    new_p0_reach = p0_reach * action_prob
                    new_p1_reach = p1_reach
                else:
                    new_p0_reach = p0_reach
                    new_p1_reach = p1_reach * action_prob

                utility = self._outcome_sampling_traversal(child, player, new_p0_reach, new_p1_reach)

                # Update regrets
                cf_reach = p0_reach if player == 1 else p1_reach
                self._update_regrets(infoset_key, strategy, utility, cf_reach, action_idx)
            else:
                # Fallback if action_idx is out of bounds
                utility = 0.0

            return utility
        else:
            # Opponent node - sample action
            strategy = self._get_strategy(infoset_key)
            
            # Ensure strategy length matches available actions
            num_actions = len(node.next_nodes)
            if len(strategy) != num_actions:
                # Pad or truncate strategy to match available actions
                if len(strategy) < num_actions:
                    # Pad with uniform probability
                    padding = torch.ones(num_actions - len(strategy), device=strategy.device) / num_actions
                    strategy = torch.cat([strategy, padding])
                    strategy = strategy / strategy.sum()  # Renormalize
                else:
                    # Truncate
                    strategy = strategy[:num_actions]
                    strategy = strategy / strategy.sum()  # Renormalize
            
            action_idx = np.random.choice(len(strategy), p=strategy.cpu().numpy())

            if action_idx < len(node.next_nodes):
                child = node.next_nodes[action_idx]
                action_prob = strategy[action_idx].item()

                if node.player == 0:
                    new_p0_reach = p0_reach * action_prob
                    new_p1_reach = p1_reach
                else:
                    new_p0_reach = p0_reach
                    new_p1_reach = p1_reach * action_prob

                return self._outcome_sampling_traversal(child, player, new_p0_reach, new_p1_reach)
            else:
                # Fallback if action_idx is out of bounds
                return 0.0

    def _get_strategy(self, infoset_key: str) -> torch.Tensor:
        """
        Get current strategy for infoset using regret matching
        """
        if infoset_key not in self.regret_sum:
            # Return uniform strategy if no regrets yet
            return torch.ones(Config.ACTION_SIZE, device=self.device) / Config.ACTION_SIZE
            
        regrets = self.regret_sum[infoset_key]
        
        # Ensure regrets are non-negative for regret matching
        positive_regrets = torch.clamp(regrets, min=0.0)
        
        # If all regrets are negative, use uniform strategy
        if torch.sum(positive_regrets) == 0:
            return torch.ones_like(regrets) / len(regrets)
            
        # Regret matching
        strategy = positive_regrets / torch.sum(positive_regrets)
        
        # Ensure no zero probabilities for sampling
        strategy = torch.clamp(strategy, min=1e-6)
        strategy = strategy / torch.sum(strategy)
        
        return strategy

    def _update_regrets(self, infoset_key: str, strategy: torch.Tensor, utility: float,
                       cf_reach: float, sampled_action: int):
        """
        Update regrets for infoset using vectorized operations
        """
        # Ensure sampled_action is within bounds
        if sampled_action >= len(strategy):
            return  # Skip update if action is out of bounds
            
        # Get or create regret sum tensor for this infoset
        if infoset_key not in self.regret_sum:
            self.regret_sum[infoset_key] = torch.zeros(len(strategy), device=self.device)
        elif len(self.regret_sum[infoset_key]) != len(strategy):
            # Resize if necessary
            old_regrets = self.regret_sum[infoset_key]
            new_regrets = torch.zeros(len(strategy), device=self.device)
            min_len = min(len(old_regrets), len(strategy))
            new_regrets[:min_len] = old_regrets[:min_len]
            self.regret_sum[infoset_key] = new_regrets
            
        # Compute counterfactual utility for each action
        cf_utilities = torch.zeros_like(strategy)

        # Only the sampled action contributes to regret update in outcome sampling
        cf_utilities[sampled_action] = utility / strategy[sampled_action].item()

        # Update regrets
        self.regret_sum[infoset_key] += cf_reach * (cf_utilities - cf_utilities[sampled_action])

        # Update strategy sum (ensure same size)
        if infoset_key not in self.strategy_sum:
            self.strategy_sum[infoset_key] = torch.zeros_like(strategy)
        elif len(self.strategy_sum[infoset_key]) != len(strategy):
            # Resize if necessary
            old_strategy_sum = self.strategy_sum[infoset_key]
            new_strategy_sum = torch.zeros_like(strategy)
            min_len = min(len(old_strategy_sum), len(strategy))
            new_strategy_sum[:min_len] = old_strategy_sum[:min_len]
            self.strategy_sum[infoset_key] = new_strategy_sum
            
        self.strategy_sum[infoset_key] += cf_reach * strategy

    def _get_infoset_key(self, node, player: int) -> str:
        """
        Get infoset key for node and player
        """
        hole_cards = node.hole_cards[player] if isinstance(node.hole_cards, (list, tuple)) and len(node.hole_cards) > player else []
        community = node.community_cards or []

        # Create compact representation
        hole_str = ''.join([f"{c.suit}{c.value}" for c in hole_cards])
        comm_str = ''.join([f"{c.suit}{c.value}" for c in community])
        history_str = ''.join([f"{p}{a}" for p, a in node.history])

        return f"{hole_str}_{comm_str}_{history_str}"

    def _get_terminal_utility(self, node, player: int) -> float:
        """
        Get terminal utility for player
        """
        if hasattr(node, 'terminal_utility'):
            utility = node.terminal_utility
            if isinstance(utility, dict):
                return utility.get(player, 0)
            return utility if player == 0 else -utility

        # Default showdown utility
        return self._compute_showdown_utility(node, player)

    def _compute_showdown_utility(self, node, player: int) -> float:
        """
        Compute showdown utility (simplified)
        """
        # Simplified equity calculation
        equity = 0.5  # Assume 50% equity for now

        if player == 0:
            return equity * node.pot_size
        else:
            return (1 - equity) * node.pot_size

    def _prune_tree(self):
        """
        Prune low-probability nodes from tree cache
        """
        if len(self.tree_cache) < self.max_cache_size:
            return

        # Sort by access frequency and keep top ratio
        sorted_items = sorted(self.tree_cache.items(), key=lambda x: x[1]['access_count'], reverse=True)
        keep_count = int(len(sorted_items) * self.pruning_ratio)

        # Clear cache and keep only top items
        self.tree_cache.clear()
        for key, value in sorted_items[:keep_count]:
            self.tree_cache[key] = value

        print(f"Pruned tree cache to {len(self.tree_cache)} nodes")

    def _get_average_strategy(self) -> Dict[str, Dict[int, float]]:
        """
        Compute average strategy from strategy sum
        """
        avg_strategy = {}

        for infoset_key, strategy_sum in self.strategy_sum.items():
            total_sum = strategy_sum.sum().item()
            if total_sum > 0:
                avg_strategy[infoset_key] = {
                    action_idx: prob.item() / total_sum
                    for action_idx, prob in enumerate(strategy_sum)
                }
            else:
                # Uniform fallback
                num_actions = len(strategy_sum)
                avg_strategy[infoset_key] = {
                    i: 1.0 / num_actions for i in range(num_actions)
                }

        return avg_strategy


class GameNode:
    """
    Game tree node for MCCFR
    """

    def __init__(self, history, player, hole_cards, community_cards, pot_size,
                 current_bet, is_terminal=False, is_chance=False, terminal_utility=None):
        self.history = history
        self.player = player
        self.hole_cards = hole_cards
        self.community_cards = community_cards
        self.pot_size = pot_size
        self.current_bet = current_bet
        self.is_terminal = is_terminal
        self.is_chance = is_chance
        self.terminal_utility = terminal_utility
        self.next_nodes = []
        self.available_actions = []


# Enhanced functions for easy integration
def run_mccfr_training(game: GTOHoldEm, num_iterations: int = 5000) -> StrategyProfile:
    """
    Run MCCFR training and return converged strategy
    """
    solver = MCCFRSolver(game, max_iterations=num_iterations)
    return solver.solve()


def compute_exploitability_with_mccfr(strategy: StrategyProfile, game: GTOHoldEm) -> float:
    """
    Compute exploitability using MCCFR-derived strategy
    """
    br_calc = BestResponseCalculator(game)
    return br_calc.compute_exploitability(strategy)


def hybrid_rl_cfr_training(model, game: GTOHoldEm, total_steps: int = 500000,
                          cfr_weight: float = 0.9) -> Dict[str, float]:
    """
    Hybrid RL-CFR training with annealing
    """
    results = {
        'exploitability_history': [],
        'win_rate_history': [],
        'cfr_weight_history': []
    }

    # Annealing schedule
    annealing_steps = 50000

    for step in range(0, total_steps, 1000):
        # Update CFR weight (anneal from cfr_weight to 0.5)
        current_cfr_weight = cfr_weight - (cfr_weight - 0.5) * min(step / annealing_steps, 1.0)

        # Run MCCFR for preflop
        if step % Config.PREFLOP_CFR_ITERATIONS == 0:
            mccfr_strategy = run_mccfr_training(game, Config.PREFLOP_CFR_ITERATIONS)

        # Mix strategies
        mixed_strategy = mix_strategies(model, mccfr_strategy, current_cfr_weight)

        # Compute exploitability
        exploitability = compute_exploitability_with_mccfr(mixed_strategy, game)
        results['exploitability_history'].append(exploitability)
        results['cfr_weight_history'].append(current_cfr_weight)

        if step % 10000 == 0:
            print(f"Step {step}: Exploitability = {exploitability:.2f} mbb/100, CFR weight = {current_cfr_weight:.3f}")

    return results


def mix_strategies(rl_strategy, cfr_strategy: StrategyProfile, cfr_weight: float) -> StrategyProfile:
    """
    Mix RL and CFR strategies
    """
    mixed_infosets = {}

    for infoset_key in set(rl_strategy.infosets.keys()) | set(cfr_strategy.infosets.keys()):
        rl_probs = rl_strategy.infosets.get(infoset_key, {})
        cfr_probs = cfr_strategy.infosets.get(infoset_key, {})

        # Combine probabilities
        all_actions = set(rl_probs.keys()) | set(cfr_probs.keys())
        mixed_probs = {}

        for action in all_actions:
            rl_prob = rl_probs.get(action, 0.0)
            cfr_prob = cfr_probs.get(action, 0.0)
            mixed_probs[action] = (1 - cfr_weight) * rl_prob + cfr_weight * cfr_prob

        mixed_infosets[infoset_key] = mixed_probs

    return StrategyProfile(mixed_infosets)


# Standalone functions for easy integration
def evaluate_exploitability(strategy_infosets: Dict[str, Dict[int, float]], game: GTOHoldEm) -> float:
    """
    Evaluate exploitability of a strategy profile
    Args:
        strategy_infosets: Dictionary mapping infoset keys to action probability distributions
        game: Game instance for tree traversal
    Returns:
        Exploitability in BB/100 hands
    """
    strategy_profile = StrategyProfile(strategy_infosets)
    br_calc = BestResponseCalculator(game)
    return br_calc.compute_exploitability(strategy_profile)


def run_validation_with_exploitability(model, game: GTOHoldEm, num_hands: int = 1000) -> Dict[str, float]:
    """
    Run validation and compute exploitability metrics
    Args:
        model: The trained model to evaluate
        game: Game instance
        num_hands: Number of hands to evaluate
    Returns:
        Dictionary with validation metrics including exploitability
    """
    # This is a placeholder - would need model integration
    # For now, return dummy values
    return {
        'exploitability': 0.0,
        'win_rate': 0.0,
        'validation_hands': num_hands
    }
