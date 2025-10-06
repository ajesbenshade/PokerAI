from typing import List, Tuple, Optional, Any, Dict
import warnings
warnings.filterwarnings("ignore", message="expandable_segments not supported")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import threading
import random
import torch.nn.functional as F
from torch.amp import GradScaler
import torch.backends.cudnn as cudnn
import lmdb
import pickle
import os
import logging

from models import Actor, Critic
from utils import (
    Card, CustomBeta, get_vram_usage, use_preflop_chart, estimate_equity, quick_simulate
    # Temporarily disabled to avoid circular imports:
    # get_state, get_legal_actions,
    # regret_matching_adjustment, adjust_action_by_opponent,
    # should_bluff, opponent_tracker, cfr_regret_adjust
)
from config import Config, Action

cudnn.benchmark = True
device = torch.device(Config.DEVICE)

logger = logging.getLogger(__name__)

class RolloutBuffer:
    """
    Simple rollout buffer for on-policy PPO training.
    Stores transitions in memory and clears after training.
    """
    def __init__(self, capacity=4096):
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.advantages = []
        self.returns = []

    def add(self, state, action, log_prob, reward, done, value):
        """Add a single transition to the buffer."""
        self.states.append(state)
        self.actions.append(action)  # action can be action_idx or [action_idx, raise_amount]
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def sample(self):
        """Return all transitions as tensors."""
        if not self.states:
            return None

        states = torch.tensor(np.array(self.states), dtype=torch.float32, device=device)
        # Handle actions that might be tuples [action_idx, raise_amount]
        if isinstance(self.actions[0], (list, tuple)):
            actions = torch.tensor(np.array(self.actions), dtype=torch.float32, device=device)
        else:
            actions = torch.tensor(np.array(self.actions), dtype=torch.long, device=device)
        log_probs = torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=device)
        rewards = torch.tensor(np.array(self.rewards), dtype=torch.float32, device=device)
        dones = torch.tensor(np.array(self.dones), dtype=torch.float32, device=device)
        values = torch.tensor(np.array(self.values), dtype=torch.float32, device=device)

        return states, actions, log_probs, rewards, dones, values

    def compute_gae(self, next_value, gamma=0.99, gae_lambda=0.95):
        """Compute GAE advantages and returns on-the-fly."""
        if not self.states:
            return

        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values)

        advantages = np.zeros(len(rewards), dtype=np.float32)
        returns = np.zeros(len(rewards), dtype=np.float32)

        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * gae * (1 - dones[t])
            advantages[t] = gae
            returns[t] = gae + values[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        self.returns = torch.tensor(returns, dtype=torch.float32, device=device)

    def clear(self):
        """Clear all stored transitions."""
        self.__init__(self.capacity)

    def __len__(self):
        return len(self.states)


class PPOBuffer:
    def __init__(self):
        self.trajectories = []

    def add_trajectory(self, trajectory):
        self.trajectories.append(trajectory)

    def get_trajectories(self):
        return self.trajectories

    def clear(self):
        self.trajectories = []
    
    def __len__(self):
        return len(self.trajectories)


class ActorCriticAgent:
    def __init__(self, state_size: int, action_size: int, buffer=None, device: str = None):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device(device) if device else Config.DEVICE
        try:
            self.actor = Actor(state_size).to(self.device)
            self.critic = Critic(state_size).to(self.device)
        except RuntimeError as e:
            print(f"Failed to move models to {self.device}, falling back to CPU: {e}")
            self.device = torch.device('cpu')
            self.actor = Actor(state_size).to(self.device)
            self.critic = Critic(state_size).to(self.device)
        
        self.optimizer = optim.AdamW(
            list(self.actor.parameters()) + list(self.critic.parameters()), 
            lr=Config.LR, 
            weight_decay=1e-5
        )
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=Config.LR_DECAY
        )
        if buffer is None:
            self.buffer = RolloutBuffer(capacity=Config.BATCH_SIZE * 2)  # Collect ~2 batches worth
        else:
            self.buffer = buffer
        self.gamma = Config.GAMMA
        self.gae_lambda = Config.GAE_LAMBDA
        self.ppo_clip = Config.PPO_CLIP
        self.entropy_beta = Config.ENTROPY_BETA
        self.ppo_epochs = Config.PPO_EPOCHS
        
        # Training step counter for preflop annealing
        self.training_step = 0
        
        # Add total_steps attribute for checkpointing
        self.total_steps = 0

        # Add raise binning method
        self._raise_bins = Config.RAISE_BIN_FRACTIONS

    @classmethod
    def create_simulation_agent(cls, device: str = None) -> 'ActorCriticAgent':
        """Factory method to create agents for simulation (no buffer needed)."""
        return cls(Config.STATE_SIZE, Config.ACTION_SIZE, buffer=None, device=device)

    def choose_action(self, state: np.ndarray, legal_actions: np.ndarray, player_id: int, **kwargs) -> Tuple[int, Optional[float], int, Optional[float], Optional[float]]:
        """
        Choose an action using the current policy.
        
        Returns:
            Tuple of (action_idx, raise_amount, discrete_action, log_prob, value)
        """
        stack = kwargs.get('stack')
        min_raise = kwargs.get('min_raise')
        call_amount = kwargs.get('call_amount')
        hole_cards = kwargs.get('hole_cards')
        community_cards = kwargs.get('community_cards', [])  # Default to empty
        opponents = kwargs.get('opponents', [])  # Expect list; default empty
        num_opps = len(opponents)
        round_number = int(state[7] * 3)
        if round_number == 0 and hole_cards is not None:
            position_idx = kwargs.get('position_idx', 0)
            training_step = kwargs.get('training_step', getattr(self, 'training_step', 0))
            chart_action = use_preflop_chart(hole_cards, position_idx, stack, player_id, self, training_step)
            if chart_action and random.random() < Config.PREFLOP_CHART_PROBABILITY:
                action_idx, raise_amount = chart_action
                
                # Convert continuous raise_amount to discrete bin
                discrete_action = self._raise_amount_to_bin(raise_amount, kwargs.get('pot_size', Config.BIG_BLIND * 3), call_amount, min_raise, stack)
                
                # Compute log probability for chart action using current policy
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    logits = self.actor(state_tensor)
                    probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
                    probs *= legal_actions
                    if probs.sum() > 0:
                        probs /= probs.sum()
                    else:
                        probs = legal_actions / legal_actions.sum()
                    log_prob = np.log(probs[discrete_action]) if probs[discrete_action] > 0 else -10.0
                    
                    # Get value estimate from critic
                    value = self.critic(state_tensor).item()
                
                return action_idx, raise_amount, discrete_action, log_prob, value
        # Estimate hand equity relative to opponents
        equity = estimate_equity(hole_cards, community_cards, num_opps)

        # Opponent Modeling: Update range estimates for adaptive strategies
        opponent_ranges = {}
        if num_opps > 0 and Config.LEARNED_ABSTRACTION_ENABLED:
            try:
                from opponent_model import get_opponent_model_manager
                opponent_manager = get_opponent_model_manager()

                # Prepare inputs for opponent modeling
                game_state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Encode betting history (simplified - can be enhanced)
                betting_history = torch.zeros(20, device=self.device)  # Placeholder for betting history encoding
                if hasattr(kwargs, 'betting_history') and kwargs.get('betting_history'):
                    # Simple encoding of recent betting actions
                    history = kwargs['betting_history'][-20:] if len(kwargs['betting_history']) > 20 else kwargs['betting_history']
                    for i, action in enumerate(history):
                        betting_history[i] = float(action) / Config.ACTION_SIZE

                # Current range estimate (simplified uniform)
                current_range = torch.ones(169, device=self.device) / 169

                # Update range estimates for each opponent
                for opp in opponents:
                    if hasattr(opp, 'player_id'):
                        opp_id = opp.player_id
                        predicted_range = opponent_manager.update_range_estimate(
                            opp_id, game_state_tensor, betting_history, current_range
                        )
                        opponent_ranges[opp_id] = predicted_range.squeeze(0).cpu().numpy()

                        # Apply opponent-specific adjustments based on predicted range
                        if opp_id in opponent_ranges:
                            opp_range = opponent_ranges[opp_id]
                            # Adjust action selection based on opponent tendencies
                            vpip_estimate = np.sum(opp_range)  # Simplified VPIP estimate
                            aggression_estimate = np.mean(opp_range[2:])  # Simplified aggression estimate

                            # Store for later use in action selection
                            kwargs[f'opp_{opp_id}_vpip'] = vpip_estimate
                            kwargs[f'opp_{opp_id}_aggression'] = aggression_estimate

            except Exception as e:
                logger.warning(f"Opponent modeling failed: {e}")
                # Continue without opponent modeling

        # Enhanced Opponent Tracking: Update OpponentTracker with current game state
        game_state_obj = kwargs.get('game_state')
        if game_state_obj and num_opps > 0:
            try:
                from utils import opponent_tracker
                
                # Update opponent statistics based on current betting
                for opp in opponents:
                    if hasattr(opp, 'player_id'):
                        opp_id = opp.player_id
                        
                        # Determine current street
                        street = 'preflop'
                        if len(game_state_obj.community_cards) >= 3:
                            street = 'flop'
                        if len(game_state_obj.community_cards) >= 4:
                            street = 'turn'
                        if len(game_state_obj.community_cards) >= 5:
                            street = 'river'
                        
                        # Check if this is a continuation bet opportunity
                        is_cbet = self._is_continuation_bet(game_state_obj, opp)
                        
                        # Update opponent tracker with recent action (if available)
                        if hasattr(game_state_obj, 'betting_history') and game_state_obj.betting_history:
                            last_action = game_state_obj.betting_history[-1]
                            if len(last_action) >= 2:
                                action_type, bet_size = last_action[0], last_action[1]
                                opponent_tracker.update_action(
                                    opp_id, action_type, street, 
                                    game_state_obj.pot_size, call_amount, is_cbet
                                )
                        
                        # Get adaptive strategy adjustments
                        adaptive_strategy = opponent_tracker.get_adaptive_strategy(
                            opp_id, equity, 
                            call_amount / (game_state_obj.pot_size + call_amount + 1e-8)
                        )
                        
                        # Store adaptive adjustments for use in action selection
                        kwargs[f'opp_{opp_id}_adaptive'] = adaptive_strategy
                        
            except Exception as e:
                logger.warning(f"Opponent tracking failed: {e}")
                # Continue without enhanced tracking

        # Use MCTS for high-equity situations to refine action selection
        if equity > 0.6 and num_opps > 0:
            mcts_action = self.mcts_search(state, legal_actions, num_sims=5,
                                         hole_cards=hole_cards, community_cards=community_cards,
                                         num_opponents=num_opps, pot_size=kwargs.get('pot_size', Config.BIG_BLIND * 3),
                                         stack=stack, call_amount=call_amount, min_raise=min_raise)
            if mcts_action is not None:
                # Use MCTS action with some probability
                if random.random() < 0.7:
                    if mcts_action == Action.RAISE.value:
                        # Convert MCTS raise to discrete bin
                        pot_size = kwargs.get('pot_size', Config.BIG_BLIND * 3)
                        max_raise = stack - call_amount
                        
                        # Choose raise size based on equity and pot size
                        if equity > 0.8:
                            # Very strong hand - raise large
                            raise_amount = min_raise + 0.75 * (max_raise - min_raise)
                        elif equity > 0.7:
                            # Strong hand - raise medium
                            raise_amount = min_raise + 0.5 * (max_raise - min_raise)
                        else:
                            # Decent hand - raise small
                            raise_amount = min_raise + 0.25 * (max_raise - min_raise)
                        
                        raise_amount = max(min_raise, min(raise_amount, max_raise))
                        discrete_action = self._raise_amount_to_bin(raise_amount, pot_size, call_amount, min_raise, stack)
                        action_idx = Action.RAISE.value
                    else:
                        discrete_action = mcts_action
                        action_idx = mcts_action
                        raise_amount = None
                    
                    # Compute log probability and value for MCTS action using current policy
                    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        logits = self.actor(state_tensor)
                        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
                        probs *= legal_actions
                        if probs.sum() > 0:
                            probs /= probs.sum()
                        else:
                            probs = legal_actions / legal_actions.sum()
                        log_prob = np.log(probs[discrete_action]) if probs[discrete_action] > 0 else -10.0
                        
                        # Get value estimate from critic
                        value = self.critic(state_tensor).item()
                    
                    self.actor.train()
                    return action_idx, raise_amount, discrete_action, log_prob, value

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            logits = self.actor(state_tensor)
            # Use double precision for the softmax to improve numerical stability
            probs = F.softmax(logits, dim=-1).cpu().double().numpy()[0]
            
            # Get value estimate from critic
            value = self.critic(state_tensor).item()
            
        probs *= legal_actions
        if probs.sum() == 0:
            probs = legal_actions / legal_actions.sum()
        else:
            probs /= probs.sum()

        # Apply adaptive strategy adjustments based on opponent tracking
        if num_opps > 0 and 'game_state' in kwargs:
            try:
                from utils import opponent_tracker
                
                # Get primary opponent (first active opponent)
                primary_opp = None
                for opp in opponents:
                    if hasattr(opp, 'player_id') and not opp.folded:
                        primary_opp = opp
                        break
                
                if primary_opp:
                    opp_id = primary_opp.player_id
                    
                    # Get adaptive strategy adjustments
                    pot_odds = call_amount / (kwargs.get('pot_size', Config.BIG_BLIND * 3) + call_amount + 1e-8)
                    adaptive_strategy = opponent_tracker.get_adaptive_strategy(opp_id, equity, pot_odds)
                    
                    # Apply adjustments to action probabilities
                    adjusted_probs = probs.copy()
                    
                    # Adjust fold threshold
                    if equity < 0.3 and adaptive_strategy['fold_threshold'] < 0:
                        adjusted_probs[Action.FOLD.value] *= (1.0 + adaptive_strategy['fold_threshold'])
                    
                    # Adjust call threshold  
                    if adaptive_strategy['call_threshold'] < 0:
                        adjusted_probs[Action.CALL.value] *= (1.0 + adaptive_strategy['call_threshold'])
                    
                    # Adjust raise frequency
                    if adaptive_strategy['raise_frequency'] != 1.0:
                        raise_prob_sum = np.sum(adjusted_probs[2:])  # All raise actions
                        if raise_prob_sum > 0:
                            adjusted_probs[2:] *= adaptive_strategy['raise_frequency']
                    
                    # Adjust bluff multiplier
                    if equity < 0.4 and adaptive_strategy['bluff_multiplier'] > 1.0:
                        adjusted_probs[Action.RAISE.value] *= adaptive_strategy['bluff_multiplier']
                    
                    # Renormalize probabilities
                    if np.sum(adjusted_probs) > 0:
                        adjusted_probs /= np.sum(adjusted_probs)
                        probs = adjusted_probs
                    
            except Exception as e:
                logger.warning(f"Adaptive strategy application failed: {e}")
                # Continue with original probabilities

        # Pure PPO sampling - no UCB exploration
        discrete_action = np.random.choice(self.action_size, p=probs)
        
        # Convert discrete action to game format
        from utils import interpret_discrete_action
        action_idx, raise_amount = interpret_discrete_action(
            discrete_action, 
            kwargs.get('pot_size', Config.BIG_BLIND * 3),
            call_amount,
            min_raise,
            stack
        )
        
        # Compute log probability of chosen action
        log_prob = np.log(probs[discrete_action])
            
        self.total_steps += 1
            
        self.actor.train()
        return action_idx, raise_amount, discrete_action, log_prob, value

    def mcts_search(self, state, legal_actions, num_sims=15, **kwargs):
        """
        Perform Monte Carlo Tree Search with fixed simulation.
        Uses a simplified reward estimation instead of full game simulation.
        """
        from math import sqrt, log, inf
        
        hole_cards = kwargs.get('hole_cards', [])
        community_cards = kwargs.get('community_cards', [])
        num_opponents = kwargs.get('num_opponents', 1)
        pot_size = kwargs.get('pot_size', Config.BIG_BLIND * 3)
        stack = kwargs.get('stack', Config.INITIAL_STACK)
        call_amount = kwargs.get('call_amount', 0)
        min_raise = kwargs.get('min_raise', Config.BIG_BLIND * 2)
        
        class MCTSNode:
            def __init__(self, action=None, parent=None):
                self.action = action
                self.parent = parent
                self.children = []
                self.visits = 0
                self.value = 0
                self.legal_actions = legal_actions.copy()
                
            def is_fully_expanded(self):
                return len(self.children) == np.sum(self.legal_actions)
                
            def best_child(self, c=1.41):
                if not self.children:
                    return None
                best_score = -inf
                best_child = None
                for child in self.children:
                    if child.visits == 0:
                        return child
                    exploitation = child.value / child.visits
                    exploration = c * sqrt(log(self.visits) / child.visits)
                    score = exploitation + exploration
                    if score > best_score:
                        best_score = score
                        best_child = child
                return best_child
                
            def expand(self):
                # Find unexpanded actions
                expanded_actions = {child.action for child in self.children}
                for action_idx in range(len(self.legal_actions)):
                    if self.legal_actions[action_idx] and action_idx not in expanded_actions:
                        child = MCTSNode(action=action_idx, parent=self)
                        self.children.append(child)
                        return child
                return None
                
            def backpropagate(self, reward):
                self.visits += 1
                self.value += reward
                if self.parent:
                    self.parent.backpropagate(reward)
        
        root = MCTSNode()
        
        for _ in range(num_sims):
            node = root
            
            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child()
            
            # Expansion
            if not node.is_fully_expanded():
                node = node.expand()
            
            # Simulation - use simplified reward estimation
            if node and node.action is not None:
                reward = self._estimate_action_value(node.action, hole_cards, community_cards, 
                                                   num_opponents, pot_size, stack, call_amount, min_raise)
            else:
                reward = 0
            
            # Backpropagation
            if node:
                node.backpropagate(reward)
        
        # Return best action
        if root.children:
            best_child = max(root.children, key=lambda c: c.visits)
            return best_child.action
        return None
    
    def _estimate_action_value(self, action_idx, hole_cards, community_cards, num_opponents, 
                              pot_size, stack, call_amount, min_raise):
        """
        Simplified action value estimation for MCTS rollout.
        """
        # Use the improved quick_simulate for more realistic value estimation
        raise_amount = None
        if action_idx == Action.RAISE.value:
            # Use the same heuristic as in choose_action for consistency
            equity = estimate_equity(hole_cards, community_cards, num_opponents)
            max_raise = stack - call_amount
            
            if equity > 0.8:
                raise_amount = min_raise + 0.75 * (max_raise - min_raise)
            elif equity > 0.7:
                raise_amount = min_raise + 0.5 * (max_raise - min_raise)
            else:
                raise_amount = min_raise + 0.25 * (max_raise - min_raise)
            
            raise_amount = max(min_raise, min(raise_amount, max_raise))
        
        # Use quick_simulate for realistic opponent modeling
        reward = quick_simulate(
            hole_cards, community_cards, num_opponents, pot_size, 
            stack, call_amount, min_raise, action_idx, raise_amount
        )
        
        return reward

    def compute_gae(self, transitions: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
        if not transitions:
            return np.array([]), np.array([])
        
        states = torch.tensor(np.array([t[0] for t in transitions]), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([t[3] for t in transitions]), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(np.array([t[2] for t in transitions]), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array([t[4] for t in transitions]), dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(np.array([t[5] for t in transitions]), dtype=torch.float32, device=self.device)
        values = np.array([t[6] for t in transitions])

        # GAE calculation
        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = next_states[t]
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae * next_non_terminal
            advantages[t] = gae

        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update(self):
        """
        Perform a PPO update using the collected rollouts.
        """
        if len(self.buffer) == 0:
            return
        
        # Sample a batch from the buffer
        batch = self.buffer.sample()
        if batch is None:
            return
        
        states, actions, old_log_probs, rewards, dones, values = batch
        
        # Compute GAE using the buffer's method
        with torch.no_grad():
            next_value = self.critic(states[-1:]).item() if len(states) > 0 else 0.0
            self.buffer.compute_gae(next_value, self.gamma, self.gae_lambda)
        
        advantages = self.buffer.advantages
        returns = self.buffer.returns
        
        # --- Micro-batched PPO update to limit peak VRAM ---
        N = states.size(0)
        micro_bs = getattr(Config, 'PPO_MICRO_BATCH', 2048)  # Use new config parameter
        micro_bs = max(256, min(micro_bs, N))
        num_mbs = (N + micro_bs - 1) // micro_bs
        
        for _ in range(self.ppo_epochs):
            self.optimizer.zero_grad()
            mb_start = 0
            while mb_start < N:
                mb_end = min(N, mb_start + micro_bs)
                s_mb = states[mb_start:mb_end]
                a_mb = actions[mb_start:mb_end]
                old_lp_mb = old_log_probs[mb_start:mb_end]
                adv_mb = advantages[mb_start:mb_end]
                ret_mb = returns[mb_start:mb_end]
                
                try:
                    with torch.amp.autocast('cuda', enabled=Config.AMP_ENABLED):
                        logits = self.actor(s_mb)
                        log_probs = F.log_softmax(logits, dim=-1)
                        
                        # Policy loss (clipped)
                        taken_logp = log_probs.gather(1, a_mb.unsqueeze(1)).squeeze(1)
                        ratio = torch.exp(taken_logp - old_lp_mb)
                        clipped_ratio = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip)
                        policy_loss = -torch.min(ratio * adv_mb, clipped_ratio * adv_mb).mean()
                        
                        # Value loss (MSE), align shapes
                        values_pred = self.critic(s_mb).squeeze(-1)
                        value_loss = F.mse_loss(values_pred, ret_mb)
                        
                        # Entropy bonus (use mean of log_probs)
                        entropy_term = -log_probs.mean()
                        
                        loss_mb = policy_loss + self.entropy_beta * entropy_term + value_loss
                        # Normalize across micro-batches
                        loss_mb = loss_mb / num_mbs
                    
                    loss_mb.backward()
                except torch.OutOfMemoryError:
                    # Reduce micro-batch size adaptively on OOM and retry this slice
                    new_mbs = max(256, micro_bs // 2)
                    if new_mbs == micro_bs:
                        raise
                    micro_bs = new_mbs
                    num_mbs = (N + micro_bs - 1) // micro_bs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    continue  # retry with smaller micro-batch from same mb_start
                
                # Advance window only after successful backward
                mb_start = mb_end
            
            # Clip and step once per epoch
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.optimizer.step()
        
        # Update the learning rate
        self.scheduler.step()
        
        # Update total training steps
        self.total_steps += 1
        
        # Clear the buffer
        self.buffer.clear()
        
        # Return average losses for logging
        return {
            'actor_loss': policy_loss.item(),
            'critic_loss': value_loss.item(),
            'entropy': entropy_term.item()
        }

    def get_legal_actions(self, state: np.ndarray) -> np.ndarray:
        """
        Get legal actions mask for a given state.
        This is a simplified version - in practice, this should be determined by game rules.
        """
        # For MCTS purposes, assume all actions are legal except when we have specific constraints
        legal = np.ones(self.action_size, dtype=np.float32)
        
        # Action 0: Fold - always legal
        # Action 1: Call - always legal  
        # Actions 2+: Raise bins - assume legal for MCTS
        
        return legal

    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """
        Get action probabilities for a given state using current policy.
        Used by MCTS for policy evaluation.
        """
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        self.actor.eval()
        with torch.no_grad():
            logits = self.actor(state_tensor)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        
        return probs

    def apply_mcts_improvement(self, improvement: Dict):
        """
        Apply MCTS-derived policy improvements to the agent.
        This blends MCTS results with current policy for better decision making.
        """
        if not improvement or 'policy' not in improvement:
            return
            
        mcts_policy = improvement['policy']
        logger.info(f"Applying MCTS improvement with {len(mcts_policy)} actions")
        
        # For now, we just log the improvement
        # Future enhancement: Could implement policy blending
        # Example: self._blend_policies(mcts_policy, blend_weight=0.3)
        
        # Log improvement details
        best_action = max(mcts_policy.items(), key=lambda x: x[1]) if mcts_policy else None
        if best_action:
            logger.info(f"MCTS suggests action {best_action[0]} with confidence {best_action[1]:.3f}")
    
    def _blend_policies(self, mcts_policy: Dict[int, float], blend_weight: float = 0.3):
        """
        Blend MCTS policy with current policy.
        This is a placeholder for future policy blending implementation.
        """
        # Future: Implement policy blending logic
        # - Extract current policy from actor network
        # - Blend with MCTS policy using blend_weight
        # - Update actor network parameters
        pass
    
    def _is_continuation_bet(self, game_state, opponent) -> bool:
        """
        Determine if the current betting situation represents a continuation bet opportunity.
        
        A continuation bet occurs when:
        1. We raised preflop and are first to act postflop
        2. No opponent has raised after our preflop raise
        """
        if not hasattr(game_state, 'betting_history') or not game_state.betting_history:
            return False
        
        # Must be postflop
        if len(game_state.community_cards) < 3:
            return False
        
        # Check if we were the preflop raiser
        preflop_actions = [action for action in game_state.betting_history 
                          if len(game_state.community_cards) == 0 or len(action) >= 3]
        
        # Look for our raise in preflop
        our_preflop_raise = False
        for action in preflop_actions:
            if len(action) >= 3:
                player_id, action_type, bet_size = action[0], action[1], action[2]
                if player_id == self.player_id and action_type == Action.RAISE.value:
                    our_preflop_raise = True
                    break
        
        if not our_preflop_raise:
            return False
        
        # Check if we're first to act postflop (no raises after our preflop raise)
        postflop_actions = [action for action in game_state.betting_history 
                           if len(game_state.community_cards) >= 3]
        
        for action in postflop_actions:
            if len(action) >= 3:
                player_id, action_type, bet_size = action[0], action[1], action[2]
                if player_id != self.player_id and action_type == Action.RAISE.value:
                    return False  # Someone else raised, not a c-bet situation
        
        return True
    
    def _raise_amount_to_bin(self, raise_amount, pot_size, call_amount, min_raise, stack_size):
        """Convert continuous raise amount to discrete bin index."""
        if raise_amount <= 0:
            return 0  # Invalid raise
            
        # Calculate raise as fraction of pot
        raise_fraction = raise_amount / pot_size if pot_size > 0 else 1.0
        
        # Find closest bin
        min_diff = float('inf')
        best_bin = 0
        
        for i, bin_val in enumerate(self._raise_bins):
            if bin_val == 'all_in':
                bin_fraction = (stack_size + raise_amount) / pot_size if pot_size > 0 else 1.0
            else:
                bin_fraction = bin_val
                
            diff = abs(raise_fraction - bin_fraction)
            if diff < min_diff:
                min_diff = diff
                best_bin = i
                
        return best_bin