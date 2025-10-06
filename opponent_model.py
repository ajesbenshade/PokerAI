"""
Opponent Modeling System for PokerAI
Neural network-based opponent range prediction with contrastive learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import logging
from collections import defaultdict
import threading
import os
import pickle
from config import Config

logger = logging.getLogger(__name__)

class OpponentRangePredictor(nn.Module):
    """
    MLP-based opponent range predictor using betting history and game state.
    Predicts opponent hand distributions (169-dimensional range vectors).
    """

    def __init__(self, input_size: int = 169 + 5 + 20, hidden_size: int = 2048, output_size: int = 169):
        """
        Args:
            input_size: Combined size of range vector + game state + betting history
            hidden_size: Hidden layer size (2048 as requested)
            output_size: Output range vector size (169 hole card combinations)
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )

        # Range normalization layer (outputs probabilities)
        self.range_normalizer = nn.Softmax(dim=-1)

        # Apply torch.compile for ROCm optimization
        self._compile_model()

    def _compile_model(self):
        """Compile model for better ROCm performance"""
        try:
            self.forward = torch.compile(self.forward, mode='reduce-overhead')
            logger.info("OpponentRangePredictor compiled successfully")
        except Exception as e:
            logger.warning(f"torch.compile not available: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Predicted range probabilities of shape (batch_size, 169)
        """
        # Encode input to range logits
        range_logits = self.encoder(x)

        # Normalize to probabilities
        range_probs = self.range_normalizer(range_logits)

        return range_probs

    def predict_range(self, game_state: torch.Tensor, betting_history: torch.Tensor,
                     current_range: torch.Tensor) -> torch.Tensor:
        """
        Predict opponent range given current game state.

        Args:
            game_state: Game state features (pot, position, etc.)
            betting_history: Encoded betting history
            current_range: Current estimated range

        Returns:
            Predicted opponent range probabilities
        """
        # Concatenate inputs
        combined_input = torch.cat([current_range, game_state, betting_history], dim=-1)

        # Ensure correct input size
        if combined_input.size(-1) != self.input_size:
            # Pad or truncate as needed
            if combined_input.size(-1) < self.input_size:
                padding = torch.zeros(*combined_input.shape[:-1], self.input_size - combined_input.size(-1),
                                    device=combined_input.device)
                combined_input = torch.cat([combined_input, padding], dim=-1)
            else:
                combined_input = combined_input[..., :self.input_size]

        return self.forward(combined_input)


class ContrastiveOpponentTrainer:
    """
    Trainer for opponent modeling using contrastive loss.
    Learns to distinguish between actual opponent hands and counterfactuals.
    """

    def __init__(self, model: OpponentRangePredictor, device: str = None):
        self.model = model
        self.device = device or Config.DEVICE
        self.model.to(self.device)

        # Optimizer with weight decay for regularization
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNED_ABSTRACTION_LEARNING_RATE,
            weight_decay=1e-4
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.99
        )

        # Contrastive loss temperature
        self.temperature = 0.1

        # Training statistics
        self.training_step = 0

    def contrastive_loss(self, predicted_ranges: torch.Tensor,
                        actual_ranges: torch.Tensor,
                        negative_ranges: torch.Tensor) -> torch.Tensor:
        """
        Contrastive loss to learn opponent ranges.

        Args:
            predicted_ranges: Model predictions (batch_size, 169)
            actual_ranges: Ground truth opponent ranges (batch_size, 169)
            negative_ranges: Negative samples (batch_size, num_negatives, 169)

        Returns:
            Contrastive loss value
        """
        batch_size = predicted_ranges.size(0)
        num_negatives = negative_ranges.size(1)

        # Flatten for batch processing
        actual_flat = actual_ranges.view(batch_size, 1, -1)  # (batch, 1, 169)
        negatives_flat = negative_ranges.view(batch_size, num_negatives, -1)  # (batch, num_neg, 169)

        # Compute similarities
        pos_sim = F.cosine_similarity(predicted_ranges, actual_ranges, dim=-1)  # (batch,)
        neg_sim = F.cosine_similarity(
            predicted_ranges.unsqueeze(1).expand(-1, num_negatives, -1),
            negative_ranges,
            dim=-1
        )  # (batch, num_neg)

        # Contrastive loss (NT-Xent style)
        pos_logits = pos_sim / self.temperature
        neg_logits = neg_sim / self.temperature

        # Combine positive and negative logits
        all_logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)  # (batch, 1+num_neg)

        # Labels: positive is at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # Cross-entropy loss
        loss = F.cross_entropy(all_logits, labels)

        return loss

    def train_step(self, game_states: torch.Tensor, betting_histories: torch.Tensor,
                  current_ranges: torch.Tensor, actual_ranges: torch.Tensor,
                  negative_ranges: torch.Tensor) -> Dict[str, float]:
        """
        Single training step for opponent modeling.

        Args:
            game_states: Game state features
            betting_histories: Encoded betting history
            current_ranges: Current range estimates
            actual_ranges: Ground truth opponent ranges
            negative_ranges: Negative samples for contrastive learning

        Returns:
            Training metrics
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        predicted_ranges = self.model.predict_range(game_states, betting_histories, current_ranges)

        # Compute contrastive loss
        loss = self.contrastive_loss(predicted_ranges, actual_ranges, negative_ranges)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # Optimizer step
        self.optimizer.step()

        # Update learning rate
        self.scheduler.step()

        # Compute metrics
        with torch.no_grad():
            # KL divergence between predicted and actual
            kl_div = F.kl_div(predicted_ranges.log(), actual_ranges, reduction='batchmean')

            # Accuracy metrics
            pred_max_idx = predicted_ranges.argmax(dim=-1)
            actual_max_idx = actual_ranges.argmax(dim=-1)
            accuracy = (pred_max_idx == actual_max_idx).float().mean()

        self.training_step += 1

        return {
            'loss': loss.item(),
            'kl_div': kl_div.item(),
            'accuracy': accuracy.item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_step': self.training_step,
            'temperature': self.temperature
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved opponent model checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.training_step = checkpoint['training_step']
            self.temperature = checkpoint.get('temperature', 0.1)
            logger.info(f"Loaded opponent model checkpoint from {path}")
        else:
            logger.warning(f"Checkpoint not found: {path}")


class OpponentModelManager:
    """
    Manages opponent modeling for multiple players with disk caching.
    Handles memory-efficient storage of large range data.
    """

    def __init__(self, cache_dir: str = "./opponent_cache"):
        self.cache_dir = cache_dir
        self.models = {}  # player_id -> OpponentRangePredictor
        self.trainers = {}  # player_id -> ContrastiveOpponentTrainer
        self.range_cache = {}  # Memory cache for recent ranges
        self.cache_size = 10000  # Max entries in memory cache
        self.disk_cache = {}  # Track what's on disk

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        # Thread lock for thread safety
        self.lock = threading.Lock()

    def get_or_create_model(self, player_id: int) -> OpponentRangePredictor:
        """Get or create opponent model for a player"""
        with self.lock:
            if player_id not in self.models:
                model = OpponentRangePredictor()
                self.models[player_id] = model

                # Create trainer
                trainer = ContrastiveOpponentTrainer(model)
                self.trainers[player_id] = trainer

                # Try to load existing checkpoint
                checkpoint_path = os.path.join(self.cache_dir, f"opponent_model_{player_id}.pth")
                trainer.load_checkpoint(checkpoint_path)

                logger.info(f"Created opponent model for player {player_id}")

            return self.models[player_id]

    def update_range_estimate(self, player_id: int, game_state: torch.Tensor,
                            betting_history: torch.Tensor, current_range: torch.Tensor) -> torch.Tensor:
        """
        Update and predict opponent range for a player.

        Args:
            player_id: ID of the opponent
            game_state: Current game state features
            betting_history: Encoded betting history
            current_range: Current range estimate

        Returns:
            Updated range prediction
        """
        model = self.get_or_create_model(player_id)

        with torch.no_grad():
            model.eval()
            predicted_range = model.predict_range(game_state, betting_history, current_range)

        # Cache the result
        cache_key = f"{player_id}_{hash(game_state.cpu().numpy().tobytes()):x}"
        self._update_cache(cache_key, predicted_range)

        return predicted_range

    def train_on_trajectory(self, player_id: int, trajectory_data: Dict[str, torch.Tensor]):
        """
        Train opponent model on trajectory data.

        Args:
            trajectory_data: Dict containing game_states, betting_histories,
                           current_ranges, actual_ranges, negative_ranges
        """
        trainer = self.trainers.get(player_id)
        if trainer is None:
            return

        try:
            metrics = trainer.train_step(
                trajectory_data['game_states'],
                trajectory_data['betting_histories'],
                trajectory_data['current_ranges'],
                trajectory_data['actual_ranges'],
                trajectory_data['negative_ranges']
            )

            # Log training progress
            if trainer.training_step % 100 == 0:
                logger.info(f"Opponent model {player_id} - Step {trainer.training_step}: "
                          f"Loss: {metrics['loss']:.4f}, KL: {metrics['kl_div']:.4f}, "
                          f"Acc: {metrics['accuracy']:.3f}")

            # Periodic checkpointing
            if trainer.training_step % 1000 == 0:
                checkpoint_path = os.path.join(self.cache_dir, f"opponent_model_{player_id}.pth")
                trainer.save_checkpoint(checkpoint_path)

        except Exception as e:
            logger.warning(f"Failed to train opponent model {player_id}: {e}")

    def _update_cache(self, key: str, value: torch.Tensor):
        """Update memory cache with LRU eviction"""
        with self.lock:
            if len(self.range_cache) >= self.cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.range_cache))
                del self.range_cache[oldest_key]

            self.range_cache[key] = value

    def get_cached_range(self, key: str) -> Optional[torch.Tensor]:
        """Get range from memory cache"""
        with self.lock:
            return self.range_cache.get(key)

    def clear_cache(self):
        """Clear memory cache"""
        with self.lock:
            self.range_cache.clear()
            logger.info("Cleared opponent model cache")

    def save_all_models(self):
        """Save all opponent models to disk"""
        for player_id, trainer in self.trainers.items():
            checkpoint_path = os.path.join(self.cache_dir, f"opponent_model_{player_id}.pth")
            trainer.save_checkpoint(checkpoint_path)

    def load_all_models(self):
        """Load all opponent models from disk"""
        for player_id in range(Config.TOTAL_PLAYERS):
            checkpoint_path = os.path.join(self.cache_dir, f"opponent_model_{player_id}.pth")
            if os.path.exists(checkpoint_path):
                model = self.get_or_create_model(player_id)
                trainer = self.trainers[player_id]
                trainer.load_checkpoint(checkpoint_path)


# Global opponent model manager instance
_opponent_model_manager = None

def get_opponent_model_manager() -> OpponentModelManager:
    """Get global opponent model manager instance"""
    global _opponent_model_manager
    if _opponent_model_manager is None:
        _opponent_model_manager = OpponentModelManager()
    return _opponent_model_manager
