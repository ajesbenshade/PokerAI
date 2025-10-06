import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class Actor(nn.Module):
    """Actor network for PPO policy with Transformer layer for betting history processing."""
    def __init__(self, state_size: int):
        super().__init__()
        self.state_size = state_size

        # Input layer
        self.input_layer = nn.Linear(state_size, Config.ACTOR_HIDDEN_SIZE)

        # Lightweight Transformer layer for betting history processing
        self.transformer = nn.TransformerEncoderLayer(
            d_model=Config.ACTOR_HIDDEN_SIZE,
            nhead=2,  # 2 heads as requested
            dim_feedforward=512,  # dim=512 as requested
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer, num_layers=1)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            nn.Linear(Config.ACTOR_HIDDEN_SIZE, Config.ACTOR_HIDDEN_SIZE)
            for _ in range(Config.NUM_RES_BLOCKS)
        ])

        # Output layers
        self.policy_head = nn.Linear(Config.ACTOR_HIDDEN_SIZE, Config.ACTION_SIZE)

        self.dropout = nn.Dropout(Config.RESIDUAL_DROPOUT)

        # Apply torch.compile for ROCm optimization
        self.compile_model()

    def compile_model(self):
        """Compile the model for better ROCm performance."""
        try:
            # Temporarily disable torch.compile to avoid initialization issues
            # self.forward = torch.compile(self.forward, mode='reduce-overhead')
            print("Actor model compilation disabled for stable initialization")
        except Exception as e:
            print(f"torch.compile not available for Actor: {e}")

    def forward(self, x):
        # Input processing with optimized activation
        x = F.gelu(self.input_layer(x))  # Use GELU instead of ReLU for better performance

        # Add sequence dimension for transformer (batch_size, seq_len=1, hidden_size)
        x = x.unsqueeze(1)

        # Apply lightweight transformer for betting history processing
        x = self.transformer_encoder(x)

        # Remove sequence dimension
        x = x.squeeze(1)

        # Residual blocks with optimized operations
        for res_block in self.res_blocks:
            residual = x
            x = F.gelu(res_block(x))
            x = self.dropout(x)
            x = x + residual  # Residual connection

        # Output discrete logits for all actions (fold, call, raise_bins)
        logits = self.policy_head(x)
        return logits

    def get_log_probs(self, states, actions):
        """Get log probabilities for given actions, handling raise amounts as part of actions."""
        logits = self(states)

        # Handle different action tensor shapes
        if actions.dim() == 1:
            action_indices = actions.long()
        elif actions.dim() == 2:
            if actions.shape[1] == 1:
                action_indices = actions.squeeze(1).long()
            else:
                # Handle [action_idx, raise_amount] format - use action_idx for probability
                action_indices = actions[:, 0].long()
        else:
            action_indices = actions.long()

        # Get log probabilities for all actions
        log_probs_all = F.log_softmax(logits, dim=-1)
        
        # Gather the log probabilities for the selected actions
        # action_indices should be 1D, log_probs_all is (batch_size, num_actions)
        selected_log_probs = log_probs_all[torch.arange(log_probs_all.size(0)), action_indices]

        # Entropy calculation
        entropy = -(log_probs_all * torch.exp(log_probs_all)).sum(dim=-1)

        return selected_log_probs, entropy


class Critic(nn.Module):
    """Critic network for PPO value estimation."""
    def __init__(self, state_size: int):
        super().__init__()
        self.state_size = state_size

        # Input layer
        self.input_layer = nn.Linear(state_size, Config.CRITIC_HIDDEN_SIZE)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            nn.Linear(Config.CRITIC_HIDDEN_SIZE, Config.CRITIC_HIDDEN_SIZE)
            for _ in range(Config.NUM_RES_BLOCKS)
        ])

        # Output layer
        self.output_layer = nn.Linear(Config.CRITIC_HIDDEN_SIZE, 1)

        self.dropout = nn.Dropout(Config.RESIDUAL_DROPOUT)

        # Apply torch.compile for ROCm optimization
        self.compile_model()

    def compile_model(self):
        """Compile the model for better ROCm performance."""
        try:
            # Temporarily disable torch.compile to avoid initialization issues
            # self.forward = torch.compile(self.forward, mode='reduce-overhead')
            print("Critic model compilation disabled for stable initialization")
        except Exception as e:
            print(f"torch.compile not available for Critic: {e}")

    def forward(self, x):
        # Input processing with optimized activation
        x = F.gelu(self.input_layer(x))  # Use GELU instead of ReLU for better performance

        # Residual blocks with optimized operations
        for res_block in self.res_blocks:
            residual = x
            x = F.gelu(res_block(x))
            x = self.dropout(x)
            x = x + residual  # Residual connection

        # Output
        value = self.output_layer(x)
        return value
