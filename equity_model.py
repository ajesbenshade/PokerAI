import torch.nn as nn
import torch
from typing import List
import random
from datatypes import Card
from config import Config

class EquityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(7 * (13 + 4), 512),  # Card embeds
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Equity 0-1
        )

    def forward(self, x):
        return self.fc(x)

class LearnedAbstraction(nn.Module):
    """Dynamic learned bucketing for game tree abstraction using auto-encoder"""

    def __init__(self, input_size=169 + 5*(13+4), embed_size=64, num_buckets=20):
        super().__init__()
        # Encoder: Compress raw state to embedding
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, embed_size)
        )

        # Decoder: Reconstruct for auto-encoder training
        self.decoder = nn.Sequential(
            nn.Linear(embed_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, input_size),
            nn.Sigmoid()  # Output in [0,1] for normalized inputs
        )

        # Bucket classifier: Soft assignment to buckets
        self.bucket_fc = nn.Sequential(
            nn.Linear(embed_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_buckets)
        )

        self.num_buckets = num_buckets
        self.embed_size = embed_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for bucketing"""
        embed = self.encoder(x)
        bucket_logits = self.bucket_fc(embed)
        return bucket_logits

    def get_bucket(self, x: torch.Tensor) -> int:
        """Get hard bucket assignment"""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=-1).item()

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Auto-encoder reconstruction for training"""
        embed = self.encoder(x)
        return self.decoder(embed)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get embedding for clustering/analysis"""
        with torch.no_grad():
            return self.encoder(x)

class HandAbstraction:
    """Enhanced hand abstraction with learned bucketing"""

    def __init__(self, learned_model: LearnedAbstraction = None):
        self.learned_model = None  # Disable learned model for now, use fallback
        self.card_to_idx = self._build_card_mapping()
        self.bucket_cache = {}  # Cache for performance

    def _build_card_mapping(self):
        """Map cards to indices for embedding"""
        mapping = {}
        idx = 0
        for suit in ['h', 'd', 'c', 's']:
            for rank in ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']:
                mapping[f'{rank}{suit}'] = idx
                idx += 1
        return mapping

    def _embed_cards(self, cards: List[Card]) -> torch.Tensor:
        """Convert cards to embedding vector"""
        embed = torch.zeros(7 * (13 + 4), device=Config.DEVICE)  # 7 cards max, 17 features each

        for i, card in enumerate(cards[:7]):  # Max 7 cards (2 hole + 5 board)
            if card:
                value_map = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T'}
                rank_str = value_map.get(card.value, str(card.value))
                card_str = f'{rank_str}{card.suit.name.lower()}'
                if card_str in self.card_to_idx:
                    base_idx = i * 17
                    embed[base_idx + self.card_to_idx[card_str]] = 1.0  # One-hot for card
                    embed[base_idx + 13 + ['h', 'd', 'c', 's'].index(card.suit.name.lower())] = 1.0  # Suit one-hot

        return embed

    def get_bucket(self, hole_cards: List[Card], community_cards: List[Card],
                   num_opponents: int = 1, bet_history: List = None) -> int:
        """Get abstraction bucket for current state"""

        # Create cache key
        cache_key = (tuple(str(c) for c in hole_cards if c),
                    tuple(str(c) for c in community_cards if c),
                    num_opponents)

        if cache_key in self.bucket_cache:
            return self.bucket_cache[cache_key]

        if self.learned_model is None:
            # Fallback to simple bucketing based on equity
            equity = self._estimate_simple_equity(hole_cards, community_cards, num_opponents)
            bucket = min(int(equity * 20), 19)  # 20 buckets based on equity
        else:
            # Use learned abstraction
            input_vec = self._embed_cards(hole_cards + community_cards)
            bucket = self.learned_model.get_bucket(input_vec.unsqueeze(0))

        self.bucket_cache[cache_key] = bucket
        return bucket

    def _estimate_simple_equity(self, hole_cards: List[Card], community_cards: List[Card],
                               num_opponents: int) -> float:
        """Simple equity estimation for fallback bucketing"""
        # Basic equity estimation - can be improved
        if not hole_cards:
            return 0.5

        # Count high cards
        high_cards = sum(1 for card in hole_cards if card.value in [14, 13, 12, 11])
        pair_bonus = 1.0 if len(hole_cards) == 2 and hole_cards[0].value == hole_cards[1].value else 0.0

        base_equity = 0.5 + (high_cards * 0.05) + (pair_bonus * 0.1)
        return min(max(base_equity, 0.1), 0.9)

    def clear_cache(self):
        """Clear bucket cache"""
        self.bucket_cache.clear()

# Global instances - create on CPU, move to GPU on first access
equity_net = EquityNet()
learned_abstraction = LearnedAbstraction()
hand_abstraction = HandAbstraction(learned_abstraction)

def _ensure_gpu(model):
    """Ensure model is on GPU device"""
    if hasattr(model, 'to') and str(next(model.parameters()).device) != str(Config.DEVICE):
        model.to(Config.DEVICE)
    return model

# Lazy GPU assignment
def get_equity_net():
    return _ensure_gpu(equity_net)

def get_learned_abstraction():
    return _ensure_gpu(learned_abstraction)

def get_hand_abstraction():
    return _ensure_gpu(hand_abstraction)

class GPUEquityEvaluator:
    """GPU-accelerated equity evaluation with batch processing and caching"""
    
    def __init__(self):
        self.device = Config.DEVICE
        self.batch_size = Config.EQUITY_BATCH_SIZE
        self.fallback_threshold = Config.EQUITY_GPU_FALLBACK_THRESHOLD
        self.cache = {}  # RAM cache for equity values
        self.cache_size = Config.EQUITY_CACHE_SIZE
        self.model = EquityNet().to(self.device)
        self.model.eval()
        
    def _get_cache_key(self, hole_cards: List[Card], community_cards: List[Card], num_opponents: int) -> str:
        """Generate cache key for equity lookup"""
        hole_str = ''.join([f'{c.value}{c.suit.value}' for c in hole_cards])
        comm_str = ''.join([f'{c.value}{c.suit.value}' for c in community_cards])
        return f"{hole_str}_{comm_str}_{num_opponents}"
    
    def _embed_cards_batch(self, hole_cards_batch: List[List[Card]], community_cards_batch: List[List[Card]]) -> torch.Tensor:
        """Convert batch of card lists to embedding tensor"""
        batch_size = len(hole_cards_batch)
        embed = torch.zeros(batch_size, 7 * (13 + 4), device=self.device)
        
        for b in range(batch_size):
            hole_cards = hole_cards_batch[b]
            community_cards = community_cards_batch[b]
            all_cards = hole_cards + community_cards
            
            for i, card in enumerate(all_cards[:7]):  # Max 7 cards
                if card:
                    base_idx = i * 17
                    rank_idx = card.value - 2  # 2-14 -> 0-12
                    suit_idx = card.suit.value  # 0-3
                    if 0 <= rank_idx < 13:
                        embed[b, base_idx + rank_idx] = 1.0
                    if 0 <= suit_idx < 4:
                        embed[b, base_idx + 13 + suit_idx] = 1.0
        
        return embed
    
    def estimate_equity_batch(self, hole_cards_batch: List[List[Card]], 
                             community_cards_batch: List[List[Card]], 
                             num_opponents_batch: List[int]) -> torch.Tensor:
        """Batch equity estimation with GPU acceleration"""
        batch_size = len(hole_cards_batch)
        
        # Fallback to CPU for small batches to avoid ROCm overhead
        if batch_size < self.fallback_threshold:
            return self._cpu_fallback(hole_cards_batch, community_cards_batch, num_opponents_batch)
        
        # Check cache first
        results = []
        uncached_indices = []
        uncached_hole = []
        uncached_comm = []
        uncached_opp = []
        
        for i in range(batch_size):
            cache_key = self._get_cache_key(hole_cards_batch[i], community_cards_batch[i], num_opponents_batch[i])
            if cache_key in self.cache:
                results.append(self.cache[cache_key])
            else:
                results.append(None)
                uncached_indices.append(i)
                uncached_hole.append(hole_cards_batch[i])
                uncached_comm.append(community_cards_batch[i])
                uncached_opp.append(num_opponents_batch[i])
        
        # Process uncached items in batch
        if uncached_indices:
            embed = self._embed_cards_batch(uncached_hole, uncached_comm)
            
            with torch.no_grad():
                nn_equities = self.model(embed).squeeze(-1).cpu()
            
            # Store in cache
            for i, idx in enumerate(uncached_indices):
                cache_key = self._get_cache_key(uncached_hole[i], uncached_comm[i], uncached_opp[i])
                equity_val = nn_equities[i].item()
                self.cache[cache_key] = equity_val
                results[idx] = equity_val
                
                # Cache size management
                if len(self.cache) > self.cache_size:
                    # Remove oldest entries (simple FIFO)
                    oldest_keys = list(self.cache.keys())[:self.cache_size // 10]
                    for key in oldest_keys:
                        del self.cache[key]
        
        return torch.tensor(results, device=self.device)
    
    def _cpu_fallback(self, hole_cards_batch: List[List[Card]], 
                     community_cards_batch: List[List[Card]], 
                     num_opponents_batch: List[int]) -> torch.Tensor:
        """CPU fallback for small batches"""
        results = []
        for hole_cards, community_cards, num_opponents in zip(hole_cards_batch, community_cards_batch, num_opponents_batch):
            # Use existing estimate_equity function
            from utils import estimate_equity
            equity = estimate_equity(hole_cards, community_cards, num_opponents)
            results.append(equity)
        return torch.tensor(results, device=self.device)
    
    def quick_simulate_batch(self, hole_cards_batch: List[List[Card]], 
                           community_cards_batch: List[List[Card]], 
                           num_opponents_batch: List[int],
                           pot_size_batch: List[float], 
                           stack_batch: List[float],
                           call_amount_batch: List[float], 
                           min_raise_batch: List[float],
                           action_idx_batch: List[int],
                           raise_amount_batch: List[float]) -> torch.Tensor:
        """Batch quick simulation for action value estimation"""
        batch_size = len(hole_cards_batch)
        
        # Get batch equities
        equities = self.estimate_equity_batch(hole_cards_batch, community_cards_batch, num_opponents_batch)
        
        # Vectorized reward calculation
        pot_sizes = torch.tensor(pot_size_batch, device=self.device)
        stacks = torch.tensor(stack_batch, device=self.device)
        call_amounts = torch.tensor(call_amount_batch, device=self.device)
        min_raises = torch.tensor(min_raise_batch, device=self.device)
        action_indices = torch.tensor(action_idx_batch, device=self.device)
        raise_amounts = torch.tensor([r if r is not None else 0.0 for r in raise_amount_batch], device=self.device)
        
        # Fold action
        fold_mask = action_indices == 0
        rewards = torch.where(fold_mask, -call_amounts, torch.zeros_like(call_amounts))
        
        # Call action
        call_mask = action_indices == 1
        call_rewards = equities * pot_sizes - (1 - equities) * call_amounts
        rewards = torch.where(call_mask, call_rewards, rewards)
        
        # Raise action
        raise_mask = action_indices >= 2
        pot_multiplier = torch.where(raise_amounts > 0, 1.5, 1.2)
        new_pots = pot_sizes * pot_multiplier
        raise_costs = call_amounts + torch.where(raise_amounts > 0, raise_amounts, min_raises)
        raise_equities = torch.clamp(equities * 1.1, max=0.95)
        raise_rewards = raise_equities * new_pots - (1 - raise_equities) * raise_costs
        rewards = torch.where(raise_mask, raise_rewards, rewards)
        
        return rewards

class LearnedAbstractionTrainer:
    """Offline training for learned abstraction model"""
    
    def __init__(self):
        self.device = Config.DEVICE
        self.model = LearnedAbstraction(
            input_size=169 + 5,  # Range vector + essential features
            embed_size=Config.LEARNED_ABSTRACTION_DIM,
            num_buckets=Config.HAND_ABSTRACTION_BUCKETS
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=Config.LEARNED_ABSTRACTION_LEARNING_RATE
        )
        self.criterion = nn.CrossEntropyLoss()
        
    def generate_training_data(self, num_hands: int = Config.LEARNED_ABSTRACTION_TRAINING_HANDS):
        """Generate synthetic training data for abstraction learning"""
        from utils import create_deck, evaluate_hand
        
        print(f"Generating {num_hands} training hands for learned abstraction...")
        
        training_data = []
        deck = create_deck()
        
        for i in range(num_hands):
            if i % 100000 == 0:
                print(f"Generated {i}/{num_hands} hands...")
                
            # Shuffle deck
            random.shuffle(deck)
            
            # Deal hands
            hole_cards = [deck[0], deck[1]]
            community_cards = deck[2:7]  # Flop, turn, river
            
            # Evaluate hand strength
            hand_rank = evaluate_hand(hole_cards, community_cards)
            
            # Create range vector (simplified uniform for training)
            range_vec = torch.ones(169, device=self.device) / 169
            
            # Essential features
            essential_features = torch.tensor([
                0.5,  # Normalized pot size
                0.5,  # Normalized stack
                0.2,  # Pot odds
                1.0,  # Street progress (river)
                0.5   # Position
            ], device=self.device)
            
            # Combine features
            state_vec = torch.cat([range_vec, essential_features])
            
            # Target bucket based on hand strength (stronger hands get higher buckets)
            target_bucket = min(int((1.0 - hand_rank / 7462) * Config.HAND_ABSTRACTION_BUCKETS), 
                              Config.HAND_ABSTRACTION_BUCKETS - 1)
            
            training_data.append((state_vec, target_bucket))
            
            # Limit memory usage
            if len(training_data) >= 10000:
                yield training_data
                training_data = []
        
        if training_data:
            yield training_data
    
    def train(self, num_epochs: int = Config.LEARNED_ABSTRACTION_TRAIN_EPOCHS):
        """Train the learned abstraction model"""
        print(f"Training learned abstraction for {num_epochs} epochs...")
        
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_data in self.generate_training_data():
                batch_states = []
                batch_targets = []
                
                for state, target in batch_data:
                    batch_states.append(state)
                    batch_targets.append(target)
                
                if not batch_states:
                    continue
                
                # Convert to tensors
                states = torch.stack(batch_states)
                targets = torch.tensor(batch_targets, device=self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                logits = self.model(states)
                
                # Compute losses
                ce_loss = self.criterion(logits, targets)
                
                # Reconstruction loss
                reconstructed = self.model.reconstruct(states)
                recon_loss = nn.MSELoss()(reconstructed, states)
                
                # Total loss
                total_loss = ce_loss + Config.LEARNED_ABSTRACTION_RECONSTRUCTION_WEIGHT * recon_loss
                
                # Backward pass
                total_loss.backward()
                self.optimizer.step()
                
                epoch_loss += total_loss.item()
                batch_count += 1
                
                # Log progress
                if batch_count % 100 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_count}, Loss: {total_loss.item():.4f}")
            
            avg_loss = epoch_loss / max(batch_count, 1)
            print(f"Epoch {epoch+1}/{num_epochs} completed. Average loss: {avg_loss:.4f}")
        
        # Save trained model
        torch.save(self.model.state_dict(), 'learned_abstraction.pth')
        print("Learned abstraction model saved to learned_abstraction.pth")
        
        return self.model
