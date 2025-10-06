# --- Memory/allocator configuration (must be set before importing torch) ---
import os
# Help ROCm allocator reduce fragmentation and handle varying alloc sizes
os.environ.setdefault(
    'PYTORCH_HIP_ALLOC_CONF',
    'garbage_collection_threshold:0.6,expandable_segments:True,max_split_size_mb:256,roundup_power2_divisions:16'
)

# Additional memory optimizations for 20GB VRAM
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:256')

# --- Minimal warnings setup ---
import warnings
warnings.filterwarnings("ignore", message="expandable_segments not supported on this platform")
warnings.filterwarnings("ignore", message="Synchronization debug mode is a prototype feature")

import torch

import logging
import csv
import gc
import traceback
from torch.utils.tensorboard import SummaryWriter
import torch
import random
import numpy as np
from typing import Optional, Dict, List
from math import inf
import copy
import multiprocessing as mp
from multiprocessing import Manager
import torch.backends.cudnn as cudnn
import psutil
import pickle
import tempfile
import glob
import time
from collections import defaultdict

from config import Config, Action
from datatypes import Player
from rl import ActorCriticAgent
from game import simulate_hand, simulate_hand_gpu
from utils import get_vram_usage, AbstractionCache, estimate_equity_batch_gpu, quick_simulate_batch_gpu, ParallelMCTS
from openCFR.minimizers import CFRPlus  # Or MCCFR_Outcome for Monte-Carlo variance reduction
from openCFR.Trainer import Trainer
from game import GTOHoldEm
from gto import evaluate_exploitability, run_validation_with_exploitability
from equity_model import GPUEquityEvaluator, LearnedAbstractionTrainer, get_hand_abstraction, LearnedAbstraction

class EloTracker:
    """Simple ELO rating tracker for population-based evolution."""
    
    def __init__(self, player_ids):
        self.ratings = {pid: 1200 for pid in player_ids}
    
    def update(self, rewards):
        """Update ratings based on game outcomes."""
        # Simple ELO update - in a real implementation this would be more sophisticated
        if rewards:
            winner = max(rewards.keys(), key=lambda k: rewards[k])
            loser = min(rewards.keys(), key=lambda k: rewards[k])
            
            # Simple rating adjustment
            self.ratings[winner] += 10
            self.ratings[loser] -= 10

# Global shared model manager for multiprocessing optimization
class SharedModelManager:
    """Manages shared models in multiprocessing to avoid serialization overhead."""
    
    def __init__(self):
        # Use direct shared memory instead of Manager to avoid AuthenticationString issues
        self.shared_models = {}
        self.shared_optimizers = {}
        self.shared_schedulers = {}
        self.model_locks = {}
        
    def register_model(self, player_id: int, agent: ActorCriticAgent):
        """Register a model for shared memory access using torch.share_memory()."""
        # Move models to CPU for sharing
        agent.actor.cpu()
        agent.critic.cpu()
        
        # Use torch.share_memory() for efficient IPC
        try:
            agent.actor.share_memory()
            agent.critic.share_memory()
            self.shared_models[player_id] = {
                'actor': agent.actor,
                'critic': agent.critic,
                'device': 'cpu'
            }
            logger.info(f"Successfully shared model for player {player_id} using torch.share_memory()")
        except Exception as e:
            # Fallback to state dict approach if share_memory fails
            logger.warning(f"torch.share_memory() failed for player {player_id}: {e}. Using state dict fallback.")
            self.shared_models[player_id] = {
                'actor_state': agent.actor.state_dict(),
                'critic_state': agent.critic.state_dict(),
                'device': 'cpu'
            }
        
        # Store optimizer and scheduler state as regular dicts (not shared)
        self.shared_optimizers[player_id] = {
            'state': agent.optimizer.state_dict(),
            'param_groups': agent.optimizer.param_groups
        }
        
        # Store scheduler state as regular dict (not shared)
        self.shared_schedulers[player_id] = {
            'state': agent.scheduler.state_dict(),
            'last_epoch': agent.scheduler.last_epoch
        }
        
        # Use multiprocessing lock for thread safety
        self.model_locks[player_id] = mp.Lock()
        
    def _share_model(self, model: torch.nn.Module):
        """Convert model parameters to shared memory."""
        # Removed - no longer using shared memory
        return model
        
    def get_model_state(self, player_id: int):
        """Get current model state for a player."""
        return self.shared_models[player_id]
        
    def update_model(self, player_id: int, new_actor_state: dict, new_critic_state: dict):
        """Update shared model with new state dicts."""
        with self.model_locks[player_id]:
            shared_model = self.shared_models[player_id]
            if 'actor' in shared_model:
                # Shared tensor approach - update in-place
                shared_model['actor'].load_state_dict(new_actor_state)
                shared_model['critic'].load_state_dict(new_critic_state)
            else:
                # State dict approach
                shared_model['actor_state'] = new_actor_state
                shared_model['critic_state'] = new_critic_state
            
    def get_simulation_agent(self, player_id: int, device: str = 'cpu'):
        """Get a simulation agent using shared model."""
        shared_model = self.shared_models[player_id]
        
        # Create new agent instance
        agent = ActorCriticAgent.create_simulation_agent(device=device)
        
        if 'actor' in shared_model:
            # Shared tensor approach - copy parameters directly
            agent.actor.load_state_dict(shared_model['actor'].state_dict())
            agent.critic.load_state_dict(shared_model['critic'].state_dict())
        else:
            # State dict approach
            agent.actor.load_state_dict(shared_model['actor_state'])
            agent.critic.load_state_dict(shared_model['critic_state'])
        
        # Move to correct device if needed
        if device != 'cpu' and str(agent.device) != device:
            agent.actor.to(device)
            agent.critic.to(device)
            
        agent.actor.eval()
        agent.critic.eval()
        
        return agent
    
    def update_all_models(self, agents: List[ActorCriticAgent]):
        """Update all shared models with current agent states."""
        for i, agent in enumerate(agents):
            if i in self.shared_models:
                self.update_model(i, agent.actor.state_dict(), agent.critic.state_dict())

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Environment Setup for PyTorch ---
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)
cudnn.benchmark = True
torch.cuda.empty_cache()

# Check for GPU availability (ROCm or CUDA)
has_gpu = torch.cuda.is_available() or (hasattr(torch.version, 'hip') and torch.version.hip is not None)
torch.cuda.empty_cache()
if has_gpu:
    try:
        torch.cuda.set_device(0)  # Try CUDA interface first (works with ROCm)
    except RuntimeError:
        # If CUDA interface fails, try ROCm-specific initialization
        pass  # ROCm should handle device selection automatically

# Enhanced memory optimization for 20GB VRAM and 64GB RAM
if has_gpu:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Enable memory efficient attention if available
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
    
    # ROCm-specific optimizations
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        # Pre-allocate some GPU memory to reduce fragmentation
        try:
            _ = torch.zeros(1, device='cuda:0')
            torch.cuda.empty_cache()
        except:
            pass

# Memory usage check before starting training
ram_percent = psutil.virtual_memory().percent
if ram_percent > 70:
    print(f"Warning: High RAM usage at startup: {ram_percent:.1f}%")
    if ram_percent > 85:
        print("Error: RAM usage too high at startup. Please free up memory before training.")
        exit(1)

if has_gpu:
    try:
        vram_usage = get_vram_usage()
        if vram_usage > 12.0:
            print(f"Warning: High VRAM usage at startup: {vram_usage:.1f}GB")
            if vram_usage > 16.0:
                print("Error: VRAM usage too high at startup. Please free up GPU memory before training.")
                exit(1)
    except:
        pass

# Optimize data loading and preprocessing
if has_gpu:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Enable async data loading if available
if hasattr(torch, 'cuda') and torch.cuda.is_available():
    torch.cuda.set_sync_debug_mode(0)  # Disable sync debug for performance

# Global memory monitor
memory_stats = {'peak_ram': 0, 'peak_vram': 0, 'gc_calls': 0}

def save_checkpoint(agent: ActorCriticAgent, player_id: int, hand_number: int, is_best: bool = False, phase: int = 1):
    """
    Save the state of an agent to disk for checkpointing or best model tracking.

    Args:
        agent (ActorCriticAgent): The agent whose state to save.
        player_id (int): The unique ID of the player/agent.
        hand_number (int): The current hand number (for resuming training).
        is_best (bool): If True, saves as the best model; else as a regular checkpoint.
        phase (int): The current training phase (1 for CFR, 2 for RL-CFR hybrid).
    """
    state = {
        "actor_state_dict": agent.actor.state_dict(),
        "critic_state_dict": agent.critic.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "scheduler_state_dict": agent.scheduler.state_dict(),
        "hand": hand_number,
        "total_steps": agent.total_steps,
        "phase": phase,
    }
    prefix = "best_model" if is_best else "checkpoint"
    filename = f"{prefix}_player_{player_id}.pth"
    torch.save(state, filename)
    logger.info(f"Saved {'best' if is_best else 'checkpoint'} for player {player_id} at hand {hand_number} (Phase {phase}).")

def load_checkpoint(agent: ActorCriticAgent, player_id: int, is_best: bool = False) -> tuple[int, int]:
    """
    Load an agent's state from disk if a checkpoint exists.

    Args:
        agent (ActorCriticAgent): The agent to load state into.
        player_id (int): The unique ID of the player/agent.
        is_best (bool): If True, loads the best model; else the latest checkpoint.

    Returns:
        tuple[int, int]: The hand number to resume from and the training phase (0 if no checkpoint found, phase defaults to 1).
    """
    filename = f"{'best_model' if is_best else 'checkpoint'}_player_{player_id}.pth"
    if not os.path.exists(filename):
        return 0, 1
    
    try:
        checkpoint = torch.load(filename, map_location=Config.DEVICE)
    except Exception as e:
        logger.error(f"Error opening checkpoint file for player {player_id}: {e}")
        return 0, 1
    # Attempt to load the actor and critic weights.  When the
    # architecture has changed (e.g. different hidden sizes) this can
    # raise a ``RuntimeError``.  To ensure training can continue
    # unhindered we catch such errors and skip loading the mismatched
    # weights.  Other components like the optimizer and scheduler are
    # loaded only if the networks match.
    start_hand = checkpoint.get('hand', 0)
    try:
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        actor_loaded = True
    except RuntimeError as e:
        logger.warning(f"Shape mismatch loading actor weights for player {player_id}: {e}\n"
                       "Proceeding with randomly initialised actor.")
        actor_loaded = False
    try:
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        critic_loaded = True
    except RuntimeError as e:
        logger.warning(f"Shape mismatch loading critic weights for player {player_id}: {e}\n"
                       "Proceeding with randomly initialised critic.")
        critic_loaded = False
    # Only restore the optimiser and scheduler state if both actor and critic
    # loaded successfully; otherwise their parameter groups will not
    # match the stored state dict structure.
    if actor_loaded and critic_loaded:
        try:
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            agent.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            logger.warning(f"Error restoring optimizer/scheduler for player {player_id}: {e}")
    # Always restore total_steps regardless of network load status
    agent.total_steps = checkpoint.get('total_steps', 0)
    phase = checkpoint.get('phase', 1)  # Default to phase 1 if not found
    logger.info(f"Loaded checkpoint for player {player_id} from hand {start_hand} (Phase {phase}).")
    return start_hand + 1, phase

def validate_agents(players: List[Player]) -> float:
    """
    Evaluate current agents against GTO baseline bot in 2-player games.

    Args:
        players (List[Player]): The list of RL agents to validate.

    Returns:
        float: Average win rate of RL agents vs. GTO bot.
    """
    total_win_rate = 0
    games_per_agent = Config.VALIDATION_GAMES // len(players)
    gto_bot = gto_baseline_bot()
    
    for p in players:
        win_rate = 0
        for _ in range(games_per_agent):
            try:
                agent_player = p.copy(sim=True)
            except AttributeError:
                agent_player = copy.deepcopy(p)
            bot_player = copy.deepcopy(gto_bot)
            
            # Alternate dealer
            dealer_idx = random.randint(0, 1)
            rewards, _ = simulate_hand([agent_player, bot_player], dealer_idx)
            
            agent_reward = rewards.get(agent_player.player_id, 0)
            bot_reward = rewards.get(bot_player.player_id, 0)
            if agent_reward > bot_reward:
                win_rate += 1
        
        win_rate /= games_per_agent
        total_win_rate += win_rate
    
    avg_win_rate = total_win_rate / len(players)
    logger.info(f"Validation win rate vs GTO bot: {avg_win_rate:.3f}")
    return avg_win_rate

def gto_baseline_bot():
    """Simple baseline bot for validation."""
    from rl import ActorCriticAgent
    bot = Player(player_id=-1, agent=ActorCriticAgent.create_simulation_agent())
    return bot

# Global instance
shared_model_manager = None

# Per-worker cached resources to avoid repeated allocations and large arg pickles
_worker_cache = {
    'version': None,
    'agents_by_pid': {},
    'gpu_evaluator': None,
    'abstraction_cache': None,
}

def _load_worker_models_if_needed(snapshot_path: str, version: int, player_ids: list):
    """Load or update per-worker cached agents from a CPU snapshot file on version change."""
    global _worker_cache
    if _worker_cache['version'] != version:
        # Create agents if not exist
        for pid in player_ids:
            if pid not in _worker_cache['agents_by_pid']:
                _worker_cache['agents_by_pid'][pid] = ActorCriticAgent.create_simulation_agent(device='cpu')
        # Load state dicts from snapshot
        try:
            state = torch.load(snapshot_path, map_location='cpu')
            for pid, sd in state.items():
                agent = _worker_cache['agents_by_pid'].get(pid)
                if agent is None:
                    agent = ActorCriticAgent.create_simulation_agent(device='cpu')
                    _worker_cache['agents_by_pid'][pid] = agent
                agent.actor.load_state_dict(sd['actor'])
                agent.critic.load_state_dict(sd['critic'])
            _worker_cache['version'] = version
        except Exception as e:
            # If snapshot load fails, keep previous models
            print(f"Worker failed to load snapshot {snapshot_path}: {e}")

def _get_worker_equity_and_cache():
    """Get or create per-worker GPUEquityEvaluator and AbstractionCache (CPU-safe)."""
    global _worker_cache
    if _worker_cache['gpu_evaluator'] is None:
        from equity_model import GPUEquityEvaluator
        _worker_cache['gpu_evaluator'] = GPUEquityEvaluator()
    if _worker_cache['abstraction_cache'] is None:
        from utils import AbstractionCache
        _worker_cache['abstraction_cache'] = AbstractionCache()
    return _worker_cache['gpu_evaluator'], _worker_cache['abstraction_cache']

def run_simulation(args):
    """Optimized simulation using per-worker cached agents and lightweight snapshots."""
    player_ids, dealer_index, hand, snapshot_path, version = args
    
    # Force CPU in workers to prevent HIP fragmentation from many contexts
    device = 'cpu'

    # Set deterministic seeds for all random number generators
    worker_seed = Config.SEED + dealer_index
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if device != 'cpu':
        torch.cuda.manual_seed(worker_seed)
        torch.cuda.manual_seed_all(worker_seed)  # For multi-GPU consistency

    # Load/update worker-cached models once per version
    _load_worker_models_if_needed(snapshot_path, version, player_ids)
    agents = [_worker_cache['agents_by_pid'][pid] for pid in player_ids]

    # Initialize per-worker singletons (CPU-safe)
    gpu_evaluator, abstraction_cache = _get_worker_equity_and_cache()

    sim_players = [Player(pid, agent=agents[i]) for i, pid in enumerate(player_ids)]
    rewards, trajectories = simulate_hand_gpu(sim_players, dealer_index, hand, gpu_evaluator, abstraction_cache)

    # No GPU allocations in workers; nothing to free

    return rewards, trajectories

def collect_raise_diversity_stats(trajectories):
    """Collect statistics on raise bin usage from trajectories"""
    raise_bins_used = defaultdict(int)
    total_actions = 0
    
    for player_trajectories in trajectories.values():
        for transition in player_trajectories:
            state, action_idx, log_prob, reward, done, value = transition
            total_actions += 1
            
            # Count raise bin usage (actions 2-11 are raise bins)
            if 2 <= action_idx < Config.ACTION_SIZE:
                bin_idx = action_idx - 2
                raise_bins_used[bin_idx] += 1
    
    # Calculate diversity metrics
    diversity_stats = {}
    if total_actions > 0:
        diversity_stats['total_actions'] = total_actions
        diversity_stats['unique_bins_used'] = len(raise_bins_used)
        diversity_stats['most_used_bin'] = max(raise_bins_used.keys(), key=lambda k: raise_bins_used[k]) if raise_bins_used else -1
        diversity_stats['bin_usage_entropy'] = 0.0
        
        # Calculate entropy of bin usage
        if raise_bins_used:
            total_raises = sum(raise_bins_used.values())
            probs = [count / total_raises for count in raise_bins_used.values()]
            diversity_stats['bin_usage_entropy'] = -sum(p * np.log(p + 1e-10) for p in probs)
    
    return diversity_stats

# Global instance
shared_model_manager = None

def main():
    """
    Main training loop for PokerAI RL agents.
    Handles initialization, parallel simulation, training, validation, population-based evolution,
    checkpointing, and logging. Designed for robust, scalable, and efficient RL training.
    """
    global shared_model_manager
    shared_model_manager = None
    
    # GPU environment variables are now set at the top of the file
    
    # Optimize multiprocessing for better GPU utilization
    try:
        mp.set_start_method('spawn', force=True)  # 'spawn' is most compatible with ROCm/HIP
    except RuntimeError:
        pass  # Already set

    # --- Initialization ---
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    random.seed(Config.SEED)
    writer = SummaryWriter()
    # Probe GPU availability once in the main process
    gpu_ok = _probe_gpu_once()
    logger.info(f"GPU probe: {'available' if gpu_ok else 'unavailable'}")
    metrics_file = "training_metrics.csv"
    TOTAL_PLAYERS = Config.TOTAL_PLAYERS
    
    # No shared replay buffer needed for on-policy PPO
    replay_buffer = None
    
    # Create agents and players
    agents = [ActorCriticAgent(Config.STATE_SIZE, Config.ACTION_SIZE) for _ in range(TOTAL_PLAYERS)]
    players = [Player(i, agent=agents[i]) for i, agent in enumerate(agents)]
    
    # Debug: Print main agent device information
    logger.info(f"Main agent device: {agents[0].device}")
    logger.info(f"Main agent actor device: {next(agents[0].actor.parameters()).device}")
    logger.info(f"Main agent critic device: {next(agents[0].critic.parameters()).device}")

    elo_tracker = EloTracker([p.player_id for p in players])
    best_model_performance = {p.player_id: -inf for p in players}

    # Load checkpoints
    start_hand = 0
    start_phase = 1
    for p in players:
        hand, phase = load_checkpoint(p.agent, p.player_id)
        start_hand = max(start_hand, hand)
        start_phase = max(start_phase, phase)  # Use the highest phase found
    
    # Update shared models after loading checkpoints
    # if shared_model_manager:
    #     shared_model_manager.update_all_models(agents)
    #     logger.info("Shared models updated after checkpoint loading")
    
    # Initialize shared model manager for multiprocessing optimization
    # if shared_model_manager is None:
    #     shared_model_manager = SharedModelManager()
    #     for i, agent in enumerate(agents):
    #         shared_model_manager.register_model(i, agent)
    #     logger.info("Shared model manager initialized with all agent models")
    
    # Setup metrics CSV
    with open(metrics_file, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["hand", "avg_reward", "ppo_actor_loss", "ppo_critic_loss", "ppo_entropy", "cfr_actor_loss", "cfr_critic_loss", "cfr_entropy", "ram_gb", "vram_gb", "cpu_usage", "unique_bins_used", "raise_entropy", "abstraction_reconstruction_error", "abstraction_bucket_diversity", "abstraction_training_data_size"])

    # Initialize training metrics tracking
    start_time = time.time()
    last_log_time = start_time
    total_actor_loss = 0.0
    total_critic_loss = 0.0
    total_entropy_loss = 0.0
    total_rewards = 0.0
    ppo_update_count = 0
    PROGRESS_LOG_INTERVAL = 100  # Log every 100 hands

    # --- Main Training Loop ---
    try:
        optimal_workers = min(8, Config.MAX_WORKERS)  # Keep modest to reduce memory pressure
        # Use default Python executable for multiprocessing (avoid version conflicts)
        # mp.set_executable('/home/aaron/PokerAI/ROCm/bin/python')
        # Prevent workers from creating HIP contexts by default
        os.environ['POKERAI_DISABLE_GPU'] = '1'
        
        # Initialize GPU equity evaluator for the main process
        from equity_model import GPUEquityEvaluator
        from utils import AbstractionCache
        gpu_equity_evaluator = GPUEquityEvaluator()
        abstraction_cache = AbstractionCache()
        
        # Initialize parallel MCTS for policy improvement
        parallel_mcts = ParallelMCTS() if False else None  # Temporarily disable MCTS
        
        with mp.Pool(processes=optimal_workers) as pool:
            logger.info(f"Started training with {optimal_workers} worker processes")

            # --- Phased Training Configuration ---
            # Phase 1: 10k hands of preflop CFR training
            # Phase 2: Hybrid RL-CFR training for the remaining hands
            PHASE_1_HANDS = 10000  # 10k hands of pure CFR (adjusted for current config)
            current_training_phase = start_phase  # Use loaded phase
            cfr_training_enabled = (current_training_phase == 1)
            rl_training_enabled = (current_training_phase == 2)

            # Initialize game object for CFR training
            game = GTOHoldEm(num_players=TOTAL_PLAYERS)

            for hand in range(start_hand, Config.NUM_HANDS):
                # --- Phase Transition Logic ---
                if hand == PHASE_1_HANDS and current_training_phase == 1:
                    logger.info("=== PHASE TRANSITION: Switching from CFR to Hybrid RL-CFR ===")
                    current_training_phase = 2
                    cfr_training_enabled = False
                    rl_training_enabled = True
                    logger.info("Phase 2: Hybrid RL-CFR training enabled")

                # --- Curriculum Learning: Dynamic number of players ---
                # More sophisticated player count scaling
                if hand < 2000:
                    num_players_this_hand = 2  # Start simple
                elif hand < 5000:
                    num_players_this_hand = 3
                elif hand < 10000:
                    num_players_this_hand = 4
                elif hand < 20000:
                    num_players_this_hand = 5
                else:
                    num_players_this_hand = TOTAL_PLAYERS  # Full complexity

                current_players = players[:num_players_this_hand]
                player_ids = [p.player_id for p in current_players]

                # Write a lightweight CPU snapshot once per hand version for workers to read
                snapshot_dir = os.path.join(tempfile.gettempdir(), 'pokerai_snapshots')
                os.makedirs(snapshot_dir, exist_ok=True)
                # Clean up older snapshots periodically
                if hand % 200 == 0:
                    for f in glob.glob(os.path.join(snapshot_dir, 'models_*.pt')):
                        try:
                            os.remove(f)
                        except Exception:
                            pass
                snapshot_path = os.path.join(snapshot_dir, f'models_{hand}.pt')
                snapshot_state = {}
                for p in current_players:
                    # Move to CPU tensors to reduce IPC payload size
                    actor_sd = {k: v.detach().cpu() for k, v in p.agent.actor.state_dict().items()}
                    critic_sd = {k: v.detach().cpu() for k, v in p.agent.critic.state_dict().items()}
                    snapshot_state[p.player_id] = {'actor': actor_sd, 'critic': critic_sd}
                torch.save(snapshot_state, snapshot_path)

                # Build simulation arguments with snapshot path and version number (hand id)
                sim_args = []
                for sim_idx in range(Config.NUM_SIMULATIONS):
                    dealer_idx = (hand * Config.NUM_SIMULATIONS + sim_idx) % len(current_players)
                    sim_args.append((player_ids, dealer_idx, hand, snapshot_path, hand))

                # Execute simulations with timeout and error handling
                # Memory cleanup before starting simulations
                if hand % 10 == 0:  # Clean memory every 10 hands
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                simulation_rewards = []
                all_trajectories = {}  # Store all trajectories for abstraction training
                try:
                    start_sim = time.time()
                    results = pool.map_async(run_simulation, sim_args)
                    # Reduced timeout for faster failure recovery
                    results = results.get(timeout=120)  # 2 minute timeout instead of 5
                    sim_time = time.time() - start_sim

                    # Only log simulation time every PROGRESS_LOG_INTERVAL hands to reduce noise
                    if hand % PROGRESS_LOG_INTERVAL == 0:
                        logger.info(f"Simulation time: {sim_time:.2f}s for {len(results)} sims ({sim_time/len(results):.3f}s per sim)")

                    for rewards, trajectories in results:
                        if rewards:  # Check for valid results
                            simulation_rewards.append(np.mean(list(rewards.values())))
                        
                        # Add trajectories to buffers
                        for p in current_players:
                            player_trajectories = trajectories.get(p.player_id, [])
                            for transition in player_trajectories:
                                state, discrete_action, log_prob, reward, done, value = transition
                                agents[p.player_id].buffer.add(
                                    state, discrete_action, log_prob, reward, done, value
                                )
                            
                            # Store trajectories for abstraction training
                            if p.player_id not in all_trajectories:
                                all_trajectories[p.player_id] = []
                            all_trajectories[p.player_id].extend(trajectories.get(p.player_id, []))
                        
                        # Update Elo ratings
                        elo_tracker.update(rewards)
                        
                except Exception as e:
                    logger.error(f"Simulation pool error: {e}")
                    # Continue with empty results rather than crashing
                    simulation_rewards = [0.0] * Config.NUM_SIMULATIONS
                    all_trajectories = {p.player_id: [] for p in current_players}
                
                # Use parallel MCTS for enhanced policy improvement
                if Config.MCTS_PARALLEL_ROLLOUTS and parallel_mcts and all_trajectories:
                    try:
                        # Extract recent states from trajectories for MCTS improvement
                        recent_states = []
                        for player_trajectories in all_trajectories.values():
                            for transition in player_trajectories[-10:]:  # Last 10 transitions
                                state, _, _, _, _, _ = transition
                                recent_states.append(state)
                        
                        if recent_states:
                            # Perform parallel MCTS rollouts on recent states
                            mcts_improvements = parallel_mcts.improve_policies(
                                agents, recent_states, num_rollouts=Config.MCTS_SIM_DEPTH
                            )
                            
                            # Apply MCTS improvements to agents
                            for pid, improvement in mcts_improvements.items():
                                if pid in agents:
                                    agents[pid].apply_mcts_improvement(improvement)
                            
                            logger.info(f"Applied MCTS improvements to {len(mcts_improvements)} agents")
                            
                    except Exception as e:
                        logger.warning(f"MCTS improvement failed: {e}")
                
                avg_reward = np.mean(simulation_rewards) if simulation_rewards else 0
                # Accumulate reward for periodic logging
                total_rewards += avg_reward
                # --- Training Phase with Optimized Frequency ---
                actor_loss, critic_loss, entropy = 0, 0, 0
                
                # --- PPO Training Phase ---
                if len(agents[0].buffer) >= Config.BATCH_SIZE:
                    start_train = time.time()
                    logger.info(f"PPO Training triggered at hand {hand} (buffer size: {len(agents[0].buffer)})")

                    # Proactive VRAM cleanup before training
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                    # Train all agents with OOM handling
                    train_results = []
                    for i, agent in enumerate(agents):
                        try:
                            result = agent.update()
                            if result is None:
                                result = {'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0}
                        except torch.OutOfMemoryError:
                            logger.warning("OOM during PPO update; reducing batch and clearing cache")
                            # Aggressively shrink batch and clear cache
                            Config.BATCH_SIZE = max(1024, Config.BATCH_SIZE // 2)
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                torch.cuda.ipc_collect()
                            # Retry once
                            result = agent.update()
                            if result is None:
                                result = {'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0}
                        train_results.append(result)

                    # Average results across agents
                    actor_loss = np.mean([r['actor_loss'] for r in train_results])
                    critic_loss = np.mean([r['critic_loss'] for r in train_results])
                    entropy = np.mean([r['entropy'] for r in train_results])

                    # Accumulate losses for periodic logging
                    total_actor_loss += actor_loss
                    total_critic_loss += critic_loss
                    total_entropy_loss += entropy
                    ppo_update_count += 1

                    # Update shared models with new parameters
                    for i, agent in enumerate(agents):
                        shared_model_manager.update_model(
                            i, 
                            agent.actor.state_dict(), 
                            agent.critic.state_dict()
                        )

                    train_time = time.time() - start_train
                    logger.info(f"PPO Training time: {train_time:.2f}s")
                else:
                    actor_loss, critic_loss, entropy = 0.0, 0.0, 0.0

                # --- CFR Training Phase (Phased) ---
                cfr_actor_loss, cfr_critic_loss, cfr_entropy = 0, 0, 0
                if cfr_training_enabled and hand % (Config.TRAINING_FREQUENCY * 2) == 0:  # Less frequent CFR training
                    start_cfr = time.time()
                    logger.info(f"CFR Training triggered at hand {hand} (Phase {current_training_phase})")
                    try:
                        # Use enhanced CFR with bucketed abstraction
                        from gto import run_mccfr_training

                        # Adjust CFR iterations based on phase
                        if current_training_phase == 1:
                            cfr_iterations = min(Config.ENHANCED_TRAINING_ITERATIONS, 10000)
                        else:
                            cfr_iterations = min(Config.ENHANCED_TRAINING_ITERATIONS // 2, 5000)

                        cfr_results = run_mccfr_training(game, num_iterations=cfr_iterations)
                        # For phase 1, we focus on building strong preflop strategies
                        if current_training_phase == 1:
                            logger.info(f"Phase 1 CFR: Building preflop strategies with {cfr_iterations} iterations")
                        else:
                            logger.info(f"Phase 2 CFR: Refining strategies with {cfr_iterations} iterations")
                    except Exception as e:
                        logger.warning(f"CFR training failed: {e}")
                    cfr_time = time.time() - start_cfr
                    logger.info(f"CFR Training time: {cfr_time:.2f}s")
                else:
                    cfr_actor_loss, cfr_critic_loss, cfr_entropy = 0.0, 0.0, 0.0

                # --- Learned Abstraction Training Phase (Periodic) ---
                if Config.LEARNED_ABSTRACTION_ENABLED and hand % Config.LEARNED_ABSTRACTION_UPDATE_FREQUENCY == 0:
                    start_abstraction = time.time()
                    logger.info(f"Learned Abstraction Training triggered at hand {hand}")

                    try:
                        # Initialize abstraction model if not exists
                        if not hasattr(Config, '_learned_abstraction'):
                            Config._learned_abstraction = LearnedAbstraction(
                                input_size=Config.STATE_SIZE,
                                embed_size=Config.LEARNED_ABSTRACTION_DIM,
                                num_buckets=20
                            ).to(Config.DEVICE)

                        abstraction_model = Config._learned_abstraction

                        # Collect training data from recent trajectories
                        abstraction_training_data = []
                        for p in current_players:
                            player_trajectories = all_trajectories.get(p.player_id, [])
                            for transition in player_trajectories:
                                state, action_idx, log_prob, reward, done, value = transition
                                if not done:  # Only use non-terminal states
                                    abstraction_training_data.append(state)

                        if len(abstraction_training_data) >= Config.LEARNED_ABSTRACTION_BATCH_SIZE:
                            # Train abstraction model
                            optimizer = torch.optim.Adam(
                                abstraction_model.parameters(),
                                lr=Config.LEARNED_ABSTRACTION_LEARNING_RATE
                            )

                            # Convert to tensor batch
                            states_tensor = torch.stack(abstraction_training_data).to(Config.DEVICE)

                            # Training loop for abstraction
                            abstraction_model.train()
                            total_abstraction_loss = 0

                            for epoch in range(Config.LEARNED_ABSTRACTION_EPOCHS):
                                # Shuffle data
                                indices = torch.randperm(len(states_tensor))
                                batch_states = states_tensor[indices[:Config.LEARNED_ABSTRACTION_BATCH_SIZE]]

                                optimizer.zero_grad()

                                # Forward pass through abstraction
                                reconstructed = abstraction_model.reconstruct(batch_states)
                                embedding = abstraction_model.get_embedding(batch_states)
                                bucket_logits = abstraction_model.forward(batch_states)

                                # Calculate losses
                                reconstruction_loss = torch.nn.functional.mse_loss(reconstructed, batch_states)
                                clustering_loss = torch.nn.functional.cross_entropy(
                                    bucket_logits.view(-1, bucket_logits.size(-1)),
                                    torch.argmax(bucket_logits, dim=-1).view(-1)
                                )

                                # Entropy regularization for bucket diversity
                                bucket_probs = torch.softmax(bucket_logits, dim=-1)
                                entropy_loss = -torch.mean(torch.sum(bucket_probs * torch.log(bucket_probs + 1e-10), dim=-1))

                                # Combined loss
                                total_loss = (
                                    Config.LEARNED_ABSTRACTION_RECONSTRUCTION_WEIGHT * reconstruction_loss +
                                    Config.LEARNED_ABSTRACTION_CLUSTERING_WEIGHT * clustering_loss +
                                    Config.LEARNED_ABSTRACTION_ENTROPY_WEIGHT * entropy_loss
                                )

                                total_loss.backward()
                                optimizer.step()

                                total_abstraction_loss += total_loss.item()

                            avg_abstraction_loss = total_abstraction_loss / Config.LEARNED_ABSTRACTION_EPOCHS
                            logger.info(f"Abstraction training completed - Avg Loss: {avg_abstraction_loss:.6f}")

                            # Update game tree with new abstraction
                            if hasattr(GTOHoldEm, '_learned_abstraction'):
                                GTOHoldEm._learned_abstraction = abstraction_model
                                logger.info("Updated game tree with new learned abstraction")

                        else:
                            logger.warning(f"Insufficient training data for abstraction: {len(abstraction_training_data)} < {Config.LEARNED_ABSTRACTION_BATCH_SIZE}")

                    except Exception as e:
                        logger.warning(f"Learned abstraction training failed: {e}")

                    abstraction_time = time.time() - start_abstraction
                    logger.info(f"Abstraction Training time: {abstraction_time:.2f}s")

                # Smart memory cleanup - less frequent but more efficient
                memory_stats['gc_calls'] += 1
                if hand % 50 == 0:  # More frequent cleanup
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # Log memory stats periodically
                    if hand % 500 == 0:
                        logger.info(f"Memory stats - Peak RAM: {memory_stats['peak_ram']:.2f}GB, "
                                  f"Peak VRAM: {memory_stats['peak_vram']:.2f}GB, "
                                  f"GC calls: {memory_stats['gc_calls']}")
                else:
                    # Smarter exploration decay (removed for PPO)
                    pass

                # Improved memory management with hysteresis
                ram_percent = psutil.virtual_memory().percent
                vram_usage = get_vram_usage() if torch.cuda.is_available() else 0
                
                # Only reduce batch size if consistently high memory usage
                if ram_percent > 80 or vram_usage > 16:  # tighten thresholds a bit
                    Config.BATCH_SIZE = max(2048, Config.BATCH_SIZE // 2)
                    logger.warning(f"Critical memory usage; reduced batch to {Config.BATCH_SIZE}")
                elif ram_percent < 60 and vram_usage < 12 and Config.BATCH_SIZE < 6144:
                    # Gradually increase batch size when memory is available
                    Config.BATCH_SIZE = min(6144, Config.BATCH_SIZE * 2)
                    logger.info(f"Memory available; increased batch to {Config.BATCH_SIZE}")
                # --- Optimized Logging and Checkpointing ---
                log_frequency = 20 if hand < 1000 else 50  # More frequent logging early, less later
                
                if hand % log_frequency == 0:
                    ram_usage = psutil.virtual_memory().used / (1024 ** 3)
                    vram_usage = get_vram_usage()
                    cpu_usage = psutil.cpu_percent(interval=0.1)  # Quick CPU check
                    
                    # Collect raise diversity stats from trajectories
                    diversity_stats = collect_raise_diversity_stats(all_trajectories)
                    
                    # Log training progress every 100 hands
                    if hand % PROGRESS_LOG_INTERVAL == 0:
                        current_time = time.time()
                        elapsed_since_last = current_time - last_log_time
                        total_elapsed = current_time - start_time
                        
                        # Calculate rates
                        hands_per_sec_overall = hand / total_elapsed
                        hands_per_sec_recent = PROGRESS_LOG_INTERVAL / elapsed_since_last
                        
                        # Calculate averages
                        avg_actor_loss = total_actor_loss / max(ppo_update_count, 1)
                        avg_critic_loss = total_critic_loss / max(ppo_update_count, 1)
                        avg_entropy_loss = total_entropy_loss / max(ppo_update_count, 1)
                        avg_reward_recent = total_rewards / PROGRESS_LOG_INTERVAL
                        
                        # Current learning rate
                        current_lr = agents[0].optimizer.param_groups[0]['lr']
                        
                        logger.info(f"Training Progress - Hand {hand}/{Config.NUM_HANDS} ({hand/Config.NUM_HANDS*100:.1f}%) - Phase {current_training_phase}")
                        logger.info(f"  Phase: {'CFR' if cfr_training_enabled else 'Hybrid RL-CFR'} (Phase {current_training_phase})")
                        logger.info(f"  Rate: {hands_per_sec_recent:.2f} hands/sec (recent), {hands_per_sec_overall:.2f} hands/sec (overall)")
                        logger.info(f"  Losses: Actor={avg_actor_loss:.4f}, Critic={avg_critic_loss:.4f}, Entropy={avg_entropy_loss:.4f}")
                        logger.info(f"  Reward: {avg_reward_recent:.2f} (avg per hand)")
                        logger.info(f"  Learning Rate: {current_lr:.6f}")
                        
                        # Reset counters for next interval
                        last_log_time = current_time
                        total_actor_loss = 0.0
                        total_critic_loss = 0.0
                        total_entropy_loss = 0.0
                        total_rewards = 0.0
                        ppo_update_count = 0
                    
                    # Monitor resource usage
                    if vram_usage > 15.0:
                        logger.warning(f"High VRAM usage: {vram_usage:.2f}GB > 15GB threshold")
                    if cpu_usage > 80.0:
                        logger.warning(f"High CPU usage: {cpu_usage:.1f}% > 80% threshold")
                    
                    # Batch TensorBoard writes for better performance
                    writer.add_scalars("PPO_Training", {
                        "Reward": avg_reward,
                        "Actor_Loss": actor_loss,
                        "Critic_Loss": critic_loss,
                        "Entropy": entropy,
                    }, hand)
                    writer.add_scalars("CFR_Training", {
                        "Actor_Loss": cfr_actor_loss,
                        "Critic_Loss": cfr_critic_loss,
                        "Entropy": cfr_entropy,
                    }, hand)
                    writer.add_scalars("Training_Phase", {
                        "Current_Phase": current_training_phase,
                        "CFR_Enabled": 1.0 if cfr_training_enabled else 0.0,
                        "RL_Enabled": 1.0 if rl_training_enabled else 0.0,
                    }, hand)
                    
                    # Log abstraction metrics if enabled
                    if Config.LEARNED_ABSTRACTION_ENABLED and hasattr(Config, '_learned_abstraction'):
                        try:
                            abstraction_model = Config._learned_abstraction
                            abstraction_model.eval()
                            
                            # Sample some states for metrics
                            sample_states = torch.stack(abstraction_training_data[:100]).to(Config.DEVICE)
                            with torch.no_grad():
                                reconstructed, embedding, bucket_logits = abstraction_model(sample_states)
                                reconstruction_error = torch.nn.functional.mse_loss(reconstructed, sample_states).item()
                                bucket_diversity = torch.mean(torch.std(bucket_logits, dim=-1)).item()
                            
                            writer.add_scalars("Learned_Abstraction", {
                                "Reconstruction_Error": reconstruction_error,
                                "Bucket_Diversity": bucket_diversity,
                                "Training_Data_Size": len(abstraction_training_data)
                            }, hand)
                            
                            logger.info(f"Abstraction Metrics - Reconstruction Error: {reconstruction_error:.6f}, "
                                      f"Bucket Diversity: {bucket_diversity:.4f}")
                        except Exception as e:
                            logger.warning(f"Failed to log abstraction metrics: {e}")
                    
                    # Write to CSV with buffering
                    with open(metrics_file, "a", newline="") as csv_file:
                        csv_writer = csv.writer(csv_file)
                        
                        # Get abstraction metrics
                        abstraction_reconstruction_error = 0.0
                        abstraction_bucket_diversity = 0.0
                        abstraction_training_data_size = len(abstraction_training_data) if 'abstraction_training_data' in locals() else 0
                        
                        if Config.LEARNED_ABSTRACTION_ENABLED and hasattr(Config, '_learned_abstraction'):
                            try:
                                abstraction_model = Config._learned_abstraction
                                abstraction_model.eval()
                                
                                sample_states = torch.stack(abstraction_training_data[:min(100, len(abstraction_training_data))]).to(Config.DEVICE)
                                with torch.no_grad():
                                    reconstructed, embedding, bucket_logits = abstraction_model(sample_states)
                                    abstraction_reconstruction_error = torch.nn.functional.mse_loss(reconstructed, sample_states).item()
                                    abstraction_bucket_diversity = torch.mean(torch.std(bucket_logits, dim=-1)).item()
                            except Exception:
                                pass
                        
                        csv_writer.writerow([
                            hand, avg_reward, actor_loss, critic_loss, entropy, 
                            cfr_actor_loss, cfr_critic_loss, cfr_entropy, 
                            ram_usage, vram_usage, cpu_usage,
                            diversity_stats.get('unique_bins_used', 0),
                            diversity_stats.get('bin_usage_entropy', 0),
                            abstraction_reconstruction_error,
                            abstraction_training_data_size
                        ])
                # --- Optimized Validation and Checkpointing ---
                validation_interval = max(500, Config.VALIDATION_INTERVAL // 2)  # More frequent validation
                
                if hand > 0 and hand % validation_interval == 0:
                    current_performance = validate_agents(players)
                    writer.add_scalar("Validation/Win_Rate_vs_GTO", current_performance, hand)
                    writer.add_scalar("Validation/Exploitability", 0.5 - current_performance, hand)
                    if current_performance > best_model_performance[0]:
                        best_model_performance[0] = current_performance
                        logger.info(f"New best performance: {current_performance:.3f}")
                        # Save best models asynchronously
                        for p in players:
                            save_checkpoint(p.agent, p.player_id, hand, is_best=True)
                    
                    # Population-based evolution with improved logic
                    ratings = elo_tracker.ratings
                    weakest_pid = min(ratings, key=ratings.get)
                    strongest_pid = max(ratings, key=ratings.get)
                    
                    if weakest_pid != strongest_pid and ratings[strongest_pid] - ratings[weakest_pid] > 100:
                        logger.info(f"Evolving: replacing player {weakest_pid} (ELO: {ratings[weakest_pid]:.0f}) "
                                  f"with mutation of player {strongest_pid} (ELO: {ratings[strongest_pid]:.0f})")
                        
                        strongest_agent = players[strongest_pid].agent
                        weakest_agent = players[weakest_pid].agent
                        
                        # Smart mutation: copy and add noise
                        weakest_agent.actor.load_state_dict(strongest_agent.actor.state_dict())
                        weakest_agent.critic.load_state_dict(strongest_agent.critic.state_dict())
                        
                        # Adaptive mutation strength based on performance gap
                        mutation_strength = min(0.05, (ratings[strongest_pid] - ratings[weakest_pid]) / 1000)
                        with torch.no_grad():
                            for param in weakest_agent.actor.parameters():
                                noise = torch.randn_like(param) * mutation_strength
                                param.add_(noise.to(param.device))
                            for param in weakest_agent.critic.parameters():
                                noise = torch.randn_like(param) * mutation_strength
                                param.add_(noise.to(param.device))
                        
                        # Reset ELO for mutated agent
                        elo_tracker.ratings[weakest_pid] = 1200
                        
                        # Update shared model with mutated parameters
                        shared_model_manager.update_model(
                            weakest_pid,
                            weakest_agent.actor.state_dict(),
                            weakest_agent.critic.state_dict()
                        )
                
                # Less frequent regular checkpointing to reduce I/O
                checkpoint_interval = 5000 if hand > 10000 else 2000
                if hand > 0 and hand % checkpoint_interval == 0:
                    logger.info("Saving regular checkpoints...")
                    for p in players:
                        save_checkpoint(p.agent, p.player_id, hand, phase=current_training_phase)
                
                # Smart memory management with final cleanup
                if hand % 50 == 0:  # More frequent cleanup
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # Log memory stats periodically
                    if hand % 500 == 0:
                        logger.info(f"Memory stats - Peak RAM: {memory_stats['peak_ram']:.2f}GB, "
                                  f"Peak VRAM: {memory_stats['peak_vram']:.2f}GB, "
                                  f"GC calls: {memory_stats['gc_calls']}")
                
                # Emergency memory management
                if psutil.virtual_memory().percent > 85:
                    logger.warning("Emergency memory cleanup triggered")
                    gc.collect()
                    torch.cuda.empty_cache()
                    # Reduce worker count if memory is critically low
                    if Config.MAX_WORKERS > 4:
                        Config.MAX_WORKERS = max(4, Config.MAX_WORKERS // 2)
                        logger.warning(f"Reduced MAX_WORKERS to {Config.MAX_WORKERS} due to memory pressure")
                    # Also reduce batch sizes if VRAM is tight
                    if get_vram_usage() > 14.0:
                        Config.PPO_MICRO_BATCH = max(128, Config.PPO_MICRO_BATCH // 2)
                        Config.EQUITY_BATCH_SIZE = max(125, Config.EQUITY_BATCH_SIZE // 2)
                        logger.warning(f"Reduced batch sizes due to high VRAM usage: PPO_MICRO_BATCH={Config.PPO_MICRO_BATCH}, EQUITY_BATCH_SIZE={Config.EQUITY_BATCH_SIZE}")
                
    except Exception as e:
        logger.error(f"Exception in training loop: {e}")
        # Use current hand if available, otherwise use start_hand
        current_hand = locals().get('hand', start_hand)
        for p in players:
            save_checkpoint(p.agent, p.player_id, current_hand)
        raise
    finally:
        # Restore parent setting: allow GPU again
        if os.environ.get('POKERAI_DISABLE_GPU') == '1':
            os.environ.pop('POKERAI_DISABLE_GPU', None)
        # Small shutdown cleanup to quiet ROCm/CUDA shared-IPC warnings
        try:
            writer.flush()
        except Exception:
            pass
        try:
            writer.close()
        except Exception:
            pass
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        gc.collect()
        logger.info("Training finished.")

 

def get_vram_usage():
    """Get current VRAM usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0.0

def get_gpu_utilization():
    """Get current GPU utilization percentage."""
    try:
        import subprocess
        result = subprocess.run(['rocm-smi', '--showuse'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GPU%' in line:
                    # Parse GPU utilization from rocm-smi output
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'GPU%':
                            return float(parts[i-1].rstrip('%'))
    except:
        pass
    return 0.0

def _probe_gpu_once() -> bool:
    try:
        if not torch.cuda.is_available():
            return False
        torch.cuda.set_device(0)
        _ = torch.empty(1, device='cuda')
        return True
    except Exception:
        return False

def test_exploitability(num_hands: int = 1000):
    """Test exploitability with simulated hands."""
    logger.info(f"Testing exploitability with {num_hands} simulated hands...")
    
    # Create a simple test agent
    test_agent = ActorCriticAgent(Config.STATE_SIZE, Config.ACTION_SIZE)
    
    # Create a simple random agent for opponent
    class RandomAgent:
        def __init__(self):
            self.total_steps = 0
            
        def choose_action(self, state, legal_actions, player_id, **kwargs):
            # Choose random legal action
            legal_indices = np.where(legal_actions)[0]
            if len(legal_indices) == 0:
                # No legal actions, return fold
                return Action.FOLD.value, None, Action.FOLD.value, -10.0, 0.0
            
            discrete_action = np.random.choice(legal_indices)
            
            # Convert discrete action to game format
            from utils import interpret_discrete_action
            action_idx, raise_amount = interpret_discrete_action(
                discrete_action, 
                kwargs.get('pot_size', Config.BIG_BLIND * 3),
                kwargs.get('call_amount', 0),
                kwargs.get('min_raise', Config.BIG_BLIND),
                kwargs.get('stack', Config.INITIAL_STACK)
            )
            
            # Return random log_prob and value
            log_prob = np.log(1.0 / len(legal_indices))
            value = 0.0
            
            return action_idx, raise_amount, discrete_action, log_prob, value
    
    random_agent = RandomAgent()
    
    total_reward = 0
    wins = 0
    
    for hand_num in range(num_hands):
        # Create players with agents
        from datatypes import Player
        from game import simulate_hand
        
        player1 = Player(0, agent=test_agent)
        player2 = Player(1, agent=random_agent)
        
        try:
            rewards, _ = simulate_hand([player1, player2], dealer_idx=hand_num % 2)
            reward = rewards.get(0, 0)
            total_reward += reward
            if reward > 0:
                wins += 1
        except Exception as e:
            logger.warning(f"Hand {hand_num} failed: {e}")
            continue
    
    win_rate = wins / num_hands
    avg_reward = total_reward / num_hands
    
    logger.info("Exploitability Test Results:")
    logger.info(f"  Hands played: {num_hands}")
    logger.info(f"  Win rate: {win_rate:.3f}")
    logger.info(f"  Average reward: {avg_reward:.3f}")
    logger.info(f"  Estimated exploitability: {1 - win_rate:.3f} (lower is better)")
    
    return win_rate, avg_reward

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_exploitability(int(sys.argv[2]) if len(sys.argv) > 2 else 1000)
    else:
        main()

# Initialize GPU-accelerated equity evaluator
gpu_equity_evaluator = GPUEquityEvaluator()
logger.info("GPU Equity Evaluator initialized")
    
# Initialize abstraction cache
abstraction_cache = AbstractionCache()
logger.info(f"Abstraction cache initialized with capacity: {Config.ABSTRACTION_CACHE_SIZE}")
    
# Initialize parallel MCTS for CPU rollouts
parallel_mcts = ParallelMCTS(num_workers=Config.MCTS_CPU_WORKERS)
logger.info(f"Parallel MCTS initialized with {Config.MCTS_CPU_WORKERS} CPU workers")