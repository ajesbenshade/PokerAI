from enum import Enum
import os
import torch

class Suit(Enum):
    HEARTS = 0
    DIAMONDS = 1
    CLUBS = 2
    SPADES = 3

class Action(Enum):
    FOLD = 0
    CALL = 1
    RAISE = 2

class Config:
    """Central configuration constants for training and simulation.

    The defaults here have been tuned for a system equipped with a Ryzen 9
    7900X CPU (12 cores/24 threads) and an AMD Radeon RX‑7900XT GPU.  On
    machines with fewer resources you may wish to further reduce
    batch sizes and model dimensions.  Conversely, on machines with
    additional VRAM you can increase these values.  Whenever possible
    computations are dispatched to the GPU via the HIP runtime on
    compatible hardware.
    """
    # Explicitly target the first visible GPU.  When running on a
    # system with an AMD 7900XT this will leverage the ROCm backend via
    # `cuda` device strings.  If no GPU is available the code will
    # gracefully fall back to the CPU.  See train.py for additional
    # HIP/ROCm environment configuration.
    import torch
    
#     # Check for GPU availability by actually trying to create a GPU tensor
#     has_gpu = False
#     try:
#         test_tensor = torch.zeros(1).cuda()
#         has_gpu = True
#         print("GPU test successful - GPU is available")
#     except Exception as e:
#         print(f"GPU test failed - using CPU: {e}")
#         has_gpu = False
    
    # Determine GPU availability (workers can force CPU via env to avoid HIP context)
    disable_gpu = os.getenv('POKERAI_DISABLE_GPU') == '1'
    has_gpu = torch.cuda.is_available() and not disable_gpu
    DEVICE = torch.device("cuda:0" if has_gpu else "cpu")
    
    # Only print config info from main process, not workers
    import multiprocessing
    if multiprocessing.current_process().name == 'MainProcess':
        print(f"Config loading - CUDA available: {torch.cuda.is_available()}")
        print(f"Config loading - ROCm version: {torch.version.hip}")
        print(f"Config DEVICE set to: {DEVICE}")

    # Use bfloat16 by default to take advantage of the 7900XT's tensor cores.
    # bfloat16 offers improved dynamic range over float16 and is fully
    # supported on modern ROCm releases.  Should bfloat16 support be
    # unavailable at runtime, PyTorch will silently cast to float32.
    DTYPE = torch.bfloat16
    # Automatic mixed precision can greatly accelerate training on
    # hardware that supports it.  Keeping this enabled ensures that
    # high‑throughput operations such as matrix multiplications run
    # at lower precision while maintaining sufficient numerical fidelity.
    AMP_ENABLED = True  # Enable automatic mixed precision for faster training on 7900XT

    LR = 3e-4  # Learning rate for PPO optimization
    LR_DECAY = 0.999  # Learning rate decay factor
    LR_MIN = 1e-5  # Minimum learning rate
    GAMMA = 0.99  # Discount factor for rewards
    GAE_LAMBDA = 0.95  # Lambda for Generalized Advantage Estimation
    PPO_CLIP = 0.2  # Clipping parameter for PPO
    ENTROPY_BETA = 0.01  # Entropy coefficient for exploration
    MAX_GRAD_NORM = 0.5  # Maximum gradient norm for clipping
    # Base batch size for on‑policy PPO updates.  The default of
    # 16384 was chosen to saturate GPUs with ample memory; however it
    # can easily exceed the 20–24 GB VRAM available on consumer cards.
    # Reducing the batch to 8192 strikes a better balance between
    # throughput and memory consumption on a 7900XT.  Should VRAM
    # pressure increase during training (monitored in train.py), the
    # batch size will be halved automatically.
    BATCH_SIZE = 16384  # Increased from 8192 to maximize GPU utilization
    PPO_EPOCHS = 4  # Number of epochs per PPO update (reduced for faster training)
    TOTAL_PLAYERS = 8  # Increased to 8 for full table training
    NUM_SIMULATIONS = 16  # Increased to 16 for maximum GPU utilization

    # PPO optimisation steps per hand.  Increasing this value will
    # improve policy updates but also prolong each training iteration.
    NUM_TRAINING_STEPS = 4  # Number of training steps (legacy, not used in PPO)
    T_MAX = 20000  # Maximum timesteps for scheduler
    VALUE_CLIP_EPS = 0.2  # Value clipping epsilon (legacy)

    INITIAL_STACK = 1000  # Starting stack size for each player
    SMALL_BLIND = 10  # Small blind amount
    BIG_BLIND = 20  # Big blind amount
    NUM_HANDS = 200000  # Overnight training run - increased from 50000
    META_TOURNAMENT_GAMES = 5000  # Games for meta-tournament evaluation
    VALIDATION_GAMES = 100  # Number of games for validation
    MAX_BETTING_ROUNDS = 10  # Maximum betting rounds per hand
    MAX_OPPONENTS = 5  # Maximum number of opponents
    SEED = 42  # Random seed for reproducibility
    MCTS_SIM_DEPTH = 500  # MCTS simulation depth

    # Capacity of the on‑disk replay buffer.  Each entry consists of a
    # tuple of states, actions and returns and typically occupies
    # several kilobytes once pickled.  A capacity of five million
    # entries can exceed 30 GB on disk and impose significant
    # overhead on a system with 64 GB of RAM.  Reducing the capacity
    # to two million still provides a rich training history while
    # freeing memory and I/O bandwidth.
    REPLAY_BUFFER_CAPACITY = 750_000  # Capacity of replay buffer (legacy, not used in PPO)
    STATE_SIZE = 174  # 169 (range vector) + 5 (essential features)
    ACTION_SIZE = 3  # Number of possible actions (fold, call, raise)
    # Actor and critic hidden dimensions.  Deep models yield better
    # representations but consume VRAM quadratically.  Given the
    # throughput of the 7900XT, a hidden width of 4096 is sufficient
    # to model complex game states while reducing VRAM consumption.
    ACTOR_HIDDEN_SIZE = 4096  # Increased to 4096 for maximum GPU utilization on 7900XT
    CRITIC_HIDDEN_SIZE = 4096  # Increased to 4096 for maximum GPU utilization on 7900XT
    # Number of residual blocks in the actor/critic networks.  Fewer
    # blocks reduce depth and memory requirements.  Adjust upwards
    # cautiously if VRAM allows and improved performance is desired.
    NUM_RES_BLOCKS = 10  # Increased to 10 for deeper networks and better GPU utilization
    RESIDUAL_DROPOUT = 0.1  # Dropout rate in residual blocks

    # Opponent modeling parameters
    OPPONENT_DECAY = 0.95  # Decay factor for opponent range estimates
    OPPONENT_MODEL_UPDATE_FREQ = 100  # Update opponent models every N hands
    OPPONENT_RANGE_CACHE_SIZE = 50000  # Cache size for opponent ranges

    EXPLORATION_FACTOR = 0.15  # Exploration factor
    EXPLORATION_DECAY = 0.9995  # Exploration decay
    MIN_EXPLORATION = 0.01  # Minimum exploration

    VALIDATION_INTERVAL = 1000  # Hands between validations
    
    # Stack management during training
    # When True: stacks reset to INITIAL_STACK each hand (isolated hand analysis)
    # When False: stacks persist across hands (realistic chip dynamics)
    RESET_STACKS_EACH_HAND = False  # Default to persistent stacks for better GTO learning
    
    # Raise amount discretization for stable training
    # When True: snap raise amounts to nearest 5-chip increments
    # When False: allow continuous raise amounts
    DISCRETE_RAISE_GRID = True  # Enable discrete grid for stable training

    # Discrete raise bins for stable training and better convergence
    RAISE_BIN_FRACTIONS = [0.25, 0.33, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 'all_in']  # 10 bins
    ACTION_SIZE = 2 + len(RAISE_BIN_FRACTIONS)  # fold=0, call=1, raise_bins=2-11

    # Hardware optimization settings for 20GB VRAM and 64GB RAM
    GRADIENT_ACCUMULATION_STEPS = 2  # Steps to accumulate gradients
    MAX_WORKERS = 6  # Reduced from 12 to prevent memory issues
    PREFETCH_FACTOR = 4  # Prefetch factor for data loading
    MEMORY_POOL_SIZE = 2  # GB of GPU memory to reserve for PyTorch memory pool
    TRAINING_FREQUENCY = 50  # Hands between CFR training updates (reduced frequency for CPU-intensive CFR)
    
    # Reward shaping parameters for better credit assignment
    REWARD_SHAPING_ENABLED = True
    IMMEDIATE_REWARD_WEIGHT = 0.1  # Weight for immediate action rewards
    STREET_REWARD_WEIGHT = 0.3     # Weight for street-level rewards
    FINAL_REWARD_WEIGHT = 0.6      # Weight for final hand reward
    POT_ODDS_REWARD_SCALE = 0.01   # Scale factor for pot odds rewards
    POSITION_BONUS_SCALE = 0.005   # Bonus for good position play
    AGGRESSION_REWARD_SCALE = 0.002 # Reward for well-timed aggression
    SHAPING_DECAY_HANDS = 50000    # Hands over which shaping decays to 0
    
    # Lazy equity model initialization (CPU-only for multiprocessing safety)
    _equity_model = None
    
    @classmethod
    def get_equity_model(cls):
        """Lazy initialization of equity model on CPU for multiprocessing safety."""
        if cls._equity_model is None:
            from equity_model import EquityNet
            cls._equity_model = EquityNet()
            cls._equity_model.eval()  # Set to eval mode
        return cls._equity_model

    RAISE_BINS = 10  # Number of discrete raise amounts

    EXPLOITABILITY_EVAL_FREQUENCY = 10  # Evaluate exploitability every N CFR training cycles
    GPU_EXPLOITABILITY = True  # Use GPU for exploitability calculations when available
    
    # Hardware optimizations for exploitability evaluation
    EXPLOITABILITY_MAX_TREE_DEPTH = 15  # Maximum tree depth for BR calculations
    EXPLOITABILITY_MEMO_CACHE_SIZE = 100000  # Maximum cache size for memoization
    EXPLOITABILITY_CPU_PARALLEL = True  # Use CPU parallelization for large trees
    EXPLOITABILITY_GPU_BATCH_SIZE = 1250  # Further reduced for memory
    
    # Preflop hybrid RL-CFR system
    PREFLOP_USE_HYBRID_CFR = True  # Use RL-guided CFR for preflop instead of hardcoded charts
    PREFLOP_CFR_ITERATIONS = 5000  # CFR iterations for preflop subgame solving
    PREFLOP_RL_GUIDANCE_WEIGHT = 0.9  # Initial weight for RL guidance (0.9 = 90% CFR, 10% RL)
    PREFLOP_ANNEALING_STEPS = 100000  # Training steps to anneal from RL to pure CFR
    PREFLOP_CHART_PROBABILITY = 0.8  # Probability of using preflop chart during training
    PREFLOP_SOLVE_BATCH_SIZE = 50  # Batch size for parallel preflop solving
    
    # Phase 3: Enhanced CFR Settings
    HAND_ABSTRACTION_BUCKETS = 75  # Increased from 50 for 8-player complexity
    SUBGAME_SOLVE_THRESHOLD = 20  # Minimum actions before subgame solving
    MCCFR_SAMPLES_PER_ITERATION = 50  # Monte Carlo samples per MCCFR iteration
    MAX_MEMORY_GB = 32.0  # Maximum memory usage before pruning
    GPU_CFR_BATCH_SIZE = 500  # Batch size for GPU CFR operations
    ENHANCED_TRAINING_ITERATIONS = 100000  # Target iterations for full training

    # Phase 4: Learned Abstraction System
    LEARNED_ABSTRACTION_ENABLED = True  # Enable learned neural abstractions
    LEARNED_ABSTRACTION_DIM = 32  # Dimension of learned abstraction embedding
    LEARNED_ABSTRACTION_LAYERS = 2  # Number of layers in abstraction encoder/decoder
    LEARNED_ABSTRACTION_LEARNING_RATE = 1e-4  # Learning rate for abstraction training
    LEARNED_ABSTRACTION_BATCH_SIZE = 256  # Batch size for abstraction training
    LEARNED_ABSTRACTION_EPOCHS = 50  # Number of epochs to train abstraction
    LEARNED_ABSTRACTION_UPDATE_FREQUENCY = 1000  # Update abstraction every N training steps
    LEARNED_ABSTRACTION_RECONSTRUCTION_WEIGHT = 0.5  # Weight for reconstruction loss
    LEARNED_ABSTRACTION_CLUSTERING_WEIGHT = 0.3  # Weight for clustering loss
    LEARNED_ABSTRACTION_ENTROPY_WEIGHT = 0.2  # Weight for entropy regularization
    
    # Tree optimization with learned abstractions
    # Constrain tree size to avoid multi-GB dictionaries during search.
    TREE_MAX_SIZE_WITH_ABSTRACTIONS = 200_000  # Maximum tree size with abstractions enabled
    TREE_ABSTRACTION_THRESHOLD = 50000  # Tree size threshold to trigger abstraction
    TREE_PRUNING_RATIO = 0.8  # Ratio of nodes to keep during pruning (0.8 = keep 80%)
    TREE_APPROXIMATION_RATIO = 0.9  # Ratio of nodes to approximate vs exact (0.9 = 90% approximated)

    # GPU-accelerated equity evaluation parameters
    EQUITY_BATCH_SIZE = 250  # Further reduced for memory
    EQUITY_GPU_FALLBACK_THRESHOLD = 100  # Fallback to CPU for batches smaller than this
    # Limit equity cache to a safe size. Large Python dicts can balloon in RAM.
    EQUITY_CACHE_SIZE = 200_000
    
    # Learned abstraction training parameters
    LEARNED_ABSTRACTION_TRAINING_HANDS = 1000000  # 1M hands for offline training
    LEARNED_ABSTRACTION_TRAIN_BATCH_SIZE = 512  # Batch size for training
    LEARNED_ABSTRACTION_TRAIN_EPOCHS = 50  # Training epochs
    LEARNED_ABSTRACTION_LEARNING_RATE = 1e-4  # Learning rate for training
    
    # MCTS parallelization parameters
    MCTS_PARALLEL_ROLLOUTS = True  # Enable parallel rollouts on CPU cores
    MCTS_CPU_WORKERS = 8  # Number of CPU workers for parallel rollouts
    MCTS_SIM_DEPTH = 500  # MCTS simulation depth (updated)
    
    # Abstraction caching parameters
    # Keep abstraction cache bounded to prevent RAM spikes.
    ABSTRACTION_CACHE_SIZE = 200_000
    ABSTRACTION_CACHE_MAX_MEMORY_GB = 8.0  # Soft guidance; pruning happens via LRU

    # Micro-batch size for PPO updates to limit peak VRAM
    PPO_MICRO_BATCH = 256

# Runtime variables (initialized here to avoid circular imports)
device = Config.DEVICE  # Alias for backward compatibility
player_models = {}  # Dictionary to store player models
logger = None  # Logger instance (set by training script)
used_player_ids = set()  # Set of used player IDs