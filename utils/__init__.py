# utils/__init__.py

# Import directly from source modules to avoid circular imports
from classes.card import Card
import importlib.util
import os
# Import from utils.py directly
spec = importlib.util.spec_from_file_location("utils_module", os.path.join(os.path.dirname(__file__), "..", "utils.py"))
utils_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_module)

CustomBeta = utils_module.CustomBeta
get_vram_usage = utils_module.get_vram_usage
use_preflop_chart = utils_module.use_preflop_chart
estimate_equity = utils_module.estimate_equity
quick_simulate = utils_module.quick_simulate
create_deck = utils_module.create_deck
burn_card = utils_module.burn_card
evaluate_hand = utils_module.evaluate_hand
count_active = utils_module.count_active
get_state = utils_module.get_state
get_legal_actions = utils_module.get_legal_actions
opponent_tracker = utils_module.opponent_tracker
AbstractionCache = utils_module.AbstractionCache
estimate_equity_batch_gpu = utils_module.estimate_equity_batch_gpu
quick_simulate_batch_gpu = utils_module.quick_simulate_batch_gpu
ParallelMCTS = utils_module.ParallelMCTS
interpret_discrete_action = utils_module.interpret_discrete_action

# Add other needed functions that don't cause circular imports
# from .helper_functions import get_state, get_legal_actions, etc. - temporarily disabled

# Temporarily disable problematic imports to avoid circular imports
# from .helper_functions import (
#     generate_unique_player_id,
#     pref,
#     postf,
#     prepare_features,
#     encode_card,
#     get_legal_bets,
#     default_strategy,
#     map_action_decision_to_action
# )

# Remove the following lines to prevent circular imports
# from .training import train_player_model
# from .evaluation import select_top_players, evaluate_agents
