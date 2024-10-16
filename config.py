# config.py

import logging
import os

# Device Configuration
import torch_directml
device = torch_directml.device()

# Caching Models
player_models = {}

# Logging Configuration
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/poker_simulation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure the models directory exists
os.makedirs("models_saved", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Unique Player ID Generation
used_player_ids = set()

TOTAL_PLAYERS = 8  # Define globally for accessibility
