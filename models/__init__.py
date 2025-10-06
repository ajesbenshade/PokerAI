# models/__init__.py

from .player_model import PlayerModel
# Also import from the main models.py file
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("model_module", os.path.join(os.path.dirname(__file__), "..", "models.py"))
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)
Actor = model_module.Actor
Critic = model_module.Critic
