# utils/evaluation.py

from typing import List, Dict
from classes.player import Player  # Import Player
from environment.poker_env import FullyObservablePokerEnv
from utils.helper_functions import prepare_features, get_legal_bets, default_strategy, map_action_decision_to_action, update_gameplay_data
from actions.player_actions import update_game_state
import torch
import os
import numpy as np
from tqdm import tqdm

def select_top_players(players: List[Player], average_pot_won: Dict[int, float], top_n: int) -> List[Player]:
    fitted_players = [p for p in players if p.is_model_fitted()]
    if not fitted_players:
        return players.copy()
    sorted_players = sorted(fitted_players, key=lambda p: average_pot_won.get(p.player_id, 0), reverse=True)
    top_players = sorted_players[:top_n]
    return top_players

def evaluate_agents(env: FullyObservablePokerEnv, players: List[Player], num_episodes: int = 1000, device: torch.device = torch.device('cpu')) -> Dict[int, float]:
    pot_won = {player.player_id: 0.0 for player in players}
    for episode in tqdm(range(num_episodes), desc="Evaluating Agents"):
        state = env.reset()
        done = False
        while not done:
            player = env.players[env.current_player_idx]
            if player.is_model_fitted():
                try:
                    features = prepare_features(env.game_state, player)
                    scaler_path = f"models_saved/scaler_player_{player.player_id}.pt"
                    if os.path.isfile(scaler_path):
                        scaler_data = torch.load(scaler_path, map_location=device)
                        mean = scaler_data['mean']
                        std = scaler_data['std']
                        features_tensor = torch.tensor(features, dtype=torch.float32, device=device)
                        features_scaled = (features_tensor - mean) / std
                    else:
                        features_scaled = torch.tensor(features, dtype=torch.float32, device=device)
                    
                    with torch.no_grad():
                        expected_values = []
                        legal_bets = get_legal_bets(env.game_state, player)
                        for bet in legal_bets:
                            action_features = features.copy()
                            action_features.append(bet)
                            if os.path.isfile(scaler_path):
                                action_features_tensor = torch.tensor(action_features, dtype=torch.float32, device=device)
                                action_features_scaled = (action_features_tensor - mean) / std
                            else:
                                action_features_scaled = torch.tensor(action_features, dtype=torch.float32, device=device)
                            output = player.model(action_features_scaled)
                            expected_value = output.item()
                            expected_values.append(expected_value)
                        if not expected_values:
                            action_decision, amount = 'fold', 0
                        else:
                            best_idx = int(np.argmax(expected_values))
                            best_bet = legal_bets[best_idx]
                            amount = best_bet

                            if best_bet == 0:
                                action_decision = 'fold' if (env.game_state.current_bet - player.current_bet) > 0 else 'check'
                            elif best_bet == (env.game_state.current_bet - player.current_bet):
                                action_decision = 'call'
                            elif best_bet > (env.game_state.current_bet - player.current_bet):
                                action_decision = 'raise' if (env.game_state.current_bet - player.current_bet) > 0 else 'bet'
                            else:
                                action_decision = 'fold'
                            amount = best_bet
                except Exception as e:
                    # Handle exception (e.g., log error, assign default action)
                    print(f"Error during agent decision: {e}")
                    action_decision, amount = default_strategy(player, env.game_state, to_call=env.game_state.current_bet - player.current_bet)
            else:
                # Use default strategy
                action_decision, amount = default_strategy(player, env.game_state, to_call=env.game_state.current_bet - player.current_bet)
            
            action = map_action_decision_to_action(action_decision, env.game_state, player)
            next_state, reward, done, info = env.step(action)
            
            if done:
                if reward > 0:
                    pot_won[player.player_id] += reward

    average_pot_won = {player_id: total_pot / num_episodes for player_id, total_pot in pot_won.items()}
    return average_pot_won
