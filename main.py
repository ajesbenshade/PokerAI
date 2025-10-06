# main.py

from classes.player import Player
from config import device, logger, TOTAL_PLAYERS
from simulation.simulate import simulate_generation
from utils.evaluation import select_top_players, evaluate_agents
from environment.poker_env import FullyObservablePokerEnv
from typing import List
import matplotlib.pyplot as plt
import json
import os
import torch

def load_player_models(players: List[Player], device: torch.device):
    """
    Loads pre-trained models for each player if available.
    """
    for player in players:
        model_path = f"models_saved/pytorch_player_{player.player_id}.pt"
        scaler_path = f"models_saved/scaler_player_{player.player_id}.pt"
        if os.path.isfile(model_path):
            try:
                player.model.load_state_dict(torch.load(model_path, map_location=device))
                player.model.to(device)
                player.model.eval()
                from config import player_models
                player_models[player.player_id] = player.model
                logger.info(f"Loaded model for Player {player.player_id}")
            except Exception as e:
                logger.error(f"Failed to load model for Player {player.player_id}: {e}")
        else:
            logger.warning(f"Model file {model_path} not found for Player {player.player_id}.")

        if os.path.isfile(scaler_path):
            try:
                scaler = torch.load(scaler_path, map_location=device)
                logger.info(f"Loaded scaler for Player {player.player_id}")
            except Exception as e:
                logger.error(f"Failed to load scaler for Player {player.player_id}: {e}")
        else:
            logger.warning(f"Scaler file {scaler_path} not found for Player {player.player_id}.")

def main():
    ENABLE_PROFILING = False
    TOP_PLAYERS = 4  # Adjusted to select top 4 players
    GAMES_PER_GENERATION = 10000

    players = [Player(num_players=TOTAL_PLAYERS) for _ in range(TOTAL_PLAYERS)]

    # Load player models if available
    load_player_models(players, device)

    average_pot_won, gameplay_data = simulate_generation(players, GAMES_PER_GENERATION, chunk_size=1000, device=device)

    print("Average Pot Won per Player:")
    for player_id, avg_pot in average_pot_won.items():
        print(f"Player {player_id}: ${avg_pot:.2f}")

    player_ids = list(average_pot_won.keys())
    avg_pots = list(average_pot_won.values())

    plt.figure(figsize=(10, 6))
    plt.bar(player_ids, avg_pots, color='skyblue')
    plt.xlabel('Player ID')
    plt.ylabel('Average Pot Won')
    plt.title('Average Pot Won per Player After Simulation')
    plt.show()

    top_players = select_top_players(players, average_pot_won, top_n=TOP_PLAYERS)

    if not top_players:
        logger.error("No players available for evaluation. Exiting.")
        return

    env = FullyObservablePokerEnv(top_players, device)
    average_pot_won_evaluation = evaluate_agents(env, top_players, num_episodes=500, device=device)

    for player_id, avg_pot in average_pot_won_evaluation.items():
        print(f"Agent {player_id} Average Pot Won: ${avg_pot:.2f}")

    for player in players:
        if player.player_id not in player_models:
            from config import player_models
            player_models[player.player_id] = player.model

    with open('simulation_results.json', 'w') as f:
        json.dump({
            'average_pot_won': average_pot_won,
            'evaluation_results': average_pot_won_evaluation
        }, f, indent=4)

if __name__ == "__main__":
    main()
