# simulation/simulate.py

import torch
from typing import List, Tuple, Dict, Any
from classes.player import Player
from classes.game_state import GameState
from environment.poker_env import FullyObservablePokerEnv
from utils.helper_functions import default_strategy, map_action_decision_to_action, get_legal_bets, update_gameplay_data
from actions.player_actions import act, update_game_state
from utils.training import train_player_model
from config import logger, device
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from treys import Card as TreysCard
from treys import Evaluator

def simulate_game(players: List[Player], gameplay_data: List[Dict[str, Any]], device: torch.device) -> Tuple[GameState, List[Player]]:
    game_state = GameState(players)
    game_state.reset_for_new_hand()
    game_state.deal_cards()
    game_state.post_blinds()

    betting_round_counter = 0
    while True:
        betting_round_counter += 1
        starting_position = (game_state.button_position + 3) % len(game_state.players) if len(game_state.players) > 0 else 0
        player_order = game_state.players[starting_position:] + game_state.players[:starting_position]

        betting_action_counter = 0
        MAX_BETTING_ACTIONS = 100

        while not game_state.is_betting_round_over():
            for player in player_order:
                if betting_action_counter >= MAX_BETTING_ACTIONS:
                    return game_state, []
                if player.folded or player.stack_size == 0:
                    continue
                action, amount = act(game_state, player, gameplay_data, device)
                update_game_state(game_state, player, action, amount)
                betting_action_counter += 1
                if game_state.is_betting_round_over():
                    break

        game_state.collect_bets()
        game_state.remove_folded_players()
        if len(game_state.active_players) == 1:
            break
        if game_state.round_stage == 3:
            break
        game_state.next_betting_round()

    if len(game_state.active_players) == 1:
        winners = [game_state.active_players[0]]
    else:
        winners, pot_share = game_state.determine_winners_and_pot()

    if winners:
        pot_share = game_state.pot_size / len(winners)
        game_state.assign_rewards(winners, pot_share, players)  # Assign rewards via GameState
    else:
        pot_share = 0

    if not winners:
        for data in gameplay_data:
            data['outcome'] = 0
    else:
        for data in gameplay_data:
            player_id = data['player_id']
            player = next(p for p in players if p.player_id == player_id)
            if player in winners:
                data['outcome'] = pot_share - data['amount']
            else:
                data['outcome'] = -data['amount']

    return game_state, winners

def simulate_generation(players: List[Player], num_games: int, chunk_size: int = 1000, device: torch.device = torch.device('cpu')) -> Tuple[Dict[int, float], List[Dict[str, Any]]]:
    wins = {player.player_id: 0.0 for player in players}
    gameplay_data = []

    for i in tqdm(range(num_games), desc="Simulating Games"):
        for player in players:
            player.stack_size = 1000
            player.folded = False
            player.acted = False

        game_state, winners = simulate_game(players, gameplay_data, device)

        if not winners:
            continue

        pot_share = game_state.pot_size / len(winners)
        for winner in winners:
            wins[winner.player_id] += pot_share

        if (i + 1) % chunk_size == 0:
            player_data = [
                (player, [data for data in gameplay_data if data['player_id'] == player.player_id])
                for player in players
            ]
            with Pool(processes=cpu_count()) as pool:
                pool.starmap(train_player_model, [(p, d, device) for p, d in player_data])
            gameplay_data.clear()

    if gameplay_data:
        player_data = [
            (player, [data for data in gameplay_data if data['player_id'] == player.player_id])
            for player in players
        ]
        with Pool(processes=cpu_count()) as pool:
            pool.starmap(train_player_model, [(p, d, device) for p, d in player_data])
        gameplay_data.clear()

    average_pot_won = {player_id: total_pot / num_games for player_id, total_pot in wins.items()}
    
    return average_pot_won, gameplay_data
