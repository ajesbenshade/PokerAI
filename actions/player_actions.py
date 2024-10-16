# actions/player_actions.py

from typing import Tuple, List, Dict, Any
from classes.game_state import GameState
from classes.player import Player
from utils.helper_functions import (
    prepare_features,
    default_strategy,
    map_action_decision_to_action,
    get_legal_bets
)
from utils.training import train_player_model
from config import player_models, logger, device
import torch
import numpy as np

def act(game_state: GameState, player: Player, gameplay_data: List[Dict[str, Any]], device: torch.device) -> Tuple[str, int]:
    pot_size = game_state.pot_size
    stack_size = player.stack_size
    to_call = game_state.current_bet - player.current_bet

    legal_actions = ['fold']
    if to_call == 0:
        legal_actions.append('check')
        if player.stack_size > 0:
            legal_actions.append('bet')
    else:
        if player.stack_size > 0:
            legal_actions.append('call')
        if player.stack_size > to_call:
            legal_actions.append('raise')

    legal_bets = set()
    for action in legal_actions:
        if action in ['fold', 'check']:
            legal_bets.add(0)
        elif action == 'call':
            legal_bets.add(to_call)
        elif action == 'bet':
            bet_sizes = [
                game_state.big_blind,
                game_state.big_blind * 2,
                min(game_state.pot_size, player.stack_size)
            ]
            legal_bets.update(bet_sizes)
        elif action == 'raise':
            raise_sizes = [
                to_call + game_state.big_blind,
                to_call + game_state.big_blind * 2,
                to_call + min(game_state.pot_size, player.stack_size)
            ]
            legal_bets.update(raise_sizes)

    legal_bets = sorted(legal_bets)

    if not legal_bets:
        return 'fold', 0

    features = prepare_features(game_state, player)
    feature_tensor = torch.tensor(features, dtype=torch.float32, device=device)

    try:
        if player.is_model_fitted():
            with torch.no_grad():
                expected_values = []
                legal_bets_list = sorted(legal_bets)
                for bet in legal_bets_list:
                    action_features = features.copy()
                    action_features.append(bet)
                    action_tensor = torch.tensor(action_features, dtype=torch.float32, device=device)
                    output = player.model(action_tensor)
                    expected_value = output.item()
                    expected_values.append(expected_value)
                best_idx = int(np.argmax(expected_values))
                best_bet = legal_bets_list[best_idx]
                amount = best_bet

                if best_bet == 0:
                    action_decision = 'fold' if to_call > 0 else 'check'
                elif best_bet == to_call:
                    action_decision = 'call'
                elif best_bet > to_call:
                    action_decision = 'raise' if to_call > 0 else 'bet'
                else:
                    action_decision = 'fold'

                gameplay_data.append({
                    'player_id': player.player_id,
                    'features': features + [best_bet],
                    'action': action_decision,
                    'amount': amount,
                    'outcome': None
                })
        else:
            action_decision, amount = default_strategy(player, game_state, to_call)
            gameplay_data.append({
                'player_id': player.player_id,
                'features': features,
                'action': action_decision,
                'amount': amount,
                'outcome': None
            })
            return action_decision, amount

    except Exception as e:
        logger.error(f"Error during player action: {e}")
        action_decision, amount = default_strategy(player, game_state, to_call)
        gameplay_data.append({
            'player_id': player.player_id,
            'features': features,
            'action': action_decision,
            'amount': amount,
            'outcome': None
        })
        return action_decision, amount

    return action_decision, amount

def update_game_state(game_state: GameState, player: Player, action: str, amount: int):
    if action == 'fold':
        player.folded = True
        game_state.bet_history.append({'player': player.player_id, 'action': 'fold', 'amount': 0})
    else:
        bet_amount = min(amount, player.stack_size)
        player.stack_size -= bet_amount
        player.current_bet += bet_amount
        if player.current_bet > game_state.current_bet:
            game_state.current_bet = player.current_bet
            game_state.last_raiser = player
            for p in game_state.players:
                if p != player and not p.folded:
                    p.acted = False
            game_state.current_raises += 1
            if game_state.current_raises > game_state.max_raises_per_round:
                game_state.current_bet = player.current_bet
                for p in game_state.players:
                    if not p.folded:
                        p.acted = True
        game_state.bet_history.append({'player': player.player_id, 'action': action, 'amount': bet_amount})
    player.acted = True
