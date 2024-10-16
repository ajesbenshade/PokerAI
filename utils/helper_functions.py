# utils/helper_functions.py

import random
from typing import List, Tuple, Dict, Any
from classes.card import Card
from enums import Suit
from config import used_player_ids, logger

def generate_unique_player_id():
    max_id = 99999
    while True:
        new_id = random.randint(1000, max_id)
        if new_id not in used_player_ids:
            used_player_ids.add(new_id)
            return new_id
        if len(used_player_ids) >= (max_id - 999):
            raise ValueError("Exhausted all unique player IDs.")

def pref(cards: List[Card]) -> List[int]:
    numbers = [card.value for card in cards]
    suits = [card.suit.value for card in cards]
    max_number = max(numbers)
    number_difference = max_number - min(numbers)
    is_suited = 1 if suits[0] == suits[1] else 0
    return [max_number, number_difference, is_suited]

def postf(cards: List[Card]) -> List[int]:
    numbers = [card.value for card in cards]
    suits = [card.suit.value for card in cards]

    number_counts = {}
    for number in numbers:
        number_counts[number] = number_counts.get(number, 0) + 1
    sorted_number_counts = sorted(number_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    top_counts = sorted_number_counts[:5]
    x1 = [count for (_, count) in top_counts] + [num for (num, _) in top_counts]

    suit_counts = {}
    for suit in suits:
        suit_counts[suit] = suit_counts.get(suit, 0) + 1
    sorted_suit_counts = sorted(suit_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    top_suits = sorted_suit_counts[:3]
    x2 = []
    for (suit, count) in top_suits:
        if suit < 0:
            continue
        x2.append(count)
        x2.append(suit)

    return x1 + x2

def encode_card(card: Card) -> List[float]:
    return [card.value / 14, card.suit_numeric() / 4]

def prepare_features(game_state, player) -> List[float]:
    """
    Prepares the feature vector for a player given the current game state.

    Parameters:
    - game_state: The current state of the game.
    - player: The player for whom features are being prepared.

    Returns:
    - A list of normalized feature values.
    """
    suit_mapping = {
        'HEARTS': 1,  # Hearts
        'DIAMONDS': 2,  # Diamonds
        'CLUBS': 3,  # Clubs
        'SPADES': 4   # Spades
    }
    
    features = [
        game_state.pot_size / 1000,                     
        player.stack_size / 1000,                       
        player.position / len(game_state.players) if len(game_state.players) > 0 else 0,  
        game_state.aggression / 100,                    
        len(game_state.players) / 10,                   
        game_state.round_stage / 4                      
    ]
    for p in game_state.players:
        for card in p.hand:
            suit_numeric = suit_mapping.get(card.suit.name, 0)  # Default to 0 if not found
            features.extend([
                card.value / 14,
                (p.hand.index(card) + 1) / 2 if p.hand else 0,
                suit_numeric / 4  # Now it's an integer divided by 4
            ])
        missing_cards = 2 - len(p.hand)
        if missing_cards > 0:
            features.extend([0.0, 0.0, 0.0] * missing_cards)
    for card in game_state.community_cards:
        suit_numeric = suit_mapping.get(card.suit.name, 0)
        features.extend([card.value / 14, suit_numeric / 4])
    missing_community_cards = 5 - len(game_state.community_cards)
    if missing_community_cards > 0:
        features.extend([0.0, 0.0] * missing_community_cards)
    return features

def get_legal_bets(game_state, player) -> List[int]:
    """
    Determines the list of legal bets for a player based on the current game state.

    Parameters:
    - game_state: The current state of the game.
    - player: The player for whom legal bets are being determined.

    Returns:
    - A sorted list of legal bet amounts.
    """
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

    return sorted(legal_bets)

def default_strategy(player, game_state, to_call: int) -> Tuple[str, int]:
    """
    Determines the default strategy for a player based on their hand and the game state.

    Parameters:
    - player: The player instance making the decision.
    - game_state: The current state of the game.
    - to_call: The amount the player needs to call.

    Returns:
    - A tuple containing the action decision and the amount.
    """
    highest_card = player.pref_features[0] if player.pref_features else 0
    is_suited = player.pref_features[2] if len(player.pref_features) > 2 else 0

    if to_call == 0:
        if highest_card >= 12 or (highest_card >= 10 and is_suited):
            amount = min(game_state.big_blind, player.stack_size)
            return 'bet', amount
        else:
            return 'check', 0
    else:
        if highest_card >= 12 or (highest_card >= 10 and is_suited):
            if player.stack_size > to_call:
                return 'call', to_call
            else:
                return 'call', player.stack_size
        else:
            return 'fold', 0

def map_action_decision_to_action(action_decision: str, game_state, player) -> int:
    """
    Maps an action decision to an integer action code.

    Parameters:
    - action_decision: The action decision as a string ('fold', 'call/check', 'raise/bet').
    - game_state: The current state of the game.
    - player: The player making the decision.

    Returns:
    - An integer representing the action.
    """
    if action_decision == 'fold':
        return 0
    elif action_decision in ['call', 'check']:
        return 1
    elif action_decision in ['raise', 'bet']:
        return 2
    else:
        return 0  # Default to 'fold' if action is unrecognized

def update_gameplay_data(winners: List, pot_share: float, gameplay_data: Dict[str, Any], players: List['Player']):
    """
    Updates the gameplay data with outcomes based on winners.

    Parameters:
    - winners: List of winning players.
    - pot_share: The share of the pot each winner receives.
    - gameplay_data: The list of gameplay data dictionaries.
    - players: The list of all players.
    """
    for data in gameplay_data:
        player_id = data['player_id']
        player = next(p for p in players if p.player_id == player_id)
        if player in winners:
            data['outcome'] = pot_share - data['amount']
        else:
            data['outcome'] = -data['amount']
