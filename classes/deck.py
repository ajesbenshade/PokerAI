# classes/deck.py

from treys import Deck as TreysDeck, Card as TreysCard
from classes.card import Card
from enums import Suit
import logging

logger = logging.getLogger(__name__)

class Deck:
    def __init__(self):
        self.deck = TreysDeck()

    def shuffle(self):
        self.deck.shuffle()

    def deal(self) -> Card:
        if len(self.deck.cards) == 0:
            raise ValueError("All cards have been dealt")
        treys_card = self.deck.draw(1)[0]
        card_str = TreysCard.int_to_str(treys_card)  # e.g., 'As', 'Kd', '5h'
        rank_char = card_str[0]
        suit_char = card_str[1]
        rank_value = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                      '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12,
                      'K': 13, 'A': 14}[rank_char]
        suit = {'h': Suit.HEARTS, 'd': Suit.DIAMONDS, 'c': Suit.CLUBS, 's': Suit.SPADES}[suit_char]
        return Card(rank_value, suit)
