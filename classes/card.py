# classes/card.py

from enums import Suit

class Card:
    def __init__(self, value: int, suit: Suit):
        self.value = value
        self.suit = suit

    def __repr__(self):
        value_str = (
            str(self.value) if self.value <= 10 else
            {11: 'Jack', 12: 'Queen', 13: 'King', 14: 'Ace'}[self.value]
        )
        suit_str = self.suit.name.capitalize()
        return f"{value_str} of {suit_str}"
    
    def to_treys_str(self) -> str:
        value_map = {
            2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'T',
            11: 'J', 12: 'Q', 13: 'K', 14: 'A'
        }
        return f"{value_map[self.value]}{self.suit.name[0].lower()}"

    def suit_numeric(self) -> int:
        suit_mapping = {
            'HEARTS': 1,
            'DIAMONDS': 2,
            'CLUBS': 3,
            'SPADES': 4
        }
        return suit_mapping.get(self.suit.name, 0)  # Default to 0 if not found
