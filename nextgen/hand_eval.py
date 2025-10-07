from __future__ import annotations

from typing import List, Tuple
from collections import Counter

from datatypes import Card


def _ranks(cards: List[Card]) -> List[int]:
    return [c.value for c in cards]


def _suits(cards: List[Card]) -> List[int]:
    # Suit is enum; use its value
    return [c.suit.value if hasattr(c.suit, "value") else int(c.suit) for c in cards]


def _best_straight(unique_ranks: List[int]) -> int:
    """Return high card of best straight (A-2-3-4-5 allowed) or 0 if none."""
    s = set(unique_ranks)
    # Account for A as low
    if 14 in s:
        s.add(1)
    best = 0
    for high in range(14, 4, -1):
        seq = [high - i for i in range(5)]
        if all(x in s for x in seq):
            best = high if high != 5 else 5  # 5-high straight (wheel)
            return best
    return best


def _flush_suit(cards: List[Card]) -> int:
    suit_counts = Counter(_suits(cards))
    for s, cnt in suit_counts.items():
        if cnt >= 5:
            return s
    return -1


def _top_n(values: List[int], n: int) -> List[int]:
    return sorted(values, reverse=True)[:n]


def classify_7(hole: List[Card], community: List[Card]) -> Tuple[int, Tuple[int, ...]]:
    """Classify 7-card hand into category 1..9 with tie-breakers.

    Categories (higher is better):
      9 Straight Flush
      8 Four of a Kind
      7 Full House
      6 Flush
      5 Straight
      4 Three of a Kind
      3 Two Pair
      2 One Pair
      1 High Card
    Returns (category, tiebreakers_desc).
    """
    cards = list(hole) + list(community)
    r = sorted(_ranks(cards), reverse=True)
    r_counts = Counter(_ranks(cards))

    # Flush/straight checks
    suit = _flush_suit(cards)
    unique_ranks = sorted(set(_ranks(cards)))
    straight_high = _best_straight(unique_ranks)

    # Straight flush
    if suit != -1:
        suited_cards = [c for c in cards if (c.suit.value if hasattr(c.suit, "value") else int(c.suit)) == suit]
        sranks = sorted(set(_ranks(suited_cards)))
        sf_high = _best_straight(sranks)
        if sf_high:
            return 9, (sf_high,)

    # Four of a kind
    quad = [rk for rk, cnt in r_counts.items() if cnt == 4]
    if quad:
        q = max(quad)
        kicker = max([rk for rk in r if rk != q])
        return 8, (q, kicker)

    # Full house
    trips = sorted([rk for rk, cnt in r_counts.items() if cnt == 3], reverse=True)
    pairs = sorted([rk for rk, cnt in r_counts.items() if cnt == 2], reverse=True)
    if trips and (len(trips) >= 2 or pairs):
        t = trips[0]
        p = trips[1] if len(trips) >= 2 else pairs[0]
        return 7, (t, p)

    # Flush
    if suit != -1:
        suited_ranks = [c.value for c in cards if (c.suit.value if hasattr(c.suit, "value") else int(c.suit)) == suit]
        top5 = _top_n(suited_ranks, 5)
        return 6, tuple(top5)

    # Straight
    if straight_high:
        return 5, (straight_high,)

    # Three of a kind
    if trips:
        t = trips[0]
        kickers = _top_n([rk for rk in r if rk != t], 2)
        return 4, (t,) + tuple(kickers)

    # Two pair
    if len(pairs) >= 2:
        p1, p2 = pairs[:2]
        if p1 < p2:
            p1, p2 = p2, p1
        kicker = max([rk for rk in r if rk not in (p1, p2)])
        return 3, (p1, p2, kicker)

    # One pair
    if len(pairs) == 1:
        p1 = pairs[0]
        kickers = _top_n([rk for rk in r if rk != p1], 3)
        return 2, (p1,) + tuple(kickers)

    # High card
    return 1, tuple(_top_n(r, 5))


def best_hand_rank(hole: List[Card], community: List[Card]) -> Tuple[int, Tuple[int, ...]]:
    """Return a tuple that is directly comparable with min() for winner detection.

    We invert values so that the best hand sorts first with min().
    """
    cat, tb = classify_7(hole, community)
    return (-cat, tuple(-x for x in tb))

