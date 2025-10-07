from config import Suit
from datatypes import Card
from nextgen.hand_eval import classify_7


def test_two_pair_kicker_tiebreak():
    # Player A: two pair Aces and Tens, K kicker
    a_hole = [Card(14, Suit.HEARTS), Card(10, Suit.DIAMONDS)]
    board = [Card(14, Suit.CLUBS), Card(10, Suit.SPADES), Card(2, Suit.HEARTS), Card(7, Suit.CLUBS), Card(13, Suit.DIAMONDS)]
    cat_a, tb_a = classify_7(a_hole, board)

    # Player B: two pair Aces and Tens, Q kicker
    b_hole = [Card(14, Suit.SPADES), Card(10, Suit.CLUBS)]
    board_b = [Card(14, Suit.CLUBS), Card(10, Suit.SPADES), Card(2, Suit.HEARTS), Card(7, Suit.CLUBS), Card(12, Suit.DIAMONDS)]  # Q instead of K
    cat_b, tb_b = classify_7(b_hole, board_b)

    assert cat_a == 3 and cat_b == 3
    # Compare tie-break tuples directly
    assert tb_a > tb_b  # K kicker beats Q kicker


def test_quads_kicker():
    # Board carries quads; kicker decides
    board = [Card(9, Suit.HEARTS), Card(9, Suit.DIAMONDS), Card(9, Suit.CLUBS), Card(9, Suit.SPADES), Card(2, Suit.HEARTS)]
    a_hole = [Card(14, Suit.SPADES), Card(3, Suit.DIAMONDS)]  # A kicker
    b_hole = [Card(13, Suit.SPADES), Card(4, Suit.CLUBS)]  # K kicker
    cat_a, tb_a = classify_7(a_hole, board)
    cat_b, tb_b = classify_7(b_hole, board)
    assert cat_a == 8 and cat_b == 8
    assert tb_a > tb_b  # Ace kicker wins over King kicker

