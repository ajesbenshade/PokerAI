import numpy as np

from config import Suit, Action
from datatypes import Card
from nextgen.hand_eval import classify_7
from nextgen.mechanics import PokerTable
from nextgen.features import compute_base_features, FeatureConfig


def test_wheel_straight_detection():
    # A-2-3-4-5 (wheel)
    hole = [Card(14, Suit.HEARTS), Card(2, Suit.DIAMONDS)]
    board = [Card(3, Suit.CLUBS), Card(4, Suit.SPADES), Card(5, Suit.HEARTS)]
    cat, tb = classify_7(hole, board)
    assert cat == 5  # Straight
    assert tb[0] == 5  # 5-high straight


def test_min_raise_logic_non_reopen():
    t = PokerTable(num_players=2, starting_stack=50, small_blind=1, big_blind=2)
    t.start_hand()
    # SB is pid 1, BB is pid 0 (depends on button=0)
    # Force to preflop actor after blinds
    pid = t.to_act
    # Try a too-small raise (all-in that does not reach min-raise)
    p = t.players[pid]
    to_call = max(t.current_bet - p.current_bet, 0)
    # Make raise less than min_raise but not all-in -> should fail
    ok = t.act_raise(pid, max(0, t.min_raise - 1))
    assert not ok
    # Now make an all-in smaller than min-raise above call -> allowed, non-reopen
    all_in_amt = max(0, p.stack - to_call)
    if all_in_amt > 0:
        ok2 = t.act_raise(pid, all_in_amt)
        assert ok2


def test_side_pot_split_tie():
    t = PokerTable(num_players=2, starting_stack=10, small_blind=1, big_blind=2)
    t.start_hand()
    # Force identical hands to cause tie at showdown by copying cards
    t.players[0].hand = [Card(10, Suit.HEARTS), Card(9, Suit.HEARTS)]
    t.players[1].hand = [Card(10, Suit.DIAMONDS), Card(9, Suit.DIAMONDS)]
    t.community = [Card(2, Suit.CLUBS), Card(3, Suit.SPADES), Card(4, Suit.CLUBS), Card(7, Suit.HEARTS), Card(8, Suit.DIAMONDS)]
    # Push chips into pot evenly
    for p in t.players:
        call = max(t.current_bet - p.current_bet, 0)
        t.act_call(p.player_id)
        if p.stack > 0:
            t.act_raise(p.player_id, p.stack)  # shove
    payouts = t.settle_showdown()
    total = sum(payouts.values())
    assert total == t.pot
    # Tie implies equal split (or off-by-one due to odd chip handling)
    assert abs(payouts[0] - payouts[1]) <= 1


def test_features_dimension():
    t = PokerTable(num_players=2)
    t.start_hand()
    p = t.players[0]
    obs = {
        "pid": p.player_id,
        "hand": p.hand,
        "community": t.community,
        "pot": t.pot,
        "stack": p.stack,
        "to_call": max(t.current_bet - p.current_bet, 0),
        "min_raise": t.min_raise,
        "alive": [q.player_id for q in t.active_players()],
        "position": (p.player_id - t.button) % t.num_players,
    }
    base = compute_base_features(
        pid=obs["pid"], hole=obs["hand"], community=obs["community"], pot=obs["pot"],
        stack=obs["stack"], to_call=obs["to_call"], min_raise=obs["min_raise"],
        num_players=len(obs["alive"]), button_pos=t.button, history=[], cfg=FeatureConfig(),
    )
    assert base.shape[0] == 480

