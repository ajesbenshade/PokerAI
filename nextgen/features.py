from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import numpy as np

from config import Action, Suit
from datatypes import Card
from utils import estimate_equity


@dataclass
class FeatureConfig:
    max_players: int = 10
    rounds: int = 4  # preflop, flop, turn, river
    base_dim: int = 480
    action_dim: int = 32
    total_dim: int = 512


def _rank_index(v: int) -> int:
    return max(2, min(14, int(v))) - 2


def _presence_vector(cards: List[Card]) -> np.ndarray:
    # 15-long vector: ranks 0..14 where 0 is unused, 1..13 for A..K mapping; include Ace as both high and low
    v = np.zeros(15, dtype=np.float32)
    for c in cards:
        v[_rank_index(c.value) + 2] = 1.0
        if c.value == 14:
            v[1] = 1.0  # Ace as 1
    return v


def _straight_draw_vector(cards: List[Card]) -> np.ndarray:
    # Sliding window of length 10 indicating 4-in-a-row opportunities and blockers
    pres = _presence_vector(cards)
    out = np.zeros(10, dtype=np.float32)
    # Windows for sequences ending 5..14 (A-high); index in pres is same rank
    idx = 0
    for high in range(5, 15):
        window = pres[high - 4 : high + 1]
        have = window.sum()
        # encode: 1.0 = open straight (4 present), 0.5 = gutshot (3 present), else 0
        if have >= 4:
            out[idx] = 1.0
        elif have == 3:
            out[idx] = 0.5
        idx += 1
    return out


def _rank_counts(cards: List[Card]) -> np.ndarray:
    counts = np.zeros(13, dtype=np.float32)
    for c in cards:
        counts[_rank_index(c.value)] += 1.0
    return counts


def _suit_counts(cards: List[Card]) -> np.ndarray:
    counts = np.zeros(4, dtype=np.float32)
    for c in cards:
        s = c.suit.value if hasattr(c.suit, "value") else int(c.suit)
        counts[s] += 1.0
    return counts


def _board_texture(community: List[Card]) -> np.ndarray:
    # Basic board texture features: paired, trips, monotone, 2-tone, straight_on_board, flush_draw, backdoor_draw
    if not community:
        return np.zeros(16, dtype=np.float32)
    rcounts = _rank_counts(community)
    sp = _suit_counts(community)
    paired = float((rcounts >= 2).any())
    trips_or_more = float((rcounts >= 3).any())
    two_pair_on_board = float((rcounts >= 2).sum() >= 2)
    quad_on_board = float((rcounts >= 4).any())
    mono = float((sp == 3).any())  # exactly 3 same suit on flop
    four_suit = float((sp == 4).any())
    two_tone = float((sp >= 2).sum() == 2)
    rainbow = float((sp > 0).sum() >= 3)
    pres = _presence_vector(community)
    straight_any = 1.0 if _straight_draw_vector(community).max() == 1.0 else 0.0
    gutshot_any = 1.0 if (_straight_draw_vector(community) == 0.5).any() else 0.0
    # Normalize ranks
    high = max([c.value for c in community]) / 14.0 if community else 0.0
    low = min([c.value for c in community]) / 2.0 if community else 0.0
    span = (high * 14 - low * 2) / 12.0 if community else 0.0
    return np.array([
        paired,
        trips_or_more,
        two_pair_on_board,
        quad_on_board,
        mono,
        two_tone,
        rainbow,
        four_suit,
        straight_any,
        gutshot_any,
        float(len(community) >= 3),
        float(len(community) >= 4),
        float(len(community) == 5),
        high,
        low,
        span,
    ], dtype=np.float32)


def _hole_features(hole: List[Card]) -> np.ndarray:
    if len(hole) < 2:
        return np.zeros(16, dtype=np.float32)
    v = sorted([hole[0].value, hole[1].value], reverse=True)
    suited = 1.0 if (hole[0].suit == hole[1].suit) else 0.0
    gap = (v[0] - v[1] - 1) / 12.0
    connectors = 1.0 if v[0] - v[1] == 1 else 0.0
    pair = 1.0 if v[0] == v[1] else 0.0
    broadway = sum([float(x >= 10) for x in v]) / 2.0
    high = v[0] / 14.0
    low = v[1] / 14.0
    # Kickers as fractional bins (0..1)
    return np.array([
        suited, gap, connectors, pair, broadway, high, low,
        float(v[0] == 14), float(v[0] == 13), float(v[0] == 12),
        float(v[1] == 14), float(v[1] == 13), float(v[1] == 12),
        float(v[0] <= 7 and v[1] <= 7),
        float(v[0] <= 5 and v[1] <= 5),
        float(v[0] >= 10 and v[1] >= 10),
    ], dtype=np.float32)


def _history_features(history: List[Dict[str, Any]], max_players: int, rounds: int) -> np.ndarray:
    # Collate per player per round statistics: raises, calls, folds, contribution
    # Dimensions: players * rounds * 3 (+ contribution)
    arr = np.zeros((max_players, rounds, 4), dtype=np.float32)
    round_map = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
    for evt in history or []:
        pid = int(evt.get("player_id", 0))
        rnd = round_map.get(evt.get("street", "preflop"), 0)
        act = evt.get("action", Action.CALL.value)
        if 0 <= pid < max_players and 0 <= rnd < rounds:
            if act == Action.RAISE.value:
                arr[pid, rnd, 0] += 1
                arr[pid, rnd, 3] += float(evt.get("amount", 0.0))
            elif act == Action.CALL.value:
                arr[pid, rnd, 1] += 1
                arr[pid, rnd, 3] += float(evt.get("amount", 0.0))
            else:
                arr[pid, rnd, 2] += 1
    return arr.reshape(-1)


def _pot_features(pot: float, stack: float, to_call: float, min_raise: float) -> np.ndarray:
    pot = float(pot)
    stack = float(stack)
    to_call = float(to_call)
    min_raise = float(min_raise)
    denom = max(1.0, pot)
    pot_odds = to_call / (pot + to_call + 1e-9)
    spr = stack / denom
    return np.array([
        pot / 1000.0, stack / 1000.0, to_call / 1000.0, min_raise / 1000.0,
        pot_odds, spr,
        float(to_call == 0.0), float(stack <= to_call), float(stack <= min_raise),
        float(pot >= 100), float(pot >= 500), float(pot >= 1000),
    ], dtype=np.float32)


def compute_base_features(
    pid: int,
    hole: List[Card],
    community: List[Card],
    pot: float,
    stack: float,
    to_call: float,
    min_raise: float,
    num_players: int,
    button_pos: int,
    history: List[Dict[str, Any]] | None = None,
    cfg: FeatureConfig | None = None,
    use_equity: bool = True,
) -> np.ndarray:
    """Compute ~500 informative features about the current state.

    The resulting vector is size FeatureConfig.base_dim (default 480). Additional
    action encoding expands to total FeatureConfig.total_dim (default 512).
    """
    cfg = cfg or FeatureConfig()
    # Card-dependent features
    all_cards = list(hole) + list(community)
    rank_counts_all = _rank_counts(all_cards)  # 13
    suit_counts_all = _suit_counts(all_cards)  # 4
    rank_counts_board = _rank_counts(community)  # 13
    suit_counts_board = _suit_counts(community)  # 4
    board_tex = _board_texture(community)  # 16
    straight_vec_all = _straight_draw_vector(all_cards)  # 10
    straight_vec_board = _straight_draw_vector(community)  # 10
    hole_feats = _hole_features(hole)  # 16

    # Equity estimate (fast NN + MC blend from utils.estimate_equity)
    if use_equity:
        try:
            eq = float(estimate_equity(hole, community, max(0, num_players - 1)))
        except Exception:
            eq = 0.5
    else:
        eq = 0.5

    pot_feats = _pot_features(pot, stack, to_call, min_raise)  # 12

    # Position
    rel_pos = (pid - button_pos) % max(1, num_players)
    pos_onehot = np.zeros(10, dtype=np.float32)
    pos_onehot[min(rel_pos, 9)] = 1.0
    player_count = np.zeros(10, dtype=np.float32)
    player_count[min(num_players, 10) - 1] = 1.0

    # History features
    hist = _history_features(history or [], cfg.max_players, cfg.rounds)  # 10*4*4 = 160

    # Presence vectors
    pres_all = _presence_vector(all_cards)  # 15
    pres_board = _presence_vector(community)  # 15

    # Aggregate and pad to cfg.base_dim
    parts = [
        rank_counts_all, suit_counts_all,
        rank_counts_board, suit_counts_board,
        board_tex, straight_vec_all, straight_vec_board,
        hole_feats, pot_feats,
        np.array([eq], dtype=np.float32),
        pos_onehot, player_count,
        hist,
        pres_all, pres_board,
    ]
    vec = np.concatenate(parts).astype(np.float32)
    if vec.shape[0] > cfg.base_dim:
        vec = vec[: cfg.base_dim]
    elif vec.shape[0] < cfg.base_dim:
        vec = np.pad(vec, (0, cfg.base_dim - vec.shape[0]))
    return vec


def build_action_features(
    base_features: np.ndarray,
    action: int,
    raise_fraction: float | None = None,
    cfg: FeatureConfig | None = None,
) -> np.ndarray:
    cfg = cfg or FeatureConfig()
    act = int(action)
    raise_fraction = float(raise_fraction or 0.0)
    act_onehot = np.zeros(8, dtype=np.float32)
    act_onehot[min(max(act, 0), 7)] = 1.0
    # Encode raise fraction into bins and raw value
    bins = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    rf_bin = np.zeros(len(bins), dtype=np.float32)
    rf_bin[np.argmin(np.abs(bins - raise_fraction))] = 1.0
    action_block = np.concatenate([
        act_onehot, rf_bin, np.array([raise_fraction], dtype=np.float32),
        np.zeros(max(0, cfg.action_dim - (len(act_onehot) + len(rf_bin) + 1)), dtype=np.float32),
    ])
    out = np.concatenate([base_features, action_block]).astype(np.float32)
    if out.shape[0] > cfg.total_dim:
        out = out[: cfg.total_dim]
    elif out.shape[0] < cfg.total_dim:
        out = np.pad(out, (0, cfg.total_dim - out.shape[0]))
    return out
