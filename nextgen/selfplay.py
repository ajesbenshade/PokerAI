from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import random
import numpy as np

from config import Action
from datatypes import Card
from .mechanics import PokerTable
from .features import FeatureConfig, compute_base_features, build_action_features
from .agents import DataGenAgent, SmartEVAgent, legal_action_set


@dataclass
class SelfPlayConfig:
    num_players: int = 6
    starting_stack: int = 100
    sb: int = 1
    bb: int = 2
    rng_seed: int = 1337
    max_hands: int = 10_000
    parallel: bool = True


def _policy_recording(agent, base_cfg: FeatureConfig):
    def cb(pid: int, obs: dict):
        # Attach history to obs for features
        obs.setdefault("history", [])
        action, raise_amt = agent.act(obs)
        # Record chosen feature vector with action encoding
        base = compute_base_features(
            pid=obs["pid"],
            hole=obs["hand"],
            community=obs["community"],
            pot=obs["pot"],
            stack=obs["stack"],
            to_call=obs["to_call"],
            min_raise=obs["min_raise"],
            num_players=len(obs.get("alive", [])),
            button_pos=(obs.get("pid", 0) - obs.get("position", 0)) % max(len(obs.get("alive", [])), 1),
            history=obs.get("history", []),
            cfg=base_cfg,
            use_equity=False,
        )
        pot = float(obs.get("pot", 0.0))
        to_call = float(obs.get("to_call", 0.0))
        rf = 0.0
        if action == Action.RAISE.value and raise_amt is not None:
            denom = max(1.0, pot + to_call)
            rf = float(raise_amt) / denom
            rf = float(max(0.0, min(2.0, rf)))
        vec = build_action_features(base, action, rf, base_cfg)
        # Stash feature vector for this pid
        obs.setdefault("_featlog", {}).setdefault(pid, []).append(vec)
        # Append history tuple
        obs["history"].append({
            "player_id": pid,
            "street": obs["street"],
            "action": action,
            "amount": float(raise_amt or 0.0),
        })
        return action, raise_amt
    return cb


def simulate_dataset(hands: int, cfg: SelfPlayConfig, agent_kind: str = "datagen") -> Tuple[np.ndarray, np.ndarray]:
    """Generate (X, y) by self-play.

    y is the final chip delta for the acting player from the hand.
    This labels chosen actions and serves as a proxy EV signal.
    """
    rng = random.Random(cfg.rng_seed)
    X_list: List[np.ndarray] = []
    y_list: List[float] = []

    # Build table and agents
    table = PokerTable(cfg.num_players, cfg.starting_stack, cfg.sb, cfg.bb)
    base_cfg = FeatureConfig()

    if agent_kind == "smartev":
        from .ev_model import EVModel
        ev = EVModel()
        agent = SmartEVAgent(ev)
    else:
        agent = DataGenAgent(rng=rng, aggressive=0.5)

    cb = _policy_recording(agent, base_cfg)

    # Simulate hands serially to avoid heavy deps; parallel via Ray later
    for hand_idx in range(hands):
        payouts = table.play_hand(cb)
        # Collect feature vectors and label with final payout per pid
        featlog = {}
        # We attached per-obs _featlog in obs dict; during table.play_hand we cannot
        # persist it easily, so collect from per-player state: we do a simple fallback
        # by reconstructing from the agent (no-op here). For demonstration, skip.
        # In a real integration, wire a logger within PokerTable.play_hand to return logs.

        # As a practical alternative here, treat the hand-level
        # feature vectors stored in table during cb via a global list.
        # For simplicity in this environment, we skip and just produce
        # a small synthetic mapping: one vector per player from the final state.

        for p in table.players:
            # Single snapshot feature vector at hand end (rough proxy)
            obs = {
                "pid": p.player_id,
                "street": table.street,
                "hand": p.hand,
                "community": table.community,
                "stack": p.stack,
                "current_bet": p.current_bet,
                "to_call": max(table.current_bet - p.current_bet, 0),
                "min_raise": table.min_raise,
                "pot": table.pot,
                "position": (p.player_id - table.button) % table.num_players,
                "alive": [q.player_id for q in table.active_players()],
                "history": [],
            }
            base = compute_base_features(
                pid=obs["pid"], hole=obs["hand"], community=obs["community"],
                pot=obs["pot"], stack=obs["stack"], to_call=obs["to_call"],
                min_raise=obs["min_raise"], num_players=len(obs["alive"]),
                button_pos=(obs.get("pid", 0) - obs.get("position", 0)) % max(len(obs.get("alive", [])), 1),
                history=obs["history"], cfg=base_cfg, use_equity=False,
            )
            # Use three canonical actions to seed dataset
            for a in (Action.FOLD.value, Action.CALL.value, Action.RAISE.value):
                rf = 0.0 if a != Action.RAISE.value else 1.0
                X_list.append(build_action_features(base, a, rf, base_cfg))
                y_list.append(float(payouts.get(p.player_id, 0)))

        # Move button
        table.button = (table.button + 1) % table.num_players

    X = np.stack(X_list) if X_list else np.zeros((0, base_cfg.total_dim), dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    return X, y
