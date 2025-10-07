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


def _policy_recording(agent, base_cfg: FeatureConfig, featlog: dict):
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
        # Store per-decision features by pid
        featlog.setdefault(pid, []).append(vec)
        # Append history tuple (minimal)
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

    # Simulate hands serially; parallel variants provided below via Ray
    for hand_idx in range(hands):
        # Per-hand feature log by pid
        featlog: dict[int, list[np.ndarray]] = {}
        cb = _policy_recording(agent, base_cfg, featlog)
        payouts = table.play_hand(cb)
        # Chip delta label per pid: payout minus contributed chips
        deltas = {p.player_id: float(payouts.get(p.player_id, 0) - p.hand_contribution) for p in table.players}
        # Flush logs to dataset
        for pid, feats in featlog.items():
            for v in feats:
                X_list.append(v)
                y_list.append(deltas.get(pid, 0.0))
        # Move button
        table.button = (table.button + 1) % table.num_players

    X = np.stack(X_list) if X_list else np.zeros((0, base_cfg.total_dim), dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    return X, y


def parallel_simulate_dataset(total_hands: int, cfg: SelfPlayConfig, workers: int = 4,
                              agent_kind: str = "datagen") -> Tuple[np.ndarray, np.ndarray]:
    """Parallel dataset generation with Ray.

    Falls back to serial if ray is unavailable.
    """
    try:
        import ray  # type: ignore
    except Exception:
        # Fallback: serial
        return simulate_dataset(total_hands, cfg, agent_kind)

    if not ray.is_initialized():
        try:
            ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
        except Exception:
            return simulate_dataset(total_hands, cfg, agent_kind)

    # Chunk work
    per = max(1, total_hands // max(1, workers))
    batches = [per] * workers
    batches[-1] += total_hands - per * workers

    @ray.remote
    def _worker(h: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        local_cfg = SelfPlayConfig(
            num_players=cfg.num_players,
            starting_stack=cfg.starting_stack,
            sb=cfg.sb,
            bb=cfg.bb,
            rng_seed=seed,
            max_hands=h,
            parallel=False,
        )
        return simulate_dataset(h, local_cfg, agent_kind)

    seeds = [int(cfg.rng_seed + i * 7919) for i in range(workers)]
    futs = [_worker.remote(batches[i], seeds[i]) for i in range(workers) if batches[i] > 0]
    outs = ray.get(futs)
    # Concatenate
    Xs, ys = zip(*outs) if outs else ([], [])
    if not Xs:
        return np.zeros((0, FeatureConfig().total_dim), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y
