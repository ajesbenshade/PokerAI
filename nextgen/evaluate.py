from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np

from .mechanics import PokerTable
from .ev_model import EVModel
from .agents import RandomAgent, SmartEVAgent
from .plotting import plot_winrates
from .cfr import toy_preflop_cfr


def load_ev_model(path_dir: str) -> EVModel:
    model = EVModel()
    with open(os.path.join(path_dir, "ev_model.meta.json"), "r") as f:
        meta = json.load(f)
    with open(os.path.join(path_dir, "ev_model.bin"), "rb") as f:
        payload = f.read()
    model.load(meta["format"], payload)
    return model


def evaluate_match(ev_dir: str, games: int = 200) -> Dict[int, float]:
    ev = load_ev_model(ev_dir)
    agent = SmartEVAgent(ev)
    rng = np.random.RandomState(123)

    table = PokerTable(num_players=6, starting_stack=100, small_blind=1, big_blind=2)
    agents = [agent] + [RandomAgent(rng) for _ in range(table.num_players - 1)]

    results = {i: 0.0 for i in range(table.num_players)}
    for g in range(games):
        def policy(pid, obs):
            return agents[pid].act(obs)
        payouts = table.play_hand(policy)
        for pid, amt in payouts.items():
            results[pid] += float(amt)
        table.button = (table.button + 1) % table.num_players
    return results


def estimate_exploitability_preflop(iters: int = 2000) -> float:
    """Return a toy exploitability-like metric using a simplified preflop CFR node.

    This is not a full-game exploitability but provides a quick sanity signal:
    lower is better. It runs a regret-matching loop on a 3-action node and
    returns the RMSE to uniform as a crude proxy.
    """
    strat = toy_preflop_cfr(iterations=iters)
    # Distance from uniform strategy
    uni = np.ones_like(strat) / len(strat)
    rmse = float(np.sqrt(np.mean((strat - uni) ** 2)))
    return rmse


def main():
    import argparse
    p = argparse.ArgumentParser(description="Evaluate EV agent vs random bots")
    p.add_argument("--model", type=str, default="artifacts", help="model dir")
    p.add_argument("--games", type=int, default=200, help="hands to play")
    p.add_argument("--plot", action="store_true", help="plot win rates")
    p.add_argument("--exploit", action="store_true", help="report toy exploitability metric")
    args = p.parse_args()

    res = evaluate_match(args.model, args.games)
    print("Results (chip deltas):", res)
    if args.plot:
        plot_winrates(res)
    if args.exploit:
        ex = estimate_exploitability_preflop(2000)
        print(f"Toy exploitability metric (lower=better): {ex:.4f}")


if __name__ == "__main__":
    main()
