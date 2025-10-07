from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np

from .mechanics import PokerTable
from .ev_model import EVModel
from .agents import RandomAgent, SmartEVAgent
from .plotting import plot_winrates


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


def main():
    import argparse
    p = argparse.ArgumentParser(description="Evaluate EV agent vs random bots")
    p.add_argument("--model", type=str, default="artifacts", help="model dir")
    p.add_argument("--games", type=int, default=200, help="hands to play")
    p.add_argument("--plot", action="store_true", help="plot win rates")
    args = p.parse_args()

    res = evaluate_match(args.model, args.games)
    print("Results (chip deltas):", res)
    if args.plot:
        plot_winrates(res)


if __name__ == "__main__":
    main()

