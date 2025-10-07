from __future__ import annotations

import argparse
import os
import sys
import time
import json
from typing import Tuple

import numpy as np

from .selfplay import SelfPlayConfig, simulate_dataset
from .ev_model import EVModel


def train_once(games: int, out_dir: str) -> Tuple[str, float]:
    cfg = SelfPlayConfig(max_hands=games)
    X, y = simulate_dataset(hands=games, cfg=cfg, agent_kind="datagen")
    n = X.shape[0]
    if n == 0:
        raise RuntimeError("No data generated; simulation failed.")
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    model = EVModel()
    t0 = time.time()
    model.fit(X_train, y_train, X_val, y_val)
    train_time = time.time() - t0
    preds = model.predict(X_val)
    rmse = float(np.sqrt(np.mean((preds - y_val) ** 2)))

    os.makedirs(out_dir, exist_ok=True)
    fmt, payload = model.save()
    with open(os.path.join(out_dir, "ev_model.bin"), "wb") as f:
        f.write(payload)
    with open(os.path.join(out_dir, "ev_model.meta.json"), "w") as f:
        json.dump({"format": fmt, "rmse": rmse, "games": games, "train_time_sec": train_time}, f, indent=2)
    return fmt, rmse


def main(argv=None):
    p = argparse.ArgumentParser(description="Train EV model via self-play dataset")
    p.add_argument("--games", type=int, default=1000, help="hands to simulate")
    p.add_argument("--out", type=str, default="artifacts", help="output directory")
    args = p.parse_args(argv)

    fmt, rmse = train_once(args.games, args.out)
    print(f"Trained EV model ({fmt}), RMSE={rmse:.3f}, saved to {args.out}")


if __name__ == "__main__":
    main()

