from __future__ import annotations

import argparse
import os
import sys
import time
import json
from typing import Tuple

import numpy as np

from .selfplay import SelfPlayConfig, simulate_dataset, parallel_simulate_dataset
from .ev_model import EVModel
from .tune import tune_ev_model


def train_once(games: int, out_dir: str, parallel: bool = False, workers: int = 4,
               tune_trials: int = 0) -> Tuple[str, float]:
    cfg = SelfPlayConfig(max_hands=games)
    if parallel:
        X, y = parallel_simulate_dataset(total_hands=games, cfg=cfg, workers=workers, agent_kind="datagen")
    else:
        X, y = simulate_dataset(hands=games, cfg=cfg, agent_kind="datagen")
    n = X.shape[0]
    if n == 0:
        raise RuntimeError("No data generated; simulation failed.")
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    model = EVModel()
    # Optional hyperparam tuning (xgboost backend only)
    if tune_trials > 0:
        try:
            best_params, best = tune_ev_model(X_train, y_train, n_trials=tune_trials)
            if best_params:
                print(f"Optuna best rmse={best:.4f} params={best_params}")
        except Exception as e:
            print(f"Tuning skipped: {e}")
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
    p.add_argument("--parallel", action="store_true", help="generate dataset in parallel via Ray")
    p.add_argument("--workers", type=int, default=4, help="number of Ray workers")
    p.add_argument("--tune", type=int, default=0, help="Optuna tuning trials (0 to disable)")
    args = p.parse_args(argv)

    fmt, rmse = train_once(args.games, args.out, parallel=args.parallel, workers=args.workers, tune_trials=args.tune)
    print(f"Trained EV model ({fmt}), RMSE={rmse:.3f}, saved to {args.out}")


if __name__ == "__main__":
    main()
