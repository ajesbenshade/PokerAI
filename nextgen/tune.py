from __future__ import annotations

from typing import Tuple
import numpy as np


def tune_ev_model(X: np.ndarray, y: np.ndarray, n_trials: int = 20) -> Tuple[dict, float]:
    try:
        import optuna
    except Exception:
        print("optuna not installed; skipping tuning.")
        return {}, float("inf")

    def objective(trial: "optuna.Trial") -> float:
        # Simple param sweep for xgboost backend
        import xgboost as xgb
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "nthread": 0,
        }
        res = xgb.cv(params, dtrain, num_boost_round=1000, nfold=3, early_stopping_rounds=50, verbose_eval=False)
        return float(res["test-rmse-mean"].min())

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, float(study.best_value)

