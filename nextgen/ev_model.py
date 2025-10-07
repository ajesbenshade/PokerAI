from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


class EVModel:
    """Hybrid EV predictor using XGBoost if available, with safe fallbacks.

    - Input: feature vector per (state+action), size ~512 floats
    - Output: predicted chip EV (float)
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.backend = None
        self.model = None
        self._init_backend()

    def _init_backend(self) -> None:
        # Prefer xgboost, then scikit-learn, else a linear ridge via numpy
        try:
            import xgboost as xgb  # noqa: F401
            self.backend = "xgboost"
        except Exception:
            try:
                import sklearn  # noqa: F401
                self.backend = "sklearn"
            except Exception:
                self.backend = "numpy"

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> None:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        if self.backend == "xgboost":
            import xgboost as xgb
            params = {
                "max_depth": 6,
                "eta": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "nthread": 0,
                "seed": self.random_state,
            }
            dtrain = xgb.DMatrix(X, label=y)
            evals = []
            if X_val is not None and y_val is not None:
                dval = xgb.DMatrix(X_val, label=y_val)
                evals = [(dtrain, "train"), (dval, "val")]
            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=2000,
                evals=evals,
                early_stopping_rounds=50 if evals else None,
                verbose_eval=False,
            )
        elif self.backend == "sklearn":
            # GradientBoostingRegressor as portable fallback
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(
                n_estimators=600,
                learning_rate=0.05,
                max_depth=3,
                random_state=self.random_state,
            )
            self.model.fit(X, y)
        else:
            # Simple ridge regression via normal equation with Tikhonov regularization
            # Not robust, but works for demo
            reg = 1e-2
            Xb = np.c_[X, np.ones((X.shape[0], 1), dtype=X.dtype)]
            A = Xb.T @ Xb + reg * np.eye(Xb.shape[1], dtype=X.dtype)
            b = Xb.T @ y
            w = np.linalg.solve(A, b)
            self.model = (w[:-1], w[-1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.backend == "xgboost":
            import xgboost as xgb
            d = xgb.DMatrix(X)
            return self.model.predict(d)
        elif self.backend == "sklearn":
            return self.model.predict(X)
        else:
            w, b = self.model
            return X @ w + b

    def save(self) -> Tuple[str, bytes]:
        """Return (format, payload) for persistence. Caller writes to disk."""
        if self.backend == "xgboost":
            payload = self.model.save_raw()
            return "xgboost", payload
        elif self.backend == "sklearn":
            import pickle
            return "sklearn", pickle.dumps(self.model)
        else:
            import pickle
            return "numpy", pickle.dumps(self.model)

    def load(self, fmt: str, payload: bytes) -> None:
        if fmt == "xgboost":
            import xgboost as xgb
            self.backend = "xgboost"
            self.model = xgb.Booster()
            self.model.load_model(payload)
        elif fmt == "sklearn":
            import pickle
            self.backend = "sklearn"
            self.model = pickle.loads(payload)
        else:
            import pickle
            self.backend = "numpy"
            self.model = pickle.loads(payload)

