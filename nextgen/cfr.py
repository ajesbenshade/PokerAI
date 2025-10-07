from __future__ import annotations

from typing import Dict, List
import numpy as np


class RegretMatcher:
    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.cum_regret = np.zeros(num_actions, dtype=np.float64)
        self.cum_policy = np.zeros(num_actions, dtype=np.float64)

    def get_strategy(self) -> np.ndarray:
        r = np.maximum(self.cum_regret, 0.0)
        s = r / r.sum() if r.sum() > 1e-9 else np.ones(self.num_actions) / self.num_actions
        self.cum_policy += s
        return s

    def get_average_strategy(self) -> np.ndarray:
        tot = self.cum_policy.sum()
        if tot <= 1e-9:
            return np.ones(self.num_actions) / self.num_actions
        return self.cum_policy / tot

    def update(self, action_utilities: np.ndarray):
        u_bar = float(action_utilities @ self.get_strategy())
        self.cum_regret += action_utilities - u_bar


def toy_preflop_cfr(iterations: int = 1000) -> np.ndarray:
    """A tiny CFR example on a 3-action node to demonstrate integration.

    Returns the average strategy after iterations.
    """
    rm = RegretMatcher(num_actions=3)
    for _ in range(iterations):
        # Toy utilities for Fold/Call/Raise varying slightly per iter
        a = np.array([0.0, 0.1, 0.2]) + np.random.randn(3) * 0.01
        rm.update(a)
    return rm.get_average_strategy()

