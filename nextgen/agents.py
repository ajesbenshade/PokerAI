from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import random
import numpy as np

from config import Action
from datatypes import Card
from .features import FeatureConfig, compute_base_features, build_action_features
from .ev_model import EVModel


def legal_action_set(obs: dict) -> List[Tuple[int, Optional[float]]]:
    """Compute a simple legal action set from observation.

    Returns list of (action, raise_amount) pairs. For raises, the raise_amount
    is the absolute raise over call (chips), not total to put in.
    """
    acts: List[Tuple[int, Optional[float]]] = []
    to_call = int(max(0, obs.get("to_call", 0)))
    stack = int(max(0, obs.get("stack", 0)))
    min_raise = int(max(0, obs.get("min_raise", 0)))

    # Fold always legal if facing a bet
    if to_call > 0:
        acts.append((Action.FOLD.value, None))
    # Call if we have chips
    if stack >= to_call and to_call > 0:
        acts.append((Action.CALL.value, None))
    if to_call == 0:
        # Check option available: treat as call
        acts.append((Action.CALL.value, None))

    # Raises
    if stack > to_call + min_raise:
        # Candidate raise sizes as pot fractions
        pot = float(obs.get("pot", 0))
        candidates = [0.5, 1.0, 1.5]
        for frac in candidates:
            amt = int(max(min_raise, frac * (pot + to_call)))
            if amt + to_call <= stack:
                acts.append((Action.RAISE.value, amt))
        # All-in raise allowed even if < min_raise; does not reopen
        all_in_amt = stack - to_call
        if all_in_amt > 0:
            acts.append((Action.RAISE.value, all_in_amt))
    elif stack > to_call:
        # Can only shove as a non-reopening all-in
        all_in_amt = stack - to_call
        acts.append((Action.RAISE.value, all_in_amt))

    # Deduplicate by amount
    seen = set()
    uniq = []
    for a, r in acts:
        key = (a, r if r is None else int(r))
        if key not in seen:
            seen.add(key)
            uniq.append((a, r))
    return uniq


@dataclass
class RandomAgent:
    rng: object  # random.Random or numpy RandomState

    def act(self, obs: dict) -> Tuple[int, Optional[int]]:
        acts = legal_action_set(obs)
        try:
            import numpy as np  # noqa
            # If numpy RNG provided
            if hasattr(self.rng, "choice") and not isinstance(self.rng, random.Random):
                idx = int(self.rng.choice(len(acts)))
                return acts[idx]
        except Exception:
            pass
        return random.choice(acts)


@dataclass
class SmartEVAgent:
    ev_model: EVModel
    cfg: FeatureConfig = None
    epsilon: float = 0.10  # exploration (10%)
    model_tol: float = 0.05  # choose any action within tol of best EV

    def act(self, obs: dict) -> Tuple[int, Optional[int]]:
        acts = legal_action_set(obs)
        if not acts:
            return Action.FOLD.value, None
        if random.random() < self.epsilon:
            return random.choice(acts)

        # Build features for each action and score
        if self.cfg is None:
            self.cfg = FeatureConfig()
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
            cfg=self.cfg,
        )
        feats = []
        act_items = []
        for a, r in acts:
            pot = float(obs.get("pot", 0.0))
            to_call = float(obs.get("to_call", 0.0))
            # Normalize raise fraction vs pot+call
            rf = 0.0
            if a == Action.RAISE.value and r is not None:
                denom = max(1.0, pot + to_call)
                rf = float(r) / denom
                rf = float(max(0.0, min(2.0, rf)))  # cap at 2x pot
            feats.append(build_action_features(base, a, rf, self.cfg))
            act_items.append((a, r))
        X = np.stack(feats)
        evs = self.ev_model.predict(X)
        # Greedy with tolerance: any action within (max - tol*|max|) is eligible
        max_ev = float(np.max(evs))
        tol = abs(max_ev) * self.model_tol
        elig = [i for i, v in enumerate(evs) if (max_ev - float(v)) <= tol]
        pick = int(random.choice(elig)) if elig else int(np.argmax(evs))
        return act_items[pick]


@dataclass
class HumanAgent:
    def act(self, obs: dict) -> Tuple[int, Optional[int]]:
        # Minimal console prompt (for quick sanity runs); use GUI/tools in VS Code instead
        print(f"You: hand={obs['hand']}, board={obs['community']}, to_call={obs['to_call']}, min_raise={obs['min_raise']} pot={obs['pot']}")
        acts = legal_action_set(obs)
        for i, (a, r) in enumerate(acts):
            print(i, ["FOLD", "CALL", "RAISE"][a], r)
        idx = int(input("Choose action index: "))
        idx = max(0, min(len(acts) - 1, idx))
        return acts[idx]


@dataclass
class DataGenAgent:
    rng: random.Random
    aggressive: float = 0.5

    def act(self, obs: dict) -> Tuple[int, Optional[int]]:
        acts = legal_action_set(obs)
        if not acts:
            return Action.FOLD.value, None
        # Bias towards raises based on aggressive
        raises = [(a, r) for a, r in acts if a == Action.RAISE.value]
        others = [(a, r) for a, r in acts if a != Action.RAISE.value]
        if raises and self.rng.random() < self.aggressive:
            return self.rng.choice(raises)
        return self.rng.choice(others) if others else self.rng.choice(acts)


# Optional PPO integration via stable-baselines3
class PPOAgent:
    def __init__(self, policy):
        self.policy = policy

    def act(self, obs: dict) -> Tuple[int, Optional[int]]:
        try:
            import numpy as np  # noqa
            action, _ = self.policy.predict(obs, deterministic=True)
            if isinstance(action, (list, tuple)):
                action = action[0]
            a = int(action)
            if a == Action.RAISE.value:
                # Default to min-raise when model chooses raise without sizing
                return a, int(max(1, obs.get("min_raise", 0)))
            return a, None
        except Exception:
            # Fallback
            acts = legal_action_set(obs)
            return random.choice(acts) if acts else (Action.FOLD.value, None)
