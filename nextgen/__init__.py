"""Next-gen PokerAI modules (mechanics, features, EV, agents).

This package is intentionally self-contained and optional. It integrates
with existing repo types (datatypes.Card/Suit/Action) but does not
modify current training code. You can adopt it incrementally.
"""

__all__ = [
    "mechanics",
    "hand_eval",
    "features",
    "ev_model",
    "agents",
    "selfplay",
    "train_agents",
    "evaluate",
    "plotting",
]

