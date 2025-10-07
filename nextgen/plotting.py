from __future__ import annotations

from typing import Dict


def plot_winrates(results: Dict[int, float]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not installed; skipping plot.")
        return
    pids = sorted(results.keys())
    vals = [results[i] for i in pids]
    plt.figure(figsize=(6, 3))
    plt.bar([str(i) for i in pids], vals)
    plt.xlabel("Player ID")
    plt.ylabel("Total chips won")
    plt.title("Win rates (chip deltas)")
    plt.tight_layout()
    plt.show()

