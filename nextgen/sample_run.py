from __future__ import annotations

from typing import Dict

from .mechanics import PokerTable
from .agents import RandomAgent


def main():
    import numpy as np
    rng = np.random.RandomState(0)
    table = PokerTable(num_players=4, starting_stack=50, small_blind=1, big_blind=2)
    agents = [RandomAgent(rng) for _ in range(table.num_players)]
    for g in range(5):
        def pol(pid, obs):
            return agents[pid].act(obs)
        payouts: Dict[int, int] = table.play_hand(pol)
        print(f"Hand {g+1} community={table.community} payouts={payouts} pot={table.pot}")
        table.button = (table.button + 1) % table.num_players


if __name__ == "__main__":
    main()

