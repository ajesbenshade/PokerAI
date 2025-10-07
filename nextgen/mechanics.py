from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from config import Action, Suit, Config
from datatypes import Card


@dataclass
class PlayerState:
    player_id: int
    stack: int
    hand: List[Card]
    folded: bool = False
    current_bet: int = 0
    all_in: bool = False
    hand_contribution: int = 0


class PokerTable:
    """Texas Hold'em engine with strict betting rules and side pots.

    - 4 rounds (preflop, flop, turn, river)
    - Min-raise enforcement; all-in exceptions do not reopen action
    - Side pot settlement and ties
    - Wheel straights and exact hand ranking delegated to hand_eval
    """

    def __init__(self, num_players: int = 6, starting_stack: int = 1000,
                 small_blind: int = 5, big_blind: int = 10):
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.sb = small_blind
        self.bb = big_blind

        self.players: List[PlayerState] = []
        self.button: int = 0
        self.deck: List[Card] = []
        self.community: List[Card] = []
        self.street: str = "preflop"  # preflop, flop, turn, river
        self.current_bet: int = 0
        self.min_raise: int = self.bb
        self.pot: int = 0
        self.last_raiser: Optional[int] = None
        self.to_act: int = 0
        self.betting_open: bool = False

        self.reset_players()

    # ----- Setup / dealing -----
    def reset_players(self):
        self.players = [
            PlayerState(player_id=i, stack=self.starting_stack, hand=[])
            for i in range(self.num_players)
        ]

    def shuffle_new_deck(self):
        from utils import create_deck
        import random
        self.deck = create_deck()
        random.shuffle(self.deck)

    def deal_hole(self):
        for p in self.players:
            p.hand = [self.deck.pop(), self.deck.pop()]

    def burn(self):
        if self.deck:
            self.deck.pop()

    def deal_flop(self):
        self.burn()
        self.community.extend([self.deck.pop(), self.deck.pop(), self.deck.pop()])

    def deal_turn(self):
        self.burn()
        self.community.append(self.deck.pop())

    def deal_river(self):
        self.burn()
        self.community.append(self.deck.pop())

    # ----- Hand lifecycle -----
    def start_hand(self):
        """Start a new hand: shuffle, deal, post blinds, set order."""
        self.community = []
        self.street = "preflop"
        self.current_bet = 0
        self.min_raise = self.bb
        self.pot = 0
        self.last_raiser = None
        for p in self.players:
            p.folded = False
            p.current_bet = 0
            p.all_in = False
            p.hand_contribution = 0

        self.shuffle_new_deck()
        self.deal_hole()
        self.post_blinds()

    def post_blinds(self):
        sb_pos = (self.button + 1) % self.num_players
        bb_pos = (self.button + 2) % self.num_players
        sb = self.players[sb_pos]
        bb = self.players[bb_pos]

        sb_pay = min(self.sb, sb.stack)
        sb.stack -= sb_pay
        sb.current_bet = sb_pay
        sb.hand_contribution += sb_pay

        bb_pay = min(self.bb, bb.stack)
        bb.stack -= bb_pay
        bb.current_bet = bb_pay
        bb.hand_contribution += bb_pay

        self.pot = sb_pay + bb_pay
        self.current_bet = bb_pay
        self.min_raise = self.bb
        # First to act preflop is UTG
        self.to_act = (bb_pos + 1) % self.num_players
        self.betting_open = True

    # ----- Action helpers -----
    def active_players(self) -> List[PlayerState]:
        return [p for p in self.players if not p.folded and (p.stack > 0 or p.current_bet > 0)]

    def next_seat(self, i: int) -> int:
        return (i + 1) % self.num_players

    def all_bets_matched(self) -> bool:
        # All non-folded players have matched current_bet or are all-in
        for p in self.active_players():
            if not p.all_in and p.current_bet != self.current_bet:
                return False
        return True

    def legal_raise_amount(self, player: PlayerState, raise_amount: int) -> Tuple[bool, bool]:
        """Return (is_legal, reopens_action)."""
        call_needed = max(self.current_bet - player.current_bet, 0)
        total = call_needed + raise_amount
        if total > player.stack:
            return False, False
        if raise_amount < self.min_raise:
            # All-in smaller than min-raise is allowed but does not reopen
            return (total == player.stack), False
        return True, True

    # ----- Player actions -----
    def act_fold(self, pid: int) -> None:
        p = self.players[pid]
        p.folded = True

    def act_call(self, pid: int) -> None:
        p = self.players[pid]
        call_needed = max(self.current_bet - p.current_bet, 0)
        pay = min(call_needed, p.stack)
        p.stack -= pay
        p.current_bet += pay
        p.hand_contribution += pay
        self.pot += pay
        if p.stack == 0:
            p.all_in = True

    def act_raise(self, pid: int, raise_amount: int) -> bool:
        p = self.players[pid]
        ok, reopens = self.legal_raise_amount(p, raise_amount)
        if not ok and not (raise_amount == 0 and p.stack == max(self.current_bet - p.current_bet, 0)):
            return False
        call_needed = max(self.current_bet - p.current_bet, 0)
        total = min(call_needed + raise_amount, p.stack)
        # Handle exact all-in when stack < total
        if call_needed + raise_amount > p.stack:
            total = p.stack
        p.stack -= total
        p.current_bet += total
        p.hand_contribution += total
        self.pot += total
        if p.current_bet > self.current_bet:
            self.min_raise = max(self.min_raise, p.current_bet - self.current_bet)
            self.current_bet = p.current_bet
            if reopens:
                self.last_raiser = pid
        if p.stack == 0:
            p.all_in = True
        return True

    # ----- Betting round flow -----
    def next_to_act(self) -> Optional[int]:
        # Find next player who can act (not folded, not all-in)
        for _ in range(self.num_players):
            self.to_act = self.next_seat(self.to_act)
            p = self.players[self.to_act]
            if not p.folded and not p.all_in:
                return self.to_act
        return None

    def is_round_done(self) -> bool:
        alive = [p for p in self.players if not p.folded]
        if len(alive) <= 1:
            return True
        return self.all_bets_matched()

    def end_betting_round(self):
        # Reset per-street bets; set next street and order
        for p in self.players:
            p.current_bet = 0
        self.current_bet = 0
        self.min_raise = self.bb
        self.last_raiser = None
        if self.street == "preflop":
            self.deal_flop()
            self.street = "flop"
        elif self.street == "flop":
            self.deal_turn()
            self.street = "turn"
        elif self.street == "turn":
            self.deal_river()
            self.street = "river"
        # First to act post-flop is left of the button
        self.to_act = (self.button + 1) % self.num_players

    # ----- Showdown and side pots -----
    def settle_showdown(self) -> Dict[int, int]:
        from nextgen.hand_eval import best_hand_rank
        alive = [p for p in self.players if not p.folded]
        if len(alive) == 1:
            # Winner takes all
            winner = alive[0]
            return {winner.player_id: self.pot}

        # Determine per-player hand ranks
        ranks: Dict[int, Tuple[int, Tuple[int, ...]]] = {}
        for p in alive:
            ranks[p.player_id] = best_hand_rank(p.hand, self.community)

        # Build side pots based on contributions
        contribs = sorted([(p.player_id, p.hand_contribution) for p in self.players], key=lambda x: x[1])
        thresholds = sorted(set([c for _, c in contribs if c > 0]))
        prev = 0
        pots: List[Tuple[int, List[int]]] = []  # (amount, eligible pids)
        if not thresholds:
            return {pid: 0 for pid in ranks}
        for t in thresholds:
            eligible = [pid for pid, c in contribs if c >= t]
            size = 0
            for _, c in contribs:
                size += max(min(c, t) - prev, 0)
            if size > 0 and eligible:
                pots.append((size, eligible))
            prev = t

        # Distribute each pot to best-ranked among eligible
        payouts = {p.player_id: 0 for p in self.players}
        for amount, elig in pots:
            # Winners among eligible
            elig_ranks = {pid: ranks[pid] for pid in elig if pid in ranks}
            if not elig_ranks:
                # Split equally if no ranks (shouldn't happen)
                share = amount // max(len(elig), 1)
                for pid in elig:
                    payouts[pid] += share
                continue
            best = min(elig_ranks.values())
            winners = [pid for pid, r in elig_ranks.items() if r == best]
            share = amount // len(winners)
            remainder = amount - share * len(winners)
            for i, pid in enumerate(winners):
                payouts[pid] += share + (1 if i < remainder else 0)
        return payouts

    # ----- Convenience: simulate a full hand with a policy callback -----
    def play_hand(self, policy_cb) -> Dict[int, int]:
        """Run a complete hand using a callback(pid, obs) -> (action, raise_amount).

        The observation is a dict with minimal info needed by agents.
        """
        self.start_hand()

        def make_obs(pid: int) -> dict:
            p = self.players[pid]
            return {
                "pid": pid,
                "street": self.street,
                "hand": p.hand,
                "community": self.community,
                "stack": p.stack,
                "current_bet": p.current_bet,
                "to_call": max(self.current_bet - p.current_bet, 0),
                "min_raise": self.min_raise,
                "pot": self.pot,
                "position": (pid - self.button) % self.num_players,
                "alive": [q.player_id for q in self.active_players()],
            }

        # Betting loop across streets
        while True:
            # If only one player left, immediate win
            if len([p for p in self.players if not p.folded]) <= 1:
                break
            # Ask actions until round ends
            acted_this_round = set()
            while True:
                pid = self.to_act
                p = self.players[pid]
                if p.folded or p.all_in:
                    nxt = self.next_to_act()
                    if nxt is None:
                        break
                    continue
                obs = make_obs(pid)
                act, amount = policy_cb(pid, obs)
                if act == Action.FOLD.value:
                    self.act_fold(pid)
                elif act == Action.CALL.value:
                    self.act_call(pid)
                else:
                    ok = self.act_raise(pid, int(amount or self.min_raise))
                    if not ok:
                        # Treat illegal raise as call if possible, else fold
                        to_call = max(self.current_bet - p.current_bet, 0)
                        if to_call > 0 and p.stack >= to_call:
                            self.act_call(pid)
                        else:
                            self.act_fold(pid)
                acted_this_round.add(pid)
                # If everyone matched and at least one bet happened, end round
                if self.is_round_done():
                    break
                nxt = self.next_to_act()
                if nxt is None:
                    break
            if self.street == "river" and self.is_round_done():
                break
            if not self.is_round_done():
                # Safety break to avoid infinite looping
                break
            self.end_betting_round()
        return self.settle_showdown()


def deal_example_deck() -> List[Card]:
    """Utility to create a deterministic shuffled deck for tests."""
    from utils import create_deck
    deck = create_deck()
    # No actual random shuffle to keep predictable order here
    return deck

