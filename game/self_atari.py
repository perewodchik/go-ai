"""
self_atari.py — the optional "no pointless self-atari" move restriction.

WHY THIS EXISTS
---------------
A self-atari is a move that leaves the group it joins with exactly one liberty,
so the opponent can capture it immediately. A weak network plays them
constantly, and each one costs a search slot and puts probability mass on a move
that simply hands over stones. Removing them sharpens the policy target and
concentrates the search budget in a single change — which is why HEURISTICS.md
ranks this the best remaining sample-efficiency lever.

HOW THIS DIFFERS FROM `restrict_eye_fill` — READ THIS FIRST
-----------------------------------------------------------
`game/eyes.py` rests on a **theorem**: filling one of your own two eyes can
never capture, connect or defend, so removing those moves cannot cost anything.

This module has no such proof. It is a **tuned assumption**, and the exemptions
below are judgement calls. Self-atari is genuinely correct in throw-ins,
snapbacks, nakade placement, eye-space reduction, ko-threat creation and seki
manipulation. The conditions carve out most of those, but "most" is the honest
word, and the stone-count threshold has nothing behind it except taste.

Consequences worth internalising before turning it on:

  * **The promotion gate cannot detect the damage.** Candidate and champion play
    under the same filter, so the gate measures relative strength under the
    handicap and is blind to the handicap's own cost. The gate Elo ladder will
    keep climbing while the model quietly gets worse. The only valid test is the
    twin-model A/B in HEURISTICS.md — train two models from identical weights,
    one filtered and one not, then play them head-to-head with the filter OFF
    for both sides.
  * **Expect a discontinuous step in `win_rate_vs_random`** at the iteration you
    switch it on, because only the network is filtered in that evaluation.

WHAT IS FORBIDDEN
-----------------
A move by `color` at `p` is a pointless self-atari iff, after placing the stone
and resolving captures:

  1. it captured **nothing**, and
  2. the group containing `p` has exactly **1** liberty, and
  3. that group is **larger than `max_stones`** (default 1).

Condition 1 exempts every ko capture and most snapbacks: if the move takes
stones off the board it is doing something, whatever its liberty count ends up
being. Condition 3 exempts throw-ins, which are the main small-group tesuji —
sacrificing one or two stones to wreck an eye space is ordinary good play, while
walking a five-stone group into atari almost never is.

Note that "the group containing p" includes any friendly groups the move
connects to. Joining a healthy group to a doomed one so that the whole thing
sits on one liberty is exactly the blunder worth removing, and it is caught here
because `own_size` and `own_liberties` describe the merged group.

COST
----
Effectively free, because it never simulates anything itself. `rules.simulate_move`
already walks the played point's neighbours to decide legality; asking it for
`need_group_facts` makes it retain the merged group's size and liberty set from
that same walk. A version that re-simulated each move would roughly double the
cost of the hottest function in the engine — see the profile in
IMPROVEMENT_PLAN.md — and could easily cost more throughput than the sharper
targets buy back.
"""

from typing import Set, Tuple

from game.board import Board
from game.rules import simulate_move

Point = Tuple[int, int]

# Default for `self_atari_max_stones`: sacrifices of a single stone stay legal.
DEFAULT_MAX_STONES = 1


def is_pointless_self_atari(simulation, max_stones: int = DEFAULT_MAX_STONES) -> bool:
    """
    Decide the rule from an already-computed `MoveSimulation`.

    The simulation must have been produced with `need_group_facts=True`;
    without it there is nothing to judge and the answer is False, which is the
    safe direction — the move stays playable.
    """
    if not simulation.is_legal:
        return False
    # Condition 1: a move that captures is never "pointless".
    if simulation.captured:
        return False
    # Group facts are only computed for non-capturing moves, so a None here
    # means the caller did not ask for them.
    if simulation.own_liberties is None or simulation.own_size is None:
        return False
    # Conditions 2 and 3.
    return len(simulation.own_liberties) == 1 and simulation.own_size > max_stones


def is_forbidden_self_atari(board: Board, color: int, row: int, col: int,
                            max_stones: int = DEFAULT_MAX_STONES,
                            ko_point=None, board_history_hashes=None) -> bool:
    """
    Single-point form, for tests and inspection. The search does not call this —
    it reuses the simulation it already performed for legality.
    """
    sim = simulate_move(board, color, row, col, ko_point, board_history_hashes,
                        need_group_facts=True)
    return is_pointless_self_atari(sim, max_stones)


def forbidden_self_ataris(board: Board, color: int,
                          max_stones: int = DEFAULT_MAX_STONES,
                          ko_point=None, board_history_hashes=None) -> Set[Point]:
    """
    Every point where `color` playing would be a pointless self-atari.

    Provided to mirror `game.eyes.forbidden_eye_fills` for tests and debugging.
    """
    forbidden: Set[Point] = set()
    for r in range(board.size):
        for c in range(board.size):
            if is_forbidden_self_atari(board, color, r, c, max_stones,
                                       ko_point, board_history_hashes):
                forbidden.add((r, c))
    return forbidden
