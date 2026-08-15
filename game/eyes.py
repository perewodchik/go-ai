"""
eyes.py — Eye detection and the optional "never fill your own last two eyes"
move restriction.

WHY THIS EXISTS
---------------
A group with two eyes is *unconditionally alive*: no sequence of opponent moves
can ever capture it. An untrained (or half-trained) network routinely plays a
stone into one of its own two eyes, which turns an immortal group into a
one-eyed group that then dies. Those moves are pure self-destruction, they never
serve any purpose, and the network wastes an enormous amount of self-play
learning that they are bad. `restrict_eye_fill` removes them from the action set
entirely so search never spends a simulation on them.

THE EYE DEFINITION USED HERE (and why it is exact)
--------------------------------------------------
An **eye of a chain C** is a liberty `p` of `C` such that *every* on-board
neighbour of `p` is a stone of `C`.

That definition is deliberately strict, and it makes the two-eye claim a
theorem rather than a heuristic:

    If a chain C has two distinct eyes p1 and p2, C can never be captured.

    Proof. The opponent can only capture C by removing its last liberty, and
    p1/p2 are liberties of C. Playing at p1 places a stone whose every
    neighbour belongs to C, so the new stone forms a single-stone group with
    zero liberties. It captures nothing (C still has p2), so the move is
    suicide — illegal. Same for p2. No other move on the board can occupy p1 or
    p2. Therefore C keeps at least two liberties forever. ∎

Everything about the classical "false eye" problem — the diagonal-count rule,
recursive false eyes, eyes that only exist if some *other* group lives — is
sidestepped, because this definition never counts a point as an eye unless the
stones around it are already one connected chain. A false eye is exactly a point
whose surrounding stones are NOT one chain, so it is never counted:

        · X ·          The marked point `a` is a classic false eye. Its four
        X a X          neighbours belong to two different chains, so `a` is not
        · X ·          an eye of either one, the restriction never fires, and
                       the (necessary!) connecting move at `a` stays legal.

The cost of the strictness is coverage, not correctness: some genuinely alive
shapes are not recognised, and the bot is simply allowed to play there as
before. That is the safe direction to be wrong in — see `forbidden_eye_fills`.

WHAT IS FORBIDDEN
-----------------
A move by `color` at `p` is forbidden iff `p` is an eye of one of `color`'s
chains and that chain has **exactly two** eyes. Rationale for each bound:

  - Exactly 2 → filling takes the chain from provably-alive to one-eyed. This is
    the blunder we are removing.
  - 3 or more → filling still leaves two eyes, so the group stays alive. Not a
    blunder, stays legal.
  - Exactly 1 → the group is not alive to begin with; filling may genuinely be
    needed (e.g. as part of making a bigger eye space), so it stays legal.
  - 0 → nothing to fill.

Multi-point eye spaces are handled for free by the definition: in a two-space
eye, each of the two points has the *other* empty point as a neighbour, so
neither is an eye of the chain. A two-space eye correctly counts as zero eyes
here, and playing inside it (which is how you split it into two real eyes)
is never restricted.
"""

from typing import FrozenSet, Set, Tuple

from game.board import Board, EMPTY

Point = Tuple[int, int]


def chain_eyes(board: Board, chain: FrozenSet[Point]) -> Set[Point]:
    """
    Eyes of `chain`: liberties whose every on-board neighbour is in `chain`.

    Board edges need no special casing — `Board.neighbors` only returns
    on-board points, so a corner eye (2 neighbours) and an edge eye (3) are
    tested exactly like a centre eye (4).
    """
    eyes = set()
    for liberty in board.get_liberties(chain):
        if all(n in chain for n in board.neighbors(*liberty)):
            eyes.add(liberty)
    return eyes


def forbidden_eye_fills(board: Board, color: int) -> Set[Point]:
    """
    All points where `color` playing would destroy one of its own chain's only
    two eyes — i.e. the moves `restrict_eye_fill` removes.

    Never returns a point that is a legal capture, a connection, or any other
    useful move: an eye point (by this module's definition) is surrounded
    entirely by one friendly chain, so playing there touches no opponent stone,
    joins no second chain of ours, and can capture nothing. The only thing such
    a move can accomplish is removing one of the group's own liberties.
    """
    forbidden: Set[Point] = set()
    for chain in board.get_all_groups(color):
        eyes = chain_eyes(board, chain)
        if len(eyes) == 2:
            forbidden |= eyes
    return forbidden


def is_forbidden_eye_fill(board: Board, color: int, row: int, col: int) -> bool:
    """
    Single-point form of `forbidden_eye_fills`, without scanning the board.

    Cheap in the overwhelmingly common case: any neighbour that is not a
    friendly stone rejects the point before a group search happens.
    """
    if not board.is_on_board(row, col) or board.grid[row, col] != EMPTY:
        return False

    neighbors = board.neighbors(row, col)
    if not neighbors:
        return False
    if any(board.grid[nr, nc] != color for nr, nc in neighbors):
        return False

    # All neighbours are friendly stones — but they must be ONE chain for this
    # to be an eye. (This is the false-eye rejection.)
    chain = board.get_group(*neighbors[0])
    if any(n not in chain for n in neighbors[1:]):
        return False

    return len(chain_eyes(board, chain)) == 2
