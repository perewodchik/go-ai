"""
rules.py — Go rule enforcement.

Handles move validation, stone captures, ko rule, and superko detection.
This module is the "referee" — it determines what's legal and applies moves.

KEY RULES:
1. Stones are placed on empty intersections.
2. After placing, opponent groups with 0 liberties are captured (removed).
3. Self-capture (suicide) is ILLEGAL — if placing would leave your own group
   with 0 liberties AND you don't capture anything, the move is illegal.
4. Ko rule: you cannot recreate the immediately previous board state.
   (Prevents infinite capture-recapture loops.)
5. Superko: SITUATIONAL superko — you cannot recreate any previous
   (board state, player to move) pair. (Prevents longer loops.)

IMPORTANT: The ko rule here follows the "situational superko" variant, which is
what the Chinese ruleset on online-go.com enforces. The difference from
POSITIONAL superko is not academic: a position can legitimately repeat with the
other player to move — e.g. Black captures two stones, White recaptures one and
the board happens to match a position from two moves ago. Positional superko
called that illegal, so a bot playing on OGS saw the opponent make a move our
board refused, and the two games diverged mid-match (game 89823532, move 58).

Superko keys are therefore (board hash, player to move), produced by
`situational_key()`. Simple ko (only checking the last position) is a subset.
"""

import numpy as np
from typing import Optional, Tuple, List, Set, FrozenSet
from game.board import Board, EMPTY, BLACK, WHITE, opponent


# Side-to-move component of a superko key. Black is 0 so that a key for a
# black-to-move position is just the board hash — which keeps stored hashes and
# any caller that predates situational superko meaning the same thing.
_SIDE_TO_MOVE_HASH = {BLACK: 0, WHITE: 0x9E3779B97F4A7C15}


def situational_key(board_hash: int, player_to_move: int) -> int:
    """
    The superko key for a position: which stones are where AND whose turn it is.
    Sets of these are what `board_history_hashes` holds everywhere below.
    """
    return board_hash ^ _SIDE_TO_MOVE_HASH[player_to_move]


class MoveResult:
    """
    Result of applying a move. Returned by apply_move().
    
    Attributes:
        is_legal: Whether the move was legal and applied.
        captured_stones: Set of positions where opponent stones were removed.
        captured_count: Number of captured stones (for prisoner counting).
        ko_point: If this capture created a ko situation, this is the point
                  that the opponent cannot immediately recapture. None otherwise.
        error: Description of why the move was illegal (if is_legal=False).
    """
    
    def __init__(self, is_legal: bool = True, captured_stones: set = None,
                 captured_count: int = 0, ko_point: Optional[Tuple[int, int]] = None,
                 error: str = ""):
        self.is_legal = is_legal
        self.captured_stones = captured_stones or set()
        self.captured_count = captured_count
        self.ko_point = ko_point
        self.error = error


def get_adjacent_opponent_groups(board: Board, row: int, col: int,
                                  color: int) -> List[FrozenSet[Tuple[int, int]]]:
    """
    Find all opponent groups adjacent to position (row, col).
    Used to check which groups might be captured by placing here.
    """
    opp = opponent(color)
    seen_groups = []
    seen_positions = set()
    
    for nr, nc in board.neighbors(row, col):
        if board.grid[nr, nc] == opp and (nr, nc) not in seen_positions:
            group = board.get_group(nr, nc)
            seen_positions |= group
            seen_groups.append(group)
    
    return seen_groups


class MoveSimulation:
    """
    Everything a caller needs to know about a hypothetical move, worked out
    without touching the board.

    Attributes:
        is_legal: Whether the move may be played.
        error: Why not, when it may not.
        captured: Positions of opponent stones the move removes.
        new_hash: Zobrist hash of the resulting position.
        ko_point: The ko point the move creates, or None.
        own_size: Stones in the group the move would belong to (itself plus any
            friendly groups it joins). Only populated when `need_group_facts`
            was requested AND the move captures nothing — see `simulate_move`.
        own_liberties: Liberties that group would have. Same condition.
    """

    __slots__ = ('is_legal', 'error', 'captured', 'new_hash', 'ko_point',
                 'own_size', 'own_liberties')

    def __init__(self, is_legal: bool, error: str = "",
                 captured: FrozenSet[Tuple[int, int]] = frozenset(),
                 new_hash: Optional[int] = None,
                 ko_point: Optional[Tuple[int, int]] = None,
                 own_size: Optional[int] = None,
                 own_liberties: Optional[Set[Tuple[int, int]]] = None):
        self.is_legal = is_legal
        self.error = error
        self.captured = captured
        self.new_hash = new_hash
        self.ko_point = ko_point
        self.own_size = own_size
        self.own_liberties = own_liberties


def simulate_move(board: Board, color: int, row: int, col: int,
                  ko_point: Optional[Tuple[int, int]] = None,
                  board_history_hashes: Optional[Set[int]] = None,
                  need_group_facts: bool = False) -> MoveSimulation:
    """
    Work out the full consequences of a move without copying or mutating
    anything. This is the single hot path of the whole engine — `is_legal_move`,
    `apply_move` and `get_legal_moves` all go through it, and a 96-simulation
    search calls it around 8,000 times.

    It used to answer by brute force: copy the board, place the stone, re-derive
    every adjacent group, remove the dead ones and read the result back. That
    made it 58% of search time, and `apply_move` paid for the whole thing twice
    — once to validate, once to actually play.

    Everything below is derived from adjacency instead, using two facts that
    hold because (row, col) is empty and orthogonally adjacent to the groups in
    question:

      * (row, col) is necessarily a liberty of every adjacent group. So an
        adjacent OPPONENT group is captured exactly when it has 1 liberty, and
        an adjacent FRIENDLY group keeps the merged group alive exactly when it
        has more than 1 — no post-move board needed to count them.
      * Zobrist hashing is XOR of per-(point, colour) values, so the resulting
        hash is the current hash XOR the placed stone XOR each captured stone.
        That gives an exact superko check with no board copy at all.

    `need_group_facts` additionally reports the size and liberties of the group
    the move would create, which is what the self-atari filter needs. Those are
    filled in only when the move could actually BE a self-atari:

      * it captures nothing (a capture is exempt from the rule, and it also
        frees points whose ownership would have to be re-derived), and
      * it has at most one empty neighbour. Every empty neighbour of the played
        point is a liberty of the resulting group, so two of them already put
        the group on two liberties and no amount of group-walking can change
        that.

    Otherwise both stay None, which `is_pointless_self_atari` reads as "not a
    self-atari". That second test is what keeps the filter cheap: it rejects the
    large majority of points — anything with elbow room — before a single group
    is walked.
    """
    size = board.size

    # Rule 1: Must be on the board
    if not (0 <= row < size and 0 <= col < size):
        return MoveSimulation(False, "Position is off the board")

    grid = board.grid

    # Rule 2: Must be empty
    if grid[row, col] != EMPTY:
        return MoveSimulation(False, "Position is already occupied")

    # Rule 3: Simple ko check (before any work)
    if ko_point is not None and (row, col) == ko_point:
        return MoveSimulation(False, "Ko rule violation: cannot recapture immediately")

    opp = opponent(color)

    neighbours = board.neighbors(row, col)
    empty_neighbours = 0          # Each one is a liberty of the resulting group
    captured = set()              # Opponent stones this move removes
    opponent_seen = set()         # Opponent stones already accounted for
    friendly_seen = set()         # Friendly stones already accounted for
    friendly_alive = False        # Some adjacent friendly group has another liberty
    friendly_neighbours = []      # Seeds of the groups this move would join
    friendly_walked = []          # Groups already resolved, kept for reuse below

    for nr, nc in neighbours:
        value = grid[nr, nc]
        if value == EMPTY:
            empty_neighbours += 1
        elif value == opp:
            if (nr, nc) in opponent_seen:
                continue
            group = board.get_group(nr, nc)
            opponent_seen |= group
            # One liberty means that liberty is (row, col) — see the docstring.
            if board.liberty_count(group) == 1:
                captured |= group
        else:
            # Recording the seed is free; walking the group is not, so keep the
            # early exit once the move is already known not to be suicide.
            friendly_neighbours.append((nr, nc))
            if friendly_alive or (nr, nc) in friendly_seen:
                continue
            group = board.get_group(nr, nc)
            friendly_seen |= group
            friendly_walked.append(group)
            if board.liberty_count(group) > 1:
                friendly_alive = True

    # Rule 4: Suicide — legal if the move captures, if the stone has a liberty
    # of its own, or if it merges with a group that has one elsewhere.
    if not captured and not empty_neighbours and not friendly_alive:
        return MoveSimulation(
            False, "Suicide: move would leave your group with no liberties")

    zobrist = board.zobrist.table
    new_hash = board.board_hash ^ zobrist[(row * size + col, color)]
    for cr, cc in captured:
        new_hash ^= zobrist[(cr * size + cc, opp)]

    # Rule 5: Superko — the resulting board must not repeat a previous state
    # WITH THE SAME PLAYER TO MOVE. See the module docstring: the same stones
    # with the other player on move is a different situation and is legal.
    if board_history_hashes is not None and \
            situational_key(new_hash, opp) in board_history_hashes:
        return MoveSimulation(
            False, "Superko violation: this board position has occurred before")

    # Ko is created when a lone stone captures exactly one stone and the point
    # it captured becomes its only liberty: no friendly neighbour to merge with
    # (so the group is one stone) and no empty neighbour (so the captured point
    # is the sole liberty).
    new_ko_point = None
    if len(captured) == 1 and not friendly_neighbours and not empty_neighbours:
        new_ko_point = next(iter(captured))

    own_size = None
    own_liberties = None
    if need_group_facts and not captured and empty_neighbours <= 1:
        # Only now is it worth resolving the groups this move would join, and
        # the suicide check above has usually already walked some of them —
        # reuse those rather than paying for the same flood fill twice.
        seen = set(friendly_seen)
        groups = list(friendly_walked)
        for nr, nc in friendly_neighbours:
            if (nr, nc) in seen:
                continue
            group = board.get_group(nr, nc)
            seen |= group
            groups.append(group)

        own_size = 1 + sum(len(g) for g in groups)
        liberties = set()
        for group in groups:
            liberties |= board.get_liberties(group)
        for nr, nc in neighbours:
            if grid[nr, nc] == EMPTY:
                liberties.add((nr, nc))
        # (row, col) is a liberty of every adjacent friendly group right now,
        # but the stone being placed there occupies it.
        liberties.discard((row, col))
        own_liberties = liberties

    return MoveSimulation(True, "", frozenset(captured), new_hash, new_ko_point,
                          own_size, own_liberties)


def is_legal_move(board: Board, color: int, row: int, col: int,
                  ko_point: Optional[Tuple[int, int]] = None,
                  board_history_hashes: Optional[Set[int]] = None) -> Tuple[bool, str]:
    """
    Check if placing a stone at (row, col) is legal.

    Args:
        board: Current board state.
        color: Color to play (BLACK or WHITE).
        row, col: Position to check.
        ko_point: If set, this position is forbidden (simple ko).
        board_history_hashes: Set of all previous board hashes (superko).

    Returns:
        (is_legal, error_message) tuple.
    """
    sim = simulate_move(board, color, row, col, ko_point, board_history_hashes)
    return sim.is_legal, sim.error


def apply_move(board: Board, color: int, row: int, col: int,
               ko_point: Optional[Tuple[int, int]] = None,
               board_history_hashes: Optional[Set[int]] = None) -> MoveResult:
    """
    Apply a move to the board (MUTATES the board in-place).
    
    This is the main function for playing moves. It:
    1. Validates the move.
    2. Places the stone.
    3. Captures any opponent groups with 0 liberties.
    4. Detects new ko points.
    
    Args:
        board: Board to modify.
        color: Color placing the stone.
        row, col: Position to play.
        ko_point: Current ko point (None if no active ko).
        board_history_hashes: Previous board hashes for superko.
    
    Returns:
        MoveResult with capture info and new ko point.
    """
    # One simulation answers legality, captures and the new ko point together.
    # This function used to validate by simulating the move on a copy and then
    # redo the identical work on the real board — every child node in the search
    # tree paid for the move twice.
    sim = simulate_move(board, color, row, col, ko_point, board_history_hashes)
    if not sim.is_legal:
        return MoveResult(is_legal=False, error=sim.error)

    board.place_stone(row, col, color)
    if sim.captured:
        # The dead stones are already known exactly, so there is no need to
        # rediscover them by re-scanning neighbours for zero-liberty groups.
        board.remove_group(sim.captured)

    return MoveResult(
        is_legal=True,
        captured_stones=set(sim.captured),
        captured_count=len(sim.captured),
        ko_point=sim.ko_point
    )


def get_legal_moves(board: Board, color: int,
                    ko_point: Optional[Tuple[int, int]] = None,
                    board_history_hashes: Optional[Set[int]] = None,
                    restrict_eye_fill: bool = False,
                    restrict_self_atari: bool = False,
                    self_atari_max_stones: int = 1) -> List[Tuple[int, int]]:
    """
    Return all legal moves for a given color.

    This is used by the AI to know which moves it can play.
    Pass is always legal but not included here (it's handled separately).

    NOTE: This is O(n²) in board size × move validation cost. For 9x9 that's
    fine (~81 checks). For 19x19 you might want to optimize with incremental
    liberty tracking, but for this project simplicity wins.

    Args:
        restrict_eye_fill: Optional playing restriction (NOT a rule of Go). When
            True, moves that would fill one of the mover's own two eyes — taking
            a provably-alive chain down to one eye — are left out. See
            game/eyes.py for the exact definition and why it is safe. Rule
            legality (is_legal_move / apply_move) is deliberately untouched, so
            a human, a stored game record or a replay is never affected by it.
        restrict_self_atari: Optional playing restriction (NOT a rule of Go).
            When True, moves that capture nothing and leave the mover's group on
            a single liberty are left out, provided that group is larger than
            `self_atari_max_stones`. See game/self_atari.py — unlike the eye
            rule this one is a tuned assumption, not a theorem.
        self_atari_max_stones: Sacrifices up to this many stones stay playable,
            which is what keeps throw-ins available.
    """
    eye_forbidden = set()
    if restrict_eye_fill:
        from game.eyes import forbidden_eye_fills
        eye_forbidden = forbidden_eye_fills(board, color)

    if restrict_self_atari:
        from game.self_atari import is_pointless_self_atari

    moves = []
    self_atari_removed = []
    for r in range(board.size):
        for c in range(board.size):
            if (r, c) in eye_forbidden:
                continue
            sim = simulate_move(board, color, r, c, ko_point, board_history_hashes,
                                need_group_facts=restrict_self_atari)
            if not sim.is_legal:
                continue
            if restrict_self_atari and is_pointless_self_atari(
                    sim, self_atari_max_stones):
                self_atari_removed.append((r, c))
                continue
            moves.append((r, c))

    # If the self-atari filter removed EVERY option, give the moves back.
    #
    # This guard is deliberately not applied to the eye rule. Filling your own
    # last two eyes is provably useless, so when that is all there is, passing
    # really is at least as good and the empty list is the right answer (MCTS
    # falls back to pass — see ai/mcts.py::_expand). Self-atari has no such
    # proof, and a position where every single move trips it is precisely the
    # tactical situation — capturing race, seki, filled endgame — where the
    # assumption behind the rule is least reliable. Forcing a pass there would
    # be the filter overriding the search on the evidence it is weakest at.
    if not moves and self_atari_removed:
        return self_atari_removed

    return moves
