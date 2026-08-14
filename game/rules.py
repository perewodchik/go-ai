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
5. Superko: positional superko — you cannot recreate ANY previous board state.
   (Stricter than simple ko; prevents longer loops.)

IMPORTANT: The ko rule here follows the "positional superko" variant used in
Chinese rules and most computer Go implementations. Simple ko (only checking
the last position) is a subset of this.
"""

import numpy as np
from typing import Optional, Tuple, List, Set, FrozenSet
from game.board import Board, EMPTY, BLACK, WHITE, opponent


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
    # Rule 1: Must be on the board
    if not board.is_on_board(row, col):
        return False, "Position is off the board"
    
    # Rule 2: Must be empty
    if board.grid[row, col] != EMPTY:
        return False, "Position is already occupied"
    
    # Rule 3: Simple ko check (fast path before the expensive simulation)
    if ko_point is not None and (row, col) == ko_point:
        return False, "Ko rule violation: cannot recapture immediately"
    
    # To check suicide and superko, we simulate the move on a copy
    test_board = board.copy()
    test_board.place_stone(row, col, color)
    
    # Check captures — opponent groups that now have 0 liberties
    captured_any = False
    for nr, nc in test_board.neighbors(row, col):
        if test_board.grid[nr, nc] == opponent(color):
            group = test_board.get_group(nr, nc)
            if test_board.liberty_count(group) == 0:
                test_board.remove_group(group)
                captured_any = True
    
    # Rule 4: Suicide check — if we didn't capture anything,
    # our own group must have at least 1 liberty
    if not captured_any:
        own_group = test_board.get_group(row, col)
        if test_board.liberty_count(own_group) == 0:
            return False, "Suicide: move would leave your group with no liberties"
    
    # Rule 5: Superko check — resulting board must not repeat any previous state
    if board_history_hashes is not None:
        if test_board.board_hash in board_history_hashes:
            return False, "Superko violation: this board position has occurred before"
    
    return True, ""


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
    # Validate
    legal, error = is_legal_move(board, color, row, col, ko_point, board_history_hashes)
    if not legal:
        return MoveResult(is_legal=False, error=error)
    
    # Place the stone
    board.place_stone(row, col, color)
    
    # Capture opponent groups with no liberties
    total_captured = 0
    all_captured_positions = set()
    opp = opponent(color)
    
    for nr, nc in board.neighbors(row, col):
        if board.grid[nr, nc] == opp:
            group = board.get_group(nr, nc)
            if board.liberty_count(group) == 0:
                removed = board.remove_group(group)
                all_captured_positions |= group
                total_captured += removed
    
    # Detect ko: if we captured exactly 1 stone, the captured position
    # becomes the ko point (opponent can't immediately recapture there).
    # This only applies to single-stone captures where we also placed
    # a single stone into atari.
    new_ko_point = None
    if total_captured == 1:
        captured_pos = next(iter(all_captured_positions))
        # Verify it's actually a ko: our placed stone must have exactly
        # 1 liberty (the position we just captured) to form a ko shape.
        own_group = board.get_group(row, col)
        if len(own_group) == 1 and board.liberty_count(own_group) == 1:
            new_ko_point = captured_pos
    
    return MoveResult(
        is_legal=True,
        captured_stones=all_captured_positions,
        captured_count=total_captured,
        ko_point=new_ko_point
    )


def get_legal_moves(board: Board, color: int,
                    ko_point: Optional[Tuple[int, int]] = None,
                    board_history_hashes: Optional[Set[int]] = None) -> List[Tuple[int, int]]:
    """
    Return all legal moves for a given color.
    
    This is used by the AI to know which moves it can play.
    Pass is always legal but not included here (it's handled separately).
    
    NOTE: This is O(n²) in board size × move validation cost. For 9x9 that's
    fine (~81 checks). For 19x19 you might want to optimize with incremental
    liberty tracking, but for this project simplicity wins.
    """
    moves = []
    for r in range(board.size):
        for c in range(board.size):
            legal, _ = is_legal_move(board, color, r, c, ko_point, board_history_hashes)
            if legal:
                moves.append((r, c))
    return moves
