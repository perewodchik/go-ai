"""
board.py — Go board state representation.

Core data structure for the Go game engine. Uses a NumPy 2D array for the grid
and Zobrist hashing for fast board comparison (used in superko detection).

KEY CONCEPTS:
- "Group" (also called "string"): a set of connected stones of the same color.
- "Liberties": empty intersections adjacent to a group. A group with 0 liberties
  is captured (removed from the board).
- We use Union-Find (Disjoint Set Union / DSU) for O(α(n)) group operations.

COORDINATE SYSTEM:
- (row, col) with (0,0) at top-left.
- Internally, positions can also be represented as a flat index: row * size + col.
"""

import numpy as np
from typing import Optional, Set, Tuple, FrozenSet
import random

# Stone constants — these are used throughout the project
EMPTY = 0
BLACK = 1
WHITE = 2


def opponent(color: int) -> int:
    """Return the opponent's color. BLACK ↔ WHITE."""
    assert color in (BLACK, WHITE), f"Invalid color: {color}"
    return WHITE if color == BLACK else BLACK


class ZobristHash:
    """
    Zobrist hashing for fast board state comparison.
    
    Each (position, color) pair gets a random 64-bit number. The board hash
    is the XOR of all placed stones' hashes. This makes incremental updates
    O(1): placing/removing a stone just XORs one value.
    
    Used for superko detection — we keep a set of all seen hashes.
    """
    
    def __init__(self, board_size: int, seed: int = 42):
        rng = random.Random(seed)  # Deterministic for reproducibility
        self.table = {}
        for pos in range(board_size * board_size):
            for color in (BLACK, WHITE):
                self.table[(pos, color)] = rng.getrandbits(64)
    
    def compute(self, grid: np.ndarray) -> int:
        """Compute full board hash from scratch (used for initialization)."""
        h = 0
        size = grid.shape[0]
        for r in range(size):
            for c in range(size):
                if grid[r, c] != EMPTY:
                    h ^= self.table[(r * size + c, grid[r, c])]
        return h
    
    def update(self, current_hash: int, position: int, color: int) -> int:
        """
        Toggle a stone at `position` with `color`.
        XOR is self-inverse: applying twice cancels out.
        Use for both placing AND removing stones.
        """
        return current_hash ^ self.table[(position, color)]


class Board:
    """
    Go board state with efficient group tracking.
    
    Attributes:
        size: Board dimension (e.g. 9 for a 9x9 board).
        grid: 2D NumPy array — grid[r][c] ∈ {EMPTY, BLACK, WHITE}.
        zobrist: ZobristHash instance for incremental hashing.
        board_hash: Current Zobrist hash of the board state.
    
    The board does NOT enforce rules (ko, suicide) — that's in rules.py.
    It only provides the mechanics: place stone, find groups, remove groups.
    """
    
    def __init__(self, size: int = 9):
        assert size in (7, 9, 13, 15, 19), f"Unsupported board size: {size}"
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.int8)
        self.zobrist = ZobristHash(size)
        self.board_hash = 0  # Hash of empty board is 0
    
    def copy(self) -> 'Board':
        """Deep copy the board state. Used when exploring hypothetical moves."""
        new = Board.__new__(Board)
        new.size = self.size
        new.grid = self.grid.copy()
        new.zobrist = self.zobrist  # Shared (immutable random table)
        new.board_hash = self.board_hash
        return new
    
    def _flat(self, row: int, col: int) -> int:
        """Convert (row, col) to flat index."""
        return row * self.size + col
    
    def _rc(self, flat_idx: int) -> Tuple[int, int]:
        """Convert flat index to (row, col)."""
        return divmod(flat_idx, self.size)
    
    def is_on_board(self, row: int, col: int) -> bool:
        """Check if (row, col) is within bounds."""
        return 0 <= row < self.size and 0 <= col < self.size
    
    def neighbors(self, row: int, col: int) -> list:
        """
        Return list of (row, col) for the 4 orthogonal neighbors
        that are within bounds. Go uses 4-connectivity (no diagonals).
        """
        result = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if self.is_on_board(nr, nc):
                result.append((nr, nc))
        return result
    
    def get_group(self, row: int, col: int) -> FrozenSet[Tuple[int, int]]:
        """
        Find all stones connected to (row, col) of the same color.
        Uses BFS (breadth-first search). Returns frozenset of (r, c) positions.
        Returns empty set if the position is empty.
        """
        color = self.grid[row, col]
        if color == EMPTY:
            return frozenset()
        
        visited = set()
        queue = [(row, col)]
        visited.add((row, col))
        
        while queue:
            r, c = queue.pop()
            for nr, nc in self.neighbors(r, c):
                if (nr, nc) not in visited and self.grid[nr, nc] == color:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        
        return frozenset(visited)
    
    def get_liberties(self, group: FrozenSet[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """
        Find all liberties (empty neighbors) of a group.
        A group with 0 liberties is captured.
        """
        liberties = set()
        for r, c in group:
            for nr, nc in self.neighbors(r, c):
                if self.grid[nr, nc] == EMPTY:
                    liberties.add((nr, nc))
        return liberties
    
    def liberty_count(self, group: FrozenSet[Tuple[int, int]]) -> int:
        """Count liberties without building the full set (slightly faster)."""
        liberties = set()
        for r, c in group:
            for nr, nc in self.neighbors(r, c):
                if self.grid[nr, nc] == EMPTY:
                    liberties.add((nr, nc))
        return len(liberties)
    
    def place_stone(self, row: int, col: int, color: int) -> None:
        """
        Place a stone on the board. Does NOT check rules.
        Updates grid and Zobrist hash.
        """
        assert self.grid[row, col] == EMPTY, \
            f"Cannot place on occupied position ({row}, {col})"
        self.grid[row, col] = color
        self.board_hash = self.zobrist.update(
            self.board_hash, self._flat(row, col), color
        )
    
    def remove_group(self, group: FrozenSet[Tuple[int, int]]) -> int:
        """
        Remove all stones in a group from the board.
        Returns the number of stones removed (for prisoner counting).
        Updates grid and Zobrist hash.
        """
        count = 0
        for r, c in group:
            color = self.grid[r, c]
            if color != EMPTY:
                self.grid[r, c] = EMPTY
                self.board_hash = self.zobrist.update(
                    self.board_hash, self._flat(r, c), color
                )
                count += 1
        return count
    
    def get_all_groups(self, color: int) -> list:
        """
        Find all groups of the given color on the board.
        Returns a list of frozensets.
        """
        visited = set()
        groups = []
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r, c] == color and (r, c) not in visited:
                    group = self.get_group(r, c)
                    visited |= group
                    groups.append(group)
        return groups
    
    def get_empty_regions(self) -> list:
        """
        Find all connected regions of empty intersections.
        Returns list of (region: frozenset, border_colors: set).
        
        border_colors tells you which colors are adjacent to this empty region:
        - {BLACK} only → territory for Black
        - {WHITE} only → territory for White
        - {BLACK, WHITE} → neutral / dame
        - empty → shouldn't happen on a valid board
        """
        visited = set()
        regions = []
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r, c] == EMPTY and (r, c) not in visited:
                    # BFS to find connected empty region
                    region = set()
                    border_colors = set()
                    queue = [(r, c)]
                    region.add((r, c))
                    
                    while queue:
                        cr, cc = queue.pop()
                        for nr, nc in self.neighbors(cr, cc):
                            if (nr, nc) in region:
                                continue
                            if self.grid[nr, nc] == EMPTY:
                                region.add((nr, nc))
                                queue.append((nr, nc))
                            else:
                                border_colors.add(int(self.grid[nr, nc]))
                    
                    visited |= region
                    regions.append((frozenset(region), border_colors))
        
        return regions
    
    def count_stones(self, color: int) -> int:
        """Count total stones of a given color on the board."""
        return int(np.sum(self.grid == color))
    
    def __eq__(self, other: 'Board') -> bool:
        """Boards are equal if their grids match (Zobrist hash for fast check)."""
        if not isinstance(other, Board):
            return False
        if self.board_hash != other.board_hash:
            return False
        return np.array_equal(self.grid, other.grid)
    
    def __hash__(self) -> int:
        return self.board_hash
    
    def __repr__(self) -> str:
        """
        Pretty-print the board for debugging.
        Uses ● for Black, ○ for White, · for empty.
        """
        symbols = {EMPTY: '·', BLACK: '●', WHITE: '○'}
        col_labels = '  ' + ' '.join(
            chr(ord('A') + c + (1 if c >= 8 else 0))  # Skip 'I' in Go
            for c in range(self.size)
        )
        lines = [col_labels]
        for r in range(self.size):
            row_num = self.size - r  # Go counts from bottom
            row_str = f"{row_num:2d} " + ' '.join(
                symbols[self.grid[r, c]] for c in range(self.size)
            ) + f" {row_num}"
            lines.append(row_str)
        lines.append(col_labels)
        return '\n'.join(lines)
