"""
test_board.py — Thorough tests for the Go board state.

Tests cover:
- Board creation and basic properties
- Stone placement and removal
- Group detection (connected stones)
- Liberty counting
- Zobrist hashing consistency
- Empty region detection
- Board copying
- Edge cases (corners, edges)
"""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import Board, EMPTY, BLACK, WHITE, opponent, ZobristHash


class TestBoardBasics:
    """Basic board operations."""

    def test_create_board(self):
        b = Board(9)
        assert b.size == 9
        assert b.grid.shape == (9, 9)
        assert np.all(b.grid == EMPTY)

    def test_supported_sizes(self):
        for size in [7, 9, 13, 15, 19]:
            b = Board(size)
            assert b.size == size

    def test_unsupported_size_raises(self):
        with pytest.raises(AssertionError):
            Board(10)

    def test_empty_board_hash_is_zero(self):
        b = Board(9)
        assert b.board_hash == 0

    def test_opponent(self):
        assert opponent(BLACK) == WHITE
        assert opponent(WHITE) == BLACK


class TestStonePlacement:
    """Place and remove stones."""

    def test_place_stone(self):
        b = Board(9)
        b.place_stone(0, 0, BLACK)
        assert b.grid[0, 0] == BLACK

    def test_place_updates_hash(self):
        b = Board(9)
        old_hash = b.board_hash
        b.place_stone(4, 4, BLACK)
        assert b.board_hash != old_hash

    def test_place_on_occupied_raises(self):
        b = Board(9)
        b.place_stone(0, 0, BLACK)
        with pytest.raises(AssertionError):
            b.place_stone(0, 0, WHITE)

    def test_remove_group(self):
        b = Board(9)
        b.place_stone(3, 3, BLACK)
        b.place_stone(3, 4, BLACK)
        group = b.get_group(3, 3)
        count = b.remove_group(group)
        assert count == 2
        assert b.grid[3, 3] == EMPTY
        assert b.grid[3, 4] == EMPTY

    def test_remove_restores_hash(self):
        """Placing then removing should restore the original hash."""
        b = Board(9)
        original_hash = b.board_hash
        b.place_stone(5, 5, WHITE)
        group = b.get_group(5, 5)
        b.remove_group(group)
        assert b.board_hash == original_hash

    def test_count_stones(self):
        b = Board(9)
        b.place_stone(0, 0, BLACK)
        b.place_stone(1, 1, BLACK)
        b.place_stone(2, 2, WHITE)
        assert b.count_stones(BLACK) == 2
        assert b.count_stones(WHITE) == 1


class TestGroups:
    """Group (connected component) detection."""

    def test_single_stone_group(self):
        b = Board(9)
        b.place_stone(4, 4, BLACK)
        group = b.get_group(4, 4)
        assert len(group) == 1
        assert (4, 4) in group

    def test_horizontal_group(self):
        b = Board(9)
        for c in range(5):
            b.place_stone(4, c, BLACK)
        group = b.get_group(4, 0)
        assert len(group) == 5

    def test_vertical_group(self):
        b = Board(9)
        for r in range(3):
            b.place_stone(r, 2, WHITE)
        group = b.get_group(0, 2)
        assert len(group) == 3

    def test_l_shaped_group(self):
        b = Board(9)
        b.place_stone(0, 0, BLACK)
        b.place_stone(0, 1, BLACK)
        b.place_stone(1, 0, BLACK)
        group = b.get_group(0, 0)
        assert len(group) == 3

    def test_separate_groups_not_connected(self):
        b = Board(9)
        b.place_stone(0, 0, BLACK)
        b.place_stone(2, 2, BLACK)  # Diagonal is NOT connected in Go
        group1 = b.get_group(0, 0)
        group2 = b.get_group(2, 2)
        assert len(group1) == 1
        assert len(group2) == 1

    def test_different_colors_not_connected(self):
        b = Board(9)
        b.place_stone(4, 4, BLACK)
        b.place_stone(4, 5, WHITE)
        group = b.get_group(4, 4)
        assert len(group) == 1  # Only black stone

    def test_empty_position_returns_empty_group(self):
        b = Board(9)
        group = b.get_group(0, 0)
        assert len(group) == 0

    def test_get_all_groups(self):
        b = Board(9)
        b.place_stone(0, 0, BLACK)
        b.place_stone(0, 1, BLACK)
        b.place_stone(5, 5, BLACK)
        groups = b.get_all_groups(BLACK)
        assert len(groups) == 2  # Two separate groups


class TestLiberties:
    """Liberty counting."""

    def test_center_stone_4_liberties(self):
        b = Board(9)
        b.place_stone(4, 4, BLACK)
        group = b.get_group(4, 4)
        assert b.liberty_count(group) == 4

    def test_corner_stone_2_liberties(self):
        b = Board(9)
        b.place_stone(0, 0, BLACK)
        group = b.get_group(0, 0)
        assert b.liberty_count(group) == 2

    def test_edge_stone_3_liberties(self):
        b = Board(9)
        b.place_stone(0, 4, BLACK)
        group = b.get_group(0, 4)
        assert b.liberty_count(group) == 3

    def test_two_stone_group_liberties(self):
        b = Board(9)
        b.place_stone(4, 4, BLACK)
        b.place_stone(4, 5, BLACK)
        group = b.get_group(4, 4)
        # Shared liberties: (3,4),(5,4),(4,3), (3,5),(5,5),(4,6) = 6
        assert b.liberty_count(group) == 6

    def test_surrounded_stone_zero_liberties(self):
        """A stone surrounded on all sides has 0 liberties."""
        b = Board(9)
        b.place_stone(4, 4, BLACK)
        # Surround with white
        b.place_stone(3, 4, WHITE)
        b.place_stone(5, 4, WHITE)
        b.place_stone(4, 3, WHITE)
        b.place_stone(4, 5, WHITE)
        group = b.get_group(4, 4)
        assert b.liberty_count(group) == 0

    def test_liberties_set_correct(self):
        b = Board(9)
        b.place_stone(4, 4, BLACK)
        group = b.get_group(4, 4)
        libs = b.get_liberties(group)
        assert libs == {(3, 4), (5, 4), (4, 3), (4, 5)}


class TestEmptyRegions:
    """Empty region detection for territory scoring."""

    def test_full_board_no_empty_regions(self):
        b = Board(7)
        for r in range(7):
            for c in range(7):
                b.place_stone(r, c, BLACK)
        regions = b.get_empty_regions()
        assert len(regions) == 0

    def test_empty_board_one_region(self):
        b = Board(9)
        regions = b.get_empty_regions()
        assert len(regions) == 1
        region, borders = regions[0]
        assert len(region) == 81  # 9x9
        assert len(borders) == 0  # No stones border the empty board

    def test_surrounded_territory(self):
        """Black stones forming a ring around empty space."""
        b = Board(9)
        # Make a small enclosure
        for c in range(3):
            b.place_stone(0, c, BLACK)  # Top wall
            b.place_stone(2, c, BLACK)  # Bottom wall
        b.place_stone(1, 0, BLACK)  # Left wall
        b.place_stone(1, 2, BLACK)  # Right wall
        # (1,1) should be surrounded by black only
        regions = b.get_empty_regions()
        # Find the small region containing (1,1)
        small_region = None
        for region, borders in regions:
            if (1, 1) in region:
                small_region = (region, borders)
                break
        assert small_region is not None
        assert small_region[1] == {BLACK}


class TestBoardCopy:
    """Board copying and equality."""

    def test_copy_equals_original(self):
        b = Board(9)
        b.place_stone(0, 0, BLACK)
        b.place_stone(4, 4, WHITE)
        c = b.copy()
        assert b == c

    def test_copy_is_independent(self):
        b = Board(9)
        b.place_stone(0, 0, BLACK)
        c = b.copy()
        c.place_stone(1, 1, WHITE)
        assert b != c
        assert b.grid[1, 1] == EMPTY


class TestZobristHash:
    """Zobrist hashing properties."""

    def test_different_positions_different_hash(self):
        b1 = Board(9)
        b1.place_stone(0, 0, BLACK)
        b2 = Board(9)
        b2.place_stone(0, 1, BLACK)
        assert b1.board_hash != b2.board_hash

    def test_same_position_same_hash(self):
        """Same stones placed in different order should give same hash."""
        b1 = Board(9)
        b1.place_stone(0, 0, BLACK)
        b1.place_stone(1, 1, WHITE)

        b2 = Board(9)
        b2.place_stone(1, 1, WHITE)
        b2.place_stone(0, 0, BLACK)

        assert b1.board_hash == b2.board_hash

    def test_hash_deterministic(self):
        """Same seed produces same hash table."""
        z1 = ZobristHash(9, seed=42)
        z2 = ZobristHash(9, seed=42)
        assert z1.table == z2.table


class TestNeighbors:
    """Neighbor computation."""

    def test_center_has_4_neighbors(self):
        b = Board(9)
        assert len(b.neighbors(4, 4)) == 4

    def test_corner_has_2_neighbors(self):
        b = Board(9)
        assert len(b.neighbors(0, 0)) == 2

    def test_edge_has_3_neighbors(self):
        b = Board(9)
        assert len(b.neighbors(0, 4)) == 3

    def test_is_on_board(self):
        b = Board(9)
        assert b.is_on_board(0, 0)
        assert b.is_on_board(8, 8)
        assert not b.is_on_board(-1, 0)
        assert not b.is_on_board(9, 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
