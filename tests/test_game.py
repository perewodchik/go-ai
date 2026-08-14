"""
test_game.py — Integration tests for full game scenarios.

Tests cover:
- Complete game flow (moves, passes, game over)
- Move history tracking
- Prisoner counting
- Game state serialization/deserialization
- Tensor encoding for neural network
- Undo functionality
- Various game endings
"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.game_state import GameState, MOVE_PASS, MOVE_RESIGN
from game.board import BLACK, WHITE


class TestGameFlow:
    """Full game flow tests."""

    def test_new_game_black_first(self):
        state = GameState(board_size=9)
        assert state.current_player == BLACK

    def test_alternating_turns(self):
        state = GameState(board_size=9)
        state.play_move(0, 0)
        assert state.current_player == WHITE
        state.play_move(1, 1)
        assert state.current_player == BLACK

    def test_two_passes_end_game(self):
        state = GameState(board_size=9)
        state.play_pass()
        assert not state.is_over
        state.play_pass()
        assert state.is_over

    def test_resign_ends_game(self):
        state = GameState(board_size=9)
        state.play_resign()
        assert state.is_over
        assert state.winner == WHITE  # Black resigned, White wins

    def test_illegal_move_returns_false(self):
        state = GameState(board_size=9)
        state.play_move(4, 4)
        assert not state.play_move(4, 4)  # Occupied

    def test_move_after_game_over_fails(self):
        state = GameState(board_size=9)
        state.play_resign()
        assert not state.play_move(0, 0)

    def test_pass_resets_ko(self):
        state = GameState(board_size=9)
        # Set up a ko point manually
        state.ko_point = (4, 4)
        state.play_pass()
        assert state.ko_point is None


class TestMoveHistory:
    """Move history tracking."""

    def test_history_records_moves(self):
        state = GameState(board_size=9)
        state.play_move(0, 0)
        state.play_move(1, 1)
        assert len(state.move_history) == 2
        assert state.move_history[0] == (BLACK, (0, 0))
        assert state.move_history[1] == (WHITE, (1, 1))

    def test_history_records_pass(self):
        state = GameState(board_size=9)
        state.play_pass()
        assert state.move_history[0] == (BLACK, MOVE_PASS)

    def test_move_number(self):
        state = GameState(board_size=9)
        assert state.move_number == 0
        state.play_move(0, 0)
        assert state.move_number == 1


class TestPrisoners:
    """Prisoner (captured stone) counting."""

    def test_capture_increments_prisoners(self):
        state = GameState(board_size=9)
        # Set up a capture: White stone at corner, Black surrounds
        state.board.place_stone(0, 0, WHITE)
        state.board.place_stone(0, 1, BLACK)
        state.current_player = BLACK
        state.play_move(1, 0)  # Captures White at (0,0)
        assert state.prisoners[BLACK] == 1  # Black captured 1 stone


class TestSerialization:
    """to_dict() and from_dict() round-trip."""

    def test_round_trip(self):
        state = GameState(board_size=9, komi=6.5)
        state.play_move(0, 0)
        state.play_move(1, 1)
        state.play_pass()

        data = state.to_dict()
        restored = GameState.from_dict(data)

        assert restored.board_size == 9
        assert restored.komi == 6.5
        assert restored.move_number == 3
        assert restored.current_player == state.current_player

    def test_dict_has_required_fields(self):
        state = GameState(board_size=9)
        data = state.to_dict()
        required = ['board_size', 'komi', 'grid', 'current_player',
                     'move_history', 'is_over', 'move_number']
        for key in required:
            assert key in data


class TestTensorEncoding:
    """Neural network tensor encoding."""

    def test_encoding_shape(self):
        state = GameState(board_size=9)
        tensor = state.encode_for_nn()
        assert tensor.shape == (10, 9, 9)  # 10 planes, 9x9 board

    def test_encoding_empty_board(self):
        state = GameState(board_size=9)
        tensor = state.encode_for_nn()
        # Stone / liberty / ko planes (0-8) are all zero on an empty board...
        assert tensor[:9].sum().item() == 0.0
        # ...but plane 9 (turn color) is all-ones because Black plays first.
        assert tensor[9].sum().item() == 9 * 9

    def test_encoding_has_stone_planes(self):
        state = GameState(board_size=9)
        state.play_move(4, 4)  # Black at center
        tensor = state.encode_for_nn()
        # Now it's White's turn, so:
        # Plane 0 = current player's stones (White) = empty
        # Plane 1 = opponent's stones (Black) = has (4,4)
        assert tensor[1, 4, 4].item() == 1.0

    def test_encoding_different_sizes(self):
        for size in [7, 9, 13]:
            state = GameState(board_size=size)
            tensor = state.encode_for_nn()
            assert tensor.shape == (10, size, size)


class TestUndo:
    """Undo functionality."""

    def test_undo_single_move(self):
        state = GameState(board_size=9)
        state.play_move(4, 4)
        state.undo_move()
        assert state.move_number == 0
        assert state.current_player == BLACK

    def test_undo_empty_history(self):
        state = GameState(board_size=9)
        assert not state.undo_move()

    def test_undo_preserves_earlier_moves(self):
        state = GameState(board_size=9)
        state.play_move(0, 0)
        state.play_move(1, 1)
        state.play_move(2, 2)
        state.undo_move()
        assert state.move_number == 2
        assert state.board.grid[2, 2] == 0  # Stone removed


class TestLegalMoves:
    """get_legal_moves() on GameState."""

    def test_legal_moves_exclude_occupied(self):
        state = GameState(board_size=9)
        state.play_move(4, 4)
        # It's now White's turn
        moves = state.get_legal_moves()
        assert (4, 4) not in moves

    def test_is_legal_check(self):
        state = GameState(board_size=9)
        assert state.is_legal(0, 0)
        state.play_move(0, 0)
        assert not state.is_legal(0, 0)


class TestGameCopy:
    """Game state copying."""

    def test_copy_independent(self):
        state = GameState(board_size=9)
        state.play_move(4, 4)
        copy = state.copy()
        copy.play_move(0, 0)
        assert state.move_number == 1
        assert copy.move_number == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
