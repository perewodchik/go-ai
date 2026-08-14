"""
test_scoring.py — Tests for scoring systems and score estimation.

Tests cover:
- Chinese scoring (area)
- Japanese scoring (territory)
- Komi application
- Scoring strategy swapping
- Score estimator (Benson's, flood-fill)
- Winner determination
- Edge cases: empty board, full board, draws
"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import Board, EMPTY, BLACK, WHITE
from game.game_state import GameState
from game.scoring.base import get_scorer, ScoringStrategy
from game.scoring.chinese import ChineseScoring
from game.scoring.japanese import JapaneseScoring
from game.scoring.estimator import ScoreEstimator


class TestChineseScoring:
    """Chinese (area) scoring tests."""

    def test_empty_board_white_wins_by_komi(self):
        """On an empty board, White wins by komi."""
        state = GameState(board_size=9, komi=6.5)
        state.play_pass()
        state.play_pass()
        scorer = ChineseScoring()
        black, white = scorer.score(state)
        # No stones, no territory for either — White gets komi
        assert black == 0
        assert white == 6.5

    def test_simple_territory(self):
        """Black occupies one half, territory counted correctly."""
        state = GameState(board_size=9, komi=6.5)
        # Place black stones along row 4 (full wall)
        for c in range(9):
            state.board.place_stone(4, c, BLACK)
        # Black "territory" is rows 0-3 (36 empty points)
        # Black stones = 9
        state.play_pass()
        state.play_pass()
        scorer = ChineseScoring()
        black, white = scorer.score(state)
        # Black: 9 stones + 36 territory = 45
        # White: 0 stones + 0 territory + 6.5 komi = 6.5
        # Actually, rows 5-8 are also empty but bordered only by Black (from row 4)
        # So those are also Black territory
        # Total empty = 72, all bordered by Black only = 72 territory
        assert black == 9 + 72  # 81 = entire board
        assert white == 6.5

    def test_both_players_territory(self):
        """Both players have territory."""
        state = GameState(board_size=7, komi=6.5)
        # Black wall at row 3, White wall at row 3 won't work...
        # Let's do: Black fills row 0, White fills row 6
        for c in range(7):
            state.board.place_stone(0, c, BLACK)
            state.board.place_stone(6, c, WHITE)
        state.play_pass()
        state.play_pass()
        scorer = ChineseScoring()
        black, white = scorer.score(state)
        # Middle rows 1-5 (35 empty points) border both colors → neutral (dame)
        # Black: 7 stones + 0 territory
        # White: 7 stones + 0 territory + 6.5
        assert black == 7
        assert white == 7 + 6.5

    def test_scorer_name(self):
        assert ChineseScoring().name == "Chinese (Area)"


class TestJapaneseScoring:
    """Japanese (territory) scoring tests."""

    def test_empty_board(self):
        state = GameState(board_size=9, komi=6.5)
        state.play_pass()
        state.play_pass()
        scorer = JapaneseScoring()
        black, white = scorer.score(state)
        assert black == 0
        assert white == 6.5

    def test_captures_count(self):
        """Japanese scoring counts captured stones."""
        state = GameState(board_size=9, komi=6.5)
        # Simulate captures
        state.prisoners = {BLACK: 5, WHITE: 3}
        state.play_pass()
        state.play_pass()
        scorer = JapaneseScoring()
        black, white = scorer.score(state)
        # Black: 0 territory + 5 captures = 5
        # White: 0 territory + 3 captures + 6.5 = 9.5
        assert black == 5
        assert white == 9.5

    def test_scorer_name(self):
        assert JapaneseScoring().name == "Japanese (Territory)"


class TestScoringStrategy:
    """Strategy pattern — swapping scoring methods."""

    def test_get_chinese_scorer(self):
        scorer = get_scorer("chinese")
        assert isinstance(scorer, ChineseScoring)

    def test_get_japanese_scorer(self):
        scorer = get_scorer("japanese")
        assert isinstance(scorer, JapaneseScoring)

    def test_invalid_scorer_raises(self):
        with pytest.raises(ValueError):
            get_scorer("invalid")


class TestWinnerDetermination:
    """determine_winner() tests."""

    def test_black_wins(self):
        state = GameState(board_size=9, komi=0.5)
        # Fill most of the board with Black
        for r in range(9):
            for c in range(9):
                state.board.place_stone(r, c, BLACK)
        state.play_pass()
        state.play_pass()
        scorer = ChineseScoring()
        winner, margin = scorer.determine_winner(state)
        assert winner == BLACK

    def test_resignation_winner(self):
        state = GameState(board_size=9, komi=6.5)
        state.play_resign()  # Black resigns
        scorer = ChineseScoring()
        winner, _ = scorer.determine_winner(state)
        assert winner == WHITE

    def test_komi_breaks_tie(self):
        """With 6.5 komi, a pure draw is impossible."""
        state = GameState(board_size=9, komi=6.5)
        state.play_pass()
        state.play_pass()
        scorer = ChineseScoring()
        winner, margin = scorer.determine_winner(state)
        assert winner == WHITE  # Komi gives White the edge
        assert margin == 6.5


class TestScoreEstimator:
    """Score estimation (display-only) tests."""

    def test_estimator_returns_dict(self):
        state = GameState(board_size=9)
        estimator = ScoreEstimator()
        result = estimator.estimate(state)
        assert 'ownership_map' in result
        assert 'black_estimate' in result
        assert 'white_estimate' in result

    def test_estimator_ownership_map_shape(self):
        state = GameState(board_size=9)
        estimator = ScoreEstimator()
        result = estimator.estimate(state)
        assert len(result['ownership_map']) == 9
        assert len(result['ownership_map'][0]) == 9

    def test_estimator_on_empty_board(self):
        state = GameState(board_size=9)
        estimator = ScoreEstimator()
        result = estimator.estimate(state)
        # Empty board should have neutral ownership
        for row in result['ownership_map']:
            for val in row:
                assert abs(val) < 0.01  # All neutral

    def test_estimator_with_stones(self):
        state = GameState(board_size=9)
        # Create a simple position with clear territory
        for c in range(9):
            state.board.place_stone(4, c, BLACK)
        estimator = ScoreEstimator()
        result = estimator.estimate(state)
        # Black stones should show black ownership
        assert result['ownership_map'][4][4] < 0  # Black = negative


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
