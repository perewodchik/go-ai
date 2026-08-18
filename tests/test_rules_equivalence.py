"""
test_rules_equivalence.py — the fast rules path must be the slow one, exactly.

`rules.is_legal_move` used to answer every question by copying the board,
placing the stone, resolving captures and re-reading the result. That is the
hottest function in the engine (58% of a 96-simulation search), so it was
rewritten to answer from adjacency alone: no board copy, no stone placement,
and the resulting Zobrist hash computed incrementally.

A rewrite of the referee is exactly the kind of change that can pass every
hand-written unit test and still be subtly wrong in a position nobody thought
of — a snapback, a multi-group capture, a superko that only triggers after a
particular capture. So this file keeps a literal copy of the ORIGINAL
simulate-everything implementation as an oracle and fuzzes thousands of real
positions against it, comparing:

  * the legality verdict for every point on the board
  * the exact set of captured stones
  * the resulting board hash
  * the ko point

The oracle below is deliberately NOT refactored or tidied. It is the old code,
kept as evidence.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from game import rules
from game.board import Board, BLACK, EMPTY, WHITE, opponent
from game.game_state import GameState


# ---------------------------------------------------------------------------
# ORACLE — the original implementation, verbatim.
# ---------------------------------------------------------------------------

def oracle_is_legal_move(board, color, row, col, ko_point=None,
                         board_history_hashes=None):
    if not board.is_on_board(row, col):
        return False, "Position is off the board"

    if board.grid[row, col] != EMPTY:
        return False, "Position is already occupied"

    if ko_point is not None and (row, col) == ko_point:
        return False, "Ko rule violation: cannot recapture immediately"

    test_board = board.copy()
    test_board.place_stone(row, col, color)

    captured_any = False
    for nr, nc in test_board.neighbors(row, col):
        if test_board.grid[nr, nc] == opponent(color):
            group = test_board.get_group(nr, nc)
            if test_board.liberty_count(group) == 0:
                test_board.remove_group(group)
                captured_any = True

    if not captured_any:
        own_group = test_board.get_group(row, col)
        if test_board.liberty_count(own_group) == 0:
            return False, "Suicide: move would leave your group with no liberties"

    if board_history_hashes is not None:
        # Situational, not positional — the key carries whose turn it becomes.
        if rules.situational_key(test_board.board_hash,
                                 opponent(color)) in board_history_hashes:
            return False, "Superko violation: this board position has occurred before"

    return True, ""


def oracle_apply(board, color, row, col, ko_point=None, history=None):
    """Original apply_move, returning (captured_set, new_hash, ko_point)."""
    legal, _ = oracle_is_legal_move(board, color, row, col, ko_point, history)
    if not legal:
        return None

    work = board.copy()
    work.place_stone(row, col, color)

    total_captured = 0
    all_captured_positions = set()
    opp = opponent(color)
    for nr, nc in work.neighbors(row, col):
        if work.grid[nr, nc] == opp:
            group = work.get_group(nr, nc)
            if work.liberty_count(group) == 0:
                removed = work.remove_group(group)
                all_captured_positions |= group
                total_captured += removed

    new_ko_point = None
    if total_captured == 1:
        captured_pos = next(iter(all_captured_positions))
        own_group = work.get_group(row, col)
        if len(own_group) == 1 and work.liberty_count(own_group) == 1:
            new_ko_point = captured_pos

    return all_captured_positions, work.board_hash, new_ko_point


# ---------------------------------------------------------------------------
# Position generators
# ---------------------------------------------------------------------------

def random_positions(board_size, num_games, max_moves, seed):
    """Play random legal games, yielding every position along the way."""
    rng = random.Random(seed)
    for g in range(num_games):
        state = GameState(board_size=board_size, komi=6.5)
        for _ in range(max_moves):
            yield state
            legal = state.get_legal_moves()
            if not legal:
                break
            # Bias towards contact play so captures actually happen.
            state.play_move(*rng.choice(legal))


def capture_heavy_positions(board_size, num_games, max_moves, seed):
    """
    Random play restricted to points adjacent to an existing stone, which
    produces far more captures, ataris and ko shapes than uniform play.
    """
    rng = random.Random(seed)
    for g in range(num_games):
        state = GameState(board_size=board_size, komi=6.5)
        for _ in range(max_moves):
            yield state
            legal = state.get_legal_moves()
            if not legal:
                break
            contact = [
                (r, c) for (r, c) in legal
                if any(state.board.grid[nr, nc] != EMPTY
                       for nr, nc in state.board.neighbors(r, c))
            ]
            state.play_move(*rng.choice(contact or legal))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLegalityMatchesOracle:

    @pytest.mark.parametrize("generator,seed", [
        (random_positions, 11),
        (capture_heavy_positions, 22),
    ])
    def test_every_point_of_every_position(self, generator, seed):
        checked = 0
        positions = 0
        for state in generator(9, num_games=12, max_moves=70, seed=seed):
            positions += 1
            board = state.board
            colour = state.current_player
            for r in range(board.size):
                for c in range(board.size):
                    got, _ = rules.is_legal_move(
                        board, colour, r, c,
                        state.ko_point, state.board_hash_history)
                    want, _ = oracle_is_legal_move(
                        board, colour, r, c,
                        state.ko_point, state.board_hash_history)
                    assert got == want, (
                        f"legality mismatch at ({r},{c}) for colour {colour}\n"
                        f"{board}"
                    )
                    checked += 1
        assert positions > 100, "generator produced too few positions to be meaningful"
        assert checked > 10_000

    def test_both_colours_are_checked(self):
        for state in random_positions(9, num_games=6, max_moves=50, seed=33):
            board = state.board
            for colour in (BLACK, WHITE):
                for r in range(board.size):
                    for c in range(board.size):
                        got, _ = rules.is_legal_move(
                            board, colour, r, c,
                            state.ko_point, state.board_hash_history)
                        want, _ = oracle_is_legal_move(
                            board, colour, r, c,
                            state.ko_point, state.board_hash_history)
                        assert got == want

    def test_get_legal_moves_matches_oracle(self):
        for state in capture_heavy_positions(9, num_games=8, max_moves=60, seed=44):
            board, colour = state.board, state.current_player
            got = set(rules.get_legal_moves(
                board, colour, state.ko_point, state.board_hash_history))
            want = {
                (r, c)
                for r in range(board.size) for c in range(board.size)
                if oracle_is_legal_move(board, colour, r, c,
                                        state.ko_point,
                                        state.board_hash_history)[0]
            }
            assert got == want

    def test_error_messages_still_distinguish_the_cases(self):
        """The reasons are surfaced in the web API, so they must not drift."""
        board = Board(9)
        board.place_stone(4, 4, BLACK)
        assert "occupied" in rules.is_legal_move(board, WHITE, 4, 4)[1]
        assert "off the board" in rules.is_legal_move(board, WHITE, -1, 0)[1]
        assert "Ko" in rules.is_legal_move(board, WHITE, 2, 2, ko_point=(2, 2))[1]

        # Surrounded point: suicide for White.
        b = Board(9)
        for p in ((0, 1), (1, 0), (1, 2), (2, 1)):
            b.place_stone(*p, BLACK)
        assert "Suicide" in rules.is_legal_move(b, WHITE, 1, 1)[1]


class TestApplyMatchesOracle:

    @pytest.mark.parametrize("generator,seed", [
        (random_positions, 55),
        (capture_heavy_positions, 66),
    ])
    def test_captures_hash_and_ko(self, generator, seed):
        capture_events = 0
        for state in generator(9, num_games=10, max_moves=70, seed=seed):
            colour = state.current_player
            for (r, c) in state.get_legal_moves():
                expected = oracle_apply(state.board, colour, r, c,
                                        state.ko_point,
                                        state.board_hash_history)
                assert expected is not None

                work = state.board.copy()
                result = rules.apply_move(work, colour, r, c, state.ko_point,
                                          state.board_hash_history)

                want_captured, want_hash, want_ko = expected
                assert result.is_legal
                assert set(result.captured_stones) == want_captured
                assert result.captured_count == len(want_captured)
                assert work.board_hash == want_hash, "incremental hash diverged"
                assert result.ko_point == want_ko
                if want_captured:
                    capture_events += 1
        assert capture_events > 20, "test never exercised a capture"

    def test_board_state_after_apply_matches_oracle_board(self):
        import numpy as np
        for state in capture_heavy_positions(9, num_games=6, max_moves=60, seed=77):
            colour = state.current_player
            for (r, c) in state.get_legal_moves()[:12]:
                mine = state.board.copy()
                rules.apply_move(mine, colour, r, c, state.ko_point,
                                 state.board_hash_history)

                theirs = state.board.copy()
                theirs.place_stone(r, c, colour)
                for nr, nc in theirs.neighbors(r, c):
                    if theirs.grid[nr, nc] == opponent(colour):
                        g = theirs.get_group(nr, nc)
                        if theirs.liberty_count(g) == 0:
                            theirs.remove_group(g)

                assert np.array_equal(mine.grid, theirs.grid)
                assert mine.board_hash == theirs.board_hash


class TestKnownShapes:
    """Named positions where a fast path is most likely to be wrong."""

    def test_snapback_capture_is_legal(self):
        b = Board(9)
        # White stone with one liberty next to a Black group it can capture back.
        for p in ((0, 0), (0, 1), (1, 1)):
            b.place_stone(*p, WHITE)
        for p in ((0, 2), (1, 0), (1, 2), (2, 0), (2, 1)):
            b.place_stone(*p, BLACK)
        for r in range(3):
            for c in range(3):
                got, _ = rules.is_legal_move(b, BLACK, r, c)
                want, _ = oracle_is_legal_move(b, BLACK, r, c)
                assert got == want

    def test_filling_own_last_liberty_next_to_friendly_group(self):
        """Suicide only if EVERY merged friendly group is also out of liberties."""
        b = Board(9)
        # Black wall with plenty of liberties; playing next to it is never suicide.
        for c in range(9):
            b.place_stone(4, c, BLACK)
        for c in range(9):
            for r in (3, 5):
                got, _ = rules.is_legal_move(b, BLACK, r, c)
                want, _ = oracle_is_legal_move(b, BLACK, r, c)
                assert got == want

    def test_capture_that_frees_the_placed_stone(self):
        """
        A move with zero liberties of its own is legal when it captures — the
        capture is what gives it a liberty.
        """
        b = Board(9)
        # White stone at (0,0) in atari, Black surrounds it.
        b.place_stone(0, 0, WHITE)
        b.place_stone(1, 0, BLACK)
        b.place_stone(0, 1, WHITE)
        b.place_stone(1, 1, BLACK)
        b.place_stone(0, 2, BLACK)
        legal, _ = rules.is_legal_move(b, BLACK, 0, 0)
        want, _ = oracle_is_legal_move(b, BLACK, 0, 0)
        assert legal == want

    def test_superko_is_still_enforced(self):
        state = GameState(board_size=9, komi=6.5)
        seen = set(state.board_hash_history)
        state.play_move(4, 4)
        # Replaying into a seen position must be refused by both paths.
        for r in range(9):
            for c in range(9):
                got, _ = rules.is_legal_move(state.board, state.current_player,
                                             r, c, None, seen | {state.board.board_hash})
                want, _ = oracle_is_legal_move(state.board, state.current_player,
                                               r, c, None,
                                               seen | {state.board.board_hash})
                assert got == want

    def test_multi_group_capture(self):
        """One stone capturing two separate groups at once."""
        b = Board(9)
        b.place_stone(4, 3, WHITE)
        b.place_stone(4, 5, WHITE)
        for p in ((3, 3), (5, 3), (4, 2), (3, 5), (5, 5), (4, 6)):
            b.place_stone(*p, BLACK)
        expected = oracle_apply(b, BLACK, 4, 4)
        assert expected is not None
        work = b.copy()
        result = rules.apply_move(work, BLACK, 4, 4)
        assert set(result.captured_stones) == expected[0]
        assert len(expected[0]) == 2
        assert work.board_hash == expected[1]
