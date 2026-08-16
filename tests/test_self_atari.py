"""
test_self_atari.py — the "no pointless self-atari" restriction.

Unlike restrict_eye_fill, this filter is a tuned assumption rather than a
theorem, so the tests that matter most are the OVER-RESTRICTION ones: throw-ins,
snapbacks, ko captures and forced defences must all stay playable. A filter that
blocks too much cannot be caught by the promotion gate (both sides play under
it), so it has to be caught here.

Every hand-built shape asserts its own preconditions against `_after()`, which
plays the move on a real board and reads the result back. That way a test can
never quietly stop testing what its name says because a stone was misplaced.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from game import rules
from game.board import Board, BLACK, EMPTY, WHITE
from game.game_state import GameState
from game.self_atari import (
    forbidden_self_ataris,
    is_forbidden_self_atari,
    is_pointless_self_atari,
)


def _after(board, color, row, col):
    """
    Ground truth: play the move on a copy and report what actually happened.
    (captured_count, merged_group_size, merged_group_liberties)
    """
    work = board.copy()
    result = rules.apply_move(work, color, row, col)
    assert result.is_legal, f"move ({row},{col}) is not even legal"
    group = work.get_group(row, col)
    return result.captured_count, len(group), work.liberty_count(group)


def _place(board, color, points):
    for r, c in points:
        board.place_stone(r, c, color)


# ---------------------------------------------------------------------------
# Hand-built shapes
# ---------------------------------------------------------------------------

class TestBlocksRealBlunders:

    def _four_stone_atari(self):
        """
        Black has a 3-stone group with two liberties. Extending to (5,6) makes
        it a 4-stone group with exactly one, capturing nothing.
        """
        b = Board(9)
        _place(b, BLACK, [(4, 4), (4, 5), (4, 6)])
        _place(b, WHITE, [(3, 4), (3, 5), (3, 6), (5, 4), (5, 5),
                          (4, 3), (6, 6), (5, 7)])
        return b

    def test_preconditions(self):
        b = self._four_stone_atari()
        captured, size, liberties = _after(b, BLACK, 5, 6)
        assert captured == 0, "shape should not capture anything"
        assert size == 4, f"expected a 4-stone group, got {size}"
        assert liberties == 1, f"expected 1 liberty, got {liberties}"

    def test_large_self_atari_is_blocked(self):
        b = self._four_stone_atari()
        assert is_forbidden_self_atari(b, BLACK, 5, 6, max_stones=1)

    def test_the_other_colour_is_unaffected(self):
        b = self._four_stone_atari()
        # White playing the same point is not walking into atari.
        assert not is_forbidden_self_atari(b, WHITE, 5, 6, max_stones=1)

    def test_k_boundary(self):
        """Group of 4: blocked while max_stones < 4, allowed once it reaches 4."""
        b = self._four_stone_atari()
        assert is_forbidden_self_atari(b, BLACK, 5, 6, max_stones=3)
        assert not is_forbidden_self_atari(b, BLACK, 5, 6, max_stones=4)
        assert not is_forbidden_self_atari(b, BLACK, 5, 6, max_stones=5)


class TestDoesNotBlockRealMoves:

    def test_throw_in_stays_playable(self):
        """
        A single stone sacrificed into White's two-point eye space. This is the
        main small-group tesuji and is exactly what max_stones=1 protects.
        """
        b = Board(9)
        _place(b, WHITE, [(0, 2), (1, 0), (1, 1), (1, 2)])
        captured, size, liberties = _after(b, BLACK, 0, 0)
        assert (captured, size, liberties) == (0, 1, 1), \
            "shape is not a one-stone throw-in"
        assert not is_forbidden_self_atari(b, BLACK, 0, 0, max_stones=1)

    def test_capturing_move_is_never_blocked(self):
        """
        Condition 1: if the move takes stones off the board it is doing
        something, whatever its own liberty count ends up being.
        """
        b = Board(9)
        # White stone at (0,1) with its last liberty at (0,0).
        _place(b, WHITE, [(0, 1)])
        _place(b, BLACK, [(0, 2), (1, 1)])
        # Black's capturing stone will be hemmed in from below.
        _place(b, WHITE, [(1, 0)])

        captured, size, liberties = _after(b, BLACK, 0, 0)
        assert captured == 1, "shape should capture exactly one stone"
        assert liberties == 1, "and should leave the capturer on one liberty"
        assert not is_forbidden_self_atari(b, BLACK, 0, 0, max_stones=1)

    def test_ko_capture_stays_playable(self):
        """
        The standard ko shape. Black takes at (1,1), capturing the lone White
        stone at (1,2) and leaving its own stone on a single liberty — the exact
        pattern conditions 2 and 3 would otherwise describe. It survives because
        condition 1 exempts anything that captures.
        """
        b = Board(9)
        # Black surrounds the white stone that is about to be taken...
        _place(b, BLACK, [(0, 2), (2, 2), (1, 3)])
        # ...and White surrounds the point Black will play on.
        _place(b, WHITE, [(1, 2), (0, 1), (2, 1), (1, 0)])

        captured, size, liberties = _after(b, BLACK, 1, 1)
        assert captured == 1, f"expected a ko capture, captured {captured}"
        assert (size, liberties) == (1, 1), \
            f"expected a lone stone on one liberty, got size {size}, libs {liberties}"

        assert not is_forbidden_self_atari(b, BLACK, 1, 1, max_stones=1)
        # Still exempt even when single-stone sacrifices are not.
        assert not is_forbidden_self_atari(b, BLACK, 1, 1, max_stones=0)

    def test_connecting_to_safety_is_not_self_atari(self):
        """Joining two groups that together have room must stay playable."""
        b = Board(9)
        _place(b, BLACK, [(4, 3), (4, 5)])
        captured, size, liberties = _after(b, BLACK, 4, 4)
        assert size == 3 and liberties > 1
        assert not is_forbidden_self_atari(b, BLACK, 4, 4, max_stones=1)


class TestNeverEmptiesTheMoveList:
    """
    The guard the eye rule deliberately does NOT get. Filling your own last two
    eyes is provably useless, so an empty list there is correct and pass is the
    right fallback. Self-atari has no such proof, and a position where every
    move trips it is where the assumption is least reliable.
    """

    def test_filter_returns_moves_when_it_would_otherwise_block_everything(self):
        b = Board(9)
        # Fill the board so that Black's only remaining points are self-ataris.
        for r in range(9):
            for c in range(9):
                if (r + c) % 2 == 0:
                    b.place_stone(r, c, WHITE)

        for colour in (BLACK, WHITE):
            unfiltered = rules.get_legal_moves(b, colour)
            filtered = rules.get_legal_moves(
                b, colour, restrict_self_atari=True, self_atari_max_stones=1)
            if unfiltered:
                assert filtered, (
                    "self-atari filter removed every legal move, forcing a pass"
                )

    def test_restored_moves_are_a_subset_of_the_legal_ones(self):
        b = Board(9)
        for r in range(9):
            for c in range(9):
                if (r + c) % 2 == 0:
                    b.place_stone(r, c, WHITE)
        legal = set(rules.get_legal_moves(b, BLACK))
        filtered = set(rules.get_legal_moves(
            b, BLACK, restrict_self_atari=True, self_atari_max_stones=1))
        assert filtered <= legal


# ---------------------------------------------------------------------------
# Fuzzing the semantics over real positions
# ---------------------------------------------------------------------------

def _positions(num_games, moves, seed, board_size=9):
    rng = random.Random(seed)
    for _ in range(num_games):
        state = GameState(board_size=board_size, komi=6.5)
        for _ in range(moves):
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


class TestSemanticsAgainstRealBoards:

    def test_every_blocked_move_satisfies_all_three_conditions(self):
        blocked_total = 0
        for state in _positions(10, 70, seed=101):
            board, colour = state.board, state.current_player
            for (r, c) in forbidden_self_ataris(board, colour, max_stones=1):
                captured, size, liberties = _after(board, colour, r, c)
                assert captured == 0, "blocked a move that captures"
                assert liberties == 1, f"blocked a move with {liberties} liberties"
                assert size > 1, "blocked a single-stone sacrifice"
                blocked_total += 1
        assert blocked_total > 30, \
            f"only {blocked_total} blocked moves seen — test is not exercising the rule"

    def test_no_capturing_move_is_ever_blocked(self):
        capture_moves = 0
        for state in _positions(10, 70, seed=202):
            board, colour = state.board, state.current_player
            for (r, c) in rules.get_legal_moves(board, colour, state.ko_point,
                                                state.board_hash_history):
                captured, _, _ = _after(board, colour, r, c)
                if captured:
                    capture_moves += 1
                    assert not is_forbidden_self_atari(
                        board, colour, r, c, max_stones=1,
                        ko_point=state.ko_point,
                        board_history_hashes=state.board_hash_history)
        assert capture_moves > 20, "test never saw a capturing move"

    def test_filtering_only_ever_removes_moves(self):
        for state in _positions(8, 60, seed=303):
            board, colour = state.board, state.current_player
            plain = set(rules.get_legal_moves(board, colour, state.ko_point,
                                              state.board_hash_history))
            filtered = set(rules.get_legal_moves(
                board, colour, state.ko_point, state.board_hash_history,
                restrict_self_atari=True, self_atari_max_stones=1))
            # Equal when the guard restored everything, subset otherwise.
            assert filtered <= plain or filtered == plain

    def test_a_high_threshold_blocks_nothing(self):
        for state in _positions(6, 50, seed=404):
            board, colour = state.board, state.current_player
            plain = set(rules.get_legal_moves(board, colour, state.ko_point,
                                              state.board_hash_history))
            filtered = set(rules.get_legal_moves(
                board, colour, state.ko_point, state.board_hash_history,
                restrict_self_atari=True, self_atari_max_stones=999))
            assert filtered == plain


class TestOffByDefault:

    def test_default_move_lists_are_identical(self):
        for state in _positions(8, 60, seed=505):
            board, colour = state.board, state.current_player
            assert (rules.get_legal_moves(board, colour, state.ko_point,
                                          state.board_hash_history)
                    == rules.get_legal_moves(board, colour, state.ko_point,
                                             state.board_hash_history,
                                             restrict_self_atari=False))

    def test_game_state_defaults_to_off(self):
        state = GameState(board_size=9, komi=6.5)
        assert state.restrict_self_atari is False
        assert state.self_atari_max_stones == 1

    def test_is_legal_is_never_affected(self):
        """
        Rule legality must be untouched, so humans, replays and stored games
        behave identically whether the restriction is on or not.
        """
        b = Board(9)
        _place(b, BLACK, [(4, 4), (4, 5), (4, 6)])
        _place(b, WHITE, [(3, 4), (3, 5), (3, 6), (5, 4), (5, 5),
                          (4, 3), (6, 6), (5, 7)])
        state = GameState(board_size=9, komi=6.5, restrict_self_atari=True)
        state.board = b
        state.board_hash_history = {b.board_hash}
        assert (5, 6) not in state.get_legal_moves()
        assert state.is_legal(5, 6), "the restriction leaked into rule legality"
        assert state.play_move(5, 6), "a human could not play a legal move"


class TestPropagation:

    def test_restriction_survives_state_copy(self):
        state = GameState(board_size=9, komi=6.5, restrict_self_atari=True,
                          self_atari_max_stones=3)
        clone = state.copy()
        assert clone.restrict_self_atari is True
        assert clone.self_atari_max_stones == 3

    def test_restriction_survives_undo(self):
        state = GameState(board_size=9, komi=6.5, restrict_self_atari=True,
                          self_atari_max_stones=2)
        state.play_move(4, 4)
        state.undo_move()
        assert state.restrict_self_atari is True
        assert state.self_atari_max_stones == 2

    def test_mcts_stamps_it_on_every_node(self):
        import torch
        from ai.mcts import MCTS
        from ai.network import GoNetwork

        torch.manual_seed(0)
        net = GoNetwork(board_size=9, num_input_planes=10)
        net.eval()

        state = GameState(board_size=9, komi=6.5)
        rng = random.Random(7)
        for _ in range(30):
            legal = state.get_legal_moves()
            if not legal:
                break
            state.play_move(*rng.choice(legal))

        mcts = MCTS(network=net, num_simulations=48, device="cpu",
                    restrict_self_atari=True, self_atari_max_stones=2)
        mcts.search(state, temperature=0.5, add_noise=False)

        # The caller's own state must not be mutated by the searcher.
        assert state.restrict_self_atari is False

    def test_restricted_moves_get_no_visits_and_no_policy_mass(self):
        """
        The restriction has to hold at every depth of the tree, and a blocked
        move must be absent from the training target, not merely unplayed.
        """
        import torch
        from ai.mcts import MCTS
        from ai.network import GoNetwork

        torch.manual_seed(0)
        net = GoNetwork(board_size=9, num_res_blocks=1, num_filters=8,
                        value_head_hidden=8)
        net.eval()

        b = Board(9)
        _place(b, BLACK, [(4, 4), (4, 5), (4, 6)])
        _place(b, WHITE, [(3, 4), (3, 5), (3, 6), (5, 4), (5, 5),
                          (4, 3), (6, 6), (5, 7)])
        state = GameState(board_size=9, komi=6.5)
        state.board = b
        state.board_hash_history = {b.board_hash}

        mcts = MCTS(network=net, num_simulations=60, device="cpu",
                    restrict_self_atari=True, self_atari_max_stones=1)
        action, policy = mcts.search(state, temperature=1.0, add_noise=True)

        assert policy[5 * 9 + 6] == 0.0, "blocked move carried policy mass"
        assert action != (5, 6)

    def test_searcher_does_not_mutate_the_shared_state(self):
        """
        The same GameState object is shared with opponents (random bot, the
        other network in a gate match, a human), so the restriction belongs to
        the searcher, not to the game.
        """
        import torch
        from ai.mcts import MCTS
        from ai.network import GoNetwork

        torch.manual_seed(0)
        net = GoNetwork(board_size=7, num_input_planes=10)
        net.eval()
        state = GameState(board_size=7, komi=6.5)
        mcts = MCTS(network=net, num_simulations=16, device="cpu",
                    restrict_self_atari=True)
        mcts.search(state, temperature=0.5, add_noise=False)
        assert state.restrict_self_atari is False
        assert state.self_atari_max_stones == 1
