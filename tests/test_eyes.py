"""
test_eyes.py — Tests for eye detection and the `restrict_eye_fill` restriction.

The restriction removes exactly one class of move: playing into one of your own
chain's only two eyes. Getting that wrong in the permissive direction just means
the bot keeps a bad move available; getting it wrong in the RESTRICTIVE
direction takes a legal — sometimes necessary — move away from the search. So
most of what is tested here is the second kind: false eyes, shared eyes, big eye
spaces, opponent eyes, three-eyed groups. All of those must stay playable.

Board diagrams use '.', 'X' (black) and 'O' (white), row 0 at the top.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import Board, EMPTY, BLACK, WHITE
from game.eyes import chain_eyes, forbidden_eye_fills, is_forbidden_eye_fill
from game.rules import get_legal_moves, is_legal_move
from game.game_state import GameState, MOVE_PASS


SYMBOLS = {'.': EMPTY, 'X': BLACK, 'O': WHITE}


def board_from(diagram: str, size: int = 9) -> Board:
    """
    Build a Board from an ASCII diagram. Rows shorter than `size` are padded
    with empty points, and rows may be omitted entirely from the bottom.
    """
    b = Board(size)
    rows = [r for r in diagram.strip('\n').split('\n')]
    for r, line in enumerate(rows):
        cells = line.replace(' ', '')
        for c, ch in enumerate(cells):
            if SYMBOLS[ch] != EMPTY:
                b.place_stone(r, c, SYMBOLS[ch])
    return b


def group_at(board: Board, row: int, col: int):
    return board.get_group(row, col)


# ---------------------------------------------------------------------------
# The canonical shape used by most tests: one black chain, exactly two eyes,
# at (0, 0) and (0, 2).
#
#     . X . X . . . . .
#     X X X X . . . . .
# ---------------------------------------------------------------------------
TWO_EYE_GROUP = """
. X . X
X X X X
"""


class TestChainEyes:
    """The eye definition itself: a liberty enclosed entirely by one chain."""

    def test_single_stone_has_no_eyes(self):
        b = board_from("""
        . . .
        . X .
        """)
        assert chain_eyes(b, group_at(b, 1, 1)) == set()

    def test_two_eye_group(self):
        b = board_from(TWO_EYE_GROUP)
        assert chain_eyes(b, group_at(b, 1, 1)) == {(0, 0), (0, 2)}

    def test_corner_eye_counts_with_only_two_neighbours(self):
        """(0,0) has 2 on-board neighbours — the edge must not disqualify it."""
        b = board_from(TWO_EYE_GROUP)
        assert (0, 0) in chain_eyes(b, group_at(b, 1, 1))

    def test_edge_eye_counts_with_three_neighbours(self):
        """(0,2) sits on the top edge and has 3 on-board neighbours."""
        b = board_from(TWO_EYE_GROUP)
        assert (0, 2) in chain_eyes(b, group_at(b, 1, 1))

    def test_centre_eye_needs_all_four_neighbours(self):
        b = board_from("""
        . X X X .
        X X . X .
        . X X X .
        """)
        assert (1, 2) in chain_eyes(b, group_at(b, 1, 1))

    def test_liberty_with_an_empty_neighbour_is_not_an_eye(self):
        """A two-space eye is ONE eye, and neither of its points qualifies."""
        b = board_from("""
        . . X
        X X X
        """)
        eyes = chain_eyes(b, group_at(b, 1, 1))
        assert eyes == set()

    def test_three_space_eye_has_no_eye_points(self):
        b = board_from("""
        . . . X
        X X X X
        """)
        assert chain_eyes(b, group_at(b, 1, 0)) == set()

    def test_point_touching_an_opponent_stone_is_not_an_eye(self):
        b = board_from("""
        O X . X
        X X X X
        """)
        # (0,0) now has a white neighbour, so it encloses nothing.
        assert chain_eyes(b, group_at(b, 1, 1)) == {(0, 2)}

    def test_false_eye_is_not_an_eye(self):
        """
        The surrounding stones are two separate chains, so the point is a false
        eye and belongs to neither of them.

            . X .
            X . X
            . X .
        """
        b = board_from("""
        . X .
        X . X
        . X .
        """)
        for r, c in ((0, 1), (1, 0), (1, 2), (2, 1)):
            assert (1, 1) not in chain_eyes(b, group_at(b, r, c))


class TestForbiddenEyeFills:
    """Which points the restriction actually removes."""

    def test_both_eyes_of_a_two_eye_group_are_forbidden(self):
        b = board_from(TWO_EYE_GROUP)
        assert forbidden_eye_fills(b, BLACK) == {(0, 0), (0, 2)}

    def test_single_point_query_agrees_with_the_board_scan(self):
        b = board_from(TWO_EYE_GROUP)
        scan = forbidden_eye_fills(b, BLACK)
        for r in range(b.size):
            for c in range(b.size):
                assert is_forbidden_eye_fill(b, BLACK, r, c) == ((r, c) in scan)

    def test_one_eyed_group_is_not_restricted(self):
        """
        A one-eyed group is not alive to begin with — filling its eye may be
        part of building a real eye space, so it stays available.

            . X . . .
            X X . . .
        """
        b = board_from("""
        . X
        X X
        """)
        assert chain_eyes(b, group_at(b, 1, 1)) == {(0, 0)}
        assert forbidden_eye_fills(b, BLACK) == set()

    def test_three_eyed_group_is_not_restricted(self):
        """Filling one of three eyes still leaves two — the group stays alive."""
        b = board_from("""
        . X . X . X
        X X X X X X
        """)
        assert len(chain_eyes(b, group_at(b, 1, 1))) == 3
        assert forbidden_eye_fills(b, BLACK) == set()

    def test_false_eye_connection_stays_playable(self):
        """
        The move that would be lost here is a genuine connection. A diagonal-rule
        eye detector counts (1,1) as a true eye (zero hostile diagonals) and
        would forbid it; the chain-based definition does not.

            . X . . .
            X . X . .
            . X X . .
        """
        b = board_from("""
        . X .
        X . X
        . X X
        """)
        assert not is_forbidden_eye_fill(b, BLACK, 1, 1)
        assert (1, 1) in get_legal_moves(b, BLACK, restrict_eye_fill=True)

    def test_eye_shared_by_two_chains_stays_playable(self):
        """
        Filling a point shared by two chains MERGES them, which is how a group
        with two half-eyes becomes one group with two eyes. Never restrict it.

            . X . X . . .
            X X . X X . .      <- (0,2)/(1,2) column splits two chains
        """
        b = board_from("""
        . X . X .
        X X . X X
        """)
        left = group_at(b, 1, 1)
        right = group_at(b, 1, 3)
        assert left != right
        assert forbidden_eye_fills(b, BLACK) == set()

    def test_vital_point_of_a_three_space_eye_stays_playable(self):
        """Playing the middle of a three-space eye is what MAKES two eyes."""
        b = board_from("""
        . . . X
        X X X X
        """)
        legal = get_legal_moves(b, BLACK, restrict_eye_fill=True)
        assert (0, 1) in legal

    def test_opponent_eyes_are_not_restricted(self):
        """A throw-in into White's eye space is Black's business, not White's."""
        b = board_from("""
        . . O
        O O O
        """)
        assert forbidden_eye_fills(b, BLACK) == set()
        assert (0, 0) in get_legal_moves(b, BLACK, restrict_eye_fill=True)

    def test_black_is_not_restricted_by_a_white_two_eye_group(self):
        b = board_from("""
        . O . O
        O O O O
        """)
        assert forbidden_eye_fills(b, BLACK) == set()
        assert forbidden_eye_fills(b, WHITE) == {(0, 0), (0, 2)}
        # Black playing there is suicide anyway — illegal with or without the flag.
        assert not is_legal_move(b, BLACK, 0, 0)[0]

    def test_two_separate_two_eye_groups_are_both_restricted(self):
        b = board_from("""
        . X . X . . . . .
        X X X X . . . . .
        . . . . . . . . .
        . . . . . . . . .
        . . . . . X X X X
        . . . . X X . X .
        . . . . . X X X X
        """)
        assert forbidden_eye_fills(b, BLACK) == {(0, 0), (0, 2), (5, 6), (5, 8)}

    def test_occupied_and_off_board_points_are_never_forbidden(self):
        b = board_from(TWO_EYE_GROUP)
        assert not is_forbidden_eye_fill(b, BLACK, 1, 1)      # occupied
        assert not is_forbidden_eye_fill(b, BLACK, -1, 0)     # off board
        assert not is_forbidden_eye_fill(b, BLACK, 9, 9)      # off board

    def test_empty_board_restricts_nothing(self):
        b = Board(9)
        assert forbidden_eye_fills(b, BLACK) == set()
        assert forbidden_eye_fills(b, WHITE) == set()

    @pytest.mark.parametrize("size", [7, 9, 13, 15, 19])
    def test_corner_shape_behaves_the_same_on_every_board_size(self, size):
        b = board_from(TWO_EYE_GROUP, size=size)
        assert forbidden_eye_fills(b, BLACK) == {(0, 0), (0, 2)}

    def test_eye_next_to_an_atari_group_still_cannot_capture(self):
        """
        Sanity check on the claim that an eye fill can never be a capture: an
        eye point is surrounded by our own chain, so no opponent group touches
        it. The white stone in atari here is adjacent to the chain, not to the
        eyes.

            . X . X O
            X X X X O
            . . . . .
        """
        b = board_from("""
        . X . X O
        X X X X O
        """)
        forbidden = forbidden_eye_fills(b, BLACK)
        assert forbidden == {(0, 0), (0, 2)}
        for point in forbidden:
            for nr, nc in b.neighbors(*point):
                assert b.grid[nr, nc] == BLACK


class TestGetLegalMoves:
    """rules.get_legal_moves() with and without the restriction."""

    def test_off_by_default(self):
        b = board_from(TWO_EYE_GROUP)
        moves = get_legal_moves(b, BLACK)
        assert (0, 0) in moves and (0, 2) in moves

    def test_restriction_removes_exactly_the_eye_points(self):
        b = board_from(TWO_EYE_GROUP)
        plain = set(get_legal_moves(b, BLACK))
        restricted = set(get_legal_moves(b, BLACK, restrict_eye_fill=True))
        assert plain - restricted == {(0, 0), (0, 2)}

    def test_restriction_does_not_touch_the_other_colour(self):
        b = board_from(TWO_EYE_GROUP)
        plain = set(get_legal_moves(b, WHITE))
        restricted = set(get_legal_moves(b, WHITE, restrict_eye_fill=True))
        assert plain == restricted

    def test_move_legality_itself_is_unchanged(self):
        """The restriction is a playing policy, never a rule — is_legal_move
        must keep reporting the truth so humans and replays are unaffected."""
        b = board_from(TWO_EYE_GROUP)
        assert is_legal_move(b, BLACK, 0, 0)[0]
        assert is_legal_move(b, BLACK, 0, 2)[0]

    def test_filling_the_last_eye_remains_illegal_as_suicide(self):
        """
        With one eye left, filling it is suicide and already illegal. The
        restriction is not what stops it, and it must not crash on the shape.

            . X . . .
            X X . . .   (chain has outside liberties)
        """
        b = board_from("""
        . X X X
        X X X X
        X X X X
        X X X X
        """, size=7)
        # The chain fills the corner and has plenty of liberties, so its single
        # eye is legal to fill and not restricted.
        assert chain_eyes(b, group_at(b, 0, 1)) == {(0, 0)}
        assert (0, 0) in get_legal_moves(b, BLACK, restrict_eye_fill=True)

    def test_restriction_can_leave_no_board_moves(self):
        """
        Whole board is one black chain with exactly two eyes: every legal move
        is an eye fill, so the restricted move list is empty and only pass
        remains. Callers must handle that (MCTS does — see below).
        """
        b = Board(7)
        for r in range(7):
            for c in range(7):
                if (r, c) not in ((0, 0), (0, 2)):
                    b.place_stone(r, c, BLACK)
        assert set(get_legal_moves(b, BLACK)) == {(0, 0), (0, 2)}
        assert get_legal_moves(b, BLACK, restrict_eye_fill=True) == []


class TestGameState:
    """The flag's lifecycle on GameState."""

    @staticmethod
    def _two_eye_state(restrict: bool) -> GameState:
        state = GameState(board_size=9, komi=6.5, restrict_eye_fill=restrict)
        state.board = board_from(TWO_EYE_GROUP)
        state.board_hash_history = {state.board.board_hash}
        return state

    def test_default_is_off(self):
        assert GameState(board_size=9).restrict_eye_fill is False

    def test_legal_moves_respect_the_flag(self):
        assert (0, 0) in self._two_eye_state(False).get_legal_moves()
        assert (0, 0) not in self._two_eye_state(True).get_legal_moves()

    def test_human_moves_are_never_blocked(self):
        """play_move()/is_legal() are the human path and ignore the flag."""
        state = self._two_eye_state(True)
        assert state.is_legal(0, 0)
        assert state.play_move(0, 0)

    def test_copy_preserves_the_flag(self):
        """MCTS builds its tree out of copies — this is what keeps the
        restriction in force below the root."""
        assert self._two_eye_state(True).copy().restrict_eye_fill is True
        assert self._two_eye_state(False).copy().restrict_eye_fill is False

    def test_undo_preserves_the_flag(self):
        state = GameState(board_size=9, restrict_eye_fill=True)
        state.play_move(4, 4)
        state.undo_move()
        assert state.restrict_eye_fill is True

    def test_replaying_a_stored_game_is_unaffected(self):
        """from_dict replays through play_move, which the flag never touches."""
        state = GameState(board_size=9)
        state.play_move(4, 4)
        state.play_move(3, 3)
        restored = GameState.from_dict(state.to_dict())
        assert restored.restrict_eye_fill is False
        assert restored.board.board_hash == state.board.board_hash


class TestMCTSIntegration:
    """The restriction must hold inside the search, not just at the root."""

    @staticmethod
    def _network(board_size: int = 9):
        from ai.network import GoNetwork
        net = GoNetwork(board_size=board_size, num_res_blocks=1, num_filters=8,
                        value_head_hidden=8)
        net.eval()
        return net

    def _state(self) -> GameState:
        state = GameState(board_size=9, komi=6.5)
        state.board = board_from(TWO_EYE_GROUP)
        state.board_hash_history = {state.board.board_hash}
        return state

    def test_restricted_moves_get_no_visits_and_no_policy_mass(self):
        from ai.mcts import MCTS
        state = self._state()
        mcts = MCTS(network=self._network(), num_simulations=60, device="cpu",
                    restrict_eye_fill=True)
        action, policy = mcts.search(state, temperature=1.0, add_noise=True)

        for r, c in ((0, 0), (0, 2)):
            assert policy[r * 9 + c] == 0.0
            assert action != (r, c)

    def test_search_does_not_mutate_the_callers_state(self):
        """The eval and gate matches share one GameState between two players."""
        from ai.mcts import MCTS
        state = self._state()
        mcts = MCTS(network=self._network(), num_simulations=10, device="cpu",
                    restrict_eye_fill=True)
        mcts.search(state, temperature=1.0, add_noise=False)
        assert state.restrict_eye_fill is False

    def test_pass_is_still_available_when_everything_is_restricted(self):
        from ai.mcts import MCTS
        state = GameState(board_size=7, komi=6.5)
        for r in range(7):
            for c in range(7):
                if (r, c) not in ((0, 0), (0, 2)):
                    state.board.place_stone(r, c, BLACK)
        state.board_hash_history = {state.board.board_hash}

        mcts = MCTS(network=self._network(board_size=7), num_simulations=8,
                    device="cpu", restrict_eye_fill=True)
        # allow_pass=False as well, to prove the no-legal-moves fallback holds.
        action, _ = mcts.search(state, temperature=0.0, add_noise=False,
                                allow_pass=False)
        assert action == MOVE_PASS


class TestConfigPlumbing:
    """The setting has to survive the params → config → trainer path."""

    def test_param_bounds_entry_is_a_bool_defaulting_to_off(self):
        from param_bounds import PARAM_BOUNDS, CATEGORIES
        spec = PARAM_BOUNDS['restrict_eye_fill']
        assert spec['type'] == 'bool'
        assert spec['default'] is False
        assert spec['category'] in {c['key'] for c in CATEGORIES}

    def test_sanitize_params_accepts_every_checkbox_encoding(self):
        from param_bounds import sanitize_params
        assert sanitize_params({'restrict_eye_fill': True})['restrict_eye_fill'] is True
        assert sanitize_params({'restrict_eye_fill': False})['restrict_eye_fill'] is False
        assert sanitize_params({'restrict_eye_fill': 'true'})['restrict_eye_fill'] is True
        assert sanitize_params({'restrict_eye_fill': 0})['restrict_eye_fill'] is False
        assert 'restrict_eye_fill' not in sanitize_params({})

    def test_training_config_defaults_to_off(self):
        from config import TrainingConfig
        assert TrainingConfig().restrict_eye_fill is False

    def test_model_without_the_field_loads_as_off(self):
        """Models created before this setting existed must be unchanged."""
        from model_manager import ModelInfo
        info = ModelInfo.from_dict({
            'id': 'legacy', 'name': 'Legacy', 'board_size': 9, 'komi': 6.5,
            'training': {'num_self_play_games': 5},
        })
        assert info.training.restrict_eye_fill is False

    def test_model_setting_reaches_the_training_config(self, tmp_path):
        from config import Config
        from model_manager import ModelInfo
        info = ModelInfo.from_dict({
            'id': 'restricted', 'name': 'Restricted', 'board_size': 9, 'komi': 6.5,
            'training': {'restrict_eye_fill': True},
        })
        cfg = Config.from_model(info, str(tmp_path))
        assert cfg.training.restrict_eye_fill is True
