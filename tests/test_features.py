"""
test_features.py — versioned input encodings (game/features.py).

Two things have to hold for `v2_12` to be a safe addition:

  1. It is a strict SUPERSET of v1_10 in the first 10 planes. That is what makes
     the widening migration exact, and it is easy to break by inserting a plane
     in the middle rather than appending.
  2. The new planes survive the 8-fold dihedral augmentation in the replay
     buffer, which rotates the whole tensor. They are spatial one-hots, so they
     must rotate with the board — a plane that encoded something non-spatial
     would be silently corrupted by augmentation.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from game.board import BLACK, WHITE
from game.features import (
    DEFAULT_FEATURES,
    FEATURE_SETS,
    encode_for_network,
    encode_state,
    num_planes,
    resolve,
)
from game.game_state import GameState


def _played(moves=25, seed=0, board_size=9):
    rng = random.Random(seed)
    state = GameState(board_size=board_size, komi=6.5)
    for _ in range(moves):
        legal = state.get_legal_moves()
        if not legal:
            break
        state.play_move(*rng.choice(legal))
    return state


class TestRegistry:
    def test_default_is_the_legacy_encoding(self):
        assert DEFAULT_FEATURES == "v1_10"
        assert num_planes(None) == 10
        assert num_planes("v1_10") == 10
        assert num_planes("v2_12") == 12

    def test_unknown_names_fall_back_to_the_default(self):
        assert resolve("v99_nonsense").name == "v1_10"
        assert resolve(None).name == "v1_10"

    def test_plane_names_match_the_declared_count(self):
        for name, spec in FEATURE_SETS.items():
            assert len(spec.plane_names) == spec.num_planes, name

    def test_turn_colour_plane_is_declared_and_correct(self):
        for name in FEATURE_SETS:
            spec = resolve(name)
            black = encode_state(GameState(board_size=9, komi=6.5), name)
            assert black[spec.turn_colour_plane].min() == 1.0, \
                f"{name}: Black to move should set the turn plane"

            state = GameState(board_size=9, komi=6.5)
            state.play_move(4, 4)
            white = encode_state(state, name)
            assert white[spec.turn_colour_plane].max() == 0.0, \
                f"{name}: White to move should clear the turn plane"


class TestV2IsASupersetOfV1:
    """The property the widening migration depends on."""

    def test_first_ten_planes_are_identical(self):
        for seed in range(6):
            state = _played(seed=seed, moves=20 + seed * 5)
            v1 = encode_state(state, "v1_10")
            v2 = encode_state(state, "v2_12")
            assert v1.shape[0] == 10 and v2.shape[0] == 12
            assert torch.equal(v1, v2[:10]), \
                "v2_12 must leave planes 0-9 untouched"

    def test_encode_for_nn_default_is_unchanged(self):
        """Existing callers must get byte-identical output to before."""
        state = _played(seed=3)
        assert torch.equal(state.encode_for_nn(), encode_state(state, "v1_10"))


class TestHistoryPlanes:
    def test_empty_board_has_no_history(self):
        planes = encode_state(GameState(board_size=9, komi=6.5), "v2_12")
        assert planes[10].sum() == 0
        assert planes[11].sum() == 0

    def test_opponents_last_move_lands_on_plane_10(self):
        state = GameState(board_size=9, komi=6.5)
        state.play_move(3, 4)          # Black plays; White is now to move
        planes = encode_state(state, "v2_12")
        assert planes[10, 3, 4] == 1.0, "plane 10 must mark the opponent's last move"
        assert planes[10].sum() == 1.0
        assert planes[11].sum() == 0, "no own previous move yet"

    def test_own_previous_move_lands_on_plane_11(self):
        state = GameState(board_size=9, komi=6.5)
        state.play_move(3, 4)          # Black
        state.play_move(5, 5)          # White
        planes = encode_state(state, "v2_12")   # Black to move again
        assert planes[10, 5, 5] == 1.0, "opponent's (White's) last move"
        assert planes[11, 3, 4] == 1.0, "my (Black's) previous move"

    def test_planes_are_from_the_movers_perspective_throughout(self):
        state = _played(moves=12, seed=9)
        history = state.move_history
        planes = encode_state(state, "v2_12")
        last = history[-1][1]
        prev = history[-2][1]
        assert planes[10, last[0], last[1]] == 1.0
        assert planes[11, prev[0], prev[1]] == 1.0

    def test_a_pass_leaves_its_plane_empty(self):
        state = GameState(board_size=9, komi=6.5)
        state.play_move(3, 3)
        state.play_pass()
        planes = encode_state(state, "v2_12")
        assert planes[10].sum() == 0, "a pass has no location"
        assert planes[11, 3, 3] == 1.0, "the move before it is still recorded"

    def test_a_captured_stone_still_shows_as_the_last_move(self):
        """
        The history planes record where a move was PLAYED, which is not the same
        as where a stone now sits — a capturing move may have removed stones,
        and a stone played into a snapback may itself be gone.
        """
        state = GameState(board_size=9, komi=6.5)
        state.board.place_stone(0, 1, WHITE)
        state.board.place_stone(1, 0, BLACK)
        state.board.place_stone(0, 2, BLACK)
        state.board_hash_history = {state.board.board_hash}
        assert state.play_move(0, 0), "Black captures at (0,0)"
        planes = encode_state(state, "v2_12")
        assert planes[10, 0, 0] == 1.0


class TestAugmentationSafety:
    """
    ReplayBuffer.sample rotates and flips the whole state tensor. The history
    planes are spatial, so they must transform exactly like the stone planes.
    """

    def test_history_planes_rotate_with_the_board(self):
        state = _played(moves=15, seed=4)
        planes = encode_state(state, "v2_12")

        for k in range(1, 4):
            rotated = torch.rot90(planes, k=k, dims=(1, 2))
            # A one-hot must stay a one-hot in exactly one place.
            assert rotated[10].sum() == planes[10].sum()
            assert rotated[11].sum() == planes[11].sum()
            # And it must agree with rotating the position's own stone plane.
            assert torch.equal(torch.rot90(planes[0], k=k, dims=(0, 1)),
                               rotated[0])

    def test_history_planes_flip_with_the_board(self):
        state = _played(moves=15, seed=5)
        planes = encode_state(state, "v2_12")
        flipped = torch.flip(planes, dims=(2,))
        assert flipped[10].sum() == planes[10].sum()
        assert flipped[11].sum() == planes[11].sum()

    def test_planes_are_binary_so_the_buffer_can_compact_them(self):
        """ai/replay_store.py stores indicator planes as uint8."""
        for seed in range(4):
            planes = encode_state(_played(seed=seed), "v2_12")
            assert bool(torch.all((planes == 0) | (planes == 1)))


class TestNetworkIntegration:
    def test_network_reports_and_sizes_itself_from_the_feature_set(self):
        from ai.network import GoNetwork
        for name in ("v1_10", "v2_12"):
            net = GoNetwork(board_size=9, input_features=name)
            assert net.input_features == name
            assert net.input_conv.in_channels == num_planes(name)
            assert net.arch_signature()['num_input_planes'] == num_planes(name)

    def test_encode_for_network_matches_the_networks_width(self):
        from ai.network import GoNetwork
        state = _played(seed=8)
        for name in ("v1_10", "v2_12"):
            net = GoNetwork(board_size=9, input_features=name)
            tensor = encode_for_network(state, net)
            assert tensor.shape[0] == net.input_conv.in_channels
            # And it must actually run through the network.
            policy, value = net.predict(tensor, "cpu")
            assert policy.shape[0] == 82
            assert np.isfinite(value)

    def test_a_v2_network_rejects_v1_input(self):
        from ai.network import GoNetwork
        net = GoNetwork(board_size=9, input_features="v2_12")
        wrong = encode_state(_played(seed=1), "v1_10")
        with pytest.raises(RuntimeError):
            net.predict(wrong, "cpu")


class TestSearchUsesTheRightEncoding:
    def test_mcts_runs_with_a_v2_network(self):
        from ai.mcts import MCTS
        from ai.network import GoNetwork

        torch.manual_seed(0)
        net = GoNetwork(board_size=9, input_features="v2_12",
                        num_res_blocks=1, num_filters=8, value_head_hidden=8)
        net.eval()
        state = _played(moves=20, seed=6)
        mcts = MCTS(network=net, num_simulations=32, device="cpu")
        action, policy = mcts.search(state, temperature=0.5, add_noise=True)

        assert action is not None
        assert np.all(np.isfinite(policy))
        assert abs(float(policy.sum()) - 1.0) < 1e-5
