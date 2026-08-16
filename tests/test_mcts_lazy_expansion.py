"""
test_mcts_lazy_expansion.py — child positions are built only when selected.

`_expand` used to build a full GameState for every legal move as soon as a node
was expanded: 3,233 of them per 96-simulation search on a mid-game 9x9 board,
of which at most 96 could ever be visited. The rest were built and thrown away,
including a copy of the superko hash-history set, which grows with game length.

Children now carry only (action, prior) until the search commits to them. These
tests pin the three things that could go wrong:

  1. laziness actually happens (otherwise the optimisation is silently absent),
  2. a materialised child's position really is parent + that one move,
  3. a terminal node's cached score equals the freshly computed one.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from ai.mcts import MCTS, MCTSNode
from ai.network import GoNetwork
from game.board import EMPTY
from game.game_state import GameState, MOVE_PASS


def _net(board_size=9):
    torch.manual_seed(0)
    net = GoNetwork(board_size=board_size, num_input_planes=10)
    net.eval()
    return net


def _position(board_size, moves, seed):
    rng = random.Random(seed)
    state = GameState(board_size=board_size, komi=6.5)
    for _ in range(moves):
        legal = state.get_legal_moves()
        if not legal:
            break
        state.play_move(*rng.choice(legal))
    return state


def _walk(node, out):
    out.append(node)
    for child in node.children.values():
        _walk(child, out)
    return out


class TestLaziness:
    def test_most_children_are_never_materialised(self):
        """The whole point: unvisited children must not own a position."""
        state = _position(9, 40, seed=1)
        mcts = MCTS(network=_net(), num_simulations=96, c_puct=1.5, device="cpu")

        root = mcts._build_tree_for_test(state) if hasattr(mcts, "_build_tree_for_test") else None
        # No test hook — drive a real search and inspect the tree it left behind.
        root = _search_capturing_root(mcts, state)

        nodes = _walk(root, [])
        materialised = [n for n in nodes if n.state is not None]

        assert len(nodes) > 500, "tree too small to prove anything"
        # At most one materialisation per simulation, plus the root.
        assert len(materialised) <= 96 + 1
        assert len(materialised) < len(nodes) * 0.5, (
            f"{len(materialised)}/{len(nodes)} children were materialised — "
            f"expansion is not lazy"
        )

    def test_every_visited_node_has_a_state(self):
        state = _position(9, 30, seed=2)
        mcts = MCTS(network=_net(), num_simulations=64, c_puct=1.5, device="cpu")
        root = _search_capturing_root(mcts, state)

        for node in _walk(root, []):
            if node.visit_count > 0:
                assert node.state is not None, \
                    "a visited node must have had its position built"

    def test_unvisited_nodes_have_no_state(self):
        state = _position(9, 30, seed=3)
        mcts = MCTS(network=_net(), num_simulations=64, c_puct=1.5, device="cpu")
        root = _search_capturing_root(mcts, state)

        # A node is materialised on selection, which also gives it a visit.
        for node in _walk(root, []):
            if node.state is None:
                assert node.visit_count == 0


class TestMaterialisedStateIsCorrect:
    def test_child_position_is_parent_plus_one_move(self):
        state = _position(9, 35, seed=4)
        mcts = MCTS(network=_net(), num_simulations=96, c_puct=1.5, device="cpu")
        root = _search_capturing_root(mcts, state)

        checked = 0
        for node in _walk(root, []):
            if node.state is None or node.parent is None:
                continue
            expected = node.parent.state.copy()
            if node.parent_action == MOVE_PASS:
                expected.play_pass()
            else:
                assert expected.play_move(*node.parent_action)

            assert np.array_equal(node.state.board.grid, expected.board.grid)
            assert node.state.board.board_hash == expected.board.board_hash
            assert node.state.current_player == expected.current_player
            assert node.state.ko_point == expected.ko_point
            assert node.state.passes == expected.passes
            assert node.state.move_number == expected.move_number
            checked += 1

        assert checked > 20, "not enough materialised children to be meaningful"

    def test_restriction_flag_is_inherited_through_materialisation(self):
        """restrict_eye_fill must hold at every depth, not just the root."""
        state = _position(9, 30, seed=5)
        mcts = MCTS(network=_net(), num_simulations=64, c_puct=1.5,
                    device="cpu", restrict_eye_fill=True)
        root = _search_capturing_root(mcts, state)

        for node in _walk(root, []):
            if node.state is not None:
                assert node.state.restrict_eye_fill is True


class TestTerminalCaching:
    def test_cached_terminal_value_matches_recomputation(self):
        # A position where passing is the ONLY action, so the search is
        # guaranteed to reach a terminal node rather than merely likely to.
        # Black owns the whole board except two eyes; White cannot play into
        # either (suicide), and Black has already passed once, so White's pass
        # ends the game.
        state = GameState(board_size=7, komi=6.5)
        for r in range(7):
            for c in range(7):
                if (r, c) not in ((0, 0), (0, 2)):
                    state.board.place_stone(r, c, 1)  # BLACK
        state.board_hash_history = {state.board.board_hash}
        state.current_player = 2  # WHITE
        state.passes = 1

        assert state.get_legal_moves() == [], "setup is wrong: White has moves"

        mcts = MCTS(network=_net(7), num_simulations=48, c_puct=1.5, device="cpu")
        root = _search_capturing_root(mcts, state)

        seen_terminal = 0
        for node in _walk(root, []):
            if node.state is not None and node.state.is_over:
                if node.terminal_value is not None:
                    assert node.terminal_value == mcts._terminal_value(node.state)
                    seen_terminal += 1
        assert seen_terminal > 0, "search never reached a terminal position"


def _search_capturing_root(mcts, state):
    """Run a search and return the root node it built."""
    captured = {}
    original = MCTS._expand

    def spy(self, node, **kwargs):
        captured.setdefault('root', node)
        return original(self, node, **kwargs)

    MCTS._expand = spy
    try:
        mcts.search(state, temperature=0.5, add_noise=True)
    finally:
        MCTS._expand = original
    return captured['root']
