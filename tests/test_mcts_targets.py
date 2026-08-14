"""
test_mcts_targets.py — MCTS training-target quality and search tuning.

Covers the fixes for near-random policy targets:
- board-scaled simulation counts (config.simulations_for_board)
- FPU reduction concentrating visits (sharper, learnable targets)
- the temperature->0 target being a single argmax (no tie-degeneracy)
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MCTSConfig, simulations_for_board
from ai.network import GoNetwork
from ai.mcts import MCTS
from game.game_state import GameState


def _fresh_net():
    torch.manual_seed(0)
    net = GoNetwork(board_size=9, num_input_planes=10)
    net.eval()
    return net


class TestSimulationScaling:
    def test_9x9_is_baseline(self):
        cfg = MCTSConfig(num_simulations=96)
        assert simulations_for_board(cfg, 9) == 96

    def test_scales_with_area(self):
        cfg = MCTSConfig(num_simulations=96, min_simulations=1, max_simulations=100000)
        # 13x13 has 169/81 ~ 2.09x the area of 9x9
        assert simulations_for_board(cfg, 13) == round(96 * 169 / 81)

    def test_respects_floor_and_ceiling(self):
        cfg = MCTSConfig(num_simulations=96, min_simulations=80, max_simulations=200)
        assert simulations_for_board(cfg, 7) == 80     # would be ~58, floored
        assert simulations_for_board(cfg, 19) == 200    # would be ~428, capped

    def test_disabled_returns_base(self):
        cfg = MCTSConfig(num_simulations=50, auto_scale_simulations=False)
        assert simulations_for_board(cfg, 19) == 50


class TestLateGameTarget:
    def test_temp_zero_target_is_single_one_hot(self):
        net = _fresh_net()
        state = GameState(board_size=9, komi=6.5)
        mcts = MCTS(network=net, num_simulations=60, c_puct=1.5, device="cpu")
        action, policy = mcts.search(state, temperature=0.001, add_noise=False)

        nonzero = policy[policy > 0]
        assert nonzero.size == 1, "temp->0 target must be a single move"
        assert policy.max() == 1.0

    def test_returned_action_matches_target_peak(self):
        net = _fresh_net()
        state = GameState(board_size=9, komi=6.5)
        mcts = MCTS(network=net, num_simulations=60, c_puct=1.5, device="cpu")
        action, policy = mcts.search(state, temperature=0.001, add_noise=False)

        peak_idx = int(np.argmax(policy))
        if action == (-1, -1) or action is None:  # pass sentinel differs; skip if pass
            return
        chosen_idx = action[0] * 9 + action[1]
        assert chosen_idx == peak_idx


class TestFpuSharpens:
    def test_fpu_reduction_lowers_target_entropy(self):
        """Higher FPU should concentrate visits -> lower-entropy policy target."""
        def worst_entropy(fpu):
            worst = 0.0
            for seed in range(4):
                torch.manual_seed(seed)
                np.random.seed(seed)
                net = GoNetwork(board_size=9, num_input_planes=10)
                net.eval()
                state = GameState(board_size=9, komi=6.5)
                mcts = MCTS(network=net, num_simulations=96, c_puct=1.5,
                            device="cpu", fpu_reduction=fpu)
                _, policy = mcts.search(state, temperature=0.5, add_noise=True)
                p = policy[policy > 0]
                worst = max(worst, -np.sum(p * np.log(p)) / np.log(82))
            return worst

        no_fpu = worst_entropy(0.0)
        with_fpu = worst_entropy(0.35)
        assert with_fpu < no_fpu
        # Shipped default should be comfortably learnable.
        assert with_fpu < 0.85
