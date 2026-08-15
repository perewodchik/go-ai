"""
test_policy_target_numerics.py — the policy target must stay a probability
distribution at every temperature the sliders allow.

The bug this pins down: the target used to be built as

    policy[i] = float(visit_count) ** (1.0 / temperature)

written straight into a float32 array. float32 tops out at ~3.4e38, so
overflow starts as soon as (1/T) * log10(N) > 38.5 — around T = 0.075 for
realistic visit counts. The entry became +inf, normalising turned it into NaN,
and one optimizer step on a NaN target turns the whole network into NaN.

param_bounds allows temperature_final down to 0.001 and MCTSConfig's own
default is 0.001, so this was reachable from the UI and from the CLI.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.mcts import MCTS, visit_distribution
from ai.network import GoNetwork
from game.game_state import GameState


# Every temperature the schedule can pass through, including the slider floor.
TEMPERATURES = [1.0, 0.8, 0.5, 0.2, 0.1, 0.075, 0.05, 0.0343, 0.02, 0.011, 0.001]

# Visit counts up to the largest sim budget the sliders allow (1000).
VISIT_SETS = [
    [1, 1, 1],
    [5, 3, 2],
    [50, 20, 10, 1],
    [96, 40, 12, 4, 1],
    [800, 120, 60, 20],
    [1000, 1, 1],
]


class TestVisitDistribution:
    """The pure helper both the training target and move selection use."""

    def test_always_a_valid_distribution(self):
        for temp in TEMPERATURES:
            for visits in VISIT_SETS:
                probs = visit_distribution(visits, temp)
                assert np.all(np.isfinite(probs)), \
                    f"non-finite probs at T={temp}, visits={visits}: {probs}"
                assert probs.min() >= 0.0
                assert abs(probs.sum() - 1.0) < 1e-9, \
                    f"probs do not sum to 1 at T={temp}, visits={visits}"

    def test_matches_naive_formula_where_the_naive_one_is_safe(self):
        """
        Log-space must be a drop-in for N^(1/T) wherever float64 can express
        it — this is a numerics fix, not a change of target.
        """
        for temp in [1.0, 0.8, 0.5, 0.2]:
            for visits in VISIT_SETS:
                naive = np.array(visits, dtype=np.float64) ** (1.0 / temp)
                naive = naive / naive.sum()
                got = visit_distribution(visits, temp)
                assert np.allclose(got, naive, rtol=1e-9, atol=1e-12)

    def test_low_temperature_concentrates_on_the_argmax(self):
        probs = visit_distribution([50, 20, 10, 1], 0.02)
        assert probs.argmax() == 0
        assert probs[0] > 0.999

    def test_higher_temperature_is_flatter(self):
        hot = visit_distribution([50, 20, 10, 1], 1.0)
        cold = visit_distribution([50, 20, 10, 1], 0.2)
        assert cold[0] > hot[0], "lower temperature must be more peaked"

    def test_zero_visits_fall_back_to_uniform(self):
        probs = visit_distribution([0, 0, 0], 0.5)
        assert np.allclose(probs, 1.0 / 3.0)

    def test_unvisited_children_get_no_mass(self):
        probs = visit_distribution([10, 0, 5], 0.5)
        assert probs[1] == 0.0


class TestSearchPolicyTarget:
    """End-to-end: the target that actually reaches the replay buffer."""

    def _net(self, board_size=7):
        torch.manual_seed(0)
        net = GoNetwork(board_size=board_size, num_input_planes=10)
        net.eval()
        return net

    def test_policy_is_finite_at_every_temperature(self):
        net = self._net()
        for temp in TEMPERATURES:
            state = GameState(board_size=7, komi=6.5)
            mcts = MCTS(network=net, num_simulations=200, c_puct=1.5, device="cpu")
            action, policy = mcts.search(state, temperature=temp, add_noise=True)

            assert np.all(np.isfinite(policy)), \
                f"NaN/inf in policy target at temperature {temp}"
            assert abs(float(policy.sum()) - 1.0) < 1e-5, \
                f"policy target does not sum to 1 at temperature {temp}"
            assert policy.min() >= 0.0
            assert action is not None

    def test_selected_action_has_support_in_the_target(self):
        """
        A move the search actually played must carry probability mass in the
        target it is stored with, at every temperature.
        """
        net = self._net()
        for temp in [0.8, 0.2, 0.05, 0.011, 0.001]:
            state = GameState(board_size=7, komi=6.5)
            mcts = MCTS(network=net, num_simulations=120, c_puct=1.5, device="cpu")
            action, policy = mcts.search(state, temperature=temp, add_noise=False)
            idx = 49 if action == (-1, -1) else action[0] * 7 + action[1]
            assert policy[idx] > 0.0, \
                f"played move has zero target probability at T={temp}"
