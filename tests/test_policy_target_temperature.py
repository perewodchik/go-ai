"""
The policy TRAINING TARGET must not follow the play temperature down.

This is the regression these tests exist for. `MCTS.search` used to build one
distribution and use it for both jobs: sampling the move to play, and labelling
the position for training. A self-play schedule that anneals temperature towards
0 to play its best move late in the game therefore also annealed the LABEL
towards one-hot.

Measured on the buffer that produced it: 62% of 50,000 targets were literally
one-hot and the median position had three actions with any mass at all, out of
200 simulations over ~60 legal moves. The policy head can only behaviour-clone
its own argmax from that, so its entropy decays monotonically and the model
overfits to the narrow move set it shares with its self-play opponent. Nothing
in the training loop notices: self-play is symmetric, the gate reads ~50%
because a network is no worse than itself, and the value head stays healthy.
The first evidence is losing to external opponents it used to beat.
"""

import inspect
import random

import numpy as np
import pytest
import torch

from ai import self_play
from ai.mcts import MCTS, visit_distribution
from ai.network import GoNetwork
from game.game_state import GameState


def _entropy(p):
    p = np.asarray(p, dtype=np.float64)
    return float(-(p * np.log(np.clip(p, 1e-12, None))).sum())


def _seeded_search(mcts, state, **kwargs):
    """
    Run a search with the RNG pinned, so two calls build the same tree.

    `MCTS._expand` shuffles legal moves on purpose — with an untrained network
    the priors are near-uniform, and without the shuffle the search would always
    walk the same first few board points. add_noise=False alone therefore does
    NOT make a search reproducible, which is what these comparisons need.
    """
    random.seed(1234)
    np.random.seed(1234)
    return mcts.search(state, **kwargs)


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return GoNetwork(board_size=9, num_res_blocks=2, num_filters=16,
                     value_head_hidden=16)


# fpu_reduction=0.0 throughout these tests, which is NOT the production default
# (0.35). These tests use add_noise=False so a search can be made reproducible,
# and that combination is degenerate: with an untrained network the prior is
# near-uniform, so every child's exploration bonus is c_puct * (1/81) * sqrt(N),
# smaller than the 0.35 penalty FPU puts on an unvisited child. The first branch
# explored keeps winning the UCB comparison and absorbs every simulation, making
# the visit distribution a single move at ANY temperature — which would make
# these tests pass or fail for a reason unrelated to what they measure.
#
# This is a property of the test harness, NOT of training. Self-play always runs
# add_noise=True, and Dirichlet noise (alpha 0.3, epsilon 0.25 on 9x9) breaks the
# flat-prior symmetry that causes it. Measured on the same untrained network at
# 200 simulations: add_noise=False gives entropy 0.00 over a support of 1;
# add_noise=True gives entropy 2.02 over a support of 8.7. See
# TestNoiseKeepsEarlyLabelsAlive below, which pins that down so nobody
# re-derives the wrong conclusion from the constant above.
NO_FPU = dict(fpu_reduction=0.0)


@pytest.fixture(scope="module")
def midgame():
    state = GameState(board_size=9, komi=6.5)
    for mv in [(4, 4), (2, 2), (6, 6), (2, 6), (6, 2), (4, 2)]:
        state.play_move(*mv)
    return state


class TestVisitDistribution:
    """The maths underneath, independent of any network."""

    def test_tau_one_is_the_visit_fraction(self):
        visits = [50, 30, 20]
        probs = visit_distribution(visits, 1.0)
        assert probs == pytest.approx([0.5, 0.3, 0.2])

    def test_low_tau_collapses_a_near_tie(self):
        """
        The number that made this a real problem rather than a cosmetic one: at
        the schedule hero-of-time actually ran (tau=0.101), a 60-vs-50 visit
        split — a genuinely close decision — becomes a near-certainty.
        """
        close = visit_distribution([60, 50], 0.101)
        assert close[0] > 0.85
        fair = visit_distribution([60, 50], 1.0)
        assert fair[0] == pytest.approx(60 / 110)


class TestSearchDecouplesTheTwo:
    def test_target_temperature_is_a_parameter(self):
        assert 'target_temperature' in inspect.signature(MCTS.search).parameters

    def test_target_defaults_to_tau_one(self):
        """A caller that says nothing must get AlphaZero's target, not a sharp one."""
        default = inspect.signature(MCTS.search).parameters['target_temperature'].default
        assert default == 1.0

    def test_label_is_unchanged_by_the_play_temperature(self, net, midgame):
        """
        The whole point: three very different play temperatures, one label.
        """
        mcts = MCTS(network=net, num_simulations=64, device="cpu", **NO_FPU)
        labels = [
            _seeded_search(mcts, midgame, temperature=play_temp,
                           add_noise=False, target_temperature=1.0)[1]
            for play_temp in (1.0, 0.101, 0.0)
        ]

        # Same seed means the same tree, so the tau=1 label built from its visit
        # counts must be identical whatever the played move was sampled at.
        for other in labels[1:]:
            np.testing.assert_allclose(labels[0], other, atol=1e-6)

    def test_sharp_target_temperature_still_collapses_the_label(self, net, midgame):
        """
        The old behaviour, reachable only by asking for it explicitly now. Kept
        as a test so the difference the fix makes stays measurable rather than
        asserted.
        """
        # 256 sims, not 64: with fewer simulations than the ~75 legal moves here
        # every child ends on exactly one visit, and a perfectly flat visit
        # vector is uniform at every tau — the test would pass trivially at
        # tau=1 and trivially fail to distinguish anything.
        mcts = MCTS(network=net, num_simulations=256, device="cpu", **NO_FPU)
        healthy = _seeded_search(mcts, midgame, temperature=0.101,
                                 add_noise=False, target_temperature=1.0)[1]
        collapsed = _seeded_search(mcts, midgame, temperature=0.101,
                                   add_noise=False, target_temperature=0.101)[1]

        assert _entropy(collapsed) < _entropy(healthy)
        assert (collapsed > 1e-6).sum() < (healthy > 1e-6).sum()
        assert healthy.max() < collapsed.max()

    def test_zero_play_temperature_plays_the_most_visited_move(self, net, midgame):
        """
        Competitive play takes the argmax. It used to run at tau=0.1, which
        SAMPLES from N^10 — a 55-vs-50 root split plays the second-best move
        about a quarter of the time, and at these sim counts near-ties are the
        common case.

        Checked against the tau=1 label from the same search, which is the visit
        distribution itself: the played move must be one of its most-visited
        actions. "One of" rather than "equal to argmax" because visit ties are
        common and the two argmaxes break them in different orders — the search
        ties over its own shuffled action list, np.argmax over board index.
        """
        mcts = MCTS(network=net, num_simulations=256, device="cpu", **NO_FPU)
        size = midgame.board_size
        for seed in range(5):
            random.seed(seed)
            np.random.seed(seed)
            action, policy = mcts.search(midgame, temperature=0.0,
                                         add_noise=False, target_temperature=1.0)
            idx = (size * size if action == self_play.MOVE_PASS
                   else action[0] * size + action[1])
            assert policy[idx] == pytest.approx(policy.max())

    def test_zero_play_temperature_is_reproducible(self, net, midgame):
        """Same tree, same move — no coin flip left in the selection step."""
        mcts = MCTS(network=net, num_simulations=64, device="cpu", **NO_FPU)
        chosen = {_seeded_search(mcts, midgame, temperature=0.0,
                                 add_noise=False)[0] for _ in range(4)}
        assert len(chosen) == 1

    def test_label_support_covers_the_sampled_move(self, net, midgame):
        """
        The invariant the old coupled version got for free and this one has to
        preserve: a position is never stored with a played move its own label
        gives zero probability. Sampling at tau < 1 only concentrates mass on
        moves that already hold mass at tau = 1, so it holds.
        """
        mcts = MCTS(network=net, num_simulations=64, device="cpu", **NO_FPU)
        size = midgame.board_size
        for _ in range(8):
            action, policy = mcts.search(midgame, temperature=0.3, add_noise=True,
                                         target_temperature=1.0)
            idx = (size * size if action == self_play.MOVE_PASS
                   else action[0] * size + action[1])
            assert policy[idx] > 0.0


class TestNoiseKeepsEarlyLabelsAlive:
    """
    A from-scratch run's first iterations produce usable labels.

    Worth pinning down because the opposite is easy to conclude by accident: with
    add_noise=False an untrained network at production fpu_reduction returns a
    one-move label, which looks like early training being doomed. It is not —
    self-play never disables the noise. This test exists so that measurement is
    made on the path training actually takes.
    """

    def test_untrained_network_still_produces_a_spread_label(self):
        # The production architecture (4x64), not the small net the other tests
        # share. The raw prior is uniform either way at initialisation, but how
        # widely the search SPREADS its visits depends on the value head: the
        # 2x16 net used elsewhere here lands at 0.38 nats where 4x64 gives 2.00,
        # from the same near-uniform prior. Since the claim being made is about
        # real training, it has to be measured on the real shape.
        torch.manual_seed(0)
        net = GoNetwork(board_size=9, num_res_blocks=4, num_filters=64,
                        value_head_hidden=64)
        net.eval()
        state = GameState(board_size=9, komi=6.5)
        for mv in [(4, 4), (2, 2), (6, 6), (2, 6), (6, 2), (4, 2)]:
            state.play_move(*mv)

        # Production settings: FPU on, noise on, as run_self_play_batch uses them.
        mcts = MCTS(network=net, num_simulations=200, device="cpu",
                    fpu_reduction=0.35)
        supports, entropies = [], []
        for seed in range(5):
            random.seed(seed)
            np.random.seed(seed)
            _, policy = mcts.search(state, temperature=0.8, add_noise=True,
                                    target_temperature=1.0)
            supports.append(int((policy > 1e-6).sum()))
            entropies.append(_entropy(policy))

        mean_support = sum(supports) / len(supports)
        mean_entropy = sum(entropies) / len(entropies)
        assert mean_support >= 5, (
            f"an untrained network should still spread its visits; got a support "
            f"of {mean_support:.1f}")
        assert mean_entropy >= 1.0, (
            f"label entropy {mean_entropy:.2f} is below what the collapse guard "
            f"considers healthy, on the very first iteration")

    def test_disabling_noise_is_what_collapses_it(self):
        """The contrast, so the cause is unambiguous rather than assumed."""
        torch.manual_seed(0)
        net = GoNetwork(board_size=9, num_res_blocks=2, num_filters=16,
                        value_head_hidden=16)
        net.eval()
        state = GameState(board_size=9, komi=6.5)
        for mv in [(4, 4), (2, 2), (6, 6), (2, 6), (6, 2), (4, 2)]:
            state.play_move(*mv)
        mcts = MCTS(network=net, num_simulations=200, device="cpu",
                    fpu_reduction=0.35)

        random.seed(0); np.random.seed(0)
        _, quiet = mcts.search(state, temperature=0.8, add_noise=False,
                               target_temperature=1.0)
        random.seed(0); np.random.seed(0)
        _, noisy = mcts.search(state, temperature=0.8, add_noise=True,
                               target_temperature=1.0)

        assert (quiet > 1e-6).sum() < (noisy > 1e-6).sum()
        assert _entropy(noisy) > _entropy(quiet)


class TestPlumbing:
    """
    A setting that stops at a function boundary is a dead setting. This repo has
    already lost temperature_init/temperature_final that way — fully wired
    through config, the model file and the tuning API, then dropped from the task
    dict, so every model silently ran the hardcoded schedule.
    """

    def test_self_play_worker_accepts_it(self):
        params = inspect.signature(self_play.play_self_play_game).parameters
        assert 'policy_target_temperature' in params
        assert params['policy_target_temperature'].default == 1.0

    def test_batch_accepts_it(self):
        params = inspect.signature(self_play.run_self_play_batch).parameters
        assert 'policy_target_temperature' in params

    def test_task_dict_carries_it_across_the_process_boundary(self):
        from tests.test_temperature_plumbing import TASK_KWARGS
        kwargs = dict(TASK_KWARGS)
        kwargs.update(policy_target_temperature=1.0)
        task = self_play.build_self_play_task(**kwargs)
        assert task['policy_target_temperature'] == 1.0

    def test_worker_actually_forwards_it_to_the_search(self):
        """
        The task dict reaching the worker is not enough — the worker has to hand
        it to MCTS.search rather than letting the default apply.
        """
        source = inspect.getsource(self_play.play_self_play_game)
        assert 'target_temperature=policy_target_temperature' in source

    def test_config_default_is_alphazero(self):
        from config import MCTSConfig
        assert MCTSConfig().policy_target_temperature == 1.0

    def test_model_config_default_is_alphazero(self):
        """
        Models written before the split carry no value, and must inherit 1.0 —
        NOT their old temperature_final, which would preserve the bug.
        """
        from model_manager import TrainingParams
        assert TrainingParams().policy_target_temperature == 1.0

    def test_from_model_defaults_old_models_to_one(self, tmp_path):
        from config import Config
        from model_manager import ModelInfo, TrainingParams

        params = TrainingParams()
        # Simulate a pre-fix model file: the attribute simply is not there.
        del params.__dict__['policy_target_temperature']
        info = ModelInfo(id="m", name="m", board_size=9, komi=6.5,
                         ruleset="chinese", training=params)
        config = Config.from_model(info, str(tmp_path))
        assert config.mcts.policy_target_temperature == 1.0

    def test_trainer_passes_the_config_value(self):
        from ai import trainer
        source = inspect.getsource(trainer.Trainer.train)
        assert 'policy_target_temperature=self.config.mcts.policy_target_temperature' in source


class TestGameRecordCarriesTheDiagnostics:
    """
    The record is what the dashboard reads, so the entropy numbers have to be in
    it — this is the only per-iteration measurement of label health that does not
    lag behind the 50k replay buffer.
    """

    def test_record_reports_label_entropy(self, net):
        _, record = self_play.play_self_play_game(
            network=net, board_size=9, komi=6.5, num_simulations=32,
            temperature_threshold=4, temperature_init=0.8,
            temperature_final=0.101, policy_target_temperature=1.0,
            device="cpu", max_moves=30, fpu_reduction=0.0,
        )
        assert record['policy_target_temperature'] == 1.0
        # Short games are discarded before they produce samples, in which case
        # there is nothing to average and None is the honest answer.
        if record['target_entropy'] is not None:
            assert record['target_entropy'] > 0.0
            assert record['target_support'] >= 1.0
            assert 0.0 < record['target_max_prob'] <= 1.0
