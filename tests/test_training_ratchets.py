"""
Two quantities that were only ever allowed to move in one direction.

1. The LEARNING RATE, which reset to full strength on every restart because the
   scheduler's state was not in the checkpoint. A model trained across many
   sessions therefore never annealed at all.

2. The GATE ELO ladder, which was credited the full implied rating gap of every
   promotion and never debited, so it climbed on nothing but the winner's curse.

Neither failure raises anything. Both make the training loop report progress it
is not making, which is worse than crashing.
"""

import math

import pytest
import torch

from ai.checkpoint import load_weights, save_weights
from ai.evaluator import performance_elo_gap, promotion_elo_gain
from ai.network import GoNetwork


def _net():
    torch.manual_seed(0)
    return GoNetwork(board_size=9, num_res_blocks=1, num_filters=8,
                     value_head_hidden=8)


def _rig(lr=0.002, t_max=100):
    net = _net()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max)
    return net, opt, sched


class TestSchedulerSurvivesARestart:
    def test_a_restart_freezes_the_cosine_without_the_fix(self, tmp_path):
        """
        The bug, stated as a test so the fix has something to be measured against.

        It is subtler than "the LR jumps back to the top". Restoring the
        optimizer DOES bring the annealed rate back, because Adam's state_dict
        carries the param group's lr. What breaks is the DESCENT RATE:
        CosineAnnealingLR steps recursively, scaling the current LR by a ratio
        that depends on last_epoch. Restarting puts last_epoch back to 0 — the
        flat top of the cosine, where the per-step change is ~0 — so the LR
        stops moving and stays wherever the restart found it, forever.

        Exactly what the real log shows: 0.001176 at iteration 92, then 0.001175
        / 0.001175 / 0.001174 across 93-95 (the restart, visible as the replay
        buffer dropping to 3183) instead of continuing down.
        """
        net, opt, sched = _rig()
        for _ in range(92):
            sched.step()
        annealed = opt.param_groups[0]['lr']

        path = str(tmp_path / "weights.pt")
        save_weights(model=net, optimizer=opt, iteration=92, elo=600.0,
                     kyu_rank="29k", total_games=1000, weights_path=path,
                     scheduler=sched)

        # A restart WITHOUT handing over the scheduler: fresh optimizer at the
        # configured base LR, fresh scheduler, checkpoint loaded.
        net2, opt2, sched2 = _rig()
        load_weights(path, net2, opt2)
        assert opt2.param_groups[0]['lr'] == pytest.approx(annealed), \
            "optimizer state restores the annealed LR"
        for _ in range(3):
            sched2.step()
        frozen = opt2.param_groups[0]['lr']
        assert frozen == pytest.approx(annealed, rel=0.01), \
            "the fresh scheduler should have stalled at its starting value"

        # The same restart WITH the scheduler restored keeps descending.
        net3, opt3, sched3 = _rig()
        load_weights(path, net3, opt3, scheduler=sched3)
        for _ in range(3):
            sched3.step()
        assert opt3.param_groups[0]['lr'] < annealed * 0.9

    def test_checkpoint_round_trip_keeps_the_annealed_rate(self, tmp_path):
        net, opt, sched = _rig()
        for _ in range(90):
            sched.step()
        annealed = opt.param_groups[0]['lr']

        path = str(tmp_path / "weights.pt")
        save_weights(model=net, optimizer=opt, iteration=90, elo=600.0,
                     kyu_rank="29k", total_games=1000, weights_path=path,
                     scheduler=sched)

        net2, opt2, sched2 = _rig()
        meta = load_weights(path, net2, opt2, scheduler=sched2)

        assert meta['scheduler_restored'] is True
        assert sched2.last_epoch == 90
        # Resuming must continue the cosine, not restart it.
        sched2.step()
        assert opt2.param_groups[0]['lr'] < annealed

    def test_reports_when_there_was_no_saved_schedule(self, tmp_path):
        """
        Pre-fix checkpoints carry no scheduler state. The loader has to say so,
        because that is what tells the trainer to fast-forward the cosine
        instead of silently handing a long-lived model a full-strength LR.
        """
        net, opt, sched = _rig()
        path = str(tmp_path / "weights.pt")
        save_weights(model=net, optimizer=opt, iteration=90, elo=600.0,
                     kyu_rank="29k", total_games=1000, weights_path=path)

        net2, opt2, sched2 = _rig()
        meta = load_weights(path, net2, opt2, scheduler=sched2)
        assert meta['scheduler_restored'] is False

    def test_a_mismatched_schedule_does_not_raise(self, tmp_path):
        """T_max changing between runs must degrade, not crash."""
        net, opt, sched = _rig(t_max=100)
        for _ in range(10):
            sched.step()
        path = str(tmp_path / "weights.pt")
        save_weights(model=net, optimizer=opt, iteration=10, elo=600.0,
                     kyu_rank="29k", total_games=100, weights_path=path,
                     scheduler=sched)

        net2, opt2, sched2 = _rig(t_max=250)
        meta = load_weights(path, net2, opt2, scheduler=sched2)
        assert 'scheduler_restored' in meta

    def test_trainer_persists_and_restores_it(self):
        """Both save paths, or a manual save quietly drops the schedule."""
        import inspect
        from ai import trainer
        assert 'scheduler=self.scheduler' in inspect.getsource(trainer.Trainer._save_weights)
        assert 'scheduler=self.scheduler' in inspect.getsource(trainer.Trainer.save_weights_now)
        assert 'scheduler=self.scheduler' in inspect.getsource(trainer.Trainer._try_load_weights)


class TestGateLadderIsNotARatchet:
    """
    The gate only ever sees results that cleared the threshold, so crediting
    each one at face value converts coin flips into rating. With 40 games at a
    0.6 bar, two equally strong networks clear it 13% of the time.
    """

    def test_a_result_exactly_on_the_bar_earns_nothing(self):
        assert promotion_elo_gain(0.6, 40, 0.6) == pytest.approx(0.0)

    def test_the_old_formula_paid_a_bare_pass_over_a_hundred_points(self):
        """What the ratchet was worth per lucky promotion."""
        assert performance_elo_gap(0.6, 40) > 60

    def test_a_genuine_win_is_still_paid(self):
        assert promotion_elo_gain(0.8, 40, 0.6) > 100

    def test_monotonic_in_win_rate(self):
        gains = [promotion_elo_gain(wr, 40, 0.6)
                 for wr in (0.60, 0.65, 0.70, 0.80, 0.95)]
        assert gains == sorted(gains)

    def test_below_threshold_never_returns_a_credit(self):
        for wr in (0.0, 0.25, 0.5, 0.59):
            assert promotion_elo_gain(wr, 40, 0.6) == 0.0

    def test_a_coin_flipping_run_gains_almost_nothing(self):
        """
        The end-to-end claim. Simulate the observed regime — a candidate of
        genuinely equal strength, 45 iterations of 40 gate games — and check the
        ladder stays put where the old formula would have added ~1500 points.

        Uses the exact binomial rather than sampling, so it cannot flake.
        """
        n, threshold = 40, 0.6
        iterations = 45
        old = new = 0.0
        for wins in range(n + 1):
            p = math.comb(n, wins) * 0.5 ** n
            wr = wins / n
            if wr >= threshold:
                old += p * iterations * performance_elo_gap(wr, n)
                new += p * iterations * promotion_elo_gain(wr, n, threshold)

        # ~550 points of pure phantom rating over 45 iterations, from a candidate
        # that is exactly as strong as the champion. (The real run climbed 2012
        # points over 57 iterations at a 27% promotion rate, so noise of this
        # kind accounts for a large part of it but not all — some gate_games
        # values were 24, where the bar is easier to clear by luck.)
        assert old > 400, f"expected the old ratchet to inflate, got {old:.0f}"
        assert new < old / 4
        # Per promotion rather than in total: ~13% of 45 iterations promote, so
        # this is the average phantom rating each lucky one used to be worth.
        assert (old - new) / (0.134 * iterations) > 50

    def test_trainer_uses_the_corrected_gain(self):
        import inspect
        from ai import trainer
        source = inspect.getsource(trainer.Trainer._run_promotion_gate)
        assert 'promotion_elo_gain(' in source
        assert 'self.gate_elo += performance_elo_gap' not in source
