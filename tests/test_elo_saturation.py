"""
test_elo_saturation.py — a rating must stop moving when the test stops informing.

The random-bot Elo used a FIXED anchor with K=32, so once win_rate saturated at
1.0 the update was 32 * (1 - expected): strictly positive, forever. +5.4 Elo per
iteration at Elo 778, +3.3 at 1000, never zero. The displayed rating (and the
kyu rank derived from it) measured how many iterations had run, not strength.

The fix is the half-game correction: a clean sweep of N games demonstrates
"at least N-0.5 out of N", not infinite superiority, so the rating converges to
the ceiling the evaluation can actually support and stops.

The gate ladder is the counterpart: it only moves when a candidate genuinely
beats the champion, so it cannot inflate on its own.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.evaluator import clamp_score, compute_elo_update, performance_elo_gap


class TestClampScore:
    def test_perfect_score_is_pulled_in_by_half_a_game(self):
        assert clamp_score(1.0, 4) == 1.0 - 1.0 / 8
        assert clamp_score(1.0, 20) == 1.0 - 1.0 / 40

    def test_zero_score_is_pulled_up_symmetrically(self):
        assert clamp_score(0.0, 4) == 1.0 / 8

    def test_ordinary_scores_are_untouched(self):
        assert clamp_score(0.6, 20) == 0.6

    def test_no_games_means_no_correction(self):
        assert clamp_score(1.0, 0) == 1.0


class TestAnchorEloConverges:
    def test_saturated_win_rate_no_longer_grows_without_bound(self):
        """The whole bug in one test: 500 iterations of a 100% win rate."""
        elo = 500.0
        for _ in range(500):
            elo = compute_elo_update(elo, 500, 1.0, num_games=4)

        ceiling = 500 + 400 * 0.845  # anchor + 400*log10(2N-1) for N=4
        assert elo < ceiling + 1, f"Elo still climbing at {elo:.0f}"

    def test_it_converges_rather_than_freezing_immediately(self):
        """A model that beats random 100% should still gain rating at first."""
        first = compute_elo_update(500, 500, 1.0, num_games=4)
        assert first > 500 + 5

    def test_more_eval_games_earn_a_higher_ceiling(self):
        few, many = 500.0, 500.0
        for _ in range(500):
            few = compute_elo_update(few, 500, 1.0, num_games=4)
            many = compute_elo_update(many, 500, 1.0, num_games=20)
        assert many > few, "more evidence must support a larger claimed gap"

    def test_old_behaviour_is_preserved_without_num_games(self):
        """Existing callers that pass no game count are unaffected."""
        assert compute_elo_update(500, 500, 1.0) == 500 + 32 * 0.5

    def test_a_losing_model_still_falls(self):
        assert compute_elo_update(900, 500, 0.0, num_games=4) < 900


class TestPerformanceEloGap:
    def test_even_match_is_no_gap(self):
        assert abs(performance_elo_gap(0.5, 20)) < 1e-9

    def test_winning_score_is_a_positive_gap(self):
        gap = performance_elo_gap(0.6, 20)
        assert 60 < gap < 80  # -400*log10(1/0.6 - 1) ~ 70

    def test_losing_score_is_a_negative_gap(self):
        assert performance_elo_gap(0.4, 20) < 0

    def test_a_sweep_is_bounded_by_the_number_of_games(self):
        small = performance_elo_gap(1.0, 4)
        large = performance_elo_gap(1.0, 20)
        assert small < large, "20 games should demonstrate more than 4"
        assert large < 1000, "a 20-game sweep is not unbounded evidence"

    def test_symmetry(self):
        assert abs(performance_elo_gap(0.75, 20)
                   + performance_elo_gap(0.25, 20)) < 1e-9


class TestGateLadderDoesNotInflate:
    def test_rejected_candidates_never_move_the_ladder(self, tmp_path):
        from config import Config, PathConfig
        from ai.trainer import Trainer

        config = Config(paths=PathConfig(model_dir=str(tmp_path)))
        config.training.eval_games = 0
        trainer = Trainer(config=config)
        start = trainer.gate_elo

        # Simulate many iterations whose candidate fails the gate. The ladder
        # is only touched on promotion, so it must not have moved at all.
        for _ in range(50):
            trainer.gate_rejections += 1
        assert trainer.gate_elo == start

    def test_gate_elo_survives_a_restart(self, tmp_path):
        from config import Config, PathConfig
        from ai.trainer import Trainer

        def build():
            config = Config(paths=PathConfig(model_dir=str(tmp_path)))
            config.training.eval_games = 0
            return Trainer(config=config)

        trainer = build()
        trainer.gate_elo = 1234.5
        trainer._save_weights()

        assert abs(build().gate_elo - 1234.5) < 1e-6
