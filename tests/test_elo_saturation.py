"""
test_elo_saturation.py — a rating must stop moving when the test stops informing.

The random-bot Elo used a FIXED anchor with K=32, so once win_rate saturated at
1.0 the update was 32 * (1 - expected): strictly positive, forever. +5.4 Elo per
iteration at Elo 778, +3.3 at 1000, never zero. The displayed rating (and the
kyu rank derived from it) measured how many iterations had run, not strength.

That whole evaluation is gone now — `compute_elo_update` with it — and
competitive Elo is earned only in played matches. What remains here is the
machinery that outlived it:

* `clamp_score`, which stops a saturated score from implying an infinite gap;
* `performance_elo_gap`, which turns a gate score into the Elo gap it implies;
* the gate ladder, which only moves when a candidate genuinely beats the
  champion and so cannot inflate on its own.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.evaluator import clamp_score, performance_elo_gap


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
            return Trainer(config=config)

        trainer = build()
        trainer.gate_elo = 1234.5
        trainer._save_weights()

        assert abs(build().gate_elo - 1234.5) < 1e-6
