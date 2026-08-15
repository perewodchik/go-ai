"""
test_resign_stats.py — Tests for the mercy-rule report (/training/api/resign_stats).

The endpoint answers one question: is resignation paying for itself, or
throwing games away? Its verdict drives a coloured banner, so the thing that
must not break is the verdict boundary — particularly the refusal to call the
rule "safe" on a handful of samples, which would be worse than saying nothing.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.game_store import PHASE_SELF_PLAY, save_game
from web.routes.training_routes import _wilson_interval, RESIGN_DANGER_RATE


def game(resigned=False, num_moves=100, would_resign_move=None,
         false_resign=False):
    """A self-play game record with only the fields the endpoint reads."""
    return {
        'board_size': 9, 'komi': 6.5, 'moves': [], 'num_moves': num_moves,
        'winner': 1, 'resigned': resigned,
        'would_resign_move': would_resign_move,
        'false_resign': false_resign,
    }


@pytest.fixture
def client(monkeypatch, tmp_path):
    """A Flask test client whose trainer points at an empty games dir."""
    from flask import Flask
    from config import Config, TrainingConfig, BoardConfig, PathConfig
    import web.app as web_app
    from web.routes.training_routes import training_bp

    class StubTrainer:
        _collapse_active = False

        def __init__(self):
            self.config = Config(
                board=BoardConfig(size=9),
                training=TrainingConfig(),
                paths=PathConfig(model_dir=str(tmp_path)),
            )

    stub = StubTrainer()
    monkeypatch.setattr(web_app, 'trainer', stub, raising=False)

    app = Flask(__name__)
    app.register_blueprint(training_bp, url_prefix='/training')

    games_dir = stub.config.paths.games_dir

    def write(iteration, records):
        for i, rec in enumerate(records):
            save_game(games_dir, iteration, PHASE_SELF_PLAY, i, rec)

    def fetch():
        return app.test_client().get('/training/api/resign_stats').get_json()

    return stub, write, fetch


class TestWilsonInterval:

    def test_empty_sample_has_no_interval(self):
        assert _wilson_interval(0, 0) == (None, None)

    def test_zero_successes_still_has_an_upper_bound(self):
        """0 wrong out of 3 is not proof of safety — the upper bound says so."""
        lo, hi = _wilson_interval(0, 3)
        assert lo == 0.0
        assert hi > 0.5

    def test_interval_narrows_as_evidence_accumulates(self):
        _, hi_small = _wilson_interval(0, 5)
        _, hi_large = _wilson_interval(0, 200)
        assert hi_large < hi_small
        assert hi_large < RESIGN_DANGER_RATE

    def test_interval_stays_inside_zero_one(self):
        for n in (1, 3, 17, 100):
            for k in range(n + 1):
                lo, hi = _wilson_interval(k, n)
                assert 0.0 <= lo <= hi <= 1.0


class TestVerdicts:

    def test_off_when_never_used(self, client):
        _, _, fetch = client
        data = fetch()
        assert data['enabled'] is False
        assert data['has_data'] is False
        assert data['summary']['verdict'] == 'off'

    def test_inactive_when_enabled_but_never_fired(self, client):
        stub, write, fetch = client
        stub.config.training.resign_enabled = True
        write(1, [game(), game(), game()])
        data = fetch()
        assert data['summary']['verdict'] == 'inactive'
        assert 'never fired' in data['summary']['headline'].lower()

    def test_unproven_while_evidence_is_thin(self, client):
        """Resigning but barely checked — must not claim to be safe."""
        stub, write, fetch = client
        stub.config.training.resign_enabled = True
        write(1, [game(resigned=True) for _ in range(8)]
                 + [game(num_moves=120, would_resign_move=90)])
        data = fetch()
        assert data['summary']['verdict'] == 'unproven'
        assert data['summary']['checked_games'] == 1

    def test_good_only_when_the_upper_bound_clears_the_danger_line(self, client):
        stub, write, fetch = client
        stub.config.training.resign_enabled = True
        # 200 clean checks: the interval is tight enough to call it.
        write(1, [game(num_moves=120, would_resign_move=90) for _ in range(200)]
                 + [game(resigned=True)])
        data = fetch()
        s = data['summary']
        assert s['false_resign_rate'] == 0.0
        assert s['ci_high'] < RESIGN_DANGER_RATE
        assert s['verdict'] == 'good'

    def test_caution_when_clean_but_undersampled(self, client):
        """
        Zero wrong resignations out of 15 looks perfect, but the true rate could
        still be ~20%. This must read as 'caution', never as 'good'.
        """
        stub, write, fetch = client
        stub.config.training.resign_enabled = True
        write(1, [game(num_moves=120, would_resign_move=90) for _ in range(15)]
                 + [game(resigned=True)])
        data = fetch()
        s = data['summary']
        assert s['false_resign_rate'] == 0.0
        assert s['ci_high'] > RESIGN_DANGER_RATE
        assert s['verdict'] == 'caution'

    def test_bad_when_the_rule_is_discarding_winnable_games(self, client):
        stub, write, fetch = client
        stub.config.training.resign_enabled = True
        checks = [game(num_moves=120, would_resign_move=90) for _ in range(16)]
        for c in checks[:4]:                      # 25% wrong
            c['false_resign'] = True
        write(1, checks + [game(resigned=True)])
        data = fetch()
        s = data['summary']
        assert s['false_resign_rate'] == 0.25
        assert s['verdict'] == 'bad'
        assert 'wrong' in s['headline'].lower()


class TestMeasurements:

    def test_resign_rate_counts_only_stored_self_play_games(self, client):
        stub, write, fetch = client
        stub.config.training.resign_enabled = True
        write(1, [game(resigned=True), game(resigned=True), game(), game()])
        s = fetch()['summary']
        assert s['total_games'] == 4
        assert s['total_resigned'] == 2
        assert s['resign_rate'] == 0.5

    def test_moves_saved_is_measured_from_the_playout_tail(self, client):
        """
        The saving is not guessed: playout games record where the rule WOULD
        have stopped, and they played on, so the tail length is observed.
        """
        stub, write, fetch = client
        stub.config.training.resign_enabled = True
        write(1, [game(num_moves=120, would_resign_move=90),   # 30 saved
                  game(num_moves=100, would_resign_move=80),   # 20 saved
                  game(resigned=True), game(resigned=True)])
        s = fetch()['summary']
        assert s['avg_moves_saved'] == 25.0
        assert s['est_moves_saved'] == 50      # 25 x 2 resigned games

    def test_a_playout_that_never_triggered_is_not_counted_as_evidence(self, client):
        stub, write, fetch = client
        stub.config.training.resign_enabled = True
        write(1, [game(resigned=True), game(num_moves=120)])   # no would_resign_move
        s = fetch()['summary']
        assert s['checked_games'] == 0
        assert s['false_resign_rate'] is None
        assert s['verdict'] == 'unproven'

    def test_negative_tails_are_ignored(self, client):
        """A trigger on the final move saves nothing and must not skew the average."""
        stub, write, fetch = client
        stub.config.training.resign_enabled = True
        write(1, [game(num_moves=90, would_resign_move=90),
                  game(num_moves=120, would_resign_move=100)])
        s = fetch()['summary']
        assert s['avg_moves_saved'] == 20.0


class TestChartSeries:

    def test_points_are_ordered_and_cumulative(self, client):
        stub, write, fetch = client
        stub.config.training.resign_enabled = True
        write(2, [game(num_moves=120, would_resign_move=90, false_resign=True)])
        write(1, [game(num_moves=120, would_resign_move=90)])
        write(3, [game(num_moves=120, would_resign_move=90)])

        points = fetch()['points']
        assert [p['iteration'] for p in points] == [1, 2, 3]
        # Cumulative checks accumulate; the rate is over everything so far.
        assert [p['cum_checked'] for p in points] == [1, 2, 3]
        assert points[0]['cum_false_rate'] == 0.0
        assert points[1]['cum_false_rate'] == 0.5
        assert points[2]['cum_false_rate'] == pytest.approx(1 / 3, abs=1e-4)

    def test_every_point_carries_a_band_the_chart_can_draw(self, client):
        stub, write, fetch = client
        stub.config.training.resign_enabled = True
        write(1, [game(num_moves=120, would_resign_move=90), game(resigned=True)])
        for p in fetch()['points']:
            if p['cum_false_rate'] is not None:
                assert p['cum_ci_low'] <= p['cum_false_rate'] <= p['cum_ci_high']

    def test_iterations_without_checks_leave_a_gap_not_a_zero(self, client):
        """
        A null rate must stay null — plotting it as 0% would draw a reassuring
        line through iterations that carry no evidence at all.
        """
        stub, write, fetch = client
        stub.config.training.resign_enabled = True
        write(1, [game(resigned=True)])
        points = fetch()['points']
        assert points[0]['cum_false_rate'] is None
        assert points[0]['cum_ci_high'] is None

    def test_danger_rate_is_published_for_the_threshold_rule(self, client):
        _, _, fetch = client
        assert fetch()['danger_rate'] == RESIGN_DANGER_RATE


class TestRobustness:

    def test_no_trainer_returns_an_empty_report(self, monkeypatch):
        from flask import Flask
        import web.app as web_app
        from web.routes.training_routes import training_bp
        monkeypatch.setattr(web_app, 'trainer', None, raising=False)
        app = Flask(__name__)
        app.register_blueprint(training_bp, url_prefix='/training')
        data = app.test_client().get('/training/api/resign_stats').get_json()
        assert data['has_data'] is False and data['points'] == []

    def test_games_from_before_the_feature_do_not_break_it(self, client):
        """Old records have no resign keys at all."""
        stub, write, fetch = client
        write(1, [{'board_size': 9, 'num_moves': 80, 'winner': 1, 'moves': []}])
        data = fetch()
        assert data['summary']['total_games'] == 1
        assert data['summary']['total_resigned'] == 0
        assert data['has_data'] is False

    def test_suppression_is_reported(self, client):
        stub, write, fetch = client
        stub.config.training.resign_enabled = True
        stub._collapse_active = True
        write(1, [game(resigned=True)])
        assert fetch()['summary']['suppressed'] is True
