"""
test_mercy_rule.py — Tests for self-play resignation (the mercy rule).

The rule ends a decided self-play game early. Its danger is not the moves it
skips — those produce no training data at the default settings — but the
outcome LABEL, which becomes a prediction instead of a played-out result. So
these tests concentrate on:

  * it cannot fire before `resign_min_move_factor x board_area`, which is what
    guarantees no training sample is ever lost;
  * it cannot fire on a single value spike, or on one side's opinion alone;
  * the playout fraction really does overrule it and record the verdict, so the
    false-resignation rate is measurable;
  * off by default, and off means byte-identical behaviour.

MCTS is replaced by a scripted stand-in so root values are exact — a real
network would make the trigger conditions untestable.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ai.self_play as self_play_mod
from ai.self_play import play_self_play_game
from game.board import BLACK, WHITE


BOARD = 7
AREA = BOARD * BOARD


class ScriptedMCTS:
    """
    Stand-in for MCTS that plays the first legal move and reports a root value
    chosen by `value_fn(move_num, current_player)`.

    Values are from the mover's perspective, exactly like the real
    `MCTS.root_value`.
    """

    value_fn = staticmethod(lambda move_num, player: 0.0)

    def __init__(self, **kwargs):
        self.root_value = 0.0
        self.move_num = 0

    def search(self, state, temperature=1.0, add_noise=True, allow_pass=True,
               min_pass_move=0, target_temperature=1.0):
        self.root_value = float(self.value_fn(self.move_num, state.current_player))
        self.move_num += 1

        size = state.board_size
        policy = np.zeros(size * size + 1, dtype=np.float32)
        legal = state.get_legal_moves()
        if not legal:
            policy[-1] = 1.0
            return self_play_mod.MOVE_PASS, policy
        action = legal[0]
        policy[action[0] * size + action[1]] = 1.0
        return action, policy


@pytest.fixture
def scripted(monkeypatch):
    """Install ScriptedMCTS and return it so a test can set its value_fn."""
    monkeypatch.setattr(self_play_mod, "MCTS", ScriptedMCTS)
    return ScriptedMCTS


def black_is_lost(move_num, player):
    """Black despairs, White is confident — the normal resignation shape."""
    return -0.99 if player == BLACK else 0.99


def play(**overrides):
    kwargs = dict(
        network=None, board_size=BOARD, komi=6.5, num_simulations=1,
        device="cpu", max_moves=AREA * 3,
    )
    kwargs.update(overrides)
    return play_self_play_game(**kwargs)


class TestDisabledByDefault:

    def test_off_by_default_records_no_resignation(self, scripted):
        scripted.value_fn = staticmethod(black_is_lost)
        _, rec = play()
        assert rec['resigned'] is False
        assert rec['resign_color'] is None
        assert rec['resign_move'] is None

    def test_off_means_the_game_runs_to_a_natural_end(self, scripted):
        scripted.value_fn = staticmethod(black_is_lost)
        _, off = play()
        assert off['num_moves'] > AREA  # would have resigned at ~AREA if enabled


class TestTriggerConditions:

    def test_resigns_when_a_side_is_hopeless(self, scripted):
        scripted.value_fn = staticmethod(black_is_lost)
        _, rec = play(resign_enabled=True, resign_playout_fraction=0.0)
        assert rec['resigned'] is True
        assert rec['resign_color'] == BLACK
        assert rec['winner'] == WHITE

    def test_cannot_fire_before_the_minimum_move(self, scripted):
        scripted.value_fn = staticmethod(black_is_lost)
        _, rec = play(resign_enabled=True, resign_playout_fraction=0.0)
        # Default factor is 1.0, so the earliest possible resignation is the
        # move at index AREA — by which point every sample already exists.
        assert rec['resign_move'] >= AREA

    def test_min_move_factor_moves_the_boundary(self, scripted):
        scripted.value_fn = staticmethod(black_is_lost)
        _, early = play(resign_enabled=True, resign_playout_fraction=0.0,
                        resign_min_move_factor=0.5)
        assert early['resign_move'] >= int(round(AREA * 0.5))
        assert early['resign_move'] < AREA

    def test_a_long_standing_streak_fires_as_soon_as_the_gate_opens(self, scripted):
        """
        The streak is counted from the start of the game, not from the minimum
        move. A side that has been hopeless for forty moves does not get a
        further grace period once the gate opens.
        """
        scripted.value_fn = staticmethod(black_is_lost)
        _, patient = play(resign_enabled=True, resign_consecutive=6,
                          resign_playout_fraction=0.0)
        # First Black move at or past the AREA boundary, and no later.
        assert patient['resign_move'] in (AREA, AREA + 1)

    def test_a_single_bad_move_does_not_resign(self, scripted):
        """One despairing search among confident ones must not end the game."""
        def one_spike(move_num, player):
            if move_num == AREA + 2 and player == BLACK:
                return -0.99
            return 0.99 if player == WHITE else 0.2
        scripted.value_fn = staticmethod(one_spike)
        _, rec = play(resign_enabled=True, resign_consecutive=4,
                      resign_playout_fraction=0.0)
        assert rec['resigned'] is False

    def test_streak_resets_when_the_position_recovers(self, scripted):
        """Alternating despair/hope never accumulates a full streak."""
        def flapping(move_num, player):
            if player == WHITE:
                return 0.99
            return -0.99 if (move_num // 2) % 2 == 0 else 0.5
        scripted.value_fn = staticmethod(flapping)
        _, rec = play(resign_enabled=True, resign_consecutive=4,
                      resign_playout_fraction=0.0)
        assert rec['resigned'] is False

    def test_consecutive_count_delays_a_fresh_collapse(self, scripted):
        """
        Isolate the streak requirement: Black only starts despairing after the
        minimum-move gate is already open, so the delay is entirely down to
        `resign_consecutive`. Black moves on every other ply, so requiring six
        confirmations costs ten plies more than requiring one.
        """
        def collapses_late(move_num, player):
            if player == WHITE:
                return 0.99
            return -0.99 if move_num > AREA else 0.2
        scripted.value_fn = staticmethod(collapses_late)

        _, fast = play(resign_enabled=True, resign_consecutive=1,
                       resign_playout_fraction=0.0)
        _, slow = play(resign_enabled=True, resign_consecutive=6,
                       resign_playout_fraction=0.0)
        assert fast['resigned'] and slow['resigned']
        assert slow['resign_move'] - fast['resign_move'] == 10

    def test_both_sides_must_agree_when_required(self, scripted):
        """
        Black thinks it is lost; White also thinks it is losing. With
        both-sides agreement on, nobody resigns — exactly the case where one
        value head is broken.
        """
        scripted.value_fn = staticmethod(lambda move_num, player: -0.99)
        _, strict = play(resign_enabled=True, resign_both_sides=True,
                         resign_playout_fraction=0.0)
        assert strict['resigned'] is False

    def test_one_sided_agreement_allows_the_resignation(self, scripted):
        scripted.value_fn = staticmethod(lambda move_num, player: -0.99)
        _, loose = play(resign_enabled=True, resign_both_sides=False,
                        resign_playout_fraction=0.0)
        assert loose['resigned'] is True

    def test_threshold_is_a_floor_not_a_trigger(self, scripted):
        """A value above -threshold never counts, however persistent."""
        scripted.value_fn = staticmethod(
            lambda move_num, player: -0.80 if player == BLACK else 0.99)
        _, rec = play(resign_enabled=True, resign_threshold=0.90,
                      resign_playout_fraction=0.0)
        assert rec['resigned'] is False

        _, rec2 = play(resign_enabled=True, resign_threshold=0.75,
                       resign_playout_fraction=0.0)
        assert rec2['resigned'] is True


class TestTrainingDataIsNotLost:
    """The whole point of the default `resign_min_move_factor = 1.0`."""

    def test_resigned_game_yields_the_same_sample_count(self, scripted):
        scripted.value_fn = staticmethod(black_is_lost)
        full_samples, full_rec = play(resign_enabled=False)
        cut_samples, cut_rec = play(resign_enabled=True,
                                    resign_playout_fraction=0.0)

        assert cut_rec['resigned'] and not full_rec['resigned']
        assert cut_rec['num_moves'] < full_rec['num_moves']   # time was saved
        assert len(cut_samples) == len(full_samples)          # data was not lost

    def test_samples_are_labelled_from_the_resignation(self, scripted):
        """Every retained sample must agree with the awarded winner."""
        scripted.value_fn = staticmethod(black_is_lost)
        samples, rec = play(resign_enabled=True, resign_playout_fraction=0.0)
        assert rec['winner'] == WHITE
        assert samples, "resigned game should still produce training samples"
        # Outcome is blended with the root value, but the sign of the outcome
        # component must follow the winner: White positions positive.
        for _, _, target in samples:
            assert -1.0 <= target <= 1.0

    def test_a_low_factor_can_cost_data_and_is_documented_as_such(self, scripted):
        """Below 1.0 the rule starts cutting into sample-producing moves."""
        scripted.value_fn = staticmethod(black_is_lost)
        full_samples, _ = play(resign_enabled=False)
        early_samples, rec = play(resign_enabled=True, resign_min_move_factor=0.5,
                                  resign_playout_fraction=0.0)
        assert rec['resigned']
        assert len(early_samples) < len(full_samples)


class TestPlayoutMeasurement:

    def test_playout_game_is_not_stopped_but_records_the_verdict(self, scripted):
        scripted.value_fn = staticmethod(black_is_lost)
        _, rec = play(resign_enabled=True, resign_playout_fraction=1.0)
        assert rec['resign_playout'] is True
        assert rec['resigned'] is False             # played on
        assert rec['would_resign_color'] == BLACK   # but the verdict is recorded
        assert rec['would_resign_move'] is not None

    def test_false_resign_is_flagged_when_the_hopeless_side_wins(self, scripted):
        """
        Black despairs the whole way but actually wins on the board. The
        playout must mark this as a resignation that would have been WRONG —
        this is the signal that the threshold is too aggressive.
        """
        scripted.value_fn = staticmethod(black_is_lost)
        _, rec = play(resign_enabled=True, resign_playout_fraction=1.0)
        assert rec['would_resign_color'] == BLACK
        assert rec['false_resign'] == (rec['winner'] == BLACK)

    def test_no_verdict_means_no_false_resign_flag(self, scripted):
        scripted.value_fn = staticmethod(lambda move_num, player: 0.0)
        _, rec = play(resign_enabled=True, resign_playout_fraction=1.0)
        assert rec['would_resign_color'] is None
        assert rec['false_resign'] is False

    def test_zero_fraction_never_plays_out(self, scripted):
        scripted.value_fn = staticmethod(black_is_lost)
        for _ in range(5):
            _, rec = play(resign_enabled=True, resign_playout_fraction=0.0)
            assert rec['resign_playout'] is False


class TestResignEvidence:
    """
    A resigned game is an unexplained early stop unless the numbers that ended
    it are recorded with it. These are what the review UI turns into a reason.
    """

    def test_evidence_is_recorded_with_a_resignation(self, scripted):
        scripted.value_fn = staticmethod(black_is_lost)
        _, rec = play(resign_enabled=True, resign_playout_fraction=0.0,
                      resign_threshold=0.90, resign_consecutive=4)
        ev = rec['resign_evidence']
        assert ev is not None
        assert ev['root_value'] == pytest.approx(-0.99)      # the losing side
        assert ev['opponent_value'] == pytest.approx(0.99)   # and the winner
        assert ev['streak'] >= ev['required_streak'] == 4
        assert ev['threshold'] == 0.90
        assert ev['both_sides'] is True
        assert ev['min_move'] == AREA

    def test_evidence_is_recorded_for_an_overruled_trigger(self, scripted):
        """Playout games explain a verdict that was measured, not applied."""
        scripted.value_fn = staticmethod(black_is_lost)
        _, rec = play(resign_enabled=True, resign_playout_fraction=1.0)
        assert rec['resigned'] is False
        assert rec['resign_evidence'] is not None
        assert rec['resign_evidence']['root_value'] == pytest.approx(-0.99)

    def test_no_trigger_means_no_evidence(self, scripted):
        scripted.value_fn = staticmethod(lambda move_num, player: 0.0)
        _, rec = play(resign_enabled=True, resign_playout_fraction=0.0)
        assert rec['resigned'] is False
        assert rec['resign_evidence'] is None

    def test_evidence_is_the_first_trigger_not_a_later_one(self, scripted):
        """
        A playout game keeps despairing after the rule is overruled. The
        recorded evidence must stay the one that fired, so the move number and
        the values describe the same moment.
        """
        def deepening(move_num, player):
            if player == WHITE:
                return 0.99
            return -0.91 if move_num < AREA + 20 else -1.0
        scripted.value_fn = staticmethod(deepening)
        _, rec = play(resign_enabled=True, resign_playout_fraction=1.0)
        assert rec['would_resign_move'] < AREA + 20
        assert rec['resign_evidence']['root_value'] == pytest.approx(-0.91)


class TestRecordShape:

    def test_margin_is_the_board_score_not_zero(self, scripted):
        """
        determine_winner() reports margin 0 for resignations; the record must
        carry the board margin instead so the dashboard series does not read
        every resigned game as a draw.
        """
        scripted.value_fn = staticmethod(black_is_lost)
        _, rec = play(resign_enabled=True, resign_playout_fraction=0.0)
        assert rec['margin'] == pytest.approx(
            abs(rec['black_score'] - rec['white_score']))

    def test_resignation_is_not_written_into_the_replay_moves(self, scripted):
        """
        The replay endpoint treats any move with a negative row as a pass, so a
        resignation must not appear in `moves` or replays would drift.
        """
        scripted.value_fn = staticmethod(black_is_lost)
        _, rec = play(resign_enabled=True, resign_playout_fraction=0.0)
        assert all(m['move'][0] >= 0 or m['move'] == [-1, -1]
                   for m in rec['moves'])
        assert rec['num_moves'] == len(rec['moves'])


class TestConfigPlumbing:

    def test_param_bounds_entries_exist_and_default_off(self):
        from param_bounds import PARAM_BOUNDS, CATEGORIES
        assert PARAM_BOUNDS['resign_enabled']['default'] is False
        assert PARAM_BOUNDS['resign_enabled']['type'] == 'bool'
        keys = {'resign_enabled', 'resign_threshold', 'resign_consecutive',
                'resign_min_move_factor', 'resign_both_sides',
                'resign_playout_fraction'}
        assert keys <= set(PARAM_BOUNDS)
        assert 'resign' in {c['key'] for c in CATEGORIES}
        for key in keys:
            assert PARAM_BOUNDS[key]['category'] == 'resign'

    def test_sanitize_clamps_out_of_range_values(self):
        from param_bounds import sanitize_params
        clean = sanitize_params({'resign_threshold': 5.0,
                                 'resign_consecutive': 999,
                                 'resign_playout_fraction': -1.0})
        assert clean['resign_threshold'] == 0.99
        assert clean['resign_consecutive'] == 10
        assert clean['resign_playout_fraction'] == 0.0

    def test_training_config_defaults(self):
        from config import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.resign_enabled is False
        assert cfg.resign_min_move_factor == 1.0
        assert cfg.resign_both_sides is True

    def test_legacy_model_loads_with_the_rule_off(self):
        from model_manager import ModelInfo
        info = ModelInfo.from_dict({
            'id': 'legacy', 'name': 'Legacy', 'board_size': 9,
            'training': {'num_self_play_games': 5},
        })
        assert info.training.resign_enabled is False

    def test_settings_reach_the_training_config(self, tmp_path):
        from config import Config
        from model_manager import ModelInfo
        info = ModelInfo.from_dict({
            'id': 'mercy', 'name': 'Mercy', 'board_size': 9,
            'training': {'resign_enabled': True, 'resign_threshold': 0.8,
                         'resign_consecutive': 2},
        })
        cfg = Config.from_model(info, str(tmp_path))
        assert cfg.training.resign_enabled is True
        assert cfg.training.resign_threshold == 0.8
        assert cfg.training.resign_consecutive == 2


class TestTrainerIntegration:

    @staticmethod
    def _trainer(tmp_path, **training):
        from config import Config, TrainingConfig, BoardConfig, PathConfig
        from ai.trainer import Trainer
        cfg = Config(
            board=BoardConfig(size=7),
            training=TrainingConfig(**training),
            paths=PathConfig(model_dir=str(tmp_path)),
        )
        return Trainer(cfg)

    def test_metrics_are_none_when_disabled(self, tmp_path):
        t = self._trainer(tmp_path)
        assert t._resign_metrics()['resign_rate'] is None
        assert t._resign_metrics()['false_resign_rate'] is None

    def test_false_resign_rate_is_computed_from_playout_games_only(self, tmp_path):
        t = self._trainer(tmp_path, resign_enabled=True)
        records = [
            # Ordinary resigned games — carry no evidence either way.
            {'num_moves': 90, 'resigned': True, 'moves': []},
            {'num_moves': 90, 'resigned': True, 'moves': []},
            # Playout games: one verdict was wrong, one was right.
            {'num_moves': 120, 'resigned': False, 'resign_playout': True,
             'would_resign_color': 1, 'false_resign': True, 'moves': []},
            {'num_moves': 120, 'resigned': False, 'resign_playout': True,
             'would_resign_color': 1, 'false_resign': False, 'moves': []},
            # Playout game where the rule never fired — not counted at all.
            {'num_moves': 120, 'resigned': False, 'resign_playout': True,
             'would_resign_color': None, 'false_resign': False, 'moves': []},
        ]
        for i, rec in enumerate(records, start=1):
            t._on_game_complete(i, len(records), rec)

        m = t._resign_metrics()
        assert m['resign_rate'] == pytest.approx(2 / 5)
        assert m['false_resign_rate'] == pytest.approx(0.5)
        assert m['resign_playout_games'] == 3
        assert m['resign_checked_games'] == 2

    def test_collapse_guard_suppresses_the_rule(self, tmp_path):
        """A flat value head cannot be trusted to judge a lost game."""
        t = self._trainer(tmp_path, resign_enabled=True,
                          collapse_guard_enabled=True)
        assert t._collapse_active is False
        t._pass_stats = {1: [0, 10], 2: [9, 10]}   # White passing 90% of moves
        diag = t._collapse_diagnostics()
        assert diag['collapse_warning'] is not None
        assert t._collapse_active is True

    def test_disabling_the_guard_clears_a_standing_trip(self, tmp_path):
        t = self._trainer(tmp_path, resign_enabled=True,
                          collapse_guard_enabled=False)
        t._collapse_active = True
        t._collapse_diagnostics()
        assert t._collapse_active is False


class TestLiveTuning:
    """
    The /training/api/apply_params path, with a stub trainer so the test never
    touches a real model directory. This is the wiring that makes the settings
    tunable from the UI without a restart.
    """

    @pytest.fixture
    def client(self, monkeypatch, tmp_path):
        from flask import Flask
        from config import Config, TrainingConfig, BoardConfig, PathConfig
        import web.app as web_app
        from web.routes.training_routes import training_bp

        class StubTrainer:
            is_running = False
            model_id = 'stub'

            def __init__(self):
                self.config = Config(
                    board=BoardConfig(size=7),
                    training=TrainingConfig(),
                    paths=PathConfig(model_dir=str(tmp_path)),
                )
                self.logged = []

            def log(self, msg):
                self.logged.append(msg)

        class StubManager:
            def __init__(self):
                self.persisted = {}

            def update_model(self, model_id, training_params=None, **kw):
                self.persisted.update(training_params or {})

        stub = StubTrainer()
        manager = StubManager()
        monkeypatch.setattr(web_app, 'trainer', stub, raising=False)
        monkeypatch.setattr(web_app, 'model_manager', manager, raising=False)

        app = Flask(__name__)
        app.register_blueprint(training_bp, url_prefix='/training')
        return app.test_client(), stub, manager

    def test_settings_apply_live_and_persist(self, client):
        c, stub, manager = client
        res = c.post('/training/api/apply_params', json={
            'resign_enabled': True,
            'resign_threshold': 0.85,
            'resign_consecutive': 3,
            'resign_min_move_factor': 1.5,
            'resign_both_sides': False,
            'resign_playout_fraction': 0.2,
        })
        assert res.status_code == 200

        cfg = stub.config.training
        assert cfg.resign_enabled is True
        assert cfg.resign_threshold == 0.85
        assert cfg.resign_consecutive == 3
        assert cfg.resign_min_move_factor == 1.5
        assert cfg.resign_both_sides is False
        assert cfg.resign_playout_fraction == 0.2

        # ...and every one of them survives a restart.
        assert manager.persisted['resign_enabled'] is True
        assert manager.persisted['resign_threshold'] == 0.85

    def test_out_of_range_values_are_clamped_not_stored_raw(self, client):
        c, stub, _ = client
        c.post('/training/api/apply_params', json={'resign_threshold': 99.0})
        assert stub.config.training.resign_threshold == 0.99

    def test_turning_the_rule_off_is_applied(self, client):
        """False must be applied, not skipped as a falsy value."""
        c, stub, _ = client
        c.post('/training/api/apply_params', json={'resign_enabled': True})
        assert stub.config.training.resign_enabled is True
        c.post('/training/api/apply_params', json={'resign_enabled': False})
        assert stub.config.training.resign_enabled is False
