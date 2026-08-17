"""
test_self_atari_plumbing.py — the setting has to survive params → config → phases.

A move restriction is only real if it reaches the worker processes. Self-play,
the promotion gate and the random-bot eval all run in separate processes and see
nothing but the dict handed to them, which is exactly how `temperature_init`
became a dead setting for every model ever trained (see
test_temperature_plumbing.py).

Also pins the deliberate asymmetry: BOTH sides of a gate match are filtered, or
the gate measures the filter instead of the networks; the random bot never is,
because it is the Elo anchor and strengthening it would make the Elo curve
incomparable across iterations.
"""

import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai import evaluator, self_play
from config import Config, PathConfig, TrainingConfig
from model_manager import TrainingParams
from param_bounds import PARAM_BOUNDS, sanitize_params


class TestDefaults:
    def test_off_everywhere_by_default(self):
        assert TrainingConfig().restrict_self_atari is False
        assert TrainingConfig().self_atari_max_stones == 1
        assert TrainingParams().restrict_self_atari is False
        assert TrainingParams().self_atari_max_stones == 1
        assert PARAM_BOUNDS['restrict_self_atari']['default'] is False
        assert PARAM_BOUNDS['self_atari_max_stones']['default'] == 1


class TestParamBounds:
    def test_sacrifice_size_is_clamped(self):
        assert sanitize_params({'self_atari_max_stones': 99})['self_atari_max_stones'] == 4
        assert sanitize_params({'self_atari_max_stones': 0})['self_atari_max_stones'] == 1

    def test_toggle_accepts_the_shapes_a_client_can_send(self):
        for raw, expected in ((True, True), (1, True), ("true", True),
                              (False, False), (0, False), ("false", False)):
            assert sanitize_params({'restrict_self_atari': raw})['restrict_self_atari'] is expected


class TestConfigFromModel:
    def _model(self, **overrides):
        from model_manager import ModelInfo
        params = TrainingParams(**overrides)
        return ModelInfo(id="t", name="t", training=params)

    def test_setting_reaches_the_training_config(self, tmp_path):
        info = self._model(restrict_self_atari=True, self_atari_max_stones=3)
        config = Config.from_model(info, str(tmp_path))
        assert config.training.restrict_self_atari is True
        assert config.training.self_atari_max_stones == 3

    def test_models_predating_the_setting_get_the_documented_default(self, tmp_path):
        """
        Old config.json files have no such key, so TrainingParams loads None
        there — it must fall back to TrainingConfig's default, not to a second
        hand-written copy of it.
        """
        info = self._model()
        info.training.restrict_self_atari = None
        info.training.self_atari_max_stones = None
        config = Config.from_model(info, str(tmp_path))
        assert config.training.restrict_self_atari is False
        assert config.training.self_atari_max_stones == 1


class TestReachesEveryPhase:
    def test_self_play_worker_accepts_it(self):
        params = inspect.signature(self_play.play_self_play_game).parameters
        assert 'restrict_self_atari' in params
        assert 'self_atari_max_stones' in params

    def test_task_dict_carries_it_across_the_process_boundary(self):
        from tests.test_temperature_plumbing import TASK_KWARGS
        kwargs = dict(TASK_KWARGS)
        kwargs.update(restrict_self_atari=True, self_atari_max_stones=2)
        task = self_play.build_self_play_task(**kwargs)
        assert task['restrict_self_atari'] is True
        assert task['self_atari_max_stones'] == 2

    def test_batch_accepts_it(self):
        params = inspect.signature(self_play.run_self_play_batch).parameters
        assert 'restrict_self_atari' in params
        assert 'self_atari_max_stones' in params

    def test_the_gate_accepts_it(self):
        params = inspect.signature(evaluator.evaluate_against_checkpoint).parameters
        assert 'restrict_self_atari' in params
        assert 'self_atari_max_stones' in params


class TestGateSymmetry:
    def test_both_sides_of_a_gate_match_are_filtered(self):
        """Otherwise the gate measures the filter, not the two networks."""
        source = inspect.getsource(evaluator._play_gate_game)
        assert source.count('restrict_self_atari=restrict_self_atari') == 2


class TestLiveTuning:
    def test_it_is_in_the_live_tune_list(self):
        source = inspect.getsource(
            __import__('web.routes.training_routes', fromlist=['apply_params']))
        assert "'restrict_self_atari'" in source
        assert "'self_atari_max_stones'" in source
