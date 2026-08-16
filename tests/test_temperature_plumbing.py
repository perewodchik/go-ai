"""
test_temperature_plumbing.py — configured settings must reach the workers.

`temperature_init` and `temperature_final` were a complete dead end: exposed in
param_bounds, stored in TrainingParams, read into config.mcts by
Config.from_model, live-tuned by /api/apply_params and persisted to
config.json — and then never passed to run_self_play_batch, which did not even
accept them. Self-play silently used play_self_play_game's own signature
defaults (1.0 -> 0.1) for every model ever trained.

Self-play games run in separate processes, so a worker only ever sees what is
in its task dict. These tests check the two links in that chain:

    Trainer.train -> run_self_play_batch -> task dict -> play_self_play_game
"""

import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai import self_play as sp
from ai.self_play import build_self_play_task, play_self_play_game


# Every argument the task builder takes. Adding a setting to self-play means
# adding it here too — which is the point: `test_every_key_is_a_real_parameter`
# then checks it against the worker's actual signature.
TASK_KWARGS = dict(
    network=None, board_size=9, komi=6.5, num_simulations=96,
    c_puct=1.5, temperature_threshold=30, temperature_init=0.8,
    temperature_final=0.2, device="cpu", fpu_reduction=0.35,
    value_target_outcome_weight=0.6, restrict_eye_fill=False,
    restrict_self_atari=False, self_atari_max_stones=1,
    resign_enabled=False, resign_threshold=0.9, resign_consecutive=4,
    resign_min_move_factor=1.0, resign_both_sides=True,
    resign_playout_fraction=0.1,
)


class TestTaskDict:
    """The dict that crosses the process boundary."""

    def test_carries_the_temperature_schedule(self):
        task = build_self_play_task(
            **dict(TASK_KWARGS)
        )
        assert task['temperature_init'] == 0.8
        assert task['temperature_final'] == 0.2
        assert task['temperature_threshold'] == 30

    def test_every_key_is_a_real_parameter(self):
        """
        Guards against the reverse failure: a key that no longer matches the
        worker signature would raise TypeError inside a subprocess, where the
        traceback is swallowed by `print(f"Worker failed: {e}")`.
        """
        task = build_self_play_task(
            **dict(TASK_KWARGS)
        )
        accepted = set(inspect.signature(play_self_play_game).parameters)
        unknown = set(task) - accepted
        assert not unknown, f"task dict carries keys the worker rejects: {unknown}"

    def test_batch_accepts_the_temperature_arguments(self):
        params = inspect.signature(sp.run_self_play_batch).parameters
        assert 'temperature_init' in params
        assert 'temperature_final' in params


class TestTrainerForwardsConfig:
    """Trainer.train must hand its own config values to the self-play batch."""

    def test_configured_temperatures_reach_the_batch(self, tmp_path, monkeypatch):
        from config import Config, PathConfig
        from ai.trainer import Trainer

        config = Config(paths=PathConfig(model_dir=str(tmp_path)))
        config.training.num_self_play_games = 1
        config.training.eval_games = 0
        config.training.gate_enabled = False
        config.mcts.temperature_init = 0.77
        config.mcts.temperature_final = 0.22
        config.mcts.temperature_threshold = 42

        captured = {}

        def fake_batch(**kwargs):
            captured.update(kwargs)
            return []

        monkeypatch.setattr('ai.trainer.run_self_play_batch', fake_batch)

        trainer = Trainer(config=config)
        trainer.train(max_iterations=1)

        assert captured, "run_self_play_batch was never called"
        assert captured['temperature_init'] == 0.77
        assert captured['temperature_final'] == 0.22
        assert captured['temperature_threshold'] == 42
