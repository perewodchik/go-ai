"""
test_replay_persistence.py — the replay buffer must survive a restart.

Trainer.__init__ always built an empty ReplayBuffer, and _try_load_weights
restored the network, the optimizer, the champion, the iteration and the Elo —
but not the samples any of that was produced from. Every server restart, and
every switch_model round trip (which constructs a fresh Trainer), silently
discarded up to replay_buffer_size samples. The iteration that followed then
trained on a few hundred samples and its candidate reliably lost the gate,
which surfaced in the UI as "training has stalled".
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, PathConfig
from ai.replay_store import buffer_path, load_buffer, save_buffer
from ai.trainer import Trainer


def _samples(n, board_size=9, planes=10, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        state = torch.from_numpy(
            rng.random((planes, board_size, board_size)).astype(np.float32))
        policy = rng.random(board_size * board_size + 1).astype(np.float32)
        policy /= policy.sum()
        out.append((state, policy, float(rng.uniform(-1, 1))))
    return out


class TestStoreRoundTrip:
    def test_round_trip_preserves_every_sample(self, tmp_path):
        path = str(tmp_path / "replay_buffer.pt")
        original = _samples(25)
        save_buffer(original, path, board_size=9, num_input_planes=10)

        loaded, reason = load_buffer(path, board_size=9, num_input_planes=10)

        assert reason is None
        assert len(loaded) == len(original)
        for (s0, p0, v0), (s1, p1, v1) in zip(original, loaded):
            assert torch.allclose(s0, s1)
            assert np.allclose(p0, p1)
            assert abs(v0 - v1) < 1e-6

    def test_missing_file_is_not_an_error(self, tmp_path):
        loaded, reason = load_buffer(str(tmp_path / "nope.pt"), 9, 10)
        assert loaded == []
        assert reason is None

    def test_board_size_mismatch_is_refused_with_a_reason(self, tmp_path):
        path = str(tmp_path / "replay_buffer.pt")
        save_buffer(_samples(5, board_size=9), path,
                    board_size=9, num_input_planes=10)

        loaded, reason = load_buffer(path, board_size=13, num_input_planes=10)

        assert loaded == []
        assert reason and "13" in reason

    def test_plane_count_mismatch_is_refused(self, tmp_path):
        path = str(tmp_path / "replay_buffer.pt")
        save_buffer(_samples(5), path, board_size=9, num_input_planes=10)

        loaded, reason = load_buffer(path, board_size=9, num_input_planes=18)

        assert loaded == []
        assert reason is not None

    def test_cap_keeps_the_most_recent(self, tmp_path):
        path = str(tmp_path / "replay_buffer.pt")
        original = _samples(30)
        written = save_buffer(original, path, board_size=9,
                              num_input_planes=10, max_samples=10)

        assert written == 10
        loaded, _ = load_buffer(path, 9, 10)
        assert len(loaded) == 10
        # FIFO semantics: the newest samples are the ones worth keeping.
        assert torch.allclose(loaded[0][0], original[-10][0])

    def test_binary_planes_are_compacted_losslessly(self, tmp_path):
        """
        The file is rewritten every iteration, so size matters: a full 50k
        buffer is ~179 MB as float32 and ~57 MB stored as indicator planes.
        The saving is only allowed to be taken when it costs nothing.
        """
        rng = np.random.default_rng(1)
        binary = [
            (torch.from_numpy((rng.random((10, 9, 9)) > 0.5).astype(np.float32)),
             np.full(82, 1 / 82, dtype=np.float32), 0.25)
            for _ in range(50)
        ]
        path = str(tmp_path / "binary.pt")
        save_buffer(binary, path, 9, 10)

        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload['states'].dtype == torch.uint8
        assert payload['states_are_binary'] is True

        loaded, _ = load_buffer(path, 9, 10)
        assert loaded[0][0].dtype == torch.float32
        for (s0, _, _), (s1, _, _) in zip(binary, loaded):
            assert torch.equal(s0, s1)

    def test_continuous_planes_keep_full_precision(self, tmp_path):
        """A future feature set with non-indicator planes must not be rounded."""
        rng = np.random.default_rng(2)
        cont = [
            (torch.from_numpy(rng.random((10, 9, 9)).astype(np.float32)),
             np.full(82, 1 / 82, dtype=np.float32), 0.25)
            for _ in range(20)
        ]
        path = str(tmp_path / "cont.pt")
        save_buffer(cont, path, 9, 10)

        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload['states'].dtype == torch.float32
        assert payload['states_are_binary'] is False

        loaded, _ = load_buffer(path, 9, 10)
        for (s0, _, _), (s1, _, _) in zip(cont, loaded):
            assert torch.equal(s0, s1)

    def test_corrupt_file_is_reported_not_raised(self, tmp_path):
        path = str(tmp_path / "replay_buffer.pt")
        with open(path, "wb") as f:
            f.write(b"not a torch archive")

        loaded, reason = load_buffer(path, 9, 10)

        assert loaded == []
        assert reason is not None


class TestTrainerRestart:
    def _trainer(self, tmp_path):
        config = Config(paths=PathConfig(model_dir=str(tmp_path)))
        config.training.num_self_play_games = 1
        config.training.eval_games = 0
        return Trainer(config=config)

    def test_buffer_survives_a_restart(self, tmp_path):
        trainer = self._trainer(tmp_path)
        trainer.replay_buffer.add(_samples(40))
        trainer._save_weights()

        reloaded = self._trainer(tmp_path)

        assert len(reloaded.replay_buffer) == 40, \
            "restart discarded the replay buffer"

    def test_restored_samples_are_trainable(self, tmp_path):
        """A restored buffer has to survive sampling and augmentation."""
        trainer = self._trainer(tmp_path)
        trainer.replay_buffer.add(_samples(40))
        trainer._save_weights()

        reloaded = self._trainer(tmp_path)
        states, policies, values = reloaded.replay_buffer.sample(16)

        assert states.shape == (16, 10, 9, 9)
        assert policies.shape == (16, 82)
        assert values.shape == (16,)
        assert torch.isfinite(states).all()
        assert torch.isfinite(policies).all()

    def test_buffer_from_another_board_size_is_ignored(self, tmp_path):
        trainer = self._trainer(tmp_path)
        trainer.replay_buffer.add(_samples(10))
        trainer._save_weights()

        config = Config(paths=PathConfig(model_dir=str(tmp_path)))
        config.board.size = 13
        config.training.eval_games = 0
        # A fresh 13x13 model in a directory holding a 9x9 buffer must start
        # empty rather than crash or mix encodings.
        reloaded = Trainer(config=config)

        assert len(reloaded.replay_buffer) == 0

    def test_buffer_is_capped_at_the_configured_size(self, tmp_path):
        config = Config(paths=PathConfig(model_dir=str(tmp_path)))
        config.training.replay_buffer_size = 20
        config.training.eval_games = 0
        trainer = Trainer(config=config)
        trainer.replay_buffer.add(_samples(50))
        trainer._save_weights()

        loaded, _ = load_buffer(buffer_path(str(tmp_path)), 9, 10)
        assert len(loaded) == 20
