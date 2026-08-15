"""
test_checkpoint_champion.py — the gated champion must survive every save path.

`Trainer.save_weights_now()` (the /training/api/save_weights route) used to
call save_weights() without champion_state_dict. The written file then had no
champion entry, and on the next restart _try_load_weights fell back to

    self.eval_network.load_state_dict(self.network.state_dict())

promoting whatever un-gated candidate happened to be in the training network to
champion — the network that then generates all self-play data. One click of
"save weights" defeated the promotion gate across a restart.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, PathConfig
from ai.trainer import Trainer


def _make_trainer(tmp_path):
    config = Config(paths=PathConfig(model_dir=str(tmp_path)))
    config.training.num_self_play_games = 1
    config.training.eval_games = 0
    return Trainer(config=config)


def _diverge_candidate(trainer):
    """Make the training network differ from the champion, as training does."""
    with torch.no_grad():
        for p in trainer.network.parameters():
            p.add_(1.0)


def _champion_matches(trainer, other_state):
    return all(
        torch.allclose(v.cpu(), other_state[k].cpu())
        for k, v in trainer.eval_network.state_dict().items()
    )


class TestManualSavePreservesChampion:
    def test_save_weights_now_writes_the_champion(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        champion_before = {k: v.clone() for k, v in trainer.eval_network.state_dict().items()}
        _diverge_candidate(trainer)

        trainer.save_weights_now()

        state = torch.load(trainer.config.paths.weights_path,
                           map_location="cpu", weights_only=False)
        assert 'champion_state_dict' in state, \
            "manual save dropped the champion from the checkpoint"
        for k, v in state['champion_state_dict'].items():
            assert torch.allclose(v, champion_before[k].cpu()), \
                "manual save stored the candidate as the champion"

    def test_restart_after_manual_save_keeps_the_gated_champion(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        champion_before = {k: v.clone() for k, v in trainer.eval_network.state_dict().items()}
        _diverge_candidate(trainer)
        trainer.save_weights_now()

        # A restart against the same model directory.
        reloaded = _make_trainer(tmp_path)

        assert _champion_matches(reloaded, champion_before), \
            "restart promoted the un-gated candidate to champion"

    def test_automatic_save_also_preserves_it(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        champion_before = {k: v.clone() for k, v in trainer.eval_network.state_dict().items()}
        _diverge_candidate(trainer)

        trainer._save_weights()
        reloaded = _make_trainer(tmp_path)

        assert _champion_matches(reloaded, champion_before)
