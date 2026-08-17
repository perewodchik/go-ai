"""
test_feature_migration.py — widening a trained model onto a wider encoding.

The whole claim of `ai/feature_migration.py` is that zero-initialising the new
input channels leaves the network computing the IDENTICAL function, so a
48-iteration model can gain two planes without losing its strength. That is an
algebraic argument in the module docstring; these tests check it numerically,
on real positions, rather than taking it on trust.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from ai.checkpoint import load_weights, save_weights
from ai.feature_migration import (
    BACKUP_SUFFIX,
    migrate_checkpoint,
    migrate_model,
    widen_input_conv,
)
from ai.network import GoNetwork
from game.features import encode_state
from game.game_state import GameState


def _trained_net(features="v1_10", seed=0):
    """A network with non-trivial weights, standing in for a trained model."""
    torch.manual_seed(seed)
    net = GoNetwork(board_size=9, input_features=features,
                    num_res_blocks=2, num_filters=16, value_head_hidden=16)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    net.eval()
    return net


def _positions(n=6):
    import random
    out = []
    rng = random.Random(0)
    state = GameState(board_size=9, komi=6.5)
    for i in range(n * 6):
        legal = state.get_legal_moves()
        if not legal:
            break
        state.play_move(*rng.choice(legal))
        if i % 6 == 0:
            out.append(state.copy())
    return out


class TestWidenIsExact:
    def test_the_widened_network_is_output_identical(self):
        """The claim, checked numerically on real positions."""
        old = _trained_net("v1_10")
        new = GoNetwork(board_size=9, input_features="v2_12",
                        num_res_blocks=2, num_filters=16, value_head_hidden=16)
        new.load_state_dict(widen_input_conv(old.state_dict(), 10, 12))
        new.eval()

        for state in _positions():
            p_old, v_old = old.predict(encode_state(state, "v1_10"), "cpu")
            p_new, v_new = new.predict(encode_state(state, "v2_12"), "cpu")
            assert torch.allclose(p_old, p_new, atol=1e-6), \
                "policy changed after widening"
            assert abs(v_old - v_new) < 1e-6, "value changed after widening"

    def test_identity_holds_whatever_the_new_planes_contain(self):
        """
        Zeroed weights mean the new planes cannot influence the output at all —
        not merely that they happen to be empty in these positions.
        """
        old = _trained_net("v1_10", seed=2)
        new = GoNetwork(board_size=9, input_features="v2_12",
                        num_res_blocks=2, num_filters=16, value_head_hidden=16)
        new.load_state_dict(widen_input_conv(old.state_dict(), 10, 12))
        new.eval()

        state = _positions()[3]
        base = encode_state(state, "v2_12")
        p_ref, v_ref = new.predict(base, "cpu")

        scrambled = base.clone()
        scrambled[10] = torch.rand(9, 9)
        scrambled[11] = torch.rand(9, 9)
        p_alt, v_alt = new.predict(scrambled, "cpu")

        assert torch.allclose(p_ref, p_alt, atol=1e-6)
        assert abs(v_ref - v_alt) < 1e-6

    def test_only_the_input_convolution_changes(self):
        old = _trained_net("v1_10", seed=3).state_dict()
        widened = widen_input_conv(old, 10, 12)
        assert set(widened) == set(old)
        for key, value in old.items():
            if key == "input_conv.weight":
                assert widened[key].shape[1] == 12
                assert torch.equal(widened[key][:, :10], value)
                assert widened[key][:, 10:].abs().sum() == 0
            else:
                assert torch.equal(widened[key], value), f"{key} was modified"


class TestGuards:
    def test_refuses_to_narrow(self):
        sd = _trained_net("v2_12").state_dict()
        with pytest.raises(ValueError, match="narrow"):
            widen_input_conv(sd, 12, 10)

    def test_refuses_a_checkpoint_that_contradicts_its_config(self):
        """Catches migrating twice, or a config that has drifted."""
        sd = _trained_net("v2_12").state_dict()
        with pytest.raises(ValueError, match="input channels"):
            widen_input_conv(sd, 10, 12)

    def test_same_width_is_a_no_op(self):
        sd = _trained_net("v1_10").state_dict()
        assert torch.equal(widen_input_conv(sd, 10, 10)["input_conv.weight"],
                           sd["input_conv.weight"])


class TestCheckpointPayload:
    def _payload(self):
        net = _trained_net("v1_10", seed=5)
        champion = _trained_net("v1_10", seed=6)
        return {
            'iteration': 48,
            'model_state_dict': net.state_dict(),
            'champion_state_dict': champion.state_dict(),
            'optimizer_state_dict': {'state': {}, 'param_groups': []},
            'elo': 778.3,
            'gate_elo': 640.0,
            'total_games': 680,
            'arch': net.arch_signature(),
        }

    def test_champion_is_widened_too(self):
        """It generates all self-play; leaving it behind breaks the next run."""
        migrated = migrate_checkpoint(self._payload(), "v1_10", "v2_12")
        assert migrated['champion_state_dict']['input_conv.weight'].shape[1] == 12

    def test_optimizer_state_is_dropped(self):
        migrated = migrate_checkpoint(self._payload(), "v1_10", "v2_12")
        assert 'optimizer_state_dict' not in migrated

    def test_training_progress_is_preserved(self):
        migrated = migrate_checkpoint(self._payload(), "v1_10", "v2_12")
        assert migrated['iteration'] == 48
        assert migrated['elo'] == 778.3
        assert migrated['gate_elo'] == 640.0
        assert migrated['total_games'] == 680

    def test_arch_tag_is_retagged_so_the_checkpoint_loads(self):
        migrated = migrate_checkpoint(self._payload(), "v1_10", "v2_12")
        assert migrated['arch']['num_input_planes'] == 12


class TestMigrateModelDirectory:
    def _make_model(self, tmp_path, features="v1_10"):
        model_dir = str(tmp_path)
        net = _trained_net(features, seed=7)
        champion = _trained_net(features, seed=8)
        optimizer = torch.optim.Adam(net.parameters())
        save_weights(model=net, optimizer=optimizer, iteration=12, elo=700.0,
                     kyu_rank="23k", total_games=240,
                     weights_path=os.path.join(model_dir, "weights.pt"),
                     champion_state_dict=champion.state_dict(),
                     gate_elo=612.0)
        with open(os.path.join(model_dir, "config.json"), "w") as fh:
            json.dump({"id": "t", "name": "t", "board_size": 9,
                       "network": {"size_preset": "small", "num_res_blocks": 2,
                                   "num_filters": 16, "value_head_hidden": 16,
                                   "input_features": features}}, fh)
        return model_dir, net

    def test_end_to_end_migration_preserves_play(self, tmp_path):
        model_dir, original = self._make_model(tmp_path)

        result = migrate_model(model_dir, "v2_12")
        assert result['migrated'] is True
        assert result['from_planes'] == 10 and result['to_planes'] == 12
        assert result['optimizer_state_dropped'] is True

        with open(os.path.join(model_dir, "config.json")) as fh:
            assert json.load(fh)["network"]["input_features"] == "v2_12"

        # The migrated checkpoint must load into a 12-plane network...
        reloaded = GoNetwork(board_size=9, input_features="v2_12",
                            num_res_blocks=2, num_filters=16,
                            value_head_hidden=16)
        meta = load_weights(os.path.join(model_dir, "weights.pt"), reloaded)
        reloaded.eval()
        assert meta['iteration'] == 12
        assert meta['gate_elo'] == 612.0

        # ...and play exactly as it did before.
        for state in _positions(4):
            p_old, v_old = original.predict(encode_state(state, "v1_10"), "cpu")
            p_new, v_new = reloaded.predict(encode_state(state, "v2_12"), "cpu")
            assert torch.allclose(p_old, p_new, atol=1e-6)
            assert abs(v_old - v_new) < 1e-6

    def test_a_backup_is_written(self, tmp_path):
        model_dir, _ = self._make_model(tmp_path)
        result = migrate_model(model_dir, "v2_12")
        assert os.path.isfile(result['backup_path'])
        assert result['backup_path'].endswith(BACKUP_SUFFIX)

    def test_migrating_twice_is_refused_not_corrupting(self, tmp_path):
        model_dir, _ = self._make_model(tmp_path)
        migrate_model(model_dir, "v2_12")
        again = migrate_model(model_dir, "v2_12")
        assert again['migrated'] is False

    def test_replay_buffer_is_reported_as_invalidated(self, tmp_path):
        model_dir, _ = self._make_model(tmp_path)
        from ai.replay_store import buffer_path, save_buffer
        import numpy as np
        save_buffer([(torch.zeros(10, 9, 9),
                      np.full(82, 1 / 82, dtype=np.float32), 0.0)],
                    buffer_path(model_dir), board_size=9, num_input_planes=10)

        result = migrate_model(model_dir, "v2_12")
        assert result['replay_buffer_invalidated'] is True

        # And the guard must actually refuse it, not silently feed 10 planes.
        from ai.replay_store import load_buffer
        samples, reason = load_buffer(buffer_path(model_dir), 9, 12)
        assert samples == []
        assert reason is not None

    def test_unknown_target_is_rejected(self, tmp_path):
        model_dir, _ = self._make_model(tmp_path)
        with pytest.raises(ValueError, match="Unknown feature set"):
            migrate_model(model_dir, "v9_nonsense")


class TestTrainerRoundTrip:
    def test_a_migrated_model_starts_and_trains(self, tmp_path):
        """The real integration: build a v1 model, migrate, run an iteration."""
        from ai.trainer import Trainer
        from config import Config, PathConfig
        from model_manager import ModelInfo, NetworkParams, TrainingParams

        model_dir = str(tmp_path)
        config = Config(paths=PathConfig(model_dir=model_dir))
        config.board.size = 7
        config.training.eval_games = 0
        config.training.gate_enabled = False
        config.training.num_self_play_games = 1
        config.mcts.num_simulations = 8
        config.mcts.auto_scale_simulations = False

        trainer = Trainer(config=config)
        trainer._save_weights()
        with open(os.path.join(model_dir, "config.json"), "w") as fh:
            json.dump({"network": {"input_features": "v1_10"}}, fh)

        result = migrate_model(model_dir, "v2_12")
        assert result['weights_migrated'] is True

        v2_config = Config(paths=PathConfig(model_dir=model_dir))
        v2_config.board.size = 7
        v2_config.network.input_features = "v2_12"
        v2_config.network.num_input_planes = 12
        v2_config.training.eval_games = 0
        v2_config.training.gate_enabled = False
        v2_config.training.num_self_play_games = 1
        v2_config.mcts.num_simulations = 8
        v2_config.mcts.auto_scale_simulations = False

        warnings = []
        migrated_trainer = Trainer(
            config=v2_config,
            progress_callback=lambda e: warnings.append(e.get('message', '')))
        assert migrated_trainer.network.input_conv.in_channels == 12
        assert not any('Failed to load weights' in w for w in warnings), warnings

        migrated_trainer.train(max_iterations=1)
        assert len(migrated_trainer.replay_buffer) > 0
        states, _, _ = migrated_trainer.replay_buffer.sample(4)
        assert states.shape[1] == 12
