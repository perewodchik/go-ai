"""
test_network_size.py — Tests for per-model configurable network size.

Covers:
- Preset resolution and custom-arch clamping
- Actual parameter counts of built networks (incl. the 1M+ "medium" preset)
- NetworkParams round-trip through ModelInfo (and backward compat with old
  configs that predate the `network` field)
- create_model persisting network arch
- Config.from_model threading the arch into NetworkConfig
- Checkpoint arch tagging: round-trip load, and a clean refusal on mismatch
"""

import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as config_mod
from config import (
    NETWORK_PRESETS, NETWORK_PRESET_ORDER, DEFAULT_NETWORK_PRESET,
    resolve_network_preset, clamp_network_arch, Config,
)
from ai.network import GoNetwork
from ai.checkpoint import save_weights, load_weights
import model_manager
from model_manager import ModelManager, ModelInfo, NetworkParams


# --------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------
class TestPresets:
    def test_default_preset_is_small_4x64(self):
        arch = resolve_network_preset(DEFAULT_NETWORK_PRESET)
        assert arch == {"num_res_blocks": 4, "num_filters": 64, "value_head_hidden": 64}

    def test_all_presets_ordered_and_monotonic(self):
        assert set(NETWORK_PRESET_ORDER) == set(NETWORK_PRESETS.keys())
        counts = [NETWORK_PRESETS[k]["approx_params"] for k in NETWORK_PRESET_ORDER]
        assert counts == sorted(counts), "presets should be ordered small→large"

    def test_unknown_preset_falls_back_to_default(self):
        assert resolve_network_preset("does-not-exist") == \
            resolve_network_preset(DEFAULT_NETWORK_PRESET)

    def test_clamp_bounds(self):
        arch = clamp_network_arch(num_res_blocks=999, num_filters=0, value_head_hidden=99999)
        rb_lo, rb_hi = config_mod.NETWORK_ARCH_BOUNDS["num_res_blocks"]
        nf_lo, nf_hi = config_mod.NETWORK_ARCH_BOUNDS["num_filters"]
        vh_lo, vh_hi = config_mod.NETWORK_ARCH_BOUNDS["value_head_hidden"]
        assert arch["num_res_blocks"] == rb_hi
        assert arch["num_filters"] == nf_lo
        assert arch["value_head_hidden"] == vh_hi


# --------------------------------------------------------------------------
# Actual parameter counts
# --------------------------------------------------------------------------
class TestParamCounts:
    def _build(self, preset_key, board_size=9):
        arch = resolve_network_preset(preset_key)
        return GoNetwork(board_size=board_size, num_input_planes=10, **arch)

    def test_small_matches_historical_default(self):
        # The historical default network — must stay ~320k so old weights load.
        net = self._build("small")
        assert 300_000 < net.count_parameters() < 340_000

    def test_medium_crosses_one_million(self):
        net = self._build("medium")
        assert net.count_parameters() > 1_000_000

    def test_preset_counts_are_monotonic_on_9x9(self):
        counts = [self._build(k).count_parameters() for k in NETWORK_PRESET_ORDER]
        assert counts == sorted(counts)

    def test_arch_signature_reflects_build(self):
        net = self._build("medium")
        sig = net.arch_signature()
        assert sig["num_res_blocks"] == 6
        assert sig["num_filters"] == 96
        assert sig["num_input_planes"] == 10
        assert sig["board_size"] == 9

    def test_bigger_network_still_forward_passes(self):
        net = self._build("medium")
        x = torch.randn(2, 10, 9, 9)
        policy, value = net(x)
        assert policy.shape == (2, 9 * 9 + 1)
        assert value.shape == (2, 1)


# --------------------------------------------------------------------------
# ModelInfo round-trip + backward compat
# --------------------------------------------------------------------------
class TestModelInfoRoundTrip:
    def test_network_params_survive_to_from_dict(self):
        info = ModelInfo(
            id="x", name="X",
            network=NetworkParams(size_preset="medium", num_res_blocks=6,
                                  num_filters=96, value_head_hidden=96),
        )
        d = info.to_dict()
        assert d["network"]["size_preset"] == "medium"
        restored = ModelInfo.from_dict(d)
        assert restored.network.num_res_blocks == 6
        assert restored.network.num_filters == 96
        assert restored.network.size_preset == "medium"

    def test_legacy_config_without_network_defaults_to_small(self):
        # Simulate a config.json written before per-model sizing existed.
        legacy = {
            "id": "old", "name": "Old", "board_size": 9,
            "training": {"num_simulations": 50},
            # no "network" key
        }
        info = ModelInfo.from_dict(legacy)
        assert info.network.num_res_blocks == 4
        assert info.network.num_filters == 64
        assert info.network.size_preset == "small"


# --------------------------------------------------------------------------
# ModelManager persistence
# --------------------------------------------------------------------------
@pytest.fixture
def temp_models_root(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(model_manager, "MODELS_ROOT", str(root))
    monkeypatch.setattr(model_manager, "ACTIVE_FILE", str(root / ".active"))
    return root


class TestManagerPersistsNetwork:
    def test_create_and_reload_stores_arch(self, temp_models_root):
        mgr = ModelManager()
        info = mgr.create_model(
            name="Big One",
            network_params={"size_preset": "medium", "num_res_blocks": 6,
                            "num_filters": 96, "value_head_hidden": 96},
        )
        reloaded = mgr.get_model(info.id)
        assert reloaded.network.size_preset == "medium"
        assert reloaded.network.num_res_blocks == 6
        assert reloaded.network.num_filters == 96

    def test_default_create_is_small(self, temp_models_root):
        mgr = ModelManager()
        info = mgr.create_model(name="Default")
        reloaded = mgr.get_model(info.id)
        assert reloaded.network.size_preset == "small"
        assert reloaded.network.num_filters == 64


# --------------------------------------------------------------------------
# Config.from_model threading
# --------------------------------------------------------------------------
class TestFromModelThreadsArch:
    def test_from_model_uses_stored_network(self, tmp_path):
        info = ModelInfo(
            id="m", name="M", board_size=9,
            network=NetworkParams(size_preset="medium", num_res_blocks=6,
                                  num_filters=96, value_head_hidden=96),
        )
        cfg = Config.from_model(info, str(tmp_path))
        assert cfg.network.num_res_blocks == 6
        assert cfg.network.num_filters == 96
        assert cfg.network.value_head_hidden == 96

    def test_built_network_matches_from_model_config(self, tmp_path):
        info = ModelInfo(
            id="m", name="M", board_size=9,
            network=NetworkParams(size_preset="medium", num_res_blocks=6,
                                  num_filters=96, value_head_hidden=96),
        )
        cfg = Config.from_model(info, str(tmp_path))
        net = GoNetwork(
            board_size=cfg.board.size,
            num_input_planes=cfg.network.num_input_planes,
            num_res_blocks=cfg.network.num_res_blocks,
            num_filters=cfg.network.num_filters,
            value_head_hidden=cfg.network.value_head_hidden,
        )
        assert net.count_parameters() > 1_000_000


# --------------------------------------------------------------------------
# Checkpoint arch safety
# --------------------------------------------------------------------------
class TestCheckpointArch:
    def _save(self, net, path):
        opt = torch.optim.Adam(net.parameters())
        save_weights(net, opt, iteration=1, elo=600, kyu_rank="29k",
                     total_games=10, weights_path=path)

    def test_roundtrip_same_arch(self, tmp_path):
        path = str(tmp_path / "weights.pt")
        arch = resolve_network_preset("medium")
        net = GoNetwork(board_size=9, num_input_planes=10, **arch)
        self._save(net, path)

        net2 = GoNetwork(board_size=9, num_input_planes=10, **arch)
        meta = load_weights(path, net2)
        assert meta["iteration"] == 1
        assert meta["arch"]["num_filters"] == 96

    def test_mismatched_arch_refuses_to_load(self, tmp_path):
        path = str(tmp_path / "weights.pt")
        big = GoNetwork(board_size=9, num_input_planes=10,
                        **resolve_network_preset("medium"))
        self._save(big, path)

        small = GoNetwork(board_size=9, num_input_planes=10,
                          **resolve_network_preset("small"))
        with pytest.raises(ValueError, match="architecture does not match"):
            load_weights(path, small)

    def test_legacy_checkpoint_without_arch_still_loads(self, tmp_path):
        # A weights file saved before arch tagging (no 'arch' key) must load
        # as long as shapes actually match.
        path = str(tmp_path / "weights.pt")
        net = GoNetwork(board_size=9, num_input_planes=10,
                        **resolve_network_preset("small"))
        state = {
            "iteration": 3,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": torch.optim.Adam(net.parameters()).state_dict(),
            "elo": 550, "kyu_rank": "29k", "total_games": 5,
            # no 'arch'
        }
        torch.save(state, path)

        net2 = GoNetwork(board_size=9, num_input_planes=10,
                         **resolve_network_preset("small"))
        meta = load_weights(path, net2)
        assert meta["iteration"] == 3
        assert meta["arch"] is None
