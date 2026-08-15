"""
model_manager.py — Manages Go AI model lifecycle.

A Model encapsulates everything for a specific training run:
  - Board configuration (size, komi, ruleset)
  - Training hyperparameters
  - Neural network weights (weights.pt)
  - Self-play game logs
  - Training metrics log

Models are stored under the `models/` directory. Each model gets its own
subdirectory named with a URL-safe slug derived from its display name.
"""

import os
import re
import json
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any


MODELS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
ACTIVE_FILE = os.path.join(MODELS_ROOT, ".active")


@dataclass
class TrainingParams:
    """Training hyperparameters stored per-model."""
    num_self_play_games: int = 5
    eval_games: int = 4
    num_simulations: int = 50
    c_puct: float = 1.5
    learning_rate: float = 0.002
    batch_size: int = 32
    num_epochs_per_iteration: int = 3
    replay_buffer_size: int = 50_000
    reflection_interval_games: int = 50
    temperature_threshold: int = 30
    temperature_init: float = 0.8
    temperature_final: float = 0.1
    # --- Champion vs candidate (promotion gate) ---
    # Defaults mirror config.TrainingConfig; models saved before these existed
    # load as None and fall back to the TrainingConfig defaults in
    # Config.from_model, so old configs keep behaving exactly as before.
    gate_enabled: bool = True
    gate_games: int = 20
    gate_threshold: float = 0.55
    gate_simulations: int = 50
    gate_stall_warning: int = 5
    # Worker processes shared by self-play, the gate, and the random-bot eval.
    num_parallel_workers: int = 4


@dataclass
class NetworkParams:
    """
    Neural-network architecture stored per-model.

    Frozen at creation time: changing it after a model has trained would make
    the saved weights.pt incompatible. Defaults match the historical "small"
    (4x64, ~320k params) network so models created before this field existed
    reload unchanged.
    """
    size_preset: str = "small"       # label only; concrete numbers below are authoritative
    num_res_blocks: int = 4
    num_filters: int = 64
    value_head_hidden: int = 64


@dataclass
class ModelInfo:
    """Complete description of a model and its current state."""
    id: str                              # URL-safe slug (directory name)
    name: str                            # Human-readable display name
    board_size: int = 9
    komi: float = 6.5
    ruleset: str = "chinese"             # "chinese" or "japanese"
    training: TrainingParams = field(default_factory=TrainingParams)
    network: NetworkParams = field(default_factory=NetworkParams)
    created_at: str = ""
    # Live state (updated by trainer, saved to config.json)
    elo: float = 500.0
    kyu_rank: str = "30k"
    iteration: int = 0
    total_games: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ModelInfo":
        training_data = data.pop("training", {})
        # `gate_workers` was briefly a gate-only setting before it became the
        # shared worker count for every game-playing phase. Carry the stored
        # value over instead of silently dropping it back to the default.
        if "gate_workers" in training_data and "num_parallel_workers" not in training_data:
            training_data["num_parallel_workers"] = training_data["gate_workers"]
        training = TrainingParams(**{
            k: v for k, v in training_data.items()
            if k in TrainingParams.__dataclass_fields__
        })
        # Network arch — absent in configs created before per-model sizing;
        # NetworkParams defaults (4x64 "small") then reproduce those models.
        network_data = data.pop("network", {}) or {}
        network = NetworkParams(**{
            k: v for k, v in network_data.items()
            if k in NetworkParams.__dataclass_fields__
        })
        # Filter to only known fields
        known = set(cls.__dataclass_fields__.keys()) - {"training", "network"}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(training=training, network=network, **filtered)


def _slugify(name: str) -> str:
    """Convert a display name to a URL-safe directory slug."""
    slug = name.lower().strip()
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    slug = slug.strip('-')
    return slug or "model"


def _unique_slug(base_slug: str) -> str:
    """Ensure the slug is unique among existing model directories."""
    if not os.path.exists(os.path.join(MODELS_ROOT, base_slug)):
        return base_slug
    counter = 2
    while os.path.exists(os.path.join(MODELS_ROOT, f"{base_slug}-{counter}")):
        counter += 1
    return f"{base_slug}-{counter}"


class ModelManager:
    """
    Manages creation, listing, selection, and lifecycle of models.

    All model data is stored under `models/<slug>/`.
    The active model ID is persisted in `models/.active`.
    """

    def __init__(self):
        os.makedirs(MODELS_ROOT, exist_ok=True)

    def list_models(self) -> List[ModelInfo]:
        """List all models, sorted by creation time (newest first)."""
        models = []
        if not os.path.exists(MODELS_ROOT):
            return models
        for entry in os.listdir(MODELS_ROOT):
            if entry.startswith("."):
                continue
            config_path = os.path.join(MODELS_ROOT, entry, "config.json")
            if os.path.isfile(config_path):
                try:
                    with open(config_path) as f:
                        data = json.load(f)
                    models.append(ModelInfo.from_dict(data))
                except (json.JSONDecodeError, TypeError, KeyError):
                    continue
        models.sort(key=lambda m: m.created_at or "", reverse=True)
        return models

    def create_model(
        self,
        name: str,
        board_size: int = 9,
        komi: float = 6.5,
        ruleset: str = "chinese",
        training_params: Optional[Dict[str, Any]] = None,
        network_params: Optional[Dict[str, Any]] = None,
    ) -> ModelInfo:
        """Create a new model with its directory structure and config."""
        slug = _unique_slug(_slugify(name))
        model_dir = os.path.join(MODELS_ROOT, slug)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(os.path.join(model_dir, "games"), exist_ok=True)
        os.makedirs(os.path.join(model_dir, "logs"), exist_ok=True)

        tp = TrainingParams()
        if training_params:
            for k, v in training_params.items():
                if hasattr(tp, k):
                    setattr(tp, k, v)

        np_ = NetworkParams()
        if network_params:
            for k, v in network_params.items():
                if hasattr(np_, k):
                    setattr(np_, k, v)

        info = ModelInfo(
            id=slug,
            name=name,
            board_size=board_size,
            komi=komi,
            ruleset=ruleset,
            training=tp,
            network=np_,
            created_at=datetime.now().isoformat(),
        )
        self._save_config(info)
        return info

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Load a model's info from its config.json."""
        config_path = os.path.join(MODELS_ROOT, model_id, "config.json")
        if not os.path.isfile(config_path):
            return None
        try:
            with open(config_path) as f:
                data = json.load(f)
            return ModelInfo.from_dict(data)
        except (json.JSONDecodeError, TypeError, KeyError):
            return None

    def update_model_state(self, model_id: str, elo: float, kyu_rank: str,
                           iteration: int, total_games: int) -> None:
        """Update the live training state in the model's config."""
        info = self.get_model(model_id)
        if not info:
            return
        info.elo = elo
        info.kyu_rank = kyu_rank
        info.iteration = iteration
        info.total_games = total_games
        self._save_config(info)

    def rename_model(self, model_id: str, new_name: str) -> Optional[ModelInfo]:
        """Rename a model (display name only, slug stays the same)."""
        info = self.get_model(model_id)
        if not info:
            return None
        info.name = new_name
        self._save_config(info)
        return info

    def update_model(
        self,
        model_id: str,
        name: Optional[str] = None,
        board_size: Optional[int] = None,
        komi: Optional[float] = None,
        ruleset: Optional[str] = None,
        training_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[ModelInfo]:
        """
        Update a model's configuration in place (slug/directory unchanged).

        Only the fields that are passed (non-None) are modified. Live training
        state (elo, iteration, total_games) is preserved. Note that changing
        board_size on an already-trained model makes its saved weights.pt
        incompatible; the trainer handles that gracefully by warning and
        starting fresh, but the caller should surface a warning to the user.
        """
        info = self.get_model(model_id)
        if not info:
            return None

        if name is not None:
            info.name = name
        if board_size is not None:
            info.board_size = board_size
        if komi is not None:
            info.komi = komi
        if ruleset is not None:
            info.ruleset = ruleset
        if training_params:
            for k, v in training_params.items():
                if hasattr(info.training, k):
                    setattr(info.training, k, v)

        self._save_config(info)
        return info

    def copy_model(self, model_id: str, new_name: str) -> Optional[ModelInfo]:
        """Deep-copy a model to a new name/directory."""
        src_dir = os.path.join(MODELS_ROOT, model_id)
        if not os.path.isdir(src_dir):
            return None
        new_slug = _unique_slug(_slugify(new_name))
        dst_dir = os.path.join(MODELS_ROOT, new_slug)
        shutil.copytree(src_dir, dst_dir)

        # Update config in the copy
        info = self.get_model(new_slug)
        if info:
            info.id = new_slug
            info.name = new_name
            info.created_at = datetime.now().isoformat()
            self._save_config(info)
        return info

    def delete_model(self, model_id: str) -> bool:
        """Delete a model and all its data."""
        model_dir = os.path.join(MODELS_ROOT, model_id)
        if not os.path.isdir(model_dir):
            return False
        shutil.rmtree(model_dir)
        # Clear active if this was the active model
        if self.get_active_model_id() == model_id:
            self._clear_active()
        return True

    def get_active_model_id(self) -> Optional[str]:
        """Get the currently active model ID."""
        if not os.path.isfile(ACTIVE_FILE):
            return None
        try:
            with open(ACTIVE_FILE) as f:
                model_id = f.read().strip()
            # Verify it still exists
            if model_id and os.path.isdir(os.path.join(MODELS_ROOT, model_id)):
                return model_id
        except IOError:
            pass
        return None

    def set_active_model_id(self, model_id: str) -> bool:
        """Set the active model. Returns False if model doesn't exist."""
        if not os.path.isdir(os.path.join(MODELS_ROOT, model_id)):
            return False
        with open(ACTIVE_FILE, "w") as f:
            f.write(model_id)
        return True

    def get_active_model(self) -> Optional[ModelInfo]:
        """Get the full ModelInfo for the active model."""
        mid = self.get_active_model_id()
        if mid:
            return self.get_model(mid)
        return None

    def get_model_dir(self, model_id: str) -> str:
        """Get the filesystem path for a model's directory."""
        return os.path.join(MODELS_ROOT, model_id)

    def _save_config(self, info: ModelInfo) -> None:
        """Write a model's config.json."""
        config_path = os.path.join(MODELS_ROOT, info.id, "config.json")
        with open(config_path, "w") as f:
            json.dump(info.to_dict(), f, indent=2)

    def _clear_active(self) -> None:
        if os.path.isfile(ACTIVE_FILE):
            os.remove(ACTIVE_FILE)
