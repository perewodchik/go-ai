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

import elo_history
from elo_history import DEFAULT_ELO, EloEntry


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
    # tau for the policy TRAINING TARGET. The three settings above decide which
    # move is played; this one decides what the network is taught. 1.0 is
    # AlphaZero's pi = N/sum(N) and should not be lowered — see MCTS.search.
    policy_target_temperature: float = 1.0
    # --- Champion vs candidate (promotion gate) ---
    # Defaults mirror config.TrainingConfig; models saved before these existed
    # load as None and fall back to the TrainingConfig defaults in
    # Config.from_model, so old configs keep behaving exactly as before.
    gate_enabled: bool = True
    gate_games: int = 20
    gate_threshold: float = 0.55
    # Tracks config.TrainingConfig.gate_simulations. These two defaults must move
    # together: a model created through ModelManager writes its own config.json
    # from THIS dataclass, so a value here that lags the one in config.py means
    # every newly created model silently keeps the old behaviour while existing
    # models pick up the new one. Raising config.py's to 200 without this left the
    # gate measuring at 50 sims for exactly the runs meant to benefit.
    gate_simulations: int = 200
    gate_stall_warning: int = 5
    # Worker processes shared by self-play, the gate, and the random-bot eval.
    num_parallel_workers: int = 4
    # --- Game recording ---
    # Whether each phase writes its full game records to disk. Off keeps every
    # statistic (those come from games/index.jsonl) and loses only the replay.
    # See config.TrainingConfig for what that saves.
    record_self_play_games: bool = True
    record_gate_games: bool = True
    # --- Move restrictions ---
    # Opt-in heuristic: forbid the bot from filling one of its own two eyes.
    # False reproduces the behaviour of every model created before it existed.
    restrict_eye_fill: bool = False
    # Opt-in heuristic (NOT a theorem, unlike restrict_eye_fill): forbid moves
    # that walk a group larger than self_atari_max_stones into atari for
    # nothing. See game/self_atari.py for what it can cost.
    restrict_self_atari: bool = False
    self_atari_max_stones: int = 1
    # --- Mercy rule (self-play resignation) ---
    # Off by default; see config.TrainingConfig for what each one does.
    resign_enabled: bool = False
    resign_threshold: float = 0.90
    resign_consecutive: int = 4
    resign_min_move_factor: float = 1.0
    resign_both_sides: bool = True
    resign_playout_fraction: float = 0.1


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
    # Versioned input encoding (game/features.py). Frozen at creation for the
    # same reason as the architecture: it changes input_conv.in_channels, so a
    # trained weights.pt cannot be reinterpreted under a different set.
    # Defaults to the 10-plane encoding every pre-existing model was trained on.
    input_features: str = "v1_10"


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
    # Why this model exists — free text the user writes on the dashboard. Forks
    # of a run are otherwise distinguishable only by their names, which is what
    # made a directory of eight near-identical models hard to read.
    notes: str = ""
    # Retired models stay on disk and keep their games; they are just folded
    # away in the model list. Default False, so every existing config.json
    # loads unarchived.
    archived: bool = False
    # Live state (updated by trainer, saved to config.json).
    # `elo` is a CACHE of the last line of `elo_history.jsonl` — it is what a
    # fleet listing reads so it does not have to open forty ledgers. The ledger
    # is the source of truth; see elo_history.reconcile.
    elo: float = DEFAULT_ELO
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
            elo=DEFAULT_ELO,
        )
        self._save_config(info)
        # Open the ledger on day one, so the Elo curve of a model created today
        # starts at its real starting point rather than at whatever its first
        # rated game happens to imply.
        elo_history.ensure_baseline(model_dir, DEFAULT_ELO)
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

    def update_training_state(self, model_id: str, iteration: int,
                              total_games: int) -> None:
        """
        Update what TRAINING owns — iteration and games played — leaving the
        rating alone.

        The trainer used to write `elo` here every iteration. It must not any
        more: the rating now moves in played matches, so a training run that
        started while the model was rated 1240 would write 1240 back over
        every point the model earned in the meantime.
        """
        info = self.get_model(model_id)
        if not info:
            return
        info.iteration = iteration
        info.total_games = total_games
        self._save_config(info)

    # --- Elo ledger --------------------------------------------------------
    #
    # The rating a model carries is the last line of its `elo_history.jsonl`,
    # not the float in its config.json. Everything that MOVES a rating goes
    # through `record_elo_result` so the move is explained; `update_model_state`
    # only refreshes the cached copy.

    def elo_history_path(self, model_id: str) -> str:
        return elo_history.history_path(self.get_model_dir(model_id))

    def read_elo_history(self, model_id: str) -> List[dict]:
        """
        A model's rating events, oldest first, migrating the ledger into
        existence on first read.

        Reading is where the migration fires because it is the first thing that
        happens to a model that predates the ledger — the dashboard opens long
        before the model plays its next rated game. `ensure_baseline` is a
        no-op once the file exists, so this costs one `os.path.exists` after
        the first call.
        """
        model_dir = self.get_model_dir(model_id)
        if not os.path.isdir(model_dir):
            return []
        info = self.get_model(model_id)
        elo_history.ensure_baseline(model_dir, info.elo if info else None)
        return elo_history.read_history(model_dir)

    def record_elo_result(self, model_id: str, entry: EloEntry) -> Optional[dict]:
        """
        Append a rating event to a model's ledger and refresh the cached `elo`.

        This is the replacement for overwriting a float: the game that caused
        the change, the opponent, and the path to the record for review all go
        down with the number. Returns the written entry, or None for an unknown
        model.
        """
        info = self.get_model(model_id)
        if info is None:
            return None

        model_dir = self.get_model_dir(model_id)
        # The rating before this game is what seeds the ledger of a model that
        # has one but was rated long before the ledger existed.
        elo_history.ensure_baseline(
            model_dir, float(entry.new_elo) - float(entry.elo_delta or 0.0))
        written = elo_history.append(model_dir, entry)

        info.elo = float(entry.new_elo)
        from config import elo_to_rank
        info.kyu_rank = elo_to_rank(info.elo)
        self._save_config(info)
        return written

    def sync_elo_from_history(self, model_id: str) -> Optional[float]:
        """
        Repair a config.json whose cached `elo` has drifted from the ledger.

        Drift means a write that appended but did not get as far as the config
        — a crash between the two, or an older build that wrote only one of
        them. Returns the corrected rating, or None if nothing needed fixing.
        """
        info = self.get_model(model_id)
        if info is None:
            return None
        corrected = elo_history.reconcile(self.get_model_dir(model_id), info.elo)
        if corrected is None:
            return None
        info.elo = corrected
        from config import elo_to_rank
        info.kyu_rank = elo_to_rank(corrected)
        self._save_config(info)
        return corrected

    def set_meta(self, model_id: str, notes: Optional[str] = None,
                 archived: Optional[bool] = None) -> Optional[ModelInfo]:
        """
        Update the fields that describe a model rather than configure it.

        Separate from `update_model` on purpose: notes and archive state say
        nothing about how training runs, so they stay editable while the model
        is training — unlike hyperparameters, which do not.
        """
        info = self.get_model(model_id)
        if not info:
            return None
        if notes is not None:
            info.notes = notes
        if archived is not None:
            info.archived = bool(archived)
        self._save_config(info)
        return info

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
            # The copy is a new thing: it inherits weights and history, not the
            # parent's description or its retirement.
            info.notes = ""
            info.archived = False
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
