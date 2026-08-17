"""
feature_migration.py — move a trained model onto a wider input encoding.

THE PROBLEM
-----------
The input encoding is frozen per model because it decides
`GoNetwork.input_conv.in_channels`. Switching a v1_10 model to v2_12 changes
that first convolution from (F, 10, 3, 3) to (F, 12, 3, 3), so the saved
`weights.pt` no longer fits and `ai/checkpoint.py` refuses to load it — which is
correct, and which would normally mean starting a 48-iteration model from
scratch to gain two planes.

THE WAY ACROSS
--------------
`v2_12` is `v1_10` plus two planes APPENDED AT THE END, leaving planes 0-9
byte-identical. So the old weights already address exactly the right channels,
and the only thing missing is what to multiply the two new planes by.

Initialise those new weights to **zero** and the widened network computes the
identical function:

    out = sum over c of W[:, c] * X[:, c]
        = sum over c < 10 of W[:, c] * X[:, c]   +   0 * X[:, 10]  +  0 * X[:, 11]

Every other parameter is untouched. On the day of the migration the model plays
exactly as it did before — same policy, same value, same Elo — and then learns
to use the new planes over subsequent training, starting from a position of
strength rather than from random weights.

`tests/test_feature_migration.py` asserts that identity numerically rather than
taking the algebra on trust.

WHAT IS NOT CARRIED OVER
------------------------
* **The optimizer state is dropped.** Adam's moments for `input_conv.weight`
  have the old shape, and rather than performing surgery on them the migration
  restarts Adam. Its moments re-warm within a few hundred steps; the weights,
  which took hours, are what actually matter. Everything else in the checkpoint
  — iteration, Elo, gate Elo, total games, the gated champion — is preserved.
* **The replay buffer is invalidated**, because its stored positions were
  encoded with 10 planes and there is no way to recover what the two new planes
  should have held: a sample does not remember which move preceded it. The
  buffer's own signature guard rejects it automatically with a message, so
  nothing silently feeds a 12-plane network 10-plane data. The next iteration
  refills it from fresh self-play.

This is a one-way widening. There is no path back from v2_12 to v1_10 that
preserves behaviour, because dropping the two planes would discard whatever the
network has learned to do with them.
"""

import os
import shutil
from typing import Dict, Optional

import torch

from game.features import FEATURE_SETS, num_planes


BACKUP_SUFFIX = ".pre-migration.bak"


def widen_input_conv(state_dict: Dict, from_planes: int, to_planes: int,
                     key: str = "input_conv.weight") -> Dict:
    """
    Return a copy of `state_dict` whose input convolution accepts `to_planes`
    channels, with the added ones zero-initialised.

    Raises ValueError if the tensor does not have the width the caller claims,
    which is the guard against migrating something twice or migrating a
    checkpoint that is not what the config says it is.
    """
    if to_planes < from_planes:
        raise ValueError(
            f"Cannot narrow an encoding ({from_planes} -> {to_planes} planes): "
            "dropping planes would discard what the network learned from them."
        )

    weight = state_dict.get(key)
    if weight is None:
        raise ValueError(f"Checkpoint has no '{key}' — is this a GoNetwork?")

    if weight.shape[1] != from_planes:
        raise ValueError(
            f"'{key}' has {weight.shape[1]} input channels, but the model is "
            f"declared as {from_planes} planes. Refusing to migrate a "
            f"checkpoint that does not match its own config."
        )

    if to_planes == from_planes:
        return dict(state_dict)

    padding = torch.zeros(
        weight.shape[0], to_planes - from_planes, *weight.shape[2:],
        dtype=weight.dtype,
    )
    widened = dict(state_dict)
    widened[key] = torch.cat([weight, padding], dim=1)
    return widened


def migrate_checkpoint(state: Dict, from_features: str, to_features: str) -> Dict:
    """
    Widen every network inside a checkpoint payload: the training weights and
    the gated champion. Drops the optimizer state and retags the architecture.
    """
    from_planes = num_planes(from_features)
    to_planes = num_planes(to_features)

    migrated = dict(state)
    migrated['model_state_dict'] = widen_input_conv(
        state['model_state_dict'], from_planes, to_planes)

    champion = state.get('champion_state_dict')
    if champion:
        # The champion is a separate network and generates ALL self-play data.
        # Leaving it un-widened would break the very next iteration.
        migrated['champion_state_dict'] = widen_input_conv(
            champion, from_planes, to_planes)

    # Adam's moments for input_conv.weight have the old shape; restart rather
    # than reshape them. See the module docstring.
    migrated.pop('optimizer_state_dict', None)

    arch = dict(state.get('arch') or {})
    if arch:
        arch['num_input_planes'] = to_planes
        migrated['arch'] = arch

    return migrated


def migrate_model(model_dir: str, to_features: str,
                  from_features: Optional[str] = None,
                  backup: bool = True) -> dict:
    """
    Migrate a model directory in place: `weights.pt` and `config.json`.

    Args:
        model_dir: The model's directory (`models/<slug>/`).
        to_features: Target feature set, e.g. "v2_12".
        from_features: Override the source set; defaults to whatever
            config.json declares (or v1_10 for models predating the field).
        backup: Copy weights.pt aside before overwriting.

    Returns:
        A summary dict describing what changed — including whether the replay
        buffer was invalidated, which the caller should surface.
    """
    import json

    if to_features not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {to_features}")

    config_path = os.path.join(model_dir, "config.json")
    weights_path = os.path.join(model_dir, "weights.pt")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"No config.json in {model_dir}")

    with open(config_path) as fh:
        config = json.load(fh)

    network_cfg = config.setdefault("network", {})
    source = from_features or network_cfg.get("input_features") or "v1_10"

    if source == to_features:
        return {'migrated': False, 'reason': f'already on {to_features}'}

    result = {
        'migrated': True,
        'model_dir': model_dir,
        'from_features': source,
        'to_features': to_features,
        'from_planes': num_planes(source),
        'to_planes': num_planes(to_features),
        'weights_migrated': False,
        'optimizer_state_dropped': False,
        'replay_buffer_invalidated': False,
        'backup_path': None,
    }

    if os.path.isfile(weights_path):
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        migrated = migrate_checkpoint(state, source, to_features)

        if backup:
            backup_path = weights_path + BACKUP_SUFFIX
            shutil.copy2(weights_path, backup_path)
            result['backup_path'] = backup_path

        tmp_path = weights_path + ".tmp"
        torch.save(migrated, tmp_path)
        os.replace(tmp_path, weights_path)

        result['weights_migrated'] = True
        result['optimizer_state_dropped'] = 'optimizer_state_dict' in state

    # The buffer holds 10-plane positions and cannot be reconstructed; its own
    # signature guard will reject it, but say so explicitly.
    from ai.replay_store import buffer_path
    if os.path.isfile(buffer_path(model_dir)):
        result['replay_buffer_invalidated'] = True

    network_cfg['input_features'] = to_features
    tmp_config = config_path + ".tmp"
    with open(tmp_config, "w") as fh:
        json.dump(config, fh, indent=2)
    os.replace(tmp_config, config_path)

    return result
