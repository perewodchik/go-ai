"""
replay_store.py — persist the replay buffer alongside weights.pt.

WHY THIS EXISTS

`Trainer.__init__` builds an empty `ReplayBuffer` every time. `_try_load_weights`
restored the network, the optimizer state, the champion, the iteration counter
and the Elo — everything except the training data those numbers were produced
from. So every server restart, and every `switch_model` round trip (which
constructs a fresh Trainer), threw away up to `replay_buffer_size` samples.

That is not just lost disk space, it actively breaks the next iteration:

  1. The buffer refills from one batch of self-play (5-20 games, a few hundred
     samples) instead of the tens of thousands it held.
  2. `_train_network` then runs its full step count against that tiny sample,
     with a *restored* Adam state carrying momentum from the old data.
  3. The resulting candidate faces a 20-game promotion gate against a champion
     that was trained on the full buffer, and loses.
  4. `gate_rejections` climbs, and after `gate_stall_warning` rejections the
     stall breaker resets the training network — which the UI reports as
     "training has stalled" when the actual cause was a restart.

STORAGE FORMAT

One torch archive holding stacked tensors rather than a list of per-sample
tuples: 50k individual tensors pickle slowly and bloat the file, while three
stacked arrays save and load in one pass.

State planes are stored as uint8 when every value in them is 0 or 1, which is
what `GameState.encode_for_nn` produces today — stone masks, liberty buckets,
the ko point and the turn colour are all indicator planes. That is a lossless
4x saving, and it matters because this file is rewritten every iteration: a full
50,000-sample buffer is ~179 MB as float32 and ~45 MB as uint8. The check is
done on the data rather than assumed from the encoding, so a future feature set
with continuous planes silently falls back to float32 instead of being
truncated.

The signature block is the safety catch. A buffer is only meaningful for the
network shape it was produced for, so board size and plane count are recorded
and checked on load. A mismatch loads nothing and says why — it never tries to
salvage a partial buffer, because a half-migrated buffer is harder to reason
about than an empty one.
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import torch


BUFFER_FILENAME = "replay_buffer.pt"

# Format version, bumped if the on-disk layout ever changes. An unknown version
# is treated exactly like a signature mismatch: ignored, with a reason.
FORMAT_VERSION = 1


def buffer_path(model_dir: str) -> str:
    """Where the replay buffer lives for a given model directory."""
    return os.path.join(model_dir, BUFFER_FILENAME)


def _signature(board_size: int, num_input_planes: int) -> dict:
    return {
        'version': FORMAT_VERSION,
        'board_size': int(board_size),
        'num_input_planes': int(num_input_planes),
    }


def save_buffer(samples: List[Tuple], path: str, board_size: int,
                num_input_planes: int, max_samples: Optional[int] = None) -> int:
    """
    Write training samples to `path` atomically.

    Args:
        samples: Sequence of (state_tensor, mcts_policy, value_target).
        path: Destination file.
        board_size: Board size these samples were generated on.
        num_input_planes: Plane count of the encoding they use.
        max_samples: Keep at most this many, taking the MOST RECENT — the same
            end the FIFO buffer would have kept.

    Returns:
        Number of samples written.
    """
    samples = list(samples)
    if max_samples is not None and len(samples) > max_samples:
        samples = samples[-max_samples:]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if not samples:
        # Still write the file: an explicit empty buffer is a fact worth
        # persisting, and it keeps load/save symmetric.
        payload = {
            'signature': _signature(board_size, num_input_planes),
            'count': 0,
        }
    else:
        states = torch.stack([s for s, _, _ in samples])
        # Lossless only while the planes really are indicators — verified here
        # rather than assumed, so a future encoding with continuous features
        # keeps its precision instead of being quietly rounded.
        binary_states = bool(torch.all((states == 0) | (states == 1)))
        if binary_states:
            states = states.to(torch.uint8)

        policies = torch.from_numpy(
            np.stack([np.asarray(p, dtype=np.float32) for _, p, _ in samples])
        )
        values = torch.tensor([float(v) for _, _, v in samples],
                              dtype=torch.float32)
        payload = {
            'signature': _signature(board_size, num_input_planes),
            'count': len(samples),
            'states': states,
            'states_are_binary': binary_states,
            'policies': policies,
            'values': values,
        }

    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    return len(samples)


def load_buffer(path: str, board_size: int,
                num_input_planes: int) -> Tuple[List[Tuple], Optional[str]]:
    """
    Load training samples written by `save_buffer`.

    Returns:
        (samples, reason). `samples` is empty whenever the buffer could not be
        used, and `reason` then explains why — for surfacing in the training log
        rather than failing silently. A missing file is not an error and returns
        (empty, None): a model simply has no buffer yet.
    """
    if not os.path.isfile(path):
        return [], None

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        return [], f"replay buffer could not be read ({e})"

    expected = _signature(board_size, num_input_planes)
    found = payload.get('signature') or {}
    if found != expected:
        return [], (f"replay buffer ignored: it was written for {found or 'an unknown shape'}, "
                    f"this model needs {expected}")

    if not payload.get('count'):
        return [], None

    states = payload['states']
    if states.dtype != torch.float32:
        # Stored compacted; the network only ever sees float32.
        states = states.to(torch.float32)
    policies = payload['policies'].numpy()
    values = payload['values']

    samples = [
        (states[i], policies[i], float(values[i]))
        for i in range(payload['count'])
    ]
    return samples, None
