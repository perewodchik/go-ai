"""
features.py — versioned input encodings for the neural network.

WHY THIS IS VERSIONED RATHER THAN A SETTING
-------------------------------------------
The number of input planes is baked into the first convolution
(`GoNetwork.input_conv.in_channels`). Change it and every saved `weights.pt`
becomes shape-incompatible — `ai/checkpoint.py` compares the stored `arch` tag
and refuses to load, which is the correct behaviour but means this is NOT a
toggle you can flip on a trained model. It is frozen per model at creation,
exactly like `NetworkParams`.

See `ai/feature_migration.py` for the one way to move an existing model across:
widening the input convolution with zero-initialised channels, which is
output-identical on the day it happens.

WHAT EACH SET CONTAINS
----------------------
`v2_12` is `v1_10` plus two planes, appended at the end and leaving planes 0-9
byte-identical. That ordering is deliberate: it is what makes the widening
migration exact, because every existing weight keeps addressing the same
channel.

Planes 0-9 (both sets), always from the CURRENT PLAYER's perspective — the
network sees "my stones" vs "their stones" regardless of colour, which halves
the effective state space:

    0  my stones
    1  their stones
    2  my groups with exactly 1 liberty (in atari)
    3  my groups with exactly 2 liberties
    4  my groups with 3 or more liberties
    5  their groups with exactly 1 liberty
    6  their groups with exactly 2 liberties
    7  their groups with 3 or more liberties
    8  the ko point
    9  turn colour: all ones when Black is to move

Plane 9 exists despite the perspective trick because komi is not symmetric —
without it the network cannot judge a close game.

Planes 10-11 (`v2_12` only):

    10 the opponent's last move (the stone just played)
    11 my own previous move

WHY THOSE TWO AND NOT ALPHAGO ZERO'S EIGHT
------------------------------------------
AlphaGo Zero used 8 past positions per colour (17 planes with colour-to-play)
and no liberty or ko planes at all. History was how it saw ko in the first
place. This project encodes the ko point explicitly AND enforces full positional
superko in the rules engine, so that motivation is already satisfied.

What history still buys here is **locality**: Go replies are overwhelmingly
local, and "where did they just play" is the single strongest hint a policy head
can get for free. That signal lives almost entirely in the last one or two
moves; planes 3 through 8 of a full history stack add very little the current
position does not already state.

A PASS ENCODES AS AN EMPTY PLANE
--------------------------------
A pass has no location, so its plane is all zeros — indistinguishable from "no
move has been played yet". This is deliberate rather than overlooked: adding a
separate pass-indicator plane costs another channel to disambiguate two states
the network can already tell apart from the stone planes (an empty board is
obvious). If it ever matters, it is a `v3` question.
"""

from typing import Callable, Dict, List, Optional

import numpy as np
import torch


DEFAULT_FEATURES = "v1_10"


class FeatureSet:
    """One versioned encoding: how many planes, what they mean, how to build."""

    __slots__ = ('name', 'num_planes', 'turn_colour_plane', 'encode',
                 'label', 'summary', 'plane_names')

    def __init__(self, name: str, num_planes: int, turn_colour_plane: int,
                 encode: Callable, label: str, summary: str,
                 plane_names: List[str]):
        self.name = name
        self.num_planes = num_planes
        # Index of the all-ones-when-Black plane. The collapse guard needs it to
        # split value-head spread by side to move; hardcoding 9 there would
        # silently break the moment a set reorders its planes.
        self.turn_colour_plane = turn_colour_plane
        self.encode = encode
        self.label = label
        self.summary = summary
        self.plane_names = plane_names


def _fill_base_planes(state, planes: np.ndarray) -> None:
    """Planes 0-9, shared by every feature set."""
    from game.board import BLACK, opponent

    board = state.board
    me = state.current_player
    them = opponent(me)

    planes[0] = (board.grid == me).astype(np.float32)
    planes[1] = (board.grid == them).astype(np.float32)

    for group in board.get_all_groups(me):
        count = board.liberty_count(group)
        idx = 2 if count == 1 else (3 if count == 2 else 4)
        for r, c in group:
            planes[idx, r, c] = 1.0

    for group in board.get_all_groups(them):
        count = board.liberty_count(group)
        idx = 5 if count == 1 else (6 if count == 2 else 7)
        for r, c in group:
            planes[idx, r, c] = 1.0

    if state.ko_point is not None:
        planes[8, state.ko_point[0], state.ko_point[1]] = 1.0

    if me == BLACK:
        planes[9, :, :] = 1.0


def _encode_v1_10(state) -> torch.Tensor:
    size = state.board_size
    planes = np.zeros((10, size, size), dtype=np.float32)
    _fill_base_planes(state, planes)
    return torch.from_numpy(planes)


def _encode_v2_12(state) -> torch.Tensor:
    size = state.board_size
    planes = np.zeros((12, size, size), dtype=np.float32)
    _fill_base_planes(state, planes)

    # history[-1] is the opponent's move (they just played); history[-2] is this
    # player's own previous move. Passes and resignations carry no location and
    # leave their plane empty.
    history = state.move_history
    for offset, plane in ((1, 10), (2, 11)):
        if len(history) >= offset:
            _, move = history[-offset]
            if move[0] >= 0:
                planes[plane, move[0], move[1]] = 1.0

    return torch.from_numpy(planes)


_BASE_PLANE_NAMES = [
    "my stones",
    "their stones",
    "my groups in atari (1 liberty)",
    "my groups with 2 liberties",
    "my groups with 3+ liberties",
    "their groups in atari (1 liberty)",
    "their groups with 2 liberties",
    "their groups with 3+ liberties",
    "ko point",
    "turn colour (all ones if Black to play)",
]


FEATURE_SETS: Dict[str, FeatureSet] = {
    "v1_10": FeatureSet(
        name="v1_10",
        num_planes=10,
        turn_colour_plane=9,
        encode=_encode_v1_10,
        label="Standard (10 planes)",
        summary=("Stones, liberty buckets, ko point and turn colour. The "
                 "encoding every model in this project was trained with."),
        plane_names=list(_BASE_PLANE_NAMES),
    ),
    "v2_12": FeatureSet(
        name="v2_12",
        num_planes=12,
        turn_colour_plane=9,
        encode=_encode_v2_12,
        label="Recent moves (12 planes)",
        summary=("Everything in Standard plus the opponent's last move and "
                 "your own previous move — a locality hint for the policy "
                 "head, which is what history buys in Go once ko is already "
                 "encoded."),
        plane_names=_BASE_PLANE_NAMES + [
            "opponent's last move",
            "my previous move",
        ],
    ),
}

# Order a chooser should present them in.
FEATURE_SET_ORDER = ["v1_10", "v2_12"]


def resolve(name: Optional[str]) -> FeatureSet:
    """Look up a feature set, falling back to the default for unknown names."""
    return FEATURE_SETS.get(name or DEFAULT_FEATURES,
                            FEATURE_SETS[DEFAULT_FEATURES])


def num_planes(name: Optional[str]) -> int:
    """Plane count for a feature set — the value NetworkConfig must carry."""
    return resolve(name).num_planes


def encode_state(state, name: Optional[str] = None) -> torch.Tensor:
    """Encode `state` with the named feature set."""
    return resolve(name).encode(state)


def encode_for_network(state, network) -> torch.Tensor:
    """
    Encode `state` the way `network` expects.

    Preferred entry point: it takes the encoding from the network itself, so a
    caller can never pair a position with the wrong plane layout.
    """
    return encode_state(state, getattr(network, "input_features", DEFAULT_FEATURES))
