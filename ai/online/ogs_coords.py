"""
ogs_coords.py — translating between OGS's coordinates and this project's.

OGS packs a coordinate into two letters counting from `a`, x first, both
indexed from the top-left corner: (row 3, col 15) is "pd". A pass is "..".
The move lists in `gamedata` use the unpacked form instead: `[x, y, time_ms]`,
with `[-1, -1]` for a pass.

This project uses `(row, col)`, also from the top-left, so the translation is
just an axis swap — which is exactly the kind of thing that is invisibly wrong
on symmetric positions and correct-looking everywhere else. Hence its own
module and its own round-trip test over every intersection.
"""

from typing import Tuple

from game.game_state import MOVE_PASS

# OGS uses a plain a-z alphabet with no letter skipped (unlike board labels,
# which skip "i").
_ALPHABET = "abcdefghijklmnopqrstuvwxyz"

PASS_STRING = ".."


def to_ogs(move: Tuple[int, int]) -> str:
    """`(row, col)` -> "pd". MOVE_PASS -> ".."."""
    if move is None or tuple(move) == MOVE_PASS:
        return PASS_STRING
    row, col = int(move[0]), int(move[1])
    if row < 0 or col < 0:
        return PASS_STRING
    if row >= len(_ALPHABET) or col >= len(_ALPHABET):
        raise ValueError(f"Board coordinate out of range for OGS: {move}")
    return _ALPHABET[col] + _ALPHABET[row]


def from_ogs(encoded: str) -> Tuple[int, int]:
    """ "pd" -> `(row, col)`. ".." -> MOVE_PASS."""
    if not encoded or encoded == PASS_STRING:
        return MOVE_PASS
    if len(encoded) < 2:
        raise ValueError(f"Unreadable OGS move: {encoded!r}")
    col = _ALPHABET.find(encoded[0])
    row = _ALPHABET.find(encoded[1])
    if col < 0 or row < 0:
        raise ValueError(f"Unreadable OGS move: {encoded!r}")
    return (row, col)


def from_ogs_pair(x: int, y: int) -> Tuple[int, int]:
    """
    An unpacked OGS move (`[x, y, …]` from a game record) -> `(row, col)`.

    OGS marks a pass as (-1, -1); some records use -1 for either axis.
    """
    if x is None or y is None or int(x) < 0 or int(y) < 0:
        return MOVE_PASS
    return (int(y), int(x))


def unpack_move(entry) -> Tuple[int, int]:
    """
    One entry of a `gamedata.moves` list -> `(row, col)`.

    Entries are `[x, y, time_ms]`, occasionally with extra trailing fields, and
    very old records store the packed string instead.
    """
    if isinstance(entry, str):
        return from_ogs(entry)
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        return from_ogs_pair(entry[0], entry[1])
    raise ValueError(f"Unreadable OGS move entry: {entry!r}")
