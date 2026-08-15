"""
game_store.py — On-disk layout for recorded games.

Every game a training iteration produces lives under that iteration's own
directory, split by the phase that produced it:

    games/
      iter_000001/
        self-play/   game_0000.json ...   (self-play data generation)
        promotion/   promo_0000.json ...  (candidate vs champion gate matches)
        eval/        eval_0000.json ...   (champion vs random bot)
      iter_000002/
        ...

Games are addressed by their path relative to `games_dir`
(e.g. `iter_000001/promotion/promo_0003.json`), which is what the web API
hands to the browser and accepts back when loading a replay.

The flat `iter_000001_game_0000.json` layout that predates this module is
migrated in place by `migrate_legacy_layout()`.
"""

import os
import re
import json
import glob
import shutil
from typing import Iterator, NamedTuple, Optional

PHASE_SELF_PLAY = 'self-play'
PHASE_PROMOTION = 'promotion'
PHASE_EVAL = 'eval'

# Display order — the order phases actually run in during an iteration.
PHASES = (PHASE_SELF_PLAY, PHASE_PROMOTION, PHASE_EVAL)

# Filename prefix used inside each phase directory.
_PHASE_PREFIX = {
    PHASE_SELF_PLAY: 'game',
    PHASE_PROMOTION: 'promo',
    PHASE_EVAL: 'eval',
}

_ITER_DIR_RE = re.compile(r'^iter_(\d+)$')
# Legacy flat filenames: iter_000001_game_0000.json / iter_000001_eval_0000.json
_LEGACY_RE = re.compile(r'^iter_(\d+)_(game|eval)_(\d+)\.json$')


class GameFile(NamedTuple):
    """A stored game located on disk."""
    rel_path: str      # path relative to games_dir, used as the public id
    abs_path: str
    iteration: int
    phase: str


def iteration_dirname(iteration: int) -> str:
    return f'iter_{iteration:06d}'


def phase_dir(games_dir: str, iteration: int, phase: str, create: bool = False) -> str:
    """Directory holding `phase` games for `iteration`."""
    path = os.path.join(games_dir, iteration_dirname(iteration), phase)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def game_rel_path(iteration: int, phase: str, index: int) -> str:
    prefix = _PHASE_PREFIX.get(phase, 'game')
    return os.path.join(iteration_dirname(iteration), phase, f'{prefix}_{index:04d}.json')


def save_game(games_dir: str, iteration: int, phase: str, index: int, record: dict) -> str:
    """
    Write a game record into its iteration/phase directory.

    Stamps `iteration`, `game_index` and `phase` onto the record so a file is
    self-describing even if it is later moved or read out of context.

    Returns the path relative to `games_dir`.
    """
    record['iteration'] = iteration
    record['game_index'] = index
    record['phase'] = phase

    rel = game_rel_path(iteration, phase, index)
    abs_path = os.path.join(games_dir, rel)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, 'w') as f:
        json.dump(record, f, indent=2)
    return rel


def resolve_game_path(games_dir: str, rel_path: str) -> Optional[str]:
    """
    Map a client-supplied relative id to an absolute path inside `games_dir`.

    Returns None if the path escapes `games_dir` or does not exist — the id
    arrives from the browser, so it is never trusted to stay in the tree.
    """
    games_root = os.path.realpath(games_dir)
    candidate = os.path.realpath(os.path.join(games_root, rel_path))
    if os.path.commonpath([games_root, candidate]) != games_root:
        return None
    if not os.path.isfile(candidate):
        return None
    return candidate


def iter_game_files(games_dir: str) -> Iterator[GameFile]:
    """
    Yield every stored game under `games_dir`, in iteration/phase order.

    Iteration and phase come from the directory structure, so a file that is
    missing its metadata still lands in the right group.
    """
    if not os.path.isdir(games_dir):
        return

    for entry in sorted(os.listdir(games_dir)):
        match = _ITER_DIR_RE.match(entry)
        if not match:
            continue
        iteration = int(match.group(1))
        iter_path = os.path.join(games_dir, entry)
        if not os.path.isdir(iter_path):
            continue

        for phase in sorted(os.listdir(iter_path)):
            phase_path = os.path.join(iter_path, phase)
            if not os.path.isdir(phase_path):
                continue
            for name in sorted(os.listdir(phase_path)):
                if not name.endswith('.json'):
                    continue
                yield GameFile(
                    rel_path=os.path.join(entry, phase, name),
                    abs_path=os.path.join(phase_path, name),
                    iteration=iteration,
                    phase=phase,
                )


def load_game_files(games_dir: str) -> Iterator[tuple]:
    """Yield (GameFile, record) for every readable game, skipping corrupt ones."""
    for gf in iter_game_files(games_dir):
        try:
            with open(gf.abs_path) as fh:
                record = json.load(fh)
        except (json.JSONDecodeError, IOError, OSError):
            continue
        yield gf, record


def migrate_legacy_layout(games_dir: str) -> int:
    """
    Move flat `iter_NNNNNN_{game,eval}_NNNN.json` files into the per-iteration
    directory layout. Safe to call repeatedly; returns the number of files moved.
    """
    if not os.path.isdir(games_dir):
        return 0

    moved = 0
    for path in sorted(glob.glob(os.path.join(games_dir, '*.json'))):
        match = _LEGACY_RE.match(os.path.basename(path))
        if not match:
            continue

        iteration = int(match.group(1))
        phase = PHASE_EVAL if match.group(2) == 'eval' else PHASE_SELF_PLAY
        index = int(match.group(3))

        target = os.path.join(games_dir, game_rel_path(iteration, phase, index))
        if os.path.exists(target):
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.move(path, target)

        # Stamp the phase so migrated records match freshly written ones.
        try:
            with open(target) as f:
                record = json.load(f)
            record.setdefault('iteration', iteration)
            record.setdefault('game_index', index)
            record['phase'] = phase
            with open(target, 'w') as f:
                json.dump(record, f, indent=2)
        except (json.JSONDecodeError, IOError, OSError):
            pass  # File is moved; a bad record is skipped by the reader anyway.

        moved += 1

    return moved
