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

Games the user creates from the Play page live outside the iteration tree, in
flat directories of their own:

    games/
      human/       game_20260815_142233.json ...   (recorded human vs bot games)
      match/       match_20260815_150411.json ...  (bot vs bot match games)

They belong to no iteration (neither is training output), they are listed on
their own at the top of the review sidebar, and they are the only games the API
lets the browser delete.

Games are addressed by their path relative to `games_dir`
(e.g. `iter_000001/promotion/promo_0003.json`, `human/game_20260815_142233.json`),
which is what the web API hands to the browser and accepts back when loading
a replay.

The flat `iter_000001_game_0000.json` layout that predates this module is
migrated in place by `migrate_legacy_layout()`.
"""

import os
import re
import json
import glob
import shutil
from datetime import datetime
from typing import Iterator, NamedTuple, Optional

PHASE_SELF_PLAY = 'self-play'
PHASE_PROMOTION = 'promotion'
PHASE_EVAL = 'eval'
# Human vs bot games recorded from the Play page. Not a training phase — it
# never appears inside an iteration directory.
PHASE_HUMAN = 'human'
# Bot vs bot exhibition matches started from the Play page (model vs model,
# model vs random, and later model vs an online opponent). Like human games
# these belong to no training iteration.
PHASE_MATCH = 'match'

# Top-level directories (under games_dir) holding non-training games. Both are
# user-created, so both are deletable from the review UI — see delete_saved_game.
HUMAN_DIR = 'human'
MATCH_DIR = 'match'
USER_DIRS = (HUMAN_DIR, MATCH_DIR)

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
    iteration: Optional[int]   # None for human games, which belong to no iteration
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


def _save_timestamped_game(games_dir: str, subdir: str, phase: str,
                           record: dict, prefix: str = 'game') -> str:
    """
    Write a non-training game into its own flat directory under `games_dir`.

    The filename is timestamp-based rather than index-based: these games can be
    deleted individually, so a running index would either reuse the id of a
    deleted game or need the whole directory scanned to stay unique.

    Stamps `phase` onto the record and returns the path relative to `games_dir`.
    """
    record['phase'] = phase

    directory = os.path.join(games_dir, subdir)
    os.makedirs(directory, exist_ok=True)

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    name = f'{prefix}_{stamp}.json'
    suffix = 2
    while os.path.exists(os.path.join(directory, name)):
        name = f'{prefix}_{stamp}_{suffix}.json'
        suffix += 1

    with open(os.path.join(directory, name), 'w') as f:
        json.dump(record, f, indent=2)
    return os.path.join(subdir, name)


def _iter_flat_game_files(games_dir: str, subdir: str, phase: str) -> Iterator[GameFile]:
    """Yield games in a flat user directory, newest first (names sort by time)."""
    directory = os.path.join(games_dir, subdir)
    if not os.path.isdir(directory):
        return

    for name in sorted(os.listdir(directory), reverse=True):
        if not name.endswith('.json'):
            continue
        yield GameFile(
            rel_path=os.path.join(subdir, name),
            abs_path=os.path.join(directory, name),
            iteration=None,
            phase=phase,
        )


def _load_flat_game_files(games_dir: str, subdir: str, phase: str) -> Iterator[tuple]:
    for gf in _iter_flat_game_files(games_dir, subdir, phase):
        try:
            with open(gf.abs_path) as fh:
                record = json.load(fh)
        except (json.JSONDecodeError, IOError, OSError):
            continue
        yield gf, record


def save_human_game(games_dir: str, record: dict) -> str:
    """Write a human vs bot game into `games/human/`."""
    return _save_timestamped_game(games_dir, HUMAN_DIR, PHASE_HUMAN, record)


def iter_human_game_files(games_dir: str) -> Iterator[GameFile]:
    """Yield recorded human games, newest first."""
    return _iter_flat_game_files(games_dir, HUMAN_DIR, PHASE_HUMAN)


def load_human_game_files(games_dir: str) -> Iterator[tuple]:
    """Yield (GameFile, record) for every readable human game, newest first."""
    return _load_flat_game_files(games_dir, HUMAN_DIR, PHASE_HUMAN)


def save_match_game(games_dir: str, record: dict) -> str:
    """Write a bot vs bot match game into `games/match/`."""
    return _save_timestamped_game(games_dir, MATCH_DIR, PHASE_MATCH, record,
                                  prefix='match')


def iter_match_game_files(games_dir: str) -> Iterator[GameFile]:
    """Yield stored match games, newest first."""
    return _iter_flat_game_files(games_dir, MATCH_DIR, PHASE_MATCH)


def load_match_game_files(games_dir: str) -> Iterator[tuple]:
    """Yield (GameFile, record) for every readable match game, newest first."""
    return _load_flat_game_files(games_dir, MATCH_DIR, PHASE_MATCH)


def delete_saved_game(games_dir: str, rel_path: str) -> bool:
    """
    Delete a user-created game (a recorded human game or a match game),
    addressed by its id (path under `games_dir`).

    Deliberately limited to the directories in USER_DIRS: the id comes from the
    browser, and training output is not the user's to delete from the review UI.
    Returns False if the path is outside those directories or does not exist.
    """
    path = resolve_game_path(games_dir, rel_path)
    if not path:
        return False

    allowed = {os.path.realpath(os.path.join(games_dir, d)) for d in USER_DIRS}
    if os.path.dirname(path) not in allowed:
        return False

    try:
        os.remove(path)
    except OSError:
        return False
    return True


# Kept as the historical name; match games are deletable through it too.
delete_human_game = delete_saved_game


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
