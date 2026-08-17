"""
game_index.py — the compact summary of every game a model has played.

WHY THIS EXISTS. A stored game record is big: its move list, its per-move
win-rate curve and its final board are most of the file, and a model with a
hundred iterations has ten thousand of them. Every statistic the training page
draws — the black/white win-rate, the signed-margin series, the average game
length, the mercy-rule report — needs about a dozen scalars per game and none
of the bulk. Reading them off the full records meant opening and parsing every
file on disk on every page load: ~3.5s for an 11k-game model, growing with the
model, and repeated for each of the three endpoints that wanted the same
numbers.

So each model keeps one line per game in `games/index.jsonl`:

    {"iteration": 96, "phase": "self-play", "game_index": 0, "winner": 1,
     "margin": 12.5, "num_moves": 84, "elapsed_seconds": 31.2, "stored": true}

The index is the ONLY thing the statistics endpoints read. The full records are
read when you open one game to replay it, and never in bulk.

`stored` is what makes the recording toggles possible: with
`record_self_play_games` off the game's numbers still land here, so no chart
loses a point — only the multi-megabyte record it would have been able to
replay is skipped.

IDENTITY. A row is identified by `(iteration, phase, game_index)`, not by its
path, because a row exists for games whose record was never written. The path
is derived from those three when one is needed.

CONSISTENCY. Nothing has to remember to keep this file up to date: `load()`
reconciles it against the directory on every read, appending rows for game
files it has never seen (a model that predates the index, or one whose index
was deleted, builds itself on first read). That check walks directory entries
only — it never opens a game file it has already indexed.
"""

import os
import json
from typing import Dict, Iterable, List, Optional, Tuple

from ai.game_store import (
    PHASE_SELF_PLAY,
    game_rel_path,
    iter_game_files,
)

INDEX_NAME = 'index.jsonl'

# The scalars every statistic on the training page is computed from. Anything
# not in here has to come from the full record — which is the point: this list
# is the contract that keeps the index small.
_SUMMARY_FIELDS = (
    'winner',            # 1 = black, 2 = white, 0/None = draw
    'margin',
    'num_moves',
    'elapsed_seconds',
    'timestamp',
    'network_color',     # eval phase: which colour the network played
    'candidate_won',     # promotion phase: gate outcome for this game
    'resigned',
    'would_resign_move',  # mercy-rule playout check — see ai/resignation.py
    'false_resign',
    'failed',
)

# (mtime, size, rows, keys) per games_dir. `keys` is the identity set, kept
# alongside the rows so reconciliation does not rebuild it on every read.
_cache: Dict[str, Tuple[float, int, List[dict], set]] = {}


def index_path(games_dir: str) -> str:
    return os.path.join(games_dir, INDEX_NAME)


def row_key(row: dict) -> tuple:
    """The identity of a row — see the module docstring."""
    return (row.get('iteration'), row.get('phase'), row.get('game_index'))


def row_path(row: dict) -> Optional[str]:
    """Where this game's full record is, or None if it was never written."""
    if not row.get('stored'):
        return None
    if row.get('path'):
        return row['path']
    iteration, phase, index = row_key(row)
    if iteration is None or phase is None or index is None:
        return None
    return game_rel_path(iteration, phase, index)


def summarize(iteration: int, phase: str, index: int, record: dict,
              stored: bool = True) -> dict:
    """Build one index row from a full game record."""
    row = {
        'iteration': iteration,
        'phase': phase,
        'game_index': index,
        'stored': bool(stored),
    }
    for field in _SUMMARY_FIELDS:
        value = record.get(field)
        # Omitting absent fields keeps the line short; every reader uses
        # .get() with its own default anyway.
        if value is not None:
            row[field] = value
    return row


def append(games_dir: str, row: dict) -> None:
    """
    Add one row to the index.

    Written with a single O_APPEND write because the promotion gate records its
    games from worker processes: one short write to a file opened for append is
    atomic, so parallel workers cannot interleave halves of two lines.
    """
    line = (json.dumps(row, separators=(',', ':')) + '\n').encode()
    os.makedirs(games_dir, exist_ok=True)
    fd = os.open(index_path(games_dir), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def record(games_dir: str, iteration: int, phase: str, index: int,
           game_record: dict, stored: bool = True) -> None:
    """Index one game. Called by `game_store.save_game`, not directly."""
    append(games_dir, summarize(iteration, phase, index, game_record, stored))


def _read_rows(path: str) -> List[dict]:
    rows = []
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue   # a line torn by a crash costs one game, not the file
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        return []
    return rows


def _write_rows(games_dir: str, rows: List[dict]) -> None:
    """Rewrite the whole index. Only used by rebuild/prune, never by a save."""
    os.makedirs(games_dir, exist_ok=True)
    tmp = index_path(games_dir) + '.tmp'
    with open(tmp, 'w') as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(',', ':')) + '\n')
    os.replace(tmp, index_path(games_dir))


def _summarize_file(game_file, index: int) -> Optional[dict]:
    """
    Read a full record off disk purely to index it.

    The index comes from the FILENAME rather than the record's own
    `game_index`, because the filename is what reconciliation matched against —
    a record whose stamped index disagreed with its name would otherwise be
    re-indexed on every read, forever.
    """
    try:
        with open(game_file.abs_path) as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    return summarize(game_file.iteration, game_file.phase, index, data, stored=True)


def _reconcile(games_dir: str, rows: List[dict], keys: set) -> Tuple[List[dict], bool]:
    """
    Add rows for game files the index has never seen.

    Only files whose identity is missing are opened, so an up-to-date index
    costs one directory walk and zero JSON parses. Rows are never removed here
    — a game whose record was deleted keeps its numbers, and an explicit delete
    calls `prune` to drop them.
    """
    added = []
    for game_file in iter_game_files(games_dir):
        index = _index_from_name(game_file)
        if index is None or (game_file.iteration, game_file.phase, index) in keys:
            continue
        row = _summarize_file(game_file, index)
        if row is None:
            continue
        keys.add(row_key(row))
        added.append(row)

    if not added:
        return rows, False

    rows = rows + added
    return rows, True


def _index_from_name(game_file) -> Optional[int]:
    """The game index a file's name encodes (`promo_0003.json` -> 3)."""
    stem = os.path.splitext(os.path.basename(game_file.abs_path))[0]
    digits = stem.rsplit('_', 1)[-1]
    return int(digits) if digits.isdigit() else None


def load(games_dir: str) -> List[dict]:
    """
    Every indexed game for one model, oldest first.

    Cached on the index file's (mtime, size), then reconciled against the
    directory so games written by another process — the trainer thread, a
    worker — show up without a restart.
    """
    if not os.path.isdir(games_dir):
        return []

    path = index_path(games_dir)
    try:
        stat = os.stat(path)
        signature = (stat.st_mtime, stat.st_size)
    except OSError:
        signature = (0.0, 0)

    cached = _cache.get(games_dir)
    fresh = not (cached and (cached[0], cached[1]) == signature)
    if fresh:
        rows = _read_rows(path)
        keys = {row_key(r) for r in rows}
    else:
        rows, keys = cached[2], cached[3]

    rows, changed = _reconcile(games_dir, rows, keys)
    if changed:
        _write_rows(games_dir, rows)
        try:
            stat = os.stat(path)
            signature = (stat.st_mtime, stat.st_size)
        except OSError:
            signature = (0.0, 0)

    if fresh or changed:
        # Appends land in completion order, which is not iteration order once
        # the gate and self-play interleave. Sorting here means every reader
        # gets a chronological series without sorting it again.
        rows.sort(key=lambda r: (r.get('iteration') if r.get('iteration') is not None else -1,
                                 r.get('game_index') or 0))
    _cache[games_dir] = (signature[0], signature[1], rows, keys)
    return rows


def rebuild(games_dir: str) -> int:
    """
    Throw the index away and rebuild it from the records on disk.

    LOSSY BY DEFINITION on a model that has run with recording off: rows for
    games whose record was never written have nothing to be rebuilt from and do
    not come back. Reconciliation in `load` is the repair path for an index
    that has fallen behind; this is for one that has gone wrong.
    """
    _cache.pop(games_dir, None)
    try:
        os.remove(index_path(games_dir))
    except OSError:
        pass
    return len(load(games_dir))


def prune(games_dir: str, rel_path: str) -> int:
    """
    Drop the rows a delete has just destroyed.

    `rel_path` is whatever `delete_saved_game` was given: one game file, a
    phase directory, an iteration directory, or the games root. Rows are
    matched by the path they describe, so deleting `iter_000096` takes its
    self-play, promotion and eval rows with it.

    Deleting games is the user's way of reclaiming space AND of clearing the
    history those games produced — which is why this is explicit, and why
    reconciliation alone never removes anything.
    """
    if not os.path.isfile(index_path(games_dir)):
        return 0

    clean = (rel_path or '').strip().strip('/')
    rows = load(games_dir)

    if clean in ('', '.'):
        kept: List[dict] = []
    else:
        prefix = clean + '/'
        kept = []
        for row in rows:
            iteration, phase, index = row_key(row)
            if iteration is None:
                kept.append(row)
                continue
            described = game_rel_path(iteration, phase, index or 0)
            if described == clean or described.startswith(prefix):
                continue
            kept.append(row)

    removed = len(rows) - len(kept)
    if removed:
        _write_rows(games_dir, kept)
        _cache.pop(games_dir, None)
    return removed


def invalidate(games_dir: Optional[str] = None) -> None:
    if games_dir is None:
        _cache.clear()
    else:
        _cache.pop(games_dir, None)


def by_phase(rows: Iterable[dict], phase: str) -> List[dict]:
    return [r for r in rows if r.get('phase') == phase]


def self_play_rows(rows: Iterable[dict]) -> List[dict]:
    return by_phase(rows, PHASE_SELF_PLAY)
