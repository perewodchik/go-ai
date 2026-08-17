"""
elo_history.py — The append-only Elo ledger, one file per model.

    models/<slug>/elo_history.jsonl

Every line is one rating event, oldest first. This replaces a single mutable
`elo` float in config.json as the SOURCE OF TRUTH for how a model got where it
is: the float only ever said where the rating landed, so a rating that moved
200 points was indistinguishable from one that never moved at all, and there
was no way back to the games that caused it.

    {"timestamp": "...", "opponent_name": "Random Bot", "opponent_id": "random",
     "game_outcome": "win", "elo_delta": 8.0, "new_elo": 508.0,
     "game_record_path": "match/match_20260817_120000.json"}

`game_record_path` is the SAME id `ai/game_store.save_match_game` returns — a
path relative to that model's `games/` directory — so every point on the Elo
curve links straight to the game review of the game that earned it. It is
stored per model rather than globally because a match is written into both
participants' `games/` directories and the two copies are different files.

`config.json`'s `elo` is kept in sync as a cache: it is what a fleet listing of
forty models reads, and re-deriving it means opening forty ledgers. The ledger
wins if they ever disagree (`reconcile`).

MIGRATION: a model whose ledger does not exist yet but which already has an
`elo` in its config gets a single baseline entry recording that rating, so the
curve starts where the model actually is instead of at the default. See
`ensure_baseline`.
"""

import io
import os
import json
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional


HISTORY_FILENAME = 'elo_history.jsonl'

# Where a model starts before it has played anything rated. Competitive-only
# ratings need a floor to be relative to; 500 keeps the scale every stored
# model and `config.elo_to_rank` (500 = 30k) is already on, and matches the
# random bot's fixed anchor in `ai/players.RANDOM_BOT_ELO` — so a brand-new
# model is rated exactly "as strong as the random bot", which is true.
DEFAULT_ELO = 500.0

# Wording for the migration entry, matched by `is_baseline` so a reader can
# tell a seeded starting point from a real game result.
BASELINE_NOTE = 'Baseline initialization'

OUTCOME_WIN = 'win'
OUTCOME_LOSS = 'loss'
OUTCOME_DRAW = 'draw'

# One lock per ledger path. Matches run on their own threads and a model can be
# in two matches at once, so two appends can race; a line-buffered append is
# atomic enough on POSIX, but the read-modify-write in `ensure_baseline` is not.
_locks: Dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(path: str) -> threading.Lock:
    key = os.path.abspath(path)
    with _locks_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _locks[key] = lock
        return lock


@dataclass
class EloEntry:
    """
    One line of the ledger.

    Built by whoever caused the rating to move (the match runner) and written
    by whoever owns the model directory (`ModelManager`), so the two never have
    to agree on a file layout.
    """
    new_elo: float
    elo_delta: float = 0.0
    opponent_name: Optional[str] = None
    # The opponent's `rating_key`: `model:<id>`, `random`, `ogs:<bot_id>`.
    # Stable across renames, which `opponent_name` is not.
    opponent_id: Optional[str] = None
    game_outcome: Optional[str] = None          # win / loss / draw
    # Path of the recorded game relative to THIS model's `games/` dir.
    game_record_path: Optional[str] = None
    match_id: Optional[str] = None
    game_index: Optional[int] = None
    # 1 = black, 2 = white; which side this model played.
    color: Optional[int] = None
    # Set only for entries that are not a game result (baseline, manual edits).
    note: Optional[str] = None
    timestamp: str = ''

    def to_json(self) -> dict:
        d = asdict(self)
        d['new_elo'] = round(float(self.new_elo), 2)
        d['elo_delta'] = round(float(self.elo_delta), 2)
        if not d['timestamp']:
            d['timestamp'] = datetime.now().isoformat()
        return d


def outcome_from_score(score: float) -> str:
    """Map a game score (1.0 / 0.5 / 0.0) to the ledger's outcome wording."""
    if score > 0.5:
        return OUTCOME_WIN
    if score < 0.5:
        return OUTCOME_LOSS
    return OUTCOME_DRAW


def history_path(model_dir: str) -> str:
    return os.path.join(model_dir, HISTORY_FILENAME)


def is_baseline(entry: dict) -> bool:
    """True for a seeded starting point rather than a played game."""
    return not entry.get('game_outcome') and bool(entry.get('note'))


def read_history(model_dir: str) -> List[dict]:
    """
    Every ledger entry for a model, oldest first.

    A truncated or half-written line is skipped rather than failing the read:
    a ledger damaged by a crash still describes the games that did complete,
    and this is on the path of a dashboard that must render regardless.
    """
    path = history_path(model_dir)
    entries: List[dict] = []
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    entries.append(row)
    except OSError:
        return []
    return entries


def append(model_dir: str, entry: EloEntry) -> dict:
    """
    Append one entry and return the dict that was written.

    Opened in append mode per call: the ledger is written a handful of times a
    minute at most, and a held file handle is one more thing to invalidate when
    a model directory is copied or deleted underneath us.
    """
    record = entry.to_json()
    path = history_path(model_dir)
    with _lock_for(path):
        os.makedirs(model_dir, exist_ok=True)
        with io.open(path, 'a') as fh:
            fh.write(json.dumps(record) + '\n')
    return record


def ensure_baseline(model_dir: str, current_elo: Optional[float] = None) -> Optional[dict]:
    """
    Seed a model's ledger with its starting rating, if it has no ledger yet.

    This is the migration for every model that existed before the ledger did:
    its config.json already holds an `elo` earned under the old anchor-based
    evaluation, and throwing that away would reset a model that has trained for
    ninety iterations back to the default. The seeded entry carries `note`
    rather than an outcome, so a reader can tell "started here" from "won a
    game" — see `is_baseline`.

    Returns the entry it wrote, or None if the ledger already existed.
    """
    path = history_path(model_dir)
    with _lock_for(path):
        if os.path.exists(path):
            return None
        elo = DEFAULT_ELO if current_elo is None else float(current_elo)
        record = EloEntry(new_elo=elo, elo_delta=0.0,
                          note=BASELINE_NOTE).to_json()
        try:
            os.makedirs(model_dir, exist_ok=True)
            with io.open(path, 'a') as fh:
                fh.write(json.dumps(record) + '\n')
        except OSError:
            # A read-only or vanished model directory must not take down the
            # dashboard that triggered the migration.
            return record
        return record


def current_elo(model_dir: str, fallback: Optional[float] = None) -> float:
    """
    The rating the ledger ends on — the authoritative current Elo.

    Falls back to `fallback` (normally config.json's cached `elo`) when there
    is no ledger yet, and to DEFAULT_ELO when there is neither.
    """
    entries = read_history(model_dir)
    for entry in reversed(entries):
        value = entry.get('new_elo')
        if value is not None:
            return float(value)
    if fallback is not None:
        return float(fallback)
    return DEFAULT_ELO


def rated_games(model_dir: str) -> int:
    """How many rated games the ledger holds (baseline entries excluded)."""
    return sum(1 for e in read_history(model_dir) if not is_baseline(e))


def record(model_dir: str, *, new_elo: float, elo_delta: float = 0.0,
           opponent_name: Optional[str] = None, opponent_id: Optional[str] = None,
           game_outcome: Optional[str] = None,
           game_record_path: Optional[str] = None,
           match_id: Optional[str] = None, game_index: Optional[int] = None,
           color: Optional[int] = None,
           baseline_elo: Optional[float] = None) -> dict:
    """
    Append a game result, seeding the baseline first if the ledger is new.

    `baseline_elo` is the rating the model held BEFORE this game, used only for
    that seeding — otherwise the very first rated game of a long-trained model
    would look like it started from the default.
    """
    ensure_baseline(model_dir, baseline_elo if baseline_elo is not None
                    else new_elo - elo_delta)
    return append(model_dir, EloEntry(
        new_elo=new_elo,
        elo_delta=elo_delta,
        opponent_name=opponent_name,
        opponent_id=opponent_id,
        game_outcome=game_outcome,
        game_record_path=game_record_path,
        match_id=match_id,
        game_index=game_index,
        color=color,
    ))


def series(model_dir: str, limit: Optional[int] = None) -> dict:
    """
    The competitive Elo curve, plotted against GAMES PLAYED rather than
    training iteration.

    Training iterations are the wrong x-axis for a rating that only moves when
    a game is played: a model can train for thirty iterations without playing
    anyone, which draws thirty points of flat line, and can play forty rated
    games inside one iteration, which draws a single point. Here index 0 is the
    starting rating and index N is the rating after the Nth rated game.

    Returns {games, elo, entries, rated_games} with `games` the sequential
    index and `elo` the `new_elo` at that point. Downsampled to `limit` points
    when given, always keeping the first and last.
    """
    entries = read_history(model_dir)
    points = [
        {
            'game': idx,
            'new_elo': round(float(e.get('new_elo') or 0.0), 1),
            'elo_delta': round(float(e.get('elo_delta') or 0.0), 1),
            'opponent_name': e.get('opponent_name'),
            'opponent_id': e.get('opponent_id'),
            'game_outcome': e.get('game_outcome'),
            'game_record_path': e.get('game_record_path'),
            'timestamp': e.get('timestamp'),
            'note': e.get('note'),
        }
        for idx, e in enumerate(entries)
    ]

    if limit and len(points) > limit:
        step = (len(points) - 1) / float(limit - 1)
        points = [points[int(round(i * step))] for i in range(limit)]

    return {
        'games': [p['game'] for p in points],
        'elo': [p['new_elo'] for p in points],
        'entries': points,
        'rated_games': sum(1 for e in entries if not is_baseline(e)),
    }


def reconcile(model_dir: str, config_elo: Optional[float]) -> Optional[float]:
    """
    The rating to trust, given a ledger and config.json's cached copy.

    Returns the ledger's last `new_elo` when they disagree, or None when there
    is nothing to correct. The ledger wins because it is append-only: a stale
    cache is a missed write, whereas a wrong ledger would need a rewritten file.
    """
    entries = read_history(model_dir)
    if not entries:
        return None
    latest = current_elo(model_dir)
    if config_elo is None or abs(float(config_elo) - latest) > 0.05:
        return latest
    return None
