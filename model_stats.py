"""
model_stats.py — Read-only facts about every model, without touching the trainer.

`ModelManager` owns a model's *configuration*; this module owns everything you
can learn about a model by reading what it has already produced — its metrics
log, its stored games, its size on disk — and the relationships between models
that nobody records explicitly but that fall out of that data.

WHY IT IS SEPARATE FROM THE TRAINER: the trainer is bound to ONE model at a
time, and pointing it at another one (`web.app.switch_model`) rebuilds the
config, reloads weights.pt and constructs a fresh `Trainer`. A dashboard that
lists eight models cannot pay that price eight times, so nothing here goes
through the trainer — it reads `models/<id>/logs/training_log.jsonl` and
`models/<id>/games/` directly.

Four things are derived rather than stored:

  * **Lineage.** `ModelManager.copy_model()` deep-copies the directory, so a
    fork inherits its parent's metrics log verbatim. Two models whose logs share
    a `(iteration, timestamp)` prefix are therefore the same run up to that
    iteration, and the length of the shared prefix IS the fork point. Nothing
    writes a parent id anywhere; see `lineage()`.
  * **Head-to-head.** Bot vs bot matches are stored in both participants'
    `games/match/` directories with `rating_key` on each side, so a win matrix
    can be tallied off disk. Because each game exists twice, every reader has to
    dedupe on `(match_id, game_index)` — see `head_to_head()`.
  * **Health.** The trainer already records `collapse_warning`, gate outcomes
    and mercy-rule rates per iteration. `health()` turns the tail of that log
    into one verdict.
  * **Cost.** Games on disk and bytes on disk, for a delete that can say what it
    is about to destroy.

Everything is cached with a short TTL: a full walk of `models/` is ~0.2s for
7k files, which is fine once and wasteful on every poll of a live dashboard.
"""

import os
import json
import time
from typing import Dict, List, Optional, Tuple

from model_manager import MODELS_ROOT, ModelInfo

# Long enough that a dashboard polling every few seconds does not re-walk the
# tree, short enough that a finished iteration shows up while you are looking.
_CACHE_TTL = 15.0

# A gate that has rejected this many candidates in a row is stalling. Mirrors
# the default of TrainingParams.gate_stall_warning, and is overridden per model
# by the model's own setting when one is available.
_DEFAULT_GATE_STALL = 5

# Above this share of checked resignations being wrong, the mercy rule is
# mislabelling training data faster than it is saving time. Same threshold the
# training page's mercy-rule card uses.
_FALSE_RESIGN_LIMIT = 0.05
_MIN_RESIGN_CHECKS = 10

_log_cache: Dict[str, Tuple[float, float, list]] = {}   # id -> (mtime, size, rows)
_disk_cache: Dict[str, Tuple[float, dict]] = {}         # id -> (fetched_at, stats)
_match_cache: Tuple[float, dict] = (0.0, {})


def _log_path(model_id: str) -> str:
    return os.path.join(MODELS_ROOT, model_id, 'logs', 'training_log.jsonl')


def read_log(model_id: str) -> List[dict]:
    """
    Parsed `training_log.jsonl` rows for a model, oldest first.

    Cached on the file's (mtime, size), so an unchanged log is parsed once and
    an appended one is re-read. Unparseable lines are skipped rather than
    failing the read — a log truncated by a crash still describes the
    iterations that did finish.
    """
    path = _log_path(model_id)
    try:
        stat = os.stat(path)
    except OSError:
        return []

    cached = _log_cache.get(model_id)
    if cached and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
        return cached[2]

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
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        return []

    rows.sort(key=lambda r: (r.get('iteration') is None, r.get('iteration') or 0))
    _log_cache[model_id] = (stat.st_mtime, stat.st_size, rows)
    return rows


def disk_stats(model_id: str) -> dict:
    """
    Stored games and bytes under a model's directory.

    This is what a delete destroys, so it is worth being exact rather than
    estimating from the iteration count.
    """
    now = time.time()
    cached = _disk_cache.get(model_id)
    if cached and now - cached[0] < _CACHE_TTL:
        return cached[1]

    model_dir = os.path.join(MODELS_ROOT, model_id)
    games = 0
    total_bytes = 0
    games_root = os.path.join(model_dir, 'games')
    for root, _dirs, files in os.walk(model_dir):
        in_games = root == games_root or root.startswith(games_root + os.sep)
        for name in files:
            try:
                total_bytes += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
            if in_games and name.endswith('.json'):
                games += 1

    stats = {'games_on_disk': games, 'bytes_on_disk': total_bytes}
    _disk_cache[model_id] = (now, stats)
    return stats


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

# Ordered worst-first; a model's level is the worst reason it has.
_LEVEL_ORDER = ['critical', 'warn', 'ok', 'idle']


def _worst(levels: List[str]) -> str:
    for level in _LEVEL_ORDER:
        if level in levels:
            return level
    return 'ok'


def health(model_id: str, info: Optional[ModelInfo] = None,
           rows: Optional[List[dict]] = None) -> dict:
    """
    One verdict per model, from the tail of its metrics log.

    Every input is already recorded by the trainer; the point here is that
    nothing was ever reading it back. A model can look like its neighbours in a
    list — same board, same network, a plausible Elo — while its last iteration
    recorded a pass-rate collapse.

    Returns {level, headline, reasons: [{level, text}]}.
    """
    rows = read_log(model_id) if rows is None else rows
    if not rows:
        return {
            'level': 'idle',
            'headline': 'Never trained',
            'reasons': [{'level': 'idle', 'text': 'No completed iterations yet.'}],
        }

    last = rows[-1]
    reasons = []

    collapse = last.get('collapse_warning')
    if collapse:
        reasons.append({'level': 'critical', 'text': f'Last iteration: {collapse}'})

    # Gate stall — count promotions backwards from the newest gated iteration.
    gated = [r for r in rows if r.get('gate_win_rate') is not None]
    if gated:
        stall = 0
        for row in reversed(gated):
            if row.get('gate_promoted'):
                break
            stall += 1
        limit = _DEFAULT_GATE_STALL
        if info is not None:
            limit = getattr(info.training, 'gate_stall_warning', None) or limit
        promoted = sum(1 for r in gated if r.get('gate_promoted'))
        if stall >= limit:
            reasons.append({
                'level': 'warn',
                'text': f'Promotion gate has rejected {stall} candidates in a row '
                        f'(warns at {limit}) — training is running without advancing.',
            })
        else:
            reasons.append({
                'level': 'ok',
                'text': f'Gate promoted {promoted} of {len(gated)} candidates.',
            })
    else:
        reasons.append({
            'level': 'warn',
            'text': 'No promotion-gate matches — Elo here rests on the random-bot '
                    'evaluation alone, which saturates.',
        })

    checked = last.get('resign_checked_games') or 0
    false_rate = last.get('false_resign_rate')
    if checked >= _MIN_RESIGN_CHECKS and false_rate is not None \
            and false_rate > _FALSE_RESIGN_LIMIT:
        reasons.append({
            'level': 'warn',
            'text': f'{false_rate:.0%} of checked resignations were wrong '
                    f'(over {_FALSE_RESIGN_LIMIT:.0%}) — the mercy rule is '
                    f'mislabelling games.',
        })

    for colour in ('black', 'white'):
        rate = last.get(f'pass_rate_{colour}')
        if rate is not None and rate > 0.15 and not collapse:
            reasons.append({
                'level': 'warn',
                'text': f'{colour.capitalize()} passed {rate:.0%} of its moves in the '
                        f'last iteration.',
            })

    level = _worst([r['level'] for r in reasons])
    critical = [r for r in reasons if r['level'] == 'critical']
    warns = [r for r in reasons if r['level'] == 'warn']
    if critical:
        headline = 'Collapse recorded'
    elif warns:
        headline = warns[0]['text'].split(' — ')[0].rstrip('.')
    else:
        headline = 'Training healthily'

    return {'level': level, 'headline': headline, 'reasons': reasons}


# ---------------------------------------------------------------------------
# Per-model summary
# ---------------------------------------------------------------------------

def _downsample(values: list, limit: int = 24) -> list:
    """Evenly thin a series to at most `limit` points, always keeping the last."""
    if len(values) <= limit:
        return list(values)
    step = (len(values) - 1) / float(limit - 1)
    return [values[int(round(i * step))] for i in range(limit)]


def summarize(info: ModelInfo) -> dict:
    """
    Everything the fleet table shows for one model.

    Deliberately includes the full `training` / `network` params: they are small,
    and having them per row is what lets the client diff two forks without a
    second round trip.
    """
    rows = read_log(info.id)
    disk = disk_stats(info.id)

    elos = [r.get('elo') for r in rows if r.get('elo') is not None]
    losses = [r.get('total_loss') for r in rows if r.get('total_loss') is not None]
    gated = [r for r in rows if r.get('gate_win_rate') is not None]

    # Elo movement over the last 10 iterations — a trajectory says more about a
    # run than its final number does.
    elo_delta_10 = None
    if len(elos) >= 2:
        window = elos[-11:] if len(elos) > 10 else elos
        elo_delta_10 = round(window[-1] - window[0], 1)

    d = info.to_dict()
    d.update({
        'iterations_logged': len(rows),
        'elo_series': [round(e, 1) for e in _downsample(elos)],
        'elo_delta_10': elo_delta_10,
        'last_loss': round(losses[-1], 4) if losses else None,
        'gate_matches': len(gated),
        'gate_promotions': sum(1 for r in gated if r.get('gate_promoted')),
        'total_train_seconds': round(sum(r.get('elapsed_seconds') or 0 for r in rows), 1),
        'last_trained': rows[-1].get('timestamp') if rows else None,
        'buffer_size': rows[-1].get('buffer_size') if rows else None,
        'health': health(info.id, info=info, rows=rows),
        **disk,
    })
    return d


def history(model_id: str, fields: List[str], limit: int = 200) -> dict:
    """
    Per-iteration series for charting, downsampled to `limit` points.

    Rows that predate a field simply contribute None for it, which Chart.js
    renders as a gap rather than a zero.
    """
    rows = _downsample(read_log(model_id), limit)
    return {
        'iterations': [r.get('iteration') for r in rows],
        'series': {f: [r.get(f) for r in rows] for f in fields},
    }


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------

def _fingerprints(model_id: str) -> List[tuple]:
    """
    Per-iteration identity used to compare two models' histories.

    `(iteration, timestamp)` is enough: a copied directory reproduces both
    exactly, and two independent runs cannot share a timestamp to the
    microsecond.
    """
    return [(r.get('iteration'), r.get('timestamp')) for r in read_log(model_id)]


def _shared_prefix(a: List[tuple], b: List[tuple]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y or x[1] is None:
            break
        n += 1
    return n


def lineage(infos: List[ModelInfo]) -> dict:
    """
    Fork graph over the given models.

    For each model, its parent is the model that shares the longest prefix with
    it AND existed first — creation order is what breaks the tie when several
    models share the same trunk. Without it, a snapshot taken early would be
    "parented" by a snapshot taken later that happens to contain its prefix.

    Returns {model_id: {parent, fork_iteration, own_iterations, root, children}}.
    """
    prints = {info.id: _fingerprints(info.id) for info in infos}
    created = {info.id: (info.created_at or '') for info in infos}

    nodes = {
        info.id: {
            'parent': None,
            'fork_iteration': None,
            'own_iterations': len(prints[info.id]),
            'children': [],
            'root': info.id,
        }
        for info in infos
    }

    def predates(candidate: str, child: str) -> bool:
        """
        Could `candidate` have been the thing `child` was copied FROM?

        Sharing a prefix is symmetric, so without this every fork pair would
        parent each other: hot-boi (48 iterations) and hot-boi-at-710 (29) share
        27, and each has rows beyond the fork. Creation order breaks it — only
        the model that already existed can have been copied. When a config
        predates `created_at` entirely, fall back to history length: the trunk
        is the longer of the two.
        """
        a, b = created[candidate], created[child]
        if a and b:
            return a < b
        if len(prints[candidate]) != len(prints[child]):
            return len(prints[candidate]) > len(prints[child])
        return candidate < child   # last resort, but deterministic

    for info in infos:
        mine = prints[info.id]
        if not mine:
            continue

        best = None  # (shared_prefix_length, parent_id)
        for other in infos:
            if other.id == info.id or not prints[other.id]:
                continue
            shared = _shared_prefix(mine, prints[other.id])
            if shared == 0:
                continue
            # Creation order is the ONLY thing separating a parent from a
            # sibling here. Requiring the parent to have rows past the fork
            # looks tempting and is wrong: a snapshot copied from a snapshot
            # that was never trained on afterwards has an identical log, and
            # that pair is exactly the one a fork tree needs to get right.
            if not predates(other.id, info.id):
                continue
            # Longest shared history wins; ties go to the older model.
            if best is None or shared > best[0] or (
                    shared == best[0] and created[other.id] < created[best[1]]):
                best = (shared, other.id)

        if best:
            shared, parent_id = best
            nodes[info.id]['parent'] = parent_id
            nodes[info.id]['fork_iteration'] = shared
            nodes[info.id]['own_iterations'] = len(mine) - shared

    # Resolve roots (a chain of forks resolves to the trunk it came from).
    for model_id in nodes:
        seen = {model_id}
        root = model_id
        while nodes[root]['parent'] and nodes[root]['parent'] not in seen:
            root = nodes[root]['parent']
            seen.add(root)
        nodes[model_id]['root'] = root

    for model_id, node in nodes.items():
        if node['parent']:
            nodes[node['parent']]['children'].append(model_id)

    return nodes


# ---------------------------------------------------------------------------
# Head to head
# ---------------------------------------------------------------------------

def _match_games() -> List[dict]:
    """
    Every stored bot vs bot game, deduplicated.

    A match is written into BOTH participants' `games/match/` directories, so
    reading the tree naively counts every game twice and doubles every score.
    """
    global _match_cache
    now = time.time()
    if now - _match_cache[0] < _CACHE_TTL:
        return _match_cache[1]

    seen = {}
    if os.path.isdir(MODELS_ROOT):
        for entry in sorted(os.listdir(MODELS_ROOT)):
            match_dir = os.path.join(MODELS_ROOT, entry, 'games', 'match')
            if not os.path.isdir(match_dir):
                continue
            for name in sorted(os.listdir(match_dir)):
                if not name.endswith('.json'):
                    continue
                try:
                    with open(os.path.join(match_dir, name)) as fh:
                        data = json.load(fh)
                except (json.JSONDecodeError, OSError):
                    continue
                key = (data.get('match_id'), data.get('game_index'))
                if key in seen:
                    continue
                data['_owner'] = entry
                data['_rel_path'] = os.path.join('match', name)
                seen[key] = data

    games = list(seen.values())
    _match_cache = (now, games)
    return games


def _side_key(player: dict) -> Optional[str]:
    """`model:<id>` / `random` → the id the dashboard keys models by."""
    key = (player or {}).get('rating_key')
    if not key:
        return None
    return key.split('model:', 1)[1] if key.startswith('model:') else key


def head_to_head() -> dict:
    """
    Win matrix over every stored match game.

    Returns {model_id: {opponent_id: {wins, losses, draws, games, last_played,
    game_ids, opponent_name, opponent_kind}}}. `random` appears as an opponent id like any model — it is the
    Elo anchor, and beating it is still evidence.
    """
    table: Dict[str, Dict[str, dict]] = {}

    def cell(a: str, b: str) -> dict:
        return table.setdefault(a, {}).setdefault(b, {
            'wins': 0, 'losses': 0, 'draws': 0, 'games': 0,
            'last_played': None, 'game_ids': [],
            'opponent_name': None,
            'opponent_kind': None,
        })

    for game in _match_games():
        bp = game.get('black_player') or {}
        wp = game.get('white_player') or {}
        black = _side_key(bp)
        white = _side_key(wp)
        if not black or not white or black == white:
            continue

        winner = game.get('winner')
        for me, them, my_colour, them_player in ((black, white, 1, wp), (white, black, 2, bp)):
            row = cell(me, them)
            row['games'] += 1
            if winner == my_colour:
                row['wins'] += 1
            elif winner in (1, 2):
                row['losses'] += 1
            else:
                row['draws'] += 1
            stamp = game.get('timestamp')
            if stamp and (row['last_played'] is None or stamp > row['last_played']):
                row['last_played'] = stamp
            if them_player:
                name = them_player.get('name')
                if name and not row.get('opponent_name'):
                    row['opponent_name'] = name
                kind = them_player.get('kind')
                if not kind:
                    key = them_player.get('rating_key', '')
                    if key.startswith('ogs:') or 'ogs' in key:
                        kind = 'ogs'
                    elif key == 'random':
                        kind = 'random'
                    elif key.startswith('model:'):
                        kind = 'model'
                if kind and not row.get('opponent_kind'):
                    row['opponent_kind'] = kind
            if not row.get('opponent_name') and them.startswith('ogs:'):
                try:
                    from ai.online.ogs_bots import registry
                    bot_id_str = them.split('ogs:', 1)[1]
                    bot = registry.get(int(bot_id_str))
                    if bot and bot.username:
                        row['opponent_name'] = bot.username
                        row['opponent_kind'] = 'ogs'
                except Exception:
                    pass
            # Enough to link a few examples into the review page; the full list
            # would bloat the payload for no benefit.
            if len(row['game_ids']) < 5:
                row['game_ids'].append({
                    'owner': game.get('_owner'),
                    'path': game.get('_rel_path'),
                })

    return table


def invalidate_caches() -> None:
    """Drop every cache. Called after a write that changes what is on disk."""
    global _match_cache
    _log_cache.clear()
    _disk_cache.clear()
    _match_cache = (0.0, {})
