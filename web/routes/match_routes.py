"""
match_routes.py — REST API for bot vs bot matches (the "Bot vs Bot" mode on
the Play page).

A match is created with one spec per side, runs on a background thread, and is
polled for a live snapshot while it plays. The specs are the same
`ai.players.create_player` specs used everywhere else, so this API already
accepts an OGS opponent the day `ai/online/ogs.py` is implemented — the only
thing that changes here is that the type flips to available.

    POST /api/match/new              create and start a match
    GET  /api/match/list             active + recent matches (summaries)
    GET  /api/match/opponents        what can be picked as a side
    GET  /api/match/<id>             live snapshot (poll this)
    POST /api/match/<id>/stop        stop between moves
    POST /api/match/<id>/pause       {paused: bool}
    POST /api/match/<id>/speed       {move_delay: seconds}
    POST /api/match/<id>/estimate    territory overlay for the live position
    DELETE /api/match/<id>           discard a finished match
"""

import os
import sys
import uuid
import threading
from collections import OrderedDict

from flask import Blueprint, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from game.scoring.estimator import ScoreEstimator
from ai.match import MatchConfig, MatchRunner, STATUS_RUNNING, STATUS_PENDING
from ai.players import create_player, PLAYER_TYPES, RANDOM_BOT_ELO

match_bp = Blueprint('match', __name__)

# Live and recently finished matches, oldest first. Bounded so a long session
# does not accumulate snapshots forever.
_matches: "OrderedDict[str, MatchRunner]" = OrderedDict()
_matches_lock = threading.Lock()

MAX_KEPT_MATCHES = 12
# Matches are CPU-bound (MCTS on CPU); running many at once just makes each of
# them crawl, and the point of this mode is to watch one.
MAX_CONCURRENT_MATCHES = 2

MAX_GAMES = 50
MAX_SIMULATIONS = 2000
MAX_MOVE_DELAY = 5.0

_estimator = ScoreEstimator()


def _get_manager():
    from web.app import model_manager
    return model_manager


def _active_count() -> int:
    return sum(1 for m in _matches.values() if m.is_active)


def _register(runner: MatchRunner) -> None:
    with _matches_lock:
        _matches[runner.match_id] = runner
        while len(_matches) > MAX_KEPT_MATCHES:
            # Evict the oldest finished match; never evict a running one.
            victim = next((mid for mid, m in _matches.items() if not m.is_active), None)
            if victim is None:
                break
            _matches.pop(victim, None)


def _get_runner(match_id: str):
    with _matches_lock:
        return _matches.get(match_id)


def _clamp(value, low, high, default):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, value))


@match_bp.route('/opponents')
def list_opponents():
    """
    Everything that can be picked as a side of a match.

    Models are grouped by board size on the client: two models can only meet on
    a board they were both trained for.
    """
    mgr = _get_manager()
    models = []
    for info in mgr.list_models():
        models.append({
            'type': 'model',
            'model_id': info.id,
            'name': info.name,
            'board_size': info.board_size,
            'komi': info.komi,
            'ruleset': info.ruleset,
            'elo': round(info.elo, 1),
            'kyu_rank': info.kyu_rank,
            'iteration': info.iteration,
            'default_simulations': info.training.num_simulations,
        })

    return jsonify({
        'models': models,
        'random': {
            'type': 'random',
            'name': 'Random Bot',
            'elo': RANDOM_BOT_ELO,
            'note': 'Elo anchor — its own rating never changes',
        },
        'player_types': PLAYER_TYPES,
        'active_model_id': mgr.get_active_model_id(),
        'limits': {
            'max_games': MAX_GAMES,
            'max_simulations': MAX_SIMULATIONS,
            'max_move_delay': MAX_MOVE_DELAY,
            'max_concurrent': MAX_CONCURRENT_MATCHES,
        },
    })


def _board_settings(specs, mgr):
    """
    Work out the board a match is played on from its participants.

    Models carry their own board size, komi and ruleset, and a model can only
    play the size it was trained for — so two models must agree. Settings come
    from the first model in the match; if neither side is a model (random vs
    random), the active model decides, falling back to a 9x9 default.

    Returns (settings_dict, error_or_None).
    """
    model_infos = []
    for spec in specs:
        if spec.get('type', 'model') != 'model':
            continue
        info = mgr.get_model(spec.get('model_id'))
        if info is None:
            return None, f"Model not found: {spec.get('model_id')}"
        model_infos.append(info)

    if not model_infos:
        active = mgr.get_active_model()
        if active:
            return {'board_size': active.board_size, 'komi': active.komi,
                    'ruleset': active.ruleset}, None
        return {'board_size': 9, 'komi': 6.5, 'ruleset': 'chinese'}, None

    sizes = {i.board_size for i in model_infos}
    if len(sizes) > 1:
        names = ', '.join(f"{i.name} ({i.board_size}x{i.board_size})" for i in model_infos)
        return None, f"These models were trained for different board sizes: {names}"

    first = model_infos[0]
    return {'board_size': first.board_size, 'komi': first.komi,
            'ruleset': first.ruleset}, None


def _record_dirs(specs, mgr):
    """games/ directory of every distinct local model taking part."""
    dirs = []
    for spec in specs:
        if spec.get('type', 'model') != 'model':
            continue
        model_id = spec.get('model_id')
        path = os.path.join(mgr.get_model_dir(model_id), 'games')
        if path not in dirs:
            dirs.append(path)
    return dirs


@match_bp.route('/new', methods=['POST'])
def new_match():
    data = request.get_json() or {}

    spec_a = data.get('player_a') or {}
    spec_b = data.get('player_b') or {}
    if not spec_a or not spec_b:
        return jsonify({'error': 'Both sides of the match must be specified'}), 400

    mgr = _get_manager()

    settings, err = _board_settings([spec_a, spec_b], mgr)
    if err:
        return jsonify({'error': err}), 400

    with _matches_lock:
        if _active_count() >= MAX_CONCURRENT_MATCHES:
            return jsonify({
                'error': f'{MAX_CONCURRENT_MATCHES} matches are already running. '
                         'Stop one before starting another.'
            }), 409

    context = {
        'board_size': settings['board_size'],
        'komi': settings['komi'],
        'ruleset': settings['ruleset'],
        # Bots give up games their own search has called lost, instead of
        # playing decided endgames out move by move. None defers to each
        # model's own resign_enabled setting.
        'mercy_resign': data.get('mercy_resign'),
    }

    try:
        player_a = create_player(spec_a, context)
        player_b = create_player(spec_b, context)
    except NotImplementedError as exc:
        return jsonify({'error': str(exc)}), 501
    except (ValueError, RuntimeError) as exc:
        return jsonify({'error': str(exc)}), 400

    config = MatchConfig(
        board_size=settings['board_size'],
        komi=settings['komi'],
        scoring_method=settings['ruleset'],
        num_games=int(_clamp(data.get('num_games', 4), 1, MAX_GAMES, 4)),
        alternate_colors=bool(data.get('alternate_colors', True)),
        move_delay=_clamp(data.get('move_delay', 0), 0, MAX_MOVE_DELAY, 0.0),
        record_games=bool(data.get('record_games', True)),
        update_ratings=bool(data.get('update_ratings', True)),
    )

    match_id = str(uuid.uuid4())[:8]
    runner = MatchRunner(
        match_id=match_id,
        player_a=player_a,
        player_b=player_b,
        config=config,
        record_dirs=_record_dirs([spec_a, spec_b], mgr) if config.record_games else [],
        name=data.get('name') or f"{player_a.name} vs {player_b.name}",
    )
    _register(runner)

    thread = threading.Thread(target=runner.run, daemon=True,
                              name=f"match-{match_id}")
    thread.start()

    return jsonify(runner.snapshot()), 201


@match_bp.route('/list')
def list_matches():
    """Summaries of known matches, newest first."""
    with _matches_lock:
        runners = list(_matches.values())

    summaries = []
    for runner in reversed(runners):
        snap = runner.snapshot()
        summaries.append({
            'match_id': snap['match_id'],
            'name': snap['name'],
            'status': snap['status'],
            'games_completed': snap['games_completed'],
            'num_games': snap['num_games'],
            'series': snap['series'],
            'players': snap['players'],
        })
    return jsonify(summaries)


@match_bp.route('/<match_id>')
def match_status(match_id):
    runner = _get_runner(match_id)
    if runner is None:
        return jsonify({'error': 'Match not found'}), 404
    return jsonify(runner.snapshot())


@match_bp.route('/<match_id>/stop', methods=['POST'])
def stop_match(match_id):
    runner = _get_runner(match_id)
    if runner is None:
        return jsonify({'error': 'Match not found'}), 404
    runner.stop()
    # A paused match would never reach the stop check — release it.
    runner.set_paused(False)
    return jsonify(runner.snapshot())


@match_bp.route('/<match_id>/pause', methods=['POST'])
def pause_match(match_id):
    runner = _get_runner(match_id)
    if runner is None:
        return jsonify({'error': 'Match not found'}), 404
    data = request.get_json() or {}
    runner.set_paused(bool(data.get('paused', True)))
    return jsonify(runner.snapshot())


@match_bp.route('/<match_id>/speed', methods=['POST'])
def set_match_speed(match_id):
    """Change the per-move delay of a running match."""
    runner = _get_runner(match_id)
    if runner is None:
        return jsonify({'error': 'Match not found'}), 404
    data = request.get_json() or {}
    runner.config.move_delay = _clamp(data.get('move_delay', 0), 0, MAX_MOVE_DELAY, 0.0)
    return jsonify({'move_delay': runner.config.move_delay})


@match_bp.route('/<match_id>/estimate', methods=['POST'])
def estimate_match(match_id):
    """Territory estimation for the position currently on the match board."""
    runner = _get_runner(match_id)
    if runner is None:
        return jsonify({'error': 'Match not found'}), 404
    state = runner.current_state()
    if state is None:
        return jsonify({'error': 'Match has not started a game yet'}), 409
    return jsonify(_estimator.estimate(state))


@match_bp.route('/<match_id>', methods=['DELETE'])
def discard_match(match_id):
    runner = _get_runner(match_id)
    if runner is None:
        return jsonify({'error': 'Match not found'}), 404
    if runner.status in (STATUS_RUNNING, STATUS_PENDING):
        return jsonify({'error': 'Stop the match before discarding it'}), 400
    with _matches_lock:
        _matches.pop(match_id, None)
    return jsonify({'discarded': True})
