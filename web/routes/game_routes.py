"""
game_routes.py — Routes for human vs bot gameplay.

Supports two modes:
- Easy: move suggestions, analysis overlay, undo/redo
- Hard: pure play, no assistance

The opponent is any model in the workspace — the Dashboard's active model is
only the default. Board size, komi and ruleset are locked to whichever model
was picked, and every session carries its own network, so two games against
two different models can be open at once.
"""

import uuid
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, session

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config
from game.game_state import GameState, MOVE_PASS, MOVE_RESIGN
from game.board import BLACK, WHITE
from game.scoring.base import get_scorer
from game.scoring.estimator import ScoreEstimator
from ai.analysis import analyze_state
from ai.game_store import save_human_game
from ai.mcts import MCTS
from ai.mercy_rule import MercyRule
from ai.model_loader import load_model_network

game_bp = Blueprint('game', __name__)

# In-memory game sessions (keyed by session ID)
game_sessions = {}
score_estimator = ScoreEstimator()


def _get_manager():
    from web.app import model_manager
    return model_manager


def _get_active_model():
    """Get active model info."""
    return _get_manager().get_active_model()


def _load_opponent(model_id: str) -> dict:
    """
    Everything a game session needs to let `model_id` play: its network and the
    settings the model was trained under.

    The ACTIVE model plays through the trainer's own CPU eval_network — that is
    the live champion, which during training is ahead of what is on disk. Any
    other model is loaded (and cached) by ai.model_loader, exactly as a bot-vs-bot
    match loads it.

    The move restrictions (`restrict_eye_fill`, `restrict_self_atari`) come from
    the chosen model's config so a bot playing a human is subject to the same
    restrictions it was trained under. They never apply to the human's own
    moves — those go through play_move(), which the restrictions deliberately
    do not touch.

    Raises ValueError if the model does not exist.
    """
    from web.app import trainer

    mgr = _get_manager()
    info = mgr.get_model(model_id)
    if info is None:
        raise ValueError(f"Model not found: {model_id}")

    config = Config.from_model(info, mgr.get_model_dir(model_id))

    if trainer is not None and model_id == mgr.get_active_model_id():
        network = trainer.eval_network
    else:
        network, _, _ = load_model_network(model_id, manager=mgr)

    return {
        'model_id': info.id,
        'model_name': info.name,
        'model_iteration': info.iteration,
        'network': network,
        'board_size': info.board_size,
        'komi': info.komi,
        'scoring_method': info.ruleset,
        'c_puct': config.mcts.c_puct,
        'restrict_eye_fill': bool(config.training.restrict_eye_fill),
        'restrict_self_atari': bool(config.training.restrict_self_atari),
        'self_atari_max_stones': int(config.training.self_atari_max_stones),
        'games_dir': config.paths.games_dir,
        'default_simulations': config.mcts.num_simulations,
        'config': config,
    }


@game_bp.route('/play')
def play_page():
    model = _get_active_model()
    return render_template('play.html', active_model=model)


@game_bp.route('/api/game/opponents')
def list_opponents():
    """
    Models that can be played against, plus which one the Dashboard has active
    (the default selection in the setup panel).
    """
    mgr = _get_manager()
    models = [{
        'model_id': info.id,
        'name': info.name,
        'board_size': info.board_size,
        'komi': info.komi,
        'ruleset': info.ruleset,
        'elo': round(info.elo, 1),
        'kyu_rank': info.kyu_rank,
        'iteration': info.iteration,
        'default_simulations': info.training.num_simulations,
    } for info in mgr.list_models()]

    return jsonify({
        'models': models,
        'active_model_id': mgr.get_active_model_id(),
    })


@game_bp.route('/api/game/new', methods=['POST'])
def new_game():
    """
    Create a new game session.

    The opponent is `model_id` when the client sends one, otherwise the active
    model; board size, komi and ruleset are locked to whichever it is.
    """
    data = request.get_json() or {}

    model_id = (data.get('model_id') or '').strip() or _get_manager().get_active_model_id()
    if not model_id:
        return jsonify({'error': 'No model selected. Create or select a model from the Dashboard.'}), 400

    try:
        opponent = _load_opponent(model_id)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 404
    except Exception as exc:
        # A weights file that no longer matches the model's architecture, most
        # likely — say so instead of failing on the first move.
        return jsonify({'error': f'Could not load {model_id}: {exc}'}), 400

    if opponent['network'] is None:
        return jsonify({'error': 'Model not loaded'}), 400

    board_size = opponent['board_size']
    komi = opponent['komi']
    mode = data.get('mode', 'hard')  # 'easy' or 'hard'
    player_color = data.get('player_color', 'black')  # 'black' or 'white'
    num_simulations = data.get('num_simulations') or opponent['default_simulations']

    game_id = str(uuid.uuid4())[:8]
    state = GameState(board_size=board_size, komi=komi)

    # Mercy rule: the bot gives up once its own search has called the game lost
    # for several moves running, instead of playing a decided endgame out.
    mercy = MercyRule.from_config(opponent['config'], board_size,
                                  enabled=data.get('mercy_resign'))

    game_sessions[game_id] = {
        'state': state,
        'mode': mode,
        'player_color': BLACK if player_color == 'black' else WHITE,
        'num_simulations': int(num_simulations),
        'mercy': mercy,
        **opponent,
    }

    response = {
        'game_id': game_id,
        'state': state.to_dict(),
        'mode': mode,
        'player_color': player_color,
        'board_size': board_size,
        'komi': komi,
        'ruleset': opponent['scoring_method'],
        'model_id': opponent['model_id'],
        # Names the game in the Play launcher's list.
        'model_name': opponent['model_name'],
        'mercy_resign': mercy.describe(),
    }

    # If player is white, bot plays first (as black)
    if player_color == 'white':
        _bot_move(game_id)
        response['state'] = game_sessions[game_id]['state'].to_dict()

    return jsonify(response)


@game_bp.route('/api/game/move', methods=['POST'])
def make_move():
    """Human plays a move, bot responds."""
    data = request.get_json() or {}
    game_id = data.get('game_id')
    row = data.get('row')
    col = data.get('col')

    if game_id not in game_sessions:
        return jsonify({'error': 'Game not found'}), 404

    sess = game_sessions[game_id]
    state = sess['state']

    if state.is_over:
        return jsonify({'error': 'Game is over'}), 400

    # Guard: the human may only move on their own turn. Without this a fast
    # double-click (two /move calls before the bot replies) would place a stone
    # for the bot's color, letting the player play both sides.
    if state.current_player != sess['player_color']:
        return jsonify({'error': "Not your turn"}), 409

    # Play human move
    if not state.play_move(row, col):
        return jsonify({'error': 'Illegal move'}), 400

    # Score info
    scoring_method = sess.get('scoring_method', 'chinese')
    scorer = get_scorer(scoring_method)
    black_score, white_score = scorer.score(state)

    return jsonify({
        'state': state.to_dict(),
        'scores': {'black': black_score, 'white': white_score},
    })


def _result_for(sess: dict, scorer, black_score: float, white_score: float):
    """
    How the game ended, or None while it is still running.

    A resignation has no margin to report — the loser gave up rather than
    being counted out — so the winner comes from the state itself.
    """
    state = sess['state']
    if not state.is_over:
        return None

    if state.resign_color:
        return {
            'winner': 'black' if state.winner == BLACK else 'white',
            'reason': 'resignation',
            'resigned_by': 'human' if state.resign_color == sess['player_color'] else 'bot',
        }

    winner, margin = scorer.determine_winner(state)
    return {
        'winner': 'black' if winner == BLACK else ('white' if winner == WHITE else 'draw'),
        'margin': margin,
        'black_score': black_score,
        'white_score': white_score,
    }


@game_bp.route('/api/game/bot_move', methods=['POST'])
def get_bot_move():
    """Trigger the bot to make a move."""
    data = request.get_json() or {}
    game_id = data.get('game_id')

    if game_id not in game_sessions:
        return jsonify({'error': 'Game not found'}), 404

    sess = game_sessions[game_id]
    state = sess['state']

    if state.is_over:
        return jsonify({'error': 'Game is over'}), 400

    bot_move = _bot_move(game_id)

    # Score info
    scoring_method = sess.get('scoring_method', 'chinese')
    scorer = get_scorer(scoring_method)
    black_score, white_score = scorer.score(state)

    return jsonify({
        'state': state.to_dict(),
        'bot_move': bot_move,
        'scores': {'black': black_score, 'white': white_score},
        # The bot's move can END the game — two passes, or the mercy rule. Say
        # so here, or the client has a finished game and no result to show.
        'result': _result_for(sess, scorer, black_score, white_score),
    })


@game_bp.route('/api/game/pass', methods=['POST'])
def pass_turn():
    """Human passes."""
    data = request.get_json() or {}
    game_id = data.get('game_id')

    if game_id not in game_sessions:
        return jsonify({'error': 'Game not found'}), 404

    sess = game_sessions[game_id]
    state = sess['state']

    if state.is_over:
        return jsonify({'error': 'Game is over'}), 400
    # Same turn guard as make_move: only pass on the human's own turn.
    if state.current_player != sess['player_color']:
        return jsonify({'error': "Not your turn"}), 409

    state.play_pass()

    scoring_method = sess.get('scoring_method', 'chinese')
    scorer = get_scorer(scoring_method)
    black_score, white_score = scorer.score(state)

    return jsonify({
        'state': state.to_dict(),
        'scores': {'black': black_score, 'white': white_score},
        'result': _result_for(sess, scorer, black_score, white_score),
    })


@game_bp.route('/api/game/resign', methods=['POST'])
def resign():
    """Human resigns."""
    data = request.get_json() or {}
    game_id = data.get('game_id')

    if game_id not in game_sessions:
        return jsonify({'error': 'Game not found'}), 404

    sess = game_sessions[game_id]
    state = sess['state']
    state.play_resign()

    scorer = get_scorer(sess.get('scoring_method', 'chinese'))
    black_score, white_score = scorer.score(state)

    return jsonify({
        'state': state.to_dict(),
        'result': _result_for(sess, scorer, black_score, white_score),
    })


@game_bp.route('/api/game/undo', methods=['POST'])
def undo_move():
    """Undo last move pair (easy mode only)."""
    data = request.get_json() or {}
    game_id = data.get('game_id')

    if game_id not in game_sessions:
        return jsonify({'error': 'Game not found'}), 404

    sess = game_sessions[game_id]
    if sess['mode'] != 'easy':
        return jsonify({'error': 'Undo only available in easy mode'}), 403

    state = sess['state']
    # Undo twice: bot's move + human's move
    state.undo_move()
    state.undo_move()

    return jsonify({'state': state.to_dict()})


@game_bp.route('/api/game/suggest', methods=['POST'])
def suggest_move():
    """Get bot's suggestion for the human (easy mode only)."""
    data = request.get_json() or {}
    game_id = data.get('game_id')

    if game_id not in game_sessions:
        return jsonify({'error': 'Game not found'}), 404

    sess = game_sessions[game_id]
    if sess['mode'] != 'easy':
        return jsonify({'error': 'Suggestions only in easy mode'}), 403

    if sess['network'] is None:
        return jsonify({'error': 'Model not loaded'}), 400

    mcts = _session_mcts(sess)
    # temperature=0.0: a hint should be the search's actual best move, not a
    # sample from it. The returned policy is still the full visit distribution
    # (target_temperature defaults to 1.0), so the top-3 list below keeps its
    # relative weights.
    action, policy = mcts.search(sess['state'], temperature=0.0, add_noise=False)

    suggestion = list(action) if action != MOVE_PASS else 'pass'
    # Get top 3 moves by policy
    top_moves = []
    board_size = sess['state'].board_size
    indices = policy.argsort()[::-1][:3]
    for idx in indices:
        if idx == board_size * board_size:
            top_moves.append({'move': 'pass', 'probability': float(policy[idx])})
        else:
            r, c = divmod(idx, board_size)
            top_moves.append({'move': [int(r), int(c)], 'probability': float(policy[idx])})

    return jsonify({'suggestion': suggestion, 'top_moves': top_moves})


@game_bp.route('/api/game/considered', methods=['POST'])
def considered_moves():
    """
    The bot's move preferences for the position on the board (easy mode only).

    This is the same search the bot runs to move, reported instead of played:
    the visit share of its best handful of moves, which the client draws as the
    considered-moves heatmap. Restricted to easy mode for the same reason
    suggestions and the win-rate curve are — hard mode gets no assistance.
    """
    data = request.get_json() or {}
    game_id = data.get('game_id')

    if game_id not in game_sessions:
        return jsonify({'error': 'Game not found'}), 404

    sess = game_sessions[game_id]
    if sess['mode'] != 'easy':
        return jsonify({'error': 'Analysis only available in easy mode'}), 403
    if sess['network'] is None:
        return jsonify({'error': 'Model not loaded'}), 400

    state = sess['state']
    if state.is_over:
        return jsonify({'move_number': state.move_number, 'moves': []})

    return jsonify(analyze_state(_session_mcts(sess), state))


@game_bp.route('/api/game/estimate', methods=['POST'])
def estimate_score():
    """Get territory estimation overlay (display only, toggleable)."""
    data = request.get_json() or {}
    game_id = data.get('game_id')

    if game_id not in game_sessions:
        return jsonify({'error': 'Game not found'}), 404

    estimation = score_estimator.estimate(game_sessions[game_id]['state'])
    return jsonify(estimation)


def _replay_positions(state: GameState):
    """
    Walk a session's move history, rebuilding it move by move.

    Returns (move_list, encoded_states, movers):
        move_list: replay entries in the same shape self-play records them
                   (`{'color', 'move', 'move_num'}`), resignation excluded —
                   it is not a board move and the replay viewer can't apply it.
        encoded_states: NN input for the position BEFORE each move, plus the
                   current position when the game is still running, so the
                   win-rate curve has a point for every position the reviewer
                   can step to.
        movers: player to move at each of those positions.
    """
    move_list, encoded, movers = [], [], []

    replay = GameState(board_size=state.board_size, komi=state.komi)
    for color, move in state.move_history:
        if tuple(move) == MOVE_RESIGN:
            break
        encoded.append(replay.encode_for_nn())
        movers.append(replay.current_player)
        move_list.append({
            'color': int(color),
            'move': list(move),
            'move_num': len(move_list),
        })
        if tuple(move) == MOVE_PASS:
            replay.play_pass()
        else:
            replay.play_move(move[0], move[1])

    if not replay.is_over:
        encoded.append(replay.encode_for_nn())
        movers.append(replay.current_player)

    return move_list, encoded, movers


def _win_rate_curve(state: GameState, network) -> list:
    """
    Black's win probability (%) at every position of a session's game.

    Evaluated in one batch with the value head of the network the session is
    playing against — the same quantity the review page charts, so a recorded
    game's curve matches what the reviewer would recompute for it. Returns []
    if no usable network.
    """
    if network is None:
        return []

    _, encoded, movers = _replay_positions(state)
    if not encoded:
        return []

    try:
        import torch
        _, values = network.predict_batch(torch.stack(encoded), "cpu")
    except Exception:
        # Board-size mismatch or a network that failed to load — the curve is
        # an extra, never a reason to fail the move the user just played.
        return []

    curve = []
    for value, player in zip(values.tolist(), movers):
        value_black = value if player == BLACK else -value
        curve.append(round(50.0 + 50.0 * value_black, 1))
    return curve


@game_bp.route('/api/game/winrate', methods=['POST'])
def game_win_rate():
    """Live win-rate curve for the running game (easy mode only)."""
    data = request.get_json() or {}
    game_id = data.get('game_id')

    if game_id not in game_sessions:
        return jsonify({'error': 'Game not found'}), 404

    sess = game_sessions[game_id]
    if sess['mode'] != 'easy':
        return jsonify({'error': 'Win rate only available in easy mode'}), 403

    curve = _win_rate_curve(sess['state'], sess['network'])
    return jsonify({
        'win_rates': curve,
        'move_number': sess['state'].move_number,
    })


@game_bp.route('/api/game/record', methods=['POST'])
def record_game():
    """
    Save the session's game to the games/human/ directory of the model it was
    played against, so it shows up in Review Games alongside that model's
    training games.
    """
    data = request.get_json() or {}
    game_id = data.get('game_id')

    if game_id not in game_sessions:
        return jsonify({'error': 'Game not found'}), 404

    sess = game_sessions[game_id]
    state = sess['state']

    move_list, _, _ = _replay_positions(state)
    if not move_list:
        return jsonify({'error': 'Nothing to record — play a move first'}), 400

    scorer = get_scorer(sess.get('scoring_method', 'chinese'))
    black_score, white_score = scorer.score(state)
    winner, margin = (scorer.determine_winner(state) if state.is_over else (None, 0.0))

    player_color = sess['player_color']
    name = (data.get('name') or '').strip()

    record = {
        'board_size': state.board_size,
        'komi': state.komi,
        'moves': move_list,
        'num_moves': len(move_list),
        'win_rates': _win_rate_curve(state, sess['network']),
        'winner': int(winner) if winner else 0,
        'black_score': black_score,
        'white_score': white_score,
        'margin': margin,
        'timestamp': datetime.now().isoformat(),
        # Human-game specifics — the review UI needs these to say who was who.
        'name': name,
        'human_color': int(player_color),
        'bot_color': int(WHITE if player_color == BLACK else BLACK),
        'mode': sess.get('mode'),
        'num_simulations': sess.get('num_simulations'),
        'resigned_by': int(state.resign_color) if state.resign_color else None,
        'unfinished': not state.is_over,
        'model_name': sess.get('model_name'),
        # Which trained version of the bot this was played against.
        'model_iteration': sess.get('model_iteration'),
    }

    rel_path = save_human_game(sess['games_dir'], record)
    return jsonify({'saved': True, 'game_id': rel_path}), 201


def _session_mcts(sess: dict) -> MCTS:
    """MCTS bound to the network and search settings of this session's model."""
    return MCTS(
        network=sess['network'],
        num_simulations=sess['num_simulations'],
        c_puct=sess.get('c_puct', 1.5),
        device="cpu",   # unbatched single positions: CPU beats MPS here
        restrict_eye_fill=sess.get('restrict_eye_fill', False),
        restrict_self_atari=sess.get('restrict_self_atari', False),
        self_atari_max_stones=sess.get('self_atari_max_stones', 1),
    )


def _bot_move(game_id: str) -> dict:
    """Have the bot make a move in the given game session."""
    sess = game_sessions[game_id]
    state = sess['state']

    if sess['network'] is None:
        # Fallback: pass if no model loaded
        state.play_pass()
        return {'type': 'pass'}

    mcts = _session_mcts(sess)
    # temperature=0.0: play the strongest move against a human opponent. Game
    # variety comes from the human, so there is nothing to buy by sampling.
    action, _ = mcts.search(state, temperature=0.0, add_noise=False)

    # The search that chose the move is also the evidence for giving up on the
    # game — a bot that has been lost for several of its own moves resigns
    # rather than grinding out a decided endgame.
    mercy = sess.get('mercy')
    if mercy is not None and mercy.observe(mcts.root_value, state.move_number):
        state.play_resign()
        return {'type': 'resign'}

    move_info = {}
    if action == MOVE_PASS:
        state.play_pass()
        move_info = {'type': 'pass'}
    else:
        state.play_move(action[0], action[1])
        move_info = {'type': 'move', 'row': action[0], 'col': action[1]}

    return move_info
