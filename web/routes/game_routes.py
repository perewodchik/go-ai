"""
game_routes.py — Routes for human vs bot gameplay.

Supports two modes:
- Easy: move suggestions, analysis overlay, undo/redo
- Hard: pure play, no assistance

Board size and komi are locked to the active model's configuration.
"""

import uuid
from flask import Blueprint, render_template, request, jsonify, session

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from game.game_state import GameState, MOVE_PASS
from game.board import BLACK, WHITE
from game.scoring.base import get_scorer
from game.scoring.estimator import ScoreEstimator
from ai.mcts import MCTS

game_bp = Blueprint('game', __name__)

# In-memory game sessions (keyed by session ID)
game_sessions = {}
score_estimator = ScoreEstimator()


def _get_network():
    """Get the current trained network from the trainer."""
    from web.app import trainer
    if trainer is None:
        return None, None
    # Use the dedicated CPU eval_network for fast unbatched human games
    return trainer.eval_network, "cpu"


def _get_active_model():
    """Get active model info."""
    from web.app import model_manager
    return model_manager.get_active_model()


@game_bp.route('/play')
def play_page():
    model = _get_active_model()
    return render_template('play.html', active_model=model)


@game_bp.route('/api/game/new', methods=['POST'])
def new_game():
    """Create a new game session. Board size and komi come from the active model."""
    model = _get_active_model()
    if not model:
        return jsonify({'error': 'No model selected. Create or select a model from the Dashboard.'}), 400

    network, device = _get_network()
    if network is None:
        return jsonify({'error': 'Model not loaded'}), 400

    data = request.get_json() or {}
    # Board size and komi are locked to model
    board_size = model.board_size
    komi = model.komi
    mode = data.get('mode', 'hard')  # 'easy' or 'hard'
    player_color = data.get('player_color', 'black')  # 'black' or 'white'
    num_simulations = data.get('num_simulations', 200)

    game_id = str(uuid.uuid4())[:8]
    state = GameState(board_size=board_size, komi=komi)

    game_sessions[game_id] = {
        'state': state,
        'mode': mode,
        'player_color': BLACK if player_color == 'black' else WHITE,
        'num_simulations': num_simulations,
        'scoring_method': model.ruleset,
    }

    response = {
        'game_id': game_id,
        'state': state.to_dict(),
        'mode': mode,
        'player_color': player_color,
        'board_size': board_size,
        'komi': komi,
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
    result = None
    if state.is_over:
        winner, margin = scorer.determine_winner(state)
        result = {
            'winner': 'black' if winner == BLACK else ('white' if winner == WHITE else 'draw'),
            'margin': margin,
            'black_score': black_score,
            'white_score': white_score,
        }

    return jsonify({
        'state': state.to_dict(),
        'scores': {'black': black_score, 'white': white_score},
        'result': result,
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

    return jsonify({
        'state': state.to_dict(),
        'result': {
            'winner': 'black' if state.winner == BLACK else 'white',
            'reason': 'resignation',
        },
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

    network, device = _get_network()
    if network is None:
        return jsonify({'error': 'Model not loaded'}), 400

    mcts = MCTS(network=network, num_simulations=sess['num_simulations'], device=device)
    action, policy = mcts.search(sess['state'], temperature=0.1, add_noise=False)

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


@game_bp.route('/api/game/estimate', methods=['POST'])
def estimate_score():
    """Get territory estimation overlay (display only, toggleable)."""
    data = request.get_json() or {}
    game_id = data.get('game_id')

    if game_id not in game_sessions:
        return jsonify({'error': 'Game not found'}), 404

    estimation = score_estimator.estimate(game_sessions[game_id]['state'])
    return jsonify(estimation)


def _bot_move(game_id: str) -> dict:
    """Have the bot make a move in the given game session."""
    sess = game_sessions[game_id]
    state = sess['state']
    network, device = _get_network()

    if network is None:
        # Fallback: pass if no model loaded
        state.play_pass()
        return {'type': 'pass'}

    mcts = MCTS(network=network, num_simulations=sess['num_simulations'], device=device)
    action, _ = mcts.search(state, temperature=0.1, add_noise=False)

    move_info = {}
    if action == MOVE_PASS:
        state.play_pass()
        move_info = {'type': 'pass'}
    else:
        state.play_move(action[0], action[1])
        move_info = {'type': 'move', 'row': action[0], 'col': action[1]}

    return move_info
