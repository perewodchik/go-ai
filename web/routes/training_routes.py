"""
training_routes.py — Routes for training control and monitoring.
"""

import os
import json
import glob
from flask import Blueprint, render_template, jsonify, request

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

training_bp = Blueprint('training', __name__)


def _get_trainer():
    from web.app import trainer
    return trainer


def _require_trainer():
    """Return the trainer or (None, error_response) if no model is selected."""
    t = _get_trainer()
    if t is None:
        return None
    return t


@training_bp.route('/')
def training_page():
    from web.app import model_manager
    active_model = model_manager.get_active_model()
    trainer = _get_trainer()
    status = trainer.get_status() if trainer else {}
    return render_template('training.html', status=status, active_model=active_model)


@training_bp.route('/review')
def review_page():
    from web.app import model_manager
    active_model = model_manager.get_active_model()
    return render_template('review.html', active_model=active_model)


@training_bp.route('/api/status')
def training_status():
    trainer = _require_trainer()
    if not trainer:
        return jsonify({'error': 'No model selected'}), 400
    return jsonify(trainer.get_status())


@training_bp.route('/api/metrics')
def training_metrics():
    trainer = _require_trainer()
    if not trainer:
        return jsonify([])
    return jsonify(trainer.get_metrics_history())


@training_bp.route('/api/games')
def list_games():
    """List stored self-play and eval games, grouped by iteration."""
    trainer = _require_trainer()
    if not trainer:
        return jsonify([])

    games_dir = trainer.config.paths.games_dir
    if not os.path.exists(games_dir):
        return jsonify([])
    
    files = sorted(glob.glob(os.path.join(games_dir, '*.json')), reverse=True)
    grouped = {}

    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)

            iter_num = data.get('iteration', 0)
            if iter_num not in grouped:
                grouped[iter_num] = []

            data['filename'] = os.path.basename(f)
            grouped[iter_num].append(data)
        except (json.JSONDecodeError, IOError):
            continue
            
    # Fetch metrics to map iteration -> elo
    metrics = trainer.get_metrics_history()
    iter_to_elo = {m.get('iteration'): round(m.get('elo', 0)) for m in metrics if 'iteration' in m}
    
    # Convert grouped dict to sorted list of dicts
    result = []
    for k in sorted(grouped.keys(), reverse=True):
        result.append({
            'iteration': k,
            'elo': iter_to_elo.get(k),
            'games': grouped[k]
        })
    return jsonify(result)


@training_bp.route('/api/games/<filename>')
def get_game(filename):
    """Get a specific game for replay."""
    trainer = _require_trainer()
    if not trainer:
        return jsonify({'error': 'No model selected'}), 400

    path = os.path.join(trainer.config.paths.games_dir, filename)
    if not os.path.exists(path):
        return jsonify({'error': 'Game not found'}), 404
        
    with open(path) as f:
        game_data = json.load(f)
        
    # Reconstruct exact board states for accurate replay (including captures)
    from game.game_state import GameState
    from game.board import BLACK
    board_size = game_data.get('board_size', 9)
    komi = game_data.get('komi', 6.5)

    # Older games predate win-rate recording — recompute the eval curve on the
    # fly by re-evaluating every pre-move position with the active eval network.
    need_win_rates = not game_data.get('win_rates')
    encoded_states = []  # tensors for each position, in move order
    move_players = []    # player to move at each position

    state = GameState(board_size=board_size, komi=komi)
    states = [state.board.grid.tolist()]

    for m in game_data.get('moves', []):
        # Evaluate the position BEFORE the move is applied, mirroring how
        # win rates are captured during self-play.
        if need_win_rates and not state.is_over:
            encoded_states.append(state.encode_for_nn())
            move_players.append(state.current_player)

        move = m['move']
        if move[0] < 0: # pass
            state.play_pass()
        else:
            state.play_move(move[0], move[1])
        states.append(state.board.grid.tolist())

    game_data['states'] = states

    if need_win_rates and encoded_states:
        try:
            import torch
            batch = torch.stack(encoded_states)
            # Value head returns the outcome from each position's mover
            # perspective; flip White-to-move positions to Black's perspective.
            _, values = trainer.eval_network.predict_batch(batch, "cpu")
            win_rates = []
            for value, player in zip(values.tolist(), move_players):
                value_black = value if player == BLACK else -value
                win_rates.append(round(50.0 + 50.0 * value_black, 1))
            game_data['win_rates'] = win_rates
        except Exception:
            # Network mismatch (e.g. different board size) — skip the curve
            # rather than failing the whole replay request.
            game_data['win_rates'] = []

    return jsonify(game_data)


@training_bp.route('/api/learning_stats')
def learning_stats():
    """Compute learning stats: timing and self-play B vs W winrate."""
    trainer = _require_trainer()
    if not trainer:
        return jsonify({})

    metrics = trainer.get_metrics_history()
    games_dir = trainer.config.paths.games_dir

    # Include all games in series output (up to 10000) for client-side toggling (All, 10, 20, 50, 100)
    SERIES_LIMIT = 10000

    result = {
        'avg_time_per_game_last_iter': None,
        'avg_time_per_game_total': None,
        'iter_time_last': None,
        'iter_time_avg': None,
        'self_play_wr_black': None,
        'self_play_wr_white': None,
        'avg_game_length': None,
        'buffer_size': len(trainer.replay_buffer),
        'buffer_capacity': trainer.config.training.replay_buffer_size,
        'learning_rate': trainer.optimizer.param_groups[0]['lr'],
        'device': trainer.device,
        # Hero-row extras
        'best_elo': None,
        'latest_win_rate_vs_random': None,
        # Per-game diverging series (signed margins) + win tallies
        'self_play_series': [],       # + = black won by N, - = white won by N
        'self_play_black_wins': 0,
        'self_play_white_wins': 0,
        'self_play_draws': 0,
        'random_series': [],          # + = AI won by N, - = AI lost by N (vs random bot)
        'random_ai_wins': 0,
        'random_ai_losses': 0,
        'random_draws': 0,
    }

    # --- Iteration timing + best elo / latest win rate from metrics history ---
    if metrics:
        elapsed_list = [m.get('elapsed_seconds') for m in metrics if m.get('elapsed_seconds')]
        if elapsed_list:
            result['iter_time_last'] = elapsed_list[-1]
            result['iter_time_avg'] = round(sum(elapsed_list) / len(elapsed_list), 1)

        elos = [m.get('elo') for m in metrics if m.get('elo') is not None]
        if elos:
            result['best_elo'] = round(max(elos))

        wrs = [m.get('win_rate_vs_random') for m in metrics if m.get('win_rate_vs_random') is not None]
        if wrs:
            result['latest_win_rate_vs_random'] = wrs[-1]

    # --- Per-game timing, winrate, and signed-margin series from game files ---
    if os.path.exists(games_dir):
        files = glob.glob(os.path.join(games_dir, '*.json'))
        all_game_times = []
        last_iter_game_times = []
        game_lengths = []
        black_wins = 0
        white_wins = 0
        total_self_play = 0

        # Collect (sort_key, signed_margin) tuples so the series reads left→right chronologically.
        self_play_points = []
        random_points = []

        current_iter = trainer.iteration

        for f in files:
            try:
                with open(f) as fh:
                    data = json.load(fh)

                iteration = data.get('iteration', 0)
                game_index = data.get('game_index', 0)
                winner = data.get('winner')
                margin = data.get('margin') or 0
                sort_key = (iteration, game_index)

                if data.get('is_eval'):
                    # vs random bot — signed from the network's perspective
                    net_color = data.get('network_color')
                    if winner == 0 or winner is None:
                        signed = 0.0
                        result['random_draws'] += 1
                    elif winner == net_color:
                        signed = float(margin)
                        result['random_ai_wins'] += 1
                    else:
                        signed = -float(margin)
                        result['random_ai_losses'] += 1
                    random_points.append((sort_key, signed))
                    continue

                # --- self-play game ---
                elapsed = data.get('elapsed_seconds')
                if elapsed is not None:
                    all_game_times.append(elapsed)
                    if iteration == current_iter:
                        last_iter_game_times.append(elapsed)

                num_moves = data.get('num_moves')
                if num_moves is not None:
                    game_lengths.append(num_moves)

                # B vs W winrate + signed margin (+ = black, - = white)
                if winner == 1:
                    black_wins += 1
                    total_self_play += 1
                    signed = float(margin)
                elif winner == 2:
                    white_wins += 1
                    total_self_play += 1
                    signed = -float(margin)
                else:
                    total_self_play += 1  # draw counts in denominator
                    signed = 0.0
                self_play_points.append((sort_key, signed))

            except (json.JSONDecodeError, IOError):
                continue

        if all_game_times:
            result['avg_time_per_game_total'] = round(sum(all_game_times) / len(all_game_times), 2)
        if last_iter_game_times:
            result['avg_time_per_game_last_iter'] = round(sum(last_iter_game_times) / len(last_iter_game_times), 2)

        if total_self_play > 0:
            result['self_play_wr_black'] = round(black_wins / total_self_play, 3)
            result['self_play_wr_white'] = round(white_wins / total_self_play, 3)

        if game_lengths:
            result['avg_game_length'] = round(sum(game_lengths) / len(game_lengths), 1)

        # Sort chronologically and keep only the most recent SERIES_LIMIT points.
        self_play_points.sort(key=lambda t: t[0])
        random_points.sort(key=lambda t: t[0])
        result['self_play_series'] = [round(m, 1) for _, m in self_play_points[-SERIES_LIMIT:]]
        result['random_series'] = [round(m, 1) for _, m in random_points[-SERIES_LIMIT:]]

        result['self_play_black_wins'] = black_wins
        result['self_play_white_wins'] = white_wins
        result['self_play_draws'] = total_self_play - black_wins - white_wins

    return jsonify(result)


@training_bp.route('/api/apply_params', methods=['POST'])
def apply_params():
    """
    Live-tune the training hyperparameters that the loop reads fresh each
    iteration. No restart is needed: the running loop picks up the new values
    on its next iteration. Structural params (board size / komi / ruleset /
    network) are NOT tunable here — those rebuild the network and are edited
    from the Dashboard while training is stopped.

    Changes are both applied to the live trainer AND persisted to the model's
    config.json so they survive a restart.
    """
    trainer = _require_trainer()
    if not trainer:
        return jsonify({'error': 'No model selected'}), 400

    data = request.get_json() or {}

    def _num(key, cast, lo, hi):
        if key not in data or data[key] is None or data[key] == '':
            return None
        try:
            v = cast(data[key])
        except (TypeError, ValueError):
            raise ValueError(f'Invalid value for {key}')
        return max(lo, min(hi, v))

    try:
        sp = _num('num_self_play_games', int, 1, 100)
        ev = _num('eval_games', int, 1, 100)
        sims = _num('num_simulations', int, 10, 1000)
        cpuct = _num('c_puct', float, 0.1, 5.0)
        lr = _num('learning_rate', float, 0.0001, 0.05)
        temp_thresh = _num('temperature_threshold', int, 0, 200)
        temp_init = _num('temperature_init', float, 0.0, 2.0)
        temp_final = _num('temperature_final', float, 0.0, 1.0)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    applied = {}

    # --- Apply live to the running trainer's config (read fresh each iteration) ---
    if sp is not None:
        trainer.config.training.num_self_play_games = sp
        applied['num_self_play_games'] = sp
    if ev is not None:
        trainer.config.training.eval_games = ev
        applied['eval_games'] = ev
    if sims is not None:
        trainer.config.mcts.num_simulations = sims
        applied['num_simulations'] = sims
    if cpuct is not None:
        trainer.config.mcts.c_puct = cpuct
        applied['c_puct'] = cpuct
    if lr is not None:
        trainer.config.training.learning_rate = lr
        # Update the optimizer immediately, and re-base the LR scheduler so its
        # cosine annealing continues from the new value instead of overwriting it.
        for group in trainer.optimizer.param_groups:
            group['lr'] = lr
        try:
            trainer.scheduler.base_lrs = [lr for _ in trainer.scheduler.base_lrs]
        except Exception:
            pass
        applied['learning_rate'] = lr
    if temp_thresh is not None:
        trainer.config.mcts.temperature_threshold = temp_thresh
        applied['temperature_threshold'] = temp_thresh
    if temp_init is not None:
        trainer.config.mcts.temperature_init = temp_init
        applied['temperature_init'] = temp_init
    if temp_final is not None:
        trainer.config.mcts.temperature_final = temp_final
        applied['temperature_final'] = temp_final

    # --- Persist to config.json so the change survives a restart ---
    model_id = getattr(trainer, 'model_id', None)
    if model_id and applied:
        try:
            from web.app import model_manager
            model_manager.update_model(model_id, training_params=applied)
        except Exception:
            pass  # Non-critical: live values are already applied.

    running = trainer.is_running
    if not applied:
        msg = 'No changes to apply.'
    elif running:
        msg = 'Parameters applied — they take effect from the next iteration.'
    else:
        msg = 'Parameters saved.'

    trainer.log(f"Parameters tuned: {applied}") if applied else None

    return jsonify({'message': msg, 'running': running, 'applied': applied})


@training_bp.route('/api/save_weights', methods=['POST'])
def save_weights_route():
    """Force an immediate weights save."""
    trainer = _require_trainer()
    if not trainer:
        return jsonify({'error': 'No model selected'}), 400
    try:
        filepath = trainer.save_weights_now()
        return jsonify({'message': 'Weights saved successfully', 'filepath': filepath})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@training_bp.route('/api/force_stop', methods=['POST'])
def force_stop_route():
    """Immediately stop training and roll back uncommitted changes."""
    trainer = _require_trainer()
    if not trainer:
        return jsonify({'error': 'No model selected'}), 400
    try:
        trainer.force_stop()
        return jsonify({'message': 'Force stop requested successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@training_bp.route('/api/start', methods=['POST'])
def start_training_route():
    """Start training via REST API."""
    from web.app import start_training_job
    data = request.get_json(silent=True) or {}
    max_iter = data.get('max_iterations') if isinstance(data, dict) else None
    ok, msg = start_training_job(max_iter)
    if not ok and msg == 'No model selected':
        return jsonify({'error': msg}), 400
    return jsonify({'message': msg})


@training_bp.route('/api/stop', methods=['POST'])
def stop_training_route():
    """Graceful stop via REST API."""
    trainer = _require_trainer()
    if not trainer:
        return jsonify({'error': 'No model selected'}), 400
    trainer.stop()
    return jsonify({'message': 'Stop requested successfully'})


