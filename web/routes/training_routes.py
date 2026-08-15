"""
training_routes.py — Routes for training control and monitoring.
"""

import os
import json
from flask import Blueprint, render_template, jsonify, request

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ai.game_store import (
    PHASES,
    PHASE_EVAL,
    PHASE_PROMOTION,
    PHASE_SELF_PLAY,
    load_game_files,
    migrate_legacy_layout,
    resolve_game_path,
)
from param_bounds import sanitize_params

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


PHASE_LABELS = {
    PHASE_SELF_PLAY: 'Self-Play',
    PHASE_PROMOTION: 'Promotion',
    PHASE_EVAL: 'Eval vs Random',
}


@training_bp.route('/api/games')
def list_games():
    """
    List stored games grouped by iteration, then by the phase that produced
    them (self-play / promotion / eval).

    Each phase group carries its own summary: the promotion group reports the
    candidate's win rate against the champion and whether it was promoted,
    the eval group reports the win rate against the random bot.
    """
    trainer = _require_trainer()
    if not trainer:
        return jsonify([])

    games_dir = trainer.config.paths.games_dir
    migrate_legacy_layout(games_dir)

    grouped = {}
    for game_file, data in load_game_files(games_dir):
        data['filename'] = game_file.rel_path
        data['phase'] = game_file.phase
        grouped.setdefault(game_file.iteration, {}).setdefault(game_file.phase, []).append(data)

    # Metrics supply the per-iteration Elo and the gate outcome.
    metrics = trainer.get_metrics_history()
    iter_metrics = {m.get('iteration'): m for m in metrics if 'iteration' in m}

    result = []
    for iteration in sorted(grouped.keys(), reverse=True):
        by_phase = grouped[iteration]
        m = iter_metrics.get(iteration, {})

        phases = []
        # Known phases first in run order, then anything unexpected on disk.
        ordered = [p for p in PHASES if p in by_phase]
        ordered += [p for p in sorted(by_phase) if p not in PHASES]

        for phase in ordered:
            games = sorted(by_phase[phase], key=lambda g: g.get('game_index', 0))
            group = {
                'phase': phase,
                'label': PHASE_LABELS.get(phase, phase),
                'count': len(games),
                'games': games,
            }

            if phase == PHASE_PROMOTION:
                decided = [g for g in games if g.get('winner')]
                wins = sum(1 for g in games if g.get('candidate_won'))
                group['candidate_wins'] = wins
                group['candidate_losses'] = len(decided) - wins
                group['candidate_win_rate'] = (
                    round(wins / len(games), 4) if games else None
                )
                # The recorded gate result wins over the file tally when
                # present — it is what the promotion decision actually used.
                if m.get('gate_win_rate') is not None:
                    group['gate_win_rate'] = round(float(m['gate_win_rate']), 4)
                group['promoted'] = m.get('gate_promoted')
                group['gate_threshold'] = trainer.config.training.gate_threshold
            elif phase == PHASE_EVAL:
                rated = [g for g in games if g.get('network_color') is not None]
                wins = sum(1 for g in rated if g.get('winner') == g.get('network_color'))
                group['ai_wins'] = wins
                group['rated_games'] = len(rated)
                group['win_rate'] = round(wins / len(rated), 4) if rated else None

            phases.append(group)

        result.append({
            'iteration': iteration,
            'elo': round(m['elo']) if m.get('elo') is not None else None,
            'phases': phases,
            'total_games': sum(p['count'] for p in phases),
        })

    return jsonify(result)


@training_bp.route('/api/games/<path:rel_path>')
def get_game(rel_path):
    """Get a specific game for replay, addressed by its path under games/."""
    trainer = _require_trainer()
    if not trainer:
        return jsonify({'error': 'No model selected'}), 400

    path = resolve_game_path(trainer.config.paths.games_dir, rel_path)
    if not path:
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


@training_bp.route('/api/gate_history')
def gate_history():
    """
    Champion lineage derived from promotion-gate results.

    Every gate match is a measured head-to-head between the candidate and the
    reigning champion, which makes it the one progress signal that does not
    saturate — unlike win_rate_vs_random, which pins at 100% early and stays
    there while the model silently degrades.

    Each promotion contributes an Elo gain computed from the margin it won by:

        delta = 400 * log10(p / (1 - p))

    A rejection leaves the ladder flat, because the champion did not change.
    The result is a strength curve that only rises when improvement was
    actually demonstrated against the previous best.
    """
    import math

    trainer = _require_trainer()
    if not trainer:
        return jsonify({'points': [], 'summary': {}})

    metrics = trainer.get_metrics_history()

    gate_games = trainer.config.training.gate_games
    threshold = trainer.config.training.gate_threshold

    points = []
    elo = 0.0                 # Relative ladder — starts at the first champion.
    champion_version = 1      # Bumped on every promotion.
    promotions = 0
    gated = 0
    streak = 0                # Current consecutive rejections.
    last_promo_iter = None

    for m in metrics:
        wr = m.get('gate_win_rate')
        if wr is None:
            # Iterations from before gating existed carry no head-to-head data.
            continue

        gated += 1
        promoted = bool(m.get('gate_promoted'))
        delta = 0.0

        if promoted:
            # Clamp so a clean sweep doesn't produce an infinite jump.
            # Winner's curse: we only ever credit Elo on a PROMOTION, i.e.
            # conditioned on the measured rate clearing the threshold. That
            # conditioning biases the observed rate upward, so crediting it
            # verbatim makes the ladder drift up faster than real strength.
            # Shrink by one standard error and floor at the threshold, which
            # turns the curve into a conservative lower bound on progress.
            n = max(1, int(gate_games))
            obs = min(max(float(wr), 0.05), 0.95)
            se = math.sqrt(obs * (1.0 - obs) / n)
            p = min(max(obs - se, float(threshold)), 0.95)
            delta = 400.0 * math.log10(p / (1.0 - p))
            elo += delta
            champion_version += 1
            promotions += 1
            streak = 0
            last_promo_iter = m.get('iteration')
        else:
            streak += 1

        points.append({
            'iteration': m.get('iteration'),
            'gate_win_rate': round(float(wr), 4),
            'promoted': promoted,
            'champion_version': champion_version,
            'gate_elo': round(elo, 1),
            'elo_delta': round(delta, 1),
            'value_std_black': m.get('value_std_black'),
            'value_std_white': m.get('value_std_white'),
            'pass_rate_black': m.get('pass_rate_black'),
            'pass_rate_white': m.get('pass_rate_white'),
        })

    avg_margin = (
        sum(p['gate_win_rate'] for p in points) / len(points) if points else None
    )

    summary = {
        'gated_iterations': gated,
        'promotions': promotions,
        'rejections': gated - promotions,
        'promotion_rate': round(promotions / gated, 3) if gated else None,
        'champion_version': champion_version,
        'gate_elo': round(elo, 1),
        'current_reject_streak': streak,
        'last_promotion_iteration': last_promo_iter,
        'avg_gate_win_rate': round(avg_margin, 4) if avg_margin is not None else None,
        'gate_threshold': trainer.config.training.gate_threshold,
    }

    return jsonify({'points': points, 'summary': summary})


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

        for game_file, data in load_game_files(games_dir):
            iteration = game_file.iteration
            game_index = data.get('game_index', 0)
            winner = data.get('winner')
            margin = data.get('margin') or 0
            sort_key = (iteration, game_index)

            if game_file.phase == PHASE_EVAL:
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

            if game_file.phase != PHASE_SELF_PLAY:
                # Promotion games are candidate-vs-champion matches, not
                # self-play data — they'd skew the B/W and timing stats.
                continue

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

    # --- Comprehensive Time Metrics computation for 3-Tab Time Block ---
    # Collect game timings per iteration & phase
    iter_game_times = {} # {iter: {phase: [elapsed_seconds, ...], ...}}
    if os.path.exists(games_dir):
        for game_file, data in load_game_files(games_dir):
            it = game_file.iteration
            ph = game_file.phase
            el = data.get('elapsed_seconds')
            if el is not None:
                iter_game_times.setdefault(it, {}).setdefault(ph, []).append(el)
            elif ph == PHASE_PROMOTION:
                # Estimate promo game duration from moves if elapsed_seconds is missing in old game records
                moves = data.get('moves', [])
                num_m = len(moves) if isinstance(moves, list) else 0
                est_el = round(max(1.0, num_m * 0.15), 1)
                iter_game_times.setdefault(it, {}).setdefault(ph, []).append(est_el)

    metrics_by_iter = {m.get('iteration'): m for m in metrics if m.get('iteration') is not None}
    all_iters = sorted(set(iter_game_times.keys()) | set(metrics_by_iter.keys()))

    history = []
    sp_total_all = 0.0
    nn_total_all = 0.0
    rand_total_all = 0.0
    champ_total_all = 0.0
    all_time_total = 0.0

    last_iter_num = all_iters[-1] if all_iters else None

    for it in all_iters:
        m = metrics_by_iter.get(it, {})
        g_times = iter_game_times.get(it, {})

        sp_list = g_times.get(PHASE_SELF_PLAY, [])
        rand_list = g_times.get(PHASE_EVAL, [])
        champ_list = g_times.get(PHASE_PROMOTION, [])

        iter_elapsed = m.get('elapsed_seconds')

        # Eval time vs random bot
        if m.get('random_eval_seconds') is not None:
            rand_time = round(float(m['random_eval_seconds']), 1)
        elif rand_list:
            rand_time = round(sum(rand_list), 1)
        else:
            rand_time = 0.0

        # Champion gate match time
        if m.get('champion_gate_seconds') is not None:
            champ_time = round(float(m['champion_gate_seconds']), 1)
        elif champ_list:
            champ_time = round(sum(champ_list), 1)
        elif m.get('gate_win_rate') is not None or m.get('gate_promoted') is not None:
            # Historical fallback estimate if gate ran
            total_ref = float(iter_elapsed) if iter_elapsed else 100.0
            champ_time = round(min(total_ref * 0.20, max(12.0, total_ref * 0.12)), 1)
        else:
            champ_time = 0.0

        # NN training time
        if m.get('nn_train_seconds') is not None:
            nn_time = round(float(m['nn_train_seconds']), 1)
        elif m.get('policy_loss') is not None or m.get('loss') is not None or m.get('buffer_size', 0) > 0:
            # Historical fallback estimate if training ran
            total_ref = float(iter_elapsed) if iter_elapsed else 100.0
            nn_time = round(min(total_ref * 0.15, max(10.0, total_ref * 0.10)), 1)
        else:
            nn_time = 0.0

        if iter_elapsed is not None:
            total_time = round(float(iter_elapsed), 1)
        else:
            raw_sp = sum(sp_list) if sp_list else 0.0
            total_time = round(raw_sp + rand_time + champ_time + nn_time, 1)

        if m.get('self_play_seconds') is not None:
            sp_time = round(float(m['self_play_seconds']), 1)
        else:
            # Self-play takes up remaining iteration wall-clock time
            sp_time = max(0.0, round(total_time - nn_time - rand_time - champ_time, 1))

        sp_total_all += sp_time
        nn_total_all += nn_time
        rand_total_all += rand_time
        champ_total_all += champ_time
        all_time_total += total_time

        history.append({
            'iteration': it,
            'total_time': total_time,
            'self_play_time': sp_time,
            'nn_train_time': nn_time,
            'random_eval_time': rand_time,
            'champion_gate_time': champ_time,
        })

    # Extract last iteration summary values
    last_h = history[-1] if history else {}

    result['time_metrics'] = {
        'summary': {
            'sp_total_last': last_h.get('self_play_time'),
            'sp_total_all': round(sp_total_all, 1),
            'nn_total_last': last_h.get('nn_train_time'),
            'nn_total_all': round(nn_total_all, 1),
            'rand_total_last': last_h.get('random_eval_time'),
            'rand_total_all': round(rand_total_all, 1),
            'champ_total_last': last_h.get('champion_gate_time'),
            'champ_total_all': round(champ_total_all, 1),
            'last_iter_total': last_h.get('total_time'),
            'all_time_total': round(all_time_total, 1),
        },
        'history': history,
    }

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

    # Clamp everything against the single source of truth in param_bounds.py,
    # so the API can never store a value the sliders would refuse to produce.
    clean = sanitize_params(data)

    sp = clean.get('num_self_play_games')
    ev = clean.get('eval_games')
    sims = clean.get('num_simulations')
    cpuct = clean.get('c_puct')
    lr = clean.get('learning_rate')
    temp_thresh = clean.get('temperature_threshold')
    temp_init = clean.get('temperature_init')
    temp_final = clean.get('temperature_final')

    applied = {}

    # --- Gate + compute settings ---
    # Every phase re-reads these when it starts, so they take effect on the
    # very next iteration with no restart.
    for key in ('gate_enabled', 'gate_games', 'gate_threshold',
                'gate_simulations', 'gate_stall_warning',
                'num_parallel_workers'):
        val = clean.get(key)
        if val is not None:
            setattr(trainer.config.training, key, val)
            applied[key] = val

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


