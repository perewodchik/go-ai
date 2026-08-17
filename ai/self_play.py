"""
self_play.py — Generate training games via self-play.

The bot plays against itself using MCTS + the current neural network.
Each game produces training data: (board_state, mcts_policy, game_outcome).

STORAGE POLICY:
Only 1 out of every N games is saved for replay in the web UI.
All games contribute training samples to the replay buffer regardless.
N is set in config (game_store_every_n, default 5).

MULTIPROCESSING NOTE:
Self-play games are independent, so `run_self_play_batch` plays them across the
shared process pool in `ai/worker_pool.py`. This is the only path — there is no
sequential branch; `num_workers=1` just means a pool of one.

That is safe on any GPU because the trainer hands this function its
CPU-resident champion (`device="cpu"`), not the training network — neither MPS
nor CUDA shares a context across processes, but nothing here touches one.
Worker count is `config.training.num_parallel_workers`, shared with the
promotion gate.
"""

import os
import json
import time
import random
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Callable
from datetime import datetime

from game.game_state import GameState, MOVE_PASS
from game.board import BLACK, WHITE, opponent
from game.features import encode_for_network
from game.scoring.base import get_scorer
from ai.mcts import MCTS
from ai.network import GoNetwork
from ai.game_store import PHASE_SELF_PLAY, save_game


# A single training sample: (encoded_state, mcts_policy, game_outcome)
TrainingSample = Tuple[torch.Tensor, np.ndarray, float]


def play_self_play_game(
    network: GoNetwork,
    board_size: int = 9,
    komi: float = 6.5,
    num_simulations: int = 200,
    c_puct: float = 1.5,
    temperature_threshold: int = 15,
    temperature_init: float = 1.0,
    temperature_final: float = 0.1,
    policy_target_temperature: float = 1.0,
    device: str = "cpu",
    max_moves: int = 200,
    fpu_reduction: float = 0.35,
    value_target_outcome_weight: float = 0.6,
    restrict_eye_fill: bool = False,
    restrict_self_atari: bool = False,
    self_atari_max_stones: int = 1,
    resign_enabled: bool = False,
    resign_threshold: float = 0.90,
    resign_consecutive: int = 4,
    resign_min_move_factor: float = 1.0,
    resign_both_sides: bool = True,
    resign_playout_fraction: float = 0.1,
) -> Tuple[List[TrainingSample], dict]:
    """
    Play one complete self-play game and collect training data.
    
    The network plays both sides. Each position is stored along with:
    - The MCTS visit-count policy (training target for policy head)
    - The eventual game outcome (training target for value head)
    
    Args:
        network: Current neural network.
        board_size: Board dimension.
        komi: Komi value.
        num_simulations: MCTS simulations per move.
        c_puct: Exploration constant.
        temperature_threshold: Move number after which temperature drops.
        temperature_init: Starting temperature (exploratory).
        temperature_final: Late-game temperature (greedy).
        device: Torch device.
        max_moves: Safety cap on game length.
        restrict_eye_fill: Hide own-two-eye-filling moves from the search (see
            game/eyes.py). Both colours are the same network here, so this
            applies to every move of the game.
        resign_enabled: Stop the game early once one side's own search says it
            is hopeless (the mercy rule — see the block comment below).
        resign_threshold: Root value at or below -threshold counts as hopeless.
            0.90 ≈ a 5% self-assessed win probability.
        resign_consecutive: Consecutive own moves that must agree before the
            game is stopped. Filters single-move value spikes.
        resign_min_move_factor: Earliest resignation move, as a multiple of the
            board area. 1.0 (the default) means no training sample can ever be
            lost to a resignation — see below.
        resign_both_sides: Also require the opponent's own last search to agree
            that it is winning. Guards against one side's value head being
            broken in a way that makes it resign won games.
        resign_playout_fraction: Share of games that ignore the resignation and
            play on, purely to measure how often it would have been wrong.

    Returns:
        (samples, game_record):
            samples: List of (state_tensor, mcts_policy, outcome) tuples.
                     Outcome is +1 if the player at that position won, -1 if lost.
            game_record: Dict with full game info for storage/replay.
    """
    state = GameState(board_size=board_size, komi=komi)
    mcts = MCTS(
        network=network,
        num_simulations=num_simulations,
        c_puct=c_puct,
        device=device,
        fpu_reduction=fpu_reduction,
        restrict_eye_fill=restrict_eye_fill,
        restrict_self_atari=restrict_self_atari,
        self_atari_max_stones=self_atari_max_stones,
    )

    # Collect training data as we play
    # Each entry: (state_tensor, mcts_policy, player_color)
    history = []
    move_list = []  # For the game record (replay in UI)
    win_rates = []  # Black's win probability (%) at each move, for the eval curve
    
    # Pass is disabled in early game (opening/midgame) to prevent premature
    # double-pass terminations that distort area scoring and flood replay buffer
    # with fake komi wins for White.
    #
    # The cutoff is JITTERED per game. A fixed boundary is a hard legal/illegal
    # discontinuity at a known move number, and the policy head learns it as a
    # feature rather than as strategy: P(pass) measured 0.000 before the cutoff
    # and 0.4-0.5 immediately after it, with the median first pass landing
    # exactly on the cutoff move. Randomizing the boundary removes the signal
    # the network was latching onto.
    base_min_pass = int(board_size * 3.3)  # 29 moves on 9x9
    min_pass_move = max(1, base_min_pass + random.randint(-4, 6))

    # ------------------------------------------------------------------
    # MERCY RULE (resignation)
    #
    # Once a side's own search has said it is lost for several moves running,
    # the rest of the game is a foregone conclusion that still costs a full
    # MCTS search per move. Stopping there is the cheapest way to buy more
    # iterations per hour.
    #
    # WHY THE DEFAULT CANNOT COST TRAINING DATA: samples are only kept for the
    # first `board_size²` moves (see max_training_moves below), so with
    # resign_min_move_factor = 1.0 every sample-producing position has already
    # been recorded before a resignation can fire. It also lands well past
    # base_min_pass, so the short-game discard below still passes. The tail
    # that gets cut was contributing nothing but wall-clock. Lower the factor
    # and that stops being true — you start trading data for speed.
    #
    # WHAT IT CAN STILL COST: the outcome label. A game that ends by
    # resignation is labelled from a PREDICTION, not from a played-out result,
    # and that label is what every retained sample of the game trains on. A
    # value head that wrongly despairs therefore gets to confirm its own
    # mistake. Three defences, in increasing order of cost:
    #   1. resign_consecutive  — ignore single-move value spikes.
    #   2. resign_both_sides   — the winner must agree that it is winning, so
    #      one broken value head is not enough to end a game.
    #   3. resign_playout_fraction — play a share of games out anyway and
    #      record whether the resignation would have been WRONG. That number
    #      (false_resign_rate) is the only real evidence that the threshold is
    #      set sanely; keep it under ~5%.
    # ------------------------------------------------------------------
    min_resign_move = int(round(resign_min_move_factor * board_size * board_size))
    # Chosen per game so the measurement sample is unbiased.
    resign_playout = resign_enabled and random.random() < resign_playout_fraction
    resign_streak = {BLACK: 0, WHITE: 0}
    last_root_value = {BLACK: None, WHITE: None}
    resigned_color = None        # Colour that actually resigned (game stopped)
    resign_move_num = None
    would_resign_color = None    # First trigger in a playout game (game continued)
    would_resign_move = None
    # The numbers behind the verdict, captured at the moment it fired. Without
    # them a resigned game in the review UI is an unexplained early stop — the
    # reader cannot tell a confident resignation from a marginal one.
    resign_evidence = None

    for move_num in range(max_moves):
        if state.is_over:
            break
        
        # Temperature schedule: smooth decay from temperature_init to temperature_final
        if temperature_threshold > 0:
            progress = min(1.0, move_num / float(temperature_threshold))
            temp = temperature_init + progress * (temperature_final - temperature_init)
        else:
            temp = temperature_final
        
        # Run MCTS to get action and policy. The opening pass ban is handed to
        # the search as a move NUMBER rather than a yes/no for this move, so it
        # is re-evaluated at every node: a line that reaches move min_pass_move
        # deep inside the tree can still end the game there.
        action, policy = mcts.search(
            state,
            temperature=temp,
            add_noise=True,  # Always add noise during training
            min_pass_move=min_pass_move,
            # The label does NOT follow the play temperature down. `temp`
            # decides which move this game plays; `policy` is what the network
            # is taught, and it stays the full visit distribution.
            target_temperature=policy_target_temperature,
        )

        # --- Mercy rule check (before the move is recorded or applied) ---
        # Running it here means a resigning side never contributes a move it
        # was never going to play, and `state.current_player` is still the
        # resigning colour when we break out.
        if resign_enabled:
            mover = state.current_player
            root_value = mcts.root_value
            last_root_value[mover] = root_value
            # The streak runs from the start of the game, not from
            # min_resign_move. A side that has been hopeless for forty moves
            # gets no further grace period once the gate opens — the gate is
            # there to protect training data, not to restart the evidence.
            if root_value <= -resign_threshold:
                resign_streak[mover] += 1
            else:
                resign_streak[mover] = 0

            opponent_value = last_root_value[opponent(mover)]
            triggered = (
                move_num >= min_resign_move
                and resign_streak[mover] >= resign_consecutive
                # The opponent has to have searched at least once for its
                # opinion to exist at all — on move 0 it never has.
                and (not resign_both_sides
                     or (opponent_value is not None
                         and opponent_value >= resign_threshold))
            )

            if triggered:
                evidence = {
                    'root_value': round(float(root_value), 4),
                    'opponent_value': (round(float(opponent_value), 4)
                                       if opponent_value is not None else None),
                    'streak': resign_streak[mover],
                    'threshold': resign_threshold,
                    'required_streak': resign_consecutive,
                    'both_sides': resign_both_sides,
                    'min_move': min_resign_move,
                }
                if resign_playout:
                    # Measurement game: note the verdict, then play on so we
                    # can find out whether it was right.
                    if would_resign_color is None:
                        would_resign_color = mover
                        would_resign_move = move_num
                        resign_evidence = evidence
                else:
                    resigned_color = mover
                    resign_move_num = move_num
                    resign_evidence = evidence
                    break

        # Store training data (before applying the move). Encoded for THIS
        # network, so the replay buffer and the network always agree on layout.
        state_tensor = encode_for_network(state, network)
        # Store the MCTS root evaluation alongside the position. It is the
        # per-position component of the value target (see value_target_outcome_weight)
        # and is already from this mover's perspective.
        history.append((state_tensor, policy, state.current_player, mcts.root_value))
        
        # Record move for game replay
        move_list.append({
            'color': int(state.current_player),
            'move': list(action),
            'move_num': move_num,
        })

        # Record the MCTS root evaluation as Black's win probability (%).
        # mcts.root_value is from the current player's perspective in [-1, +1];
        # flip to Black's perspective, then map [-1, +1] -> [0%, 100%].
        value_black = mcts.root_value if state.current_player == BLACK else -mcts.root_value
        win_rates.append(round(50.0 + 50.0 * value_black, 1))
        
        # Apply the move
        if action == MOVE_PASS:
            state.play_pass()
        else:
            success = state.play_move(action[0], action[1])
            if not success:
                # This shouldn't happen — MCTS only returns legal moves.
                # But if it does, pass instead.
                state.play_pass()
    
    if resigned_color is not None:
        # We broke out before playing, so the resigning colour is still to
        # move. play_resign() hands the win to the opponent, and
        # determine_winner() already honours state.resign_color.
        state.play_resign()
    elif not state.is_over:
        # Game didn't end naturally — force two passes
        state.play_pass()
        if not state.is_over:
            state.play_pass()

    # Determine winner
    scorer = get_scorer("chinese")
    winner_color, margin = scorer.determine_winner(state)
    black_score, white_score = scorer.score(state)

    if resigned_color is not None:
        # determine_winner() reports margin 0 for a resignation because the
        # margin is undefined. The dashboard reads `margin` as a signed score
        # difference, so give it the board score instead of a flat zero that
        # would render every resigned game as a draw.
        margin = abs(black_score - white_score)
    
    # Convert history into training samples with outcomes
    # Outcome is +1 if the player at that position won, -1 if lost, 0 for draw
    #
    # IMPORTANT: Only include samples from opening + midgame (first board_size²
    # moves). Late-game filling moves produce near-uniform MCTS policy targets.
    max_training_moves = board_size * board_size  # 81 for 9x9
    samples = []
    
    # Discard samples from abnormally short games to avoid corrupting the
    # replay buffer with premature game noise. Uses the un-jittered base so the
    # discard threshold stays stable from game to game.
    if len(move_list) >= base_min_pass:
        w = value_target_outcome_weight
        for move_idx, (state_tensor, policy, player_color, root_q) in enumerate(history):
            if move_idx >= max_training_moves:
                break  # Skip late-game filler positions

            if winner_color is None:
                outcome = 0.0  # Draw
            elif player_color == winner_color:
                outcome = 1.0  # This player won
            else:
                outcome = -1.0  # This player lost

            # Blend the game outcome with this position's own search evaluation.
            # Outcome alone gives every position in a game the SAME label, so
            # when one colour wins nearly every game the label becomes
            # predictable from the turn-colour plane and the value head stops
            # reading the board entirely. root_q varies position-to-position,
            # which keeps the target grounded in the actual position.
            target = w * outcome + (1.0 - w) * float(root_q)
            samples.append((state_tensor, policy, max(-1.0, min(1.0, target))))

    # --- Overfitting telemetry: how much of the search survives into the label ---
    # Measured on the samples that are actually TRAINED ON, not on every position
    # played, because the late-game filler positions dropped above have a
    # different entropy profile and would wash out the trend.
    #
    # These three numbers are what makes policy-target collapse visible. A
    # healthy 9x9 target at tau=1 sits around 1.5-2.5 nats with ~10-20 actions
    # holding mass; a target sharpened towards one-hot drops under 0.5 nats with
    # a support of 2-3, and the policy head that trains on it has nothing left to
    # learn but its own argmax.
    target_entropy = None
    target_support = None
    target_max_prob = None
    if samples:
        pol = np.asarray([s[1] for s in samples], dtype=np.float64)
        safe = np.clip(pol, 1e-12, None)
        target_entropy = float(np.mean(-(pol * np.log(safe)).sum(axis=1)))
        target_support = float(np.mean((pol > 1e-6).sum(axis=1)))
        target_max_prob = float(np.mean(pol.max(axis=1)))


    # Build game record for storage/replay
    game_record = {
        'board_size': board_size,
        'komi': komi,
        'moves': move_list,
        'num_moves': len(move_list),
        'win_rates': win_rates,
        'winner': int(winner_color) if winner_color else 0,
        'black_score': black_score,
        'white_score': white_score,
        'margin': margin,
        'timestamp': datetime.now().isoformat(),
        # --- Policy-target health (see the block that computes these) ---
        'target_entropy': target_entropy,
        'target_support': target_support,
        'target_max_prob': target_max_prob,
        'policy_target_temperature': policy_target_temperature,
        # --- Mercy rule bookkeeping ---
        # `resigned` marks a game whose outcome label is a PREDICTION rather
        # than a played-out result. `false_resign` is the payload: on playout
        # games the rule was overruled, so we know whether it would have
        # thrown away a game the "hopeless" side went on to win.
        'resigned': resigned_color is not None,
        'resign_color': int(resigned_color) if resigned_color is not None else None,
        'resign_move': resign_move_num,
        # The evaluation that triggered the rule, so the review UI can say WHY
        # the game stopped instead of just that it did. Also set on playout
        # games, where it explains a trigger the game went on to overrule.
        'resign_evidence': resign_evidence,
        'resign_playout': resign_playout,
        'would_resign_color': (int(would_resign_color)
                               if would_resign_color is not None else None),
        'would_resign_move': would_resign_move,
        'false_resign': (would_resign_color is not None
                         and winner_color == would_resign_color),
    }

    return samples, game_record


def build_self_play_task(
    network: GoNetwork,
    board_size: int,
    komi: float,
    num_simulations: int,
    c_puct: float,
    temperature_threshold: int,
    temperature_init: float,
    temperature_final: float,
    policy_target_temperature: float,
    device: str,
    fpu_reduction: float,
    value_target_outcome_weight: float,
    restrict_eye_fill: bool,
    restrict_self_atari: bool,
    self_atari_max_stones: int,
    resign_enabled: bool,
    resign_threshold: float,
    resign_consecutive: int,
    resign_min_move_factor: float,
    resign_both_sides: bool,
    resign_playout_fraction: float,
) -> dict:
    """
    Build the kwargs dict for one self-play worker.

    Workers are separate processes, so this dict is the ONLY thing a game sees —
    anything missing from it silently falls back to `play_self_play_game`'s
    signature defaults instead of the model's configuration. That is exactly how
    `temperature_init` / `temperature_final` came to be dead settings: fully
    plumbed through param_bounds, TrainingParams, Config.from_model and the
    live-tuning API, and then dropped here, so every model ever trained ran the
    hardcoded 1.0 -> 0.1 schedule regardless of what the UI showed.

    Keeping the construction in one named function means the task dict can be
    tested against the worker's signature without starting a process pool.
    """
    return {
        'network': network,
        'board_size': board_size,
        'komi': komi,
        'num_simulations': num_simulations,
        'c_puct': c_puct,
        'temperature_threshold': temperature_threshold,
        'temperature_init': temperature_init,
        'temperature_final': temperature_final,
        'policy_target_temperature': policy_target_temperature,
        'device': device,
        'fpu_reduction': fpu_reduction,
        'value_target_outcome_weight': value_target_outcome_weight,
        'restrict_eye_fill': restrict_eye_fill,
        'restrict_self_atari': restrict_self_atari,
        'self_atari_max_stones': self_atari_max_stones,
        'resign_enabled': resign_enabled,
        'resign_threshold': resign_threshold,
        'resign_consecutive': resign_consecutive,
        'resign_min_move_factor': resign_min_move_factor,
        'resign_both_sides': resign_both_sides,
        'resign_playout_fraction': resign_playout_fraction,
    }


def _play_worker(kwargs: dict) -> Tuple[List[TrainingSample], dict]:
    """
    Worker function for multiprocessing.

    The thread cap and the per-worker reseeding both live in
    `worker_pool._init_worker`, which runs once per process rather than once
    per game. This is kept as a belt-and-braces call because the function is
    also invoked directly by tests, outside any pool.
    """
    import torch
    import time

    # Prevent PyTorch from spawning many threads per worker, which causes CPU thrashing
    torch.set_num_threads(1)

    start_time = time.time()
    samples, record = play_self_play_game(**kwargs)
    record['elapsed_seconds'] = round(time.time() - start_time, 2)
    return samples, record


def run_self_play_batch(
    network: GoNetwork,
    num_games: int,
    board_size: int = 9,
    komi: float = 6.5,
    num_simulations: int = 200,
    c_puct: float = 1.5,
    temperature_threshold: int = 15,
    temperature_init: float = 1.0,
    temperature_final: float = 0.1,
    policy_target_temperature: float = 1.0,
    device: str = "cpu",
    game_store_every_n: int = 5,
    games_dir: str = "data/games",
    record_games: bool = True,
    iteration: int = 0,
    progress_callback: Optional[Callable] = None,
    fpu_reduction: float = 0.35,
    value_target_outcome_weight: float = 0.6,
    stop_checker: Optional[Callable[[], bool]] = None,
    num_workers: int = 4,
    restrict_eye_fill: bool = False,
    restrict_self_atari: bool = False,
    self_atari_max_stones: int = 1,
    resign_enabled: bool = False,
    resign_threshold: float = 0.90,
    resign_consecutive: int = 4,
    resign_min_move_factor: float = 1.0,
    resign_both_sides: bool = True,
    resign_playout_fraction: float = 0.1,
) -> List[TrainingSample]:
    """
    Run a batch of self-play games and return all training samples.
    
    Only stores 1 out of every `game_store_every_n` games to disk.
    All games contribute training data regardless.
    
    Args:
        network: Current neural network.
        num_games: Number of games to play.
        board_size: Board size.
        komi: Komi.
        num_simulations: MCTS simulations per move.
        c_puct: Exploration constant.
        temperature_threshold: Move number for temperature switch.
        temperature_init: Opening temperature (exploratory).
        temperature_final: Temperature after the decay window (greedy).
        policy_target_temperature: tau for the policy LABEL. Independent of the
            three settings above, which only decide which move gets played.
            1.0 = AlphaZero's pi = N/sum(N). See MCTS.search.
        device: Torch device.
        game_store_every_n: Save every Nth game for replay.
        games_dir: Directory to save game records.
        record_games: Write the full record of each stored game. False still
            indexes every game (so the training charts are unchanged) but skips
            the move lists, which are almost all of the bytes.
        iteration: Current training iteration (for filename).
        progress_callback: Called after each game with (game_num, total, game_record).
        stop_checker: Function returning True if training loop wants immediate stop.
        num_workers: Games to play concurrently (capped by num_games and CPU count).
        restrict_eye_fill: Forbid own-two-eye fills in every game of the batch.
        resign_*: Mercy rule settings, applied per game — see
            play_self_play_game. Deliberately NOT used by the promotion gate or
            the random-bot eval, where a wrong resignation would corrupt a
            measurement rather than just one training game.
    
    Returns:
        List of all training samples from all games.
    """
    all_samples = []
    os.makedirs(games_dir, exist_ok=True)

    import concurrent.futures

    from ai import worker_pool

    # Never more workers than there are games to play, or cores to play them on.
    num_workers = max(1, min(num_games, os.cpu_count() or 4, num_workers))
    
    tasks = [
        build_self_play_task(
            network=network,
            board_size=board_size,
            komi=komi,
            num_simulations=num_simulations,
            c_puct=c_puct,
            temperature_threshold=temperature_threshold,
            temperature_init=temperature_init,
            temperature_final=temperature_final,
            policy_target_temperature=policy_target_temperature,
            device=device,
            fpu_reduction=fpu_reduction,
            value_target_outcome_weight=value_target_outcome_weight,
            restrict_eye_fill=restrict_eye_fill,
            restrict_self_atari=restrict_self_atari,
            self_atari_max_stones=self_atari_max_stones,
            resign_enabled=resign_enabled,
            resign_threshold=resign_threshold,
            resign_consecutive=resign_consecutive,
            resign_min_move_factor=resign_min_move_factor,
            resign_both_sides=resign_both_sides,
            resign_playout_fraction=resign_playout_fraction,
        )
        for _ in range(num_games)
    ]
    
    completed_games = 0
    active_futures = {}  # future -> game_index
    next_task_idx = 0

    def _notify_progress(rec=None):
        if not progress_callback:
            return
        active_nums = sorted([idx + 1 for idx in active_futures.values()])
        try:
            progress_callback(completed_games, num_games, rec, active_games=active_nums, num_workers=num_workers)
        except TypeError:
            try:
                progress_callback(completed_games, num_games, rec, active_nums, num_workers)
            except TypeError:
                if rec is not None:
                    progress_callback(completed_games, num_games, rec)

    def _finish_game(game_index: int, samples, record) -> None:
        """Book one finished game: samples, storage, progress. Shared by both paths."""
        nonlocal completed_games
        record['iteration'] = iteration
        record['game_index'] = game_index

        all_samples.extend(samples)

        # Index every Nth game, and write its full record only when
        # recording is on — the index is what the charts read, so
        # turning recording off costs replays, not statistics.
        if completed_games % game_store_every_n == 0:
            save_game(games_dir, iteration, PHASE_SELF_PLAY, game_index,
                      record, store_full=record_games)

        completed_games += 1

    # The pool is shared and long-lived (see ai/worker_pool.py), so this block
    # deliberately does NOT use `with`: shutting it down here would throw away
    # every worker interpreter between the self-play phase and the gate.
    try:
        executor = worker_pool.get_executor(num_workers)
    except Exception as e:
        # A machine that cannot start a process pool at all still has to be
        # able to train. One game at a time, in this process.
        print(f"Worker pool unavailable ({e}); playing self-play games sequentially")
        for game_index in range(num_games):
            if stop_checker and stop_checker():
                break
            samples, record = _play_worker(tasks[game_index])
            _finish_game(game_index, samples, record)
            _notify_progress(record)
        return all_samples

    # Submit initial batch up to num_workers
    while next_task_idx < min(num_workers, num_games):
        f = executor.submit(_play_worker, tasks[next_task_idx])
        active_futures[f] = next_task_idx
        next_task_idx += 1

    # Initial notification of active games
    _notify_progress(None)

    while active_futures:
        if stop_checker and stop_checker():
            # Cancel what has not started; the games already in flight are
            # abandoned rather than waited on — their results are simply never
            # read. Nothing shuts the shared pool down here.
            for pending in active_futures:
                pending.cancel()
            active_futures.clear()
            break

        done, _ = concurrent.futures.wait(
            set(active_futures.keys()), timeout=0.1, return_when=concurrent.futures.FIRST_COMPLETED
        )

        for future in done:
            game_index = active_futures.pop(future)

            try:
                samples, record = future.result()
            except Exception as e:
                print(f"Worker failed: {e}")
                # Tagged so the trainer does not count a game that never
                # happened towards total_games or the mercy-rule denominator.
                samples, record = [], {'moves': [], 'winner': 0, 'num_moves': 0,
                                       'elapsed_seconds': 0, 'failed': True}

            _finish_game(game_index, samples, record)

            # Submit next task if available
            if next_task_idx < num_games and not (stop_checker and stop_checker()):
                new_f = executor.submit(_play_worker, tasks[next_task_idx])
                active_futures[new_f] = next_task_idx
                next_task_idx += 1

            # Report progress after game completion
            _notify_progress(record)

    return all_samples

