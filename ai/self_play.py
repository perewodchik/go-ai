"""
self_play.py — Generate training games via self-play.

The bot plays against itself using MCTS + the current neural network.
Each game produces training data: (board_state, mcts_policy, game_outcome).

STORAGE POLICY:
Only 1 out of every N games is saved for replay in the web UI.
All games contribute training samples to the replay buffer regardless.
N is set in config (game_store_every_n, default 5).

MULTIPROCESSING NOTE:
On M2 MacBook Air, we use a single process for self-play because:
1. MPS (Apple GPU) can't easily be shared across processes.
2. Sequential play with MPS inference is fast enough (~5-10 games/min on 9x9).
3. Multiprocessing adds complexity with diminishing returns on MPS.
If running on CPU-only, multiprocessing IS beneficial — uncomment the parallel code.
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
from game.board import BLACK, WHITE
from game.scoring.base import get_scorer
from ai.mcts import MCTS
from ai.network import GoNetwork


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
    device: str = "cpu",
    max_moves: int = 200,
    fpu_reduction: float = 0.35,
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
    
    for move_num in range(max_moves):
        if state.is_over:
            break
        
        # Temperature schedule: smooth decay from temperature_init to temperature_final
        if temperature_threshold > 0:
            progress = min(1.0, move_num / float(temperature_threshold))
            temp = temperature_init + progress * (temperature_final - temperature_init)
        else:
            temp = temperature_final
        
        # Disallow pass during opening to force stone development
        allow_pass = (move_num >= min_pass_move)
        
        # Run MCTS to get action and policy
        action, policy = mcts.search(
            state,
            temperature=temp,
            add_noise=True,  # Always add noise during training
            allow_pass=allow_pass,
        )
        
        # Store training data (before applying the move)
        state_tensor = state.encode_for_nn()
        history.append((state_tensor, policy, state.current_player))
        
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
    
    # If game didn't end naturally, force two passes
    if not state.is_over:
        state.play_pass()
        if not state.is_over:
            state.play_pass()
    
    # Determine winner
    scorer = get_scorer("chinese")
    winner_color, margin = scorer.determine_winner(state)
    black_score, white_score = scorer.score(state)
    
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
        for move_idx, (state_tensor, policy, player_color) in enumerate(history):
            if move_idx >= max_training_moves:
                break  # Skip late-game filler positions
            
            if winner_color is None:
                outcome = 0.0  # Draw
            elif player_color == winner_color:
                outcome = 1.0  # This player won
            else:
                outcome = -1.0  # This player lost
            
            samples.append((state_tensor, policy, outcome))
    
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
    }
    
    return samples, game_record


def _play_worker(kwargs: dict) -> Tuple[List[TrainingSample], dict]:
    """Worker function for multiprocessing."""
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
    device: str = "cpu",
    game_store_every_n: int = 5,
    games_dir: str = "data/games",
    iteration: int = 0,
    progress_callback: Optional[Callable] = None,
    fpu_reduction: float = 0.35,
    stop_checker: Optional[Callable[[], bool]] = None,
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
        device: Torch device.
        game_store_every_n: Save every Nth game for replay.
        games_dir: Directory to save game records.
        iteration: Current training iteration (for filename).
        progress_callback: Called after each game with (game_num, total, game_record).
        stop_checker: Function returning True if training loop wants immediate stop.
    
    Returns:
        List of all training samples from all games.
    """
    all_samples = []
    os.makedirs(games_dir, exist_ok=True)
    
    import concurrent.futures
    import multiprocessing as mp
    
    # Restrict to at most 4 parallel CPU workers to prevent thermal throttling on Macs
    num_workers = min(num_games, os.cpu_count() or 4, 4)
    
    tasks = []
    for _ in range(num_games):
        tasks.append({
            'network': network,
            'board_size': board_size,
            'komi': komi,
            'num_simulations': num_simulations,
            'c_puct': c_puct,
            'temperature_threshold': temperature_threshold,
            'device': device,
            'fpu_reduction': fpu_reduction,
        })
    
    completed_games = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_play_worker, task): i for i, task in enumerate(tasks)}
        pending = set(futures.keys())
        
        while pending:
            if stop_checker and stop_checker():
                executor.shutdown(wait=False, cancel_futures=True)
                break
            
            done, pending = concurrent.futures.wait(
                pending, timeout=0.1, return_when=concurrent.futures.FIRST_COMPLETED
            )
            
            for future in done:
                game_index = futures[future]
                
                try:
                    samples, record = future.result()
                except Exception as e:
                    print(f"Worker failed: {e}")
                    continue
                
                record['iteration'] = iteration
                record['game_index'] = game_index
                
                all_samples.extend(samples)
                
                # Store every Nth game for replay in the web UI
                if completed_games % game_store_every_n == 0:
                    game_filename = f"iter_{iteration:06d}_game_{game_index:04d}.json"
                    game_path = os.path.join(games_dir, game_filename)
                    with open(game_path, 'w') as f:
                        json.dump(record, f, indent=2)
                
                completed_games += 1
                
                # Report progress
                if progress_callback:
                    progress_callback(completed_games, num_games, record)
    
    return all_samples
