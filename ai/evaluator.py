"""
evaluator.py — Elo rating and strength estimation.

Evaluates the current model by playing matches against:
1. The random bot (Elo anchor at ~500)
2. Previous checkpoints (to track improvement)

Elo is computed using the standard formula and mapped to kyu/dan ranks.
"""

import math
import numpy as np
from typing import Callable, Optional
from game.game_state import GameState, MOVE_PASS
from game.board import BLACK, WHITE
from game.scoring.base import get_scorer
from ai.mcts import MCTS
from ai.random_bot import RandomBot
from ai.network import GoNetwork
from ai.game_store import PHASE_EVAL, PHASE_PROMOTION, save_game


def _eval_worker(kwargs: dict) -> int:
    """Worker function for multiprocessing evaluation. Returns 1 if network won, else 0."""
    import torch
    torch.set_num_threads(1)
    
    network = kwargs['network']
    board_size = kwargs['board_size']
    komi = kwargs['komi']
    num_simulations = kwargs['num_simulations']
    device = kwargs['device']
    game_idx = kwargs['game_idx']
    
    import time
    start_time = time.time()

    mcts = MCTS(network=network, num_simulations=num_simulations, device=device)
    random_bot = RandomBot(pass_probability=0.05)
    scorer = get_scorer("chinese")
    
    state = GameState(board_size=board_size, komi=komi)
    network_color = BLACK if game_idx % 2 == 0 else WHITE

    move_list = []
    
    for move_num in range(board_size * board_size * 3):
        if state.is_over:
            break

        if state.current_player == network_color:
            action, _ = mcts.search(state, temperature=0.1, add_noise=False)
        else:
            action = random_bot.select_move(state)
            
        move_list.append({
            'color': int(state.current_player),
            'move': list(action) if action != MOVE_PASS else [-1, -1],
            'move_num': move_num,
        })

        if action == MOVE_PASS:
            state.play_pass()
        else:
            if not state.play_move(action[0], action[1]):
                state.play_pass()

    if not state.is_over:
        state.play_pass()
        if not state.is_over:
            state.play_pass()

    winner, margin = scorer.determine_winner(state)
    black_score, white_score = scorer.score(state)
    
    # Save the evaluation game to disk
    from datetime import datetime

    games_dir = kwargs.get('games_dir')
    iteration = kwargs.get('iteration', 0)

    if games_dir:
        elapsed = round(time.time() - start_time, 2)
        game_record = {
            'board_size': board_size,
            'komi': komi,
            'moves': move_list,
            'num_moves': len(move_list),
            'winner': int(winner) if winner else 0,
            'black_score': black_score,
            'white_score': white_score,
            'margin': margin,
            'timestamp': datetime.now().isoformat(),
            'is_eval': True,
            'network_color': int(network_color),
            'elapsed_seconds': elapsed,
        }
        save_game(games_dir, iteration, PHASE_EVAL, game_idx, game_record)

    return 1 if winner == network_color else 0


def evaluate_against_random(
    network: GoNetwork,
    board_size: int = 9,
    komi: float = 6.5,
    num_simulations: int = 50,
    num_games: int = 20,
    device: str = "cpu",
    iteration: int = 0,
    games_dir: Optional[str] = None,
    stop_checker: Optional[Callable[[], bool]] = None,
    num_workers: int = 4,
) -> float:
    """
    Play games against the random bot and return win rate.

    Uses fewer MCTS simulations than training for speed.
    The network alternates playing Black and White.

    Returns:
        Win rate as a float between 0.0 and 1.0.
    """
    if num_games <= 0:
        return 0.5

    import concurrent.futures
    import os

    num_workers = max(1, min(num_games, os.cpu_count() or 4, num_workers))
    tasks = []
    for game_idx in range(num_games):
        tasks.append({
            'network': network,
            'board_size': board_size,
            'komi': komi,
            'num_simulations': num_simulations,
            'device': device,
            'game_idx': game_idx,
            'iteration': iteration,
            'games_dir': games_dir,
        })
        
    wins = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_eval_worker, task) for task in tasks]
        pending = set(futures)
        
        while pending:
            if stop_checker and stop_checker():
                executor.shutdown(wait=False, cancel_futures=True)
                break
            
            done, pending = concurrent.futures.wait(
                pending, timeout=0.1, return_when=concurrent.futures.FIRST_COMPLETED
            )
            
            for future in done:
                try:
                    wins += future.result()
                except Exception as e:
                    print(f"Eval worker failed: {e}")

    return wins / max(num_games, 1)


def _play_gate_game(
    current_network: GoNetwork,
    opponent_network: GoNetwork,
    board_size: int,
    komi: float,
    num_simulations: int,
    device: str,
    game_idx: int,
    iteration: int,
    games_dir: Optional[str],
) -> int:
    """
    Play one candidate-vs-champion game and record it. Returns 1 if the
    candidate won, else 0.

    Colours alternate with `game_idx` so a network that only plays one side
    well cannot pass the gate.
    """
    import time
    from datetime import datetime

    game_start = time.time()
    current_mcts = MCTS(network=current_network, num_simulations=num_simulations, device=device)
    opponent_mcts = MCTS(network=opponent_network, num_simulations=num_simulations, device=device)
    scorer = get_scorer("chinese")

    state = GameState(board_size=board_size, komi=komi)
    current_color = BLACK if game_idx % 2 == 0 else WHITE
    move_list = []

    for move_num in range(board_size * board_size * 3):
        if state.is_over:
            break
        if state.current_player == current_color:
            action, _ = current_mcts.search(state, temperature=0.1, add_noise=False)
        else:
            action, _ = opponent_mcts.search(state, temperature=0.1, add_noise=False)

        move_list.append({
            'color': int(state.current_player),
            'move': list(action) if action != MOVE_PASS else [-1, -1],
            'move_num': move_num,
        })

        if action == MOVE_PASS:
            state.play_pass()
        else:
            if not state.play_move(action[0], action[1]):
                state.play_pass()

    if not state.is_over:
        state.play_pass()
        if not state.is_over:
            state.play_pass()

    winner, margin = scorer.determine_winner(state)
    candidate_won = (winner == current_color)

    if games_dir:
        elapsed = round(time.time() - game_start, 2)
        black_score, white_score = scorer.score(state)
        champion_color = WHITE if current_color == BLACK else BLACK
        save_game(games_dir, iteration, PHASE_PROMOTION, game_idx, {
            'board_size': board_size,
            'komi': komi,
            'moves': move_list,
            'num_moves': len(move_list),
            'winner': int(winner) if winner else 0,
            'black_score': black_score,
            'white_score': white_score,
            'margin': margin,
            'timestamp': datetime.now().isoformat(),
            'is_promotion': True,
            'candidate_color': int(current_color),
            'champion_color': int(champion_color),
            'candidate_won': candidate_won,
            'elapsed_seconds': elapsed,
        })

    return 1 if candidate_won else 0


def _gate_worker(kwargs: dict) -> int:
    """Worker for parallel promotion-gate games. Returns 1 if candidate won."""
    import torch
    torch.set_num_threads(1)
    return _play_gate_game(**kwargs)


def evaluate_against_checkpoint(
    current_network: GoNetwork,
    opponent_network: GoNetwork,
    board_size: int = 9,
    komi: float = 6.5,
    num_simulations: int = 50,
    num_games: int = 10,
    device: str = "cpu",
    iteration: int = 0,
    games_dir: Optional[str] = None,
    num_workers: int = 4,
    stop_checker: Optional[Callable[[], bool]] = None,
) -> float:
    """
    Play the promotion-gate match between the candidate (`current_network`) and
    the reigning champion (`opponent_network`). Returns the candidate's win rate.

    Games are independent, so they are played across a process pool exactly the
    way `evaluate_against_random` does — the gate is the most expensive phase of
    an iteration (two MCTS searches per move instead of one), so running it
    sequentially was leaving most of the machine idle. `num_workers=1` forces
    the old in-process path, which is also the automatic fallback if the pool
    cannot be created.

    Every game is recorded to `games_dir` (when given) under the iteration's
    `promotion/` directory, tagged with which side each network played, so the
    match that decided a promotion can be replayed move by move afterwards.
    """
    if num_games <= 0:
        return 0.0

    import concurrent.futures
    import os

    def _task(game_idx: int) -> dict:
        return {
            'current_network': current_network,
            'opponent_network': opponent_network,
            'board_size': board_size,
            'komi': komi,
            'num_simulations': num_simulations,
            'device': device,
            'game_idx': game_idx,
            'iteration': iteration,
            'games_dir': games_dir,
        }

    workers = max(1, min(num_games, num_workers, os.cpu_count() or 4))

    if workers > 1:
        try:
            wins = 0
            completed = 0
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                pending = {executor.submit(_gate_worker, _task(i)) for i in range(num_games)}
                while pending:
                    if stop_checker and stop_checker():
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    done, pending = concurrent.futures.wait(
                        pending, timeout=0.1,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        try:
                            wins += future.result()
                            completed += 1
                        except Exception as e:
                            print(f"Gate worker failed: {e}")
            # Score against the games that actually finished, so an interrupted
            # or partially failed match is not read as a pile of losses.
            return wins / max(completed, 1)
        except Exception as e:
            print(f"Gate pool unavailable ({e}); falling back to sequential")

    wins = 0
    played = 0
    for game_idx in range(num_games):
        if stop_checker and stop_checker():
            break
        wins += _play_gate_game(**_task(game_idx))
        played += 1

    return wins / max(played, 1)


def compute_elo_update(current_elo: float, opponent_elo: float, win_rate: float) -> float:
    """
    Update Elo rating based on win rate against an opponent.
    
    Uses standard Elo formula:
        expected = 1 / (1 + 10^((opponent - current) / 400))
        new_elo = current + K * (actual - expected)
    
    K-factor is 32 (standard for developing players).
    """
    K = 32
    expected = 1.0 / (1.0 + math.pow(10, (opponent_elo - current_elo) / 400))
    new_elo = current_elo + K * (win_rate - expected)
    return max(0, new_elo)  # Floor at 0
