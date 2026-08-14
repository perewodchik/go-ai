"""
evaluator.py — Elo rating and strength estimation.

Evaluates the current model by playing matches against:
1. The random bot (Elo anchor at ~500)
2. Previous checkpoints (to track improvement)

Elo is computed using the standard formula and mapped to kyu/dan ranks.
"""

import math
import numpy as np
from typing import Optional
from game.game_state import GameState, MOVE_PASS
from game.board import BLACK, WHITE
from game.scoring.base import get_scorer
from ai.mcts import MCTS
from ai.random_bot import RandomBot
from ai.network import GoNetwork


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
    import json
    import os
    from datetime import datetime
    
    games_dir = kwargs.get('games_dir')
    iteration = kwargs.get('iteration', 0)
    
    if games_dir:
        os.makedirs(games_dir, exist_ok=True)
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
            'iteration': iteration,
            'game_index': game_idx,
            'is_eval': True,
            'network_color': int(network_color),
        }
        
        game_filename = f"iter_{iteration:06d}_eval_{game_idx:04d}.json"
        game_path = os.path.join(games_dir, game_filename)
        with open(game_path, 'w') as f:
            json.dump(game_record, f, indent=2)

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
) -> float:
    """
    Play games against the random bot and return win rate.
    
    Uses fewer MCTS simulations than training for speed.
    The network alternates playing Black and White.
    
    Returns:
        Win rate as a float between 0.0 and 1.0.
    """
    import concurrent.futures
    import os
    
    num_workers = min(num_games, os.cpu_count() or 4, 4)
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


def evaluate_against_checkpoint(
    current_network: GoNetwork,
    opponent_network: GoNetwork,
    board_size: int = 9,
    komi: float = 6.5,
    num_simulations: int = 50,
    num_games: int = 10,
    device: str = "cpu",
) -> float:
    """Play games between two networks. Returns win rate of current_network."""
    current_mcts = MCTS(network=current_network, num_simulations=num_simulations, device=device)
    opponent_mcts = MCTS(network=opponent_network, num_simulations=num_simulations, device=device)
    scorer = get_scorer("chinese")
    wins = 0

    for game_idx in range(num_games):
        state = GameState(board_size=board_size, komi=komi)
        current_color = BLACK if game_idx % 2 == 0 else WHITE

        for _ in range(board_size * board_size * 3):
            if state.is_over:
                break
            if state.current_player == current_color:
                action, _ = current_mcts.search(state, temperature=0.1, add_noise=False)
            else:
                action, _ = opponent_mcts.search(state, temperature=0.1, add_noise=False)

            if action == MOVE_PASS:
                state.play_pass()
            else:
                if not state.play_move(action[0], action[1]):
                    state.play_pass()

        if not state.is_over:
            state.play_pass()
            if not state.is_over:
                state.play_pass()

        winner, _ = scorer.determine_winner(state)
        if winner == current_color:
            wins += 1

    return wins / max(num_games, 1)


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
