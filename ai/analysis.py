"""
analysis.py — the search's own move preferences, in the shape a board overlay
can draw.

MCTS already produces exactly what the "considered moves" heatmap wants: the
visit distribution over the root's children, which is the probability the bot
would play each move if it sampled at temperature 1. `top_moves()` trims that
distribution to the handful worth drawing — everything below `MIN_PROBABILITY`
is search noise and would put a dot on half the board — and `analyze_state()`
runs a search to get one for an arbitrary position.

The trimming is deliberately a RANGE, not a fixed k: a sharp position where the
search put 95% of its visits on one point should show a couple of circles, an
open one should show ten. `MIN_MOVES` keeps at least the top few even when they
are all below the threshold, so the overlay is never empty on a position the
bot does have an opinion about.
"""

from typing import List, Optional

from game.game_state import GameState, MOVE_PASS
from ai.mcts import MCTS

MIN_MOVES = 4
MAX_MOVES = 10
MIN_PROBABILITY = 0.01


def top_moves(policy, board_size: int,
              min_moves: int = MIN_MOVES,
              max_moves: int = MAX_MOVES,
              min_probability: float = MIN_PROBABILITY) -> List[dict]:
    """
    The best `min_moves`..`max_moves` actions of a visit distribution.

    `policy` is the vector MCTS.search returns: one entry per point plus a
    trailing pass. Entries come back sorted strongest first as
    `{'move': [row, col] | 'pass', 'probability': float}`.
    """
    if policy is None:
        return []

    pass_index = board_size * board_size
    ranked = sorted(range(len(policy)), key=lambda i: float(policy[i]), reverse=True)

    moves = []
    for rank, idx in enumerate(ranked[:max_moves]):
        probability = float(policy[idx])
        if probability <= 0.0:
            break
        if rank >= min_moves and probability < min_probability:
            break
        if idx == pass_index:
            moves.append({'move': 'pass', 'probability': probability})
        else:
            r, c = divmod(idx, board_size)
            moves.append({'move': [int(r), int(c)], 'probability': probability})

    return moves


def analyze_state(mcts: MCTS, state: GameState) -> dict:
    """
    Search `state` and report what the search considered.

    temperature=0.0 only decides which move would be PLAYED; the policy vector
    is built at the target temperature (1.0), so the reported probabilities are
    the raw visit shares regardless.
    """
    action, policy = mcts.search(state, temperature=0.0, add_noise=False)
    return {
        'move_number': state.move_number,
        'to_play': int(state.current_player),
        'best': ('pass' if tuple(action) == MOVE_PASS else [int(action[0]), int(action[1])]),
        'moves': top_moves(policy, state.board_size),
    }
