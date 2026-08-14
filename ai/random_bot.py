"""
random_bot.py — Baseline bot that plays uniformly random legal moves.

Used as:
1. The starting point before any training (iteration 0).
2. The Elo anchor — random bot is fixed at ~500 Elo (≈30 kyu).
3. A sanity check opponent to verify the game engine works.

The random bot can also play full games for generating initial training data,
though the data quality is very low (random games are mostly nonsense).
"""

import random
from typing import Tuple, Optional
from game.game_state import GameState, MOVE_PASS


class RandomBot:
    """
    Bot that plays uniformly random legal moves.
    
    Pass probability controls how often the bot passes instead of playing.
    Default is 0.05 (5%) — without this, random games can run forever
    on an empty board since random moves keep filling and capturing.
    """
    
    def __init__(self, pass_probability: float = 0.05):
        self.pass_probability = pass_probability
    
    def select_move(self, state: GameState) -> Tuple[int, int]:
        """
        Select a random legal move.
        
        Returns:
            (row, col) for a board move, or MOVE_PASS.
        """
        # Small chance of passing (ensures games eventually end)
        if random.random() < self.pass_probability:
            return MOVE_PASS
        
        legal_moves = state.get_legal_moves()
        
        if not legal_moves:
            return MOVE_PASS
        
        return random.choice(legal_moves)
    
    def play_game(self, state: GameState, max_moves: int = 500) -> GameState:
        """
        Play a complete game (both sides) using random moves.
        
        Args:
            state: Starting game state.
            max_moves: Safety limit to prevent infinite games.
        
        Returns:
            The final game state.
        """
        for _ in range(max_moves):
            if state.is_over:
                break
            
            move = self.select_move(state)
            if move == MOVE_PASS:
                state.play_pass()
            else:
                success = state.play_move(move[0], move[1])
                if not success:
                    # Shouldn't happen with legal moves, but just in case
                    state.play_pass()
        
        # If we hit max_moves, force game over
        if not state.is_over:
            state.play_pass()
            state.play_pass()
        
        return state
