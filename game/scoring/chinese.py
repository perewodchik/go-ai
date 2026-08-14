"""
chinese.py — Chinese scoring rules (area scoring).

Chinese rules count:
  Score = stones_on_board + surrounded_territory + komi (for white)

This is simpler than Japanese scoring because you don't need to track
captured stones — only the final board position matters.

KEY DIFFERENCE FROM JAPANESE:
- Captured stones don't matter (they're already off the board).
- Stones on the board count as points (so filling your own territory
  doesn't lose points, unlike Japanese rules).
- This makes Chinese rules preferred for computer Go because the game
  outcome is the same regardless of how many "unnecessary" moves are played.
"""

from typing import Tuple
from game.scoring.base import ScoringStrategy
from game.game_state import GameState
from game.board import BLACK, WHITE


class ChineseScoring(ScoringStrategy):
    """
    Chinese area scoring.
    
    black_score = black_stones + black_territory
    white_score = white_stones + white_territory + komi
    
    Territory = empty regions surrounded entirely by one color.
    """
    
    @property
    def name(self) -> str:
        return "Chinese (Area)"
    
    def score(self, state: GameState) -> Tuple[float, float]:
        board = state.board
        
        # Count stones on the board
        black_stones = board.count_stones(BLACK)
        white_stones = board.count_stones(WHITE)
        
        # Count territory (empty regions surrounded by one color only)
        black_territory = 0
        white_territory = 0
        
        for region, border_colors in board.get_empty_regions():
            if border_colors == {BLACK}:
                # Entirely surrounded by black → black territory
                black_territory += len(region)
            elif border_colors == {WHITE}:
                # Entirely surrounded by white → white territory
                white_territory += len(region)
            # Mixed borders = dame (neutral) → no points for either
        
        black_score = float(black_stones + black_territory)
        white_score = float(white_stones + white_territory) + state.komi
        
        return black_score, white_score
