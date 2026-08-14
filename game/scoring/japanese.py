"""
japanese.py — Japanese scoring rules (territory scoring).

Japanese rules count:
  Score = surrounded_territory + prisoners_captured + komi (for white)

KEY DIFFERENCE FROM CHINESE:
- Stones on the board do NOT count as points.
- Captured stones (prisoners) DO count as points for the capturer.
- Dead stones at game end are counted as prisoners (requires agreement
  or playout to determine which stones are dead).

DEAD STONE HANDLING:
In this implementation, we assume that when the game ends (two passes),
all remaining stones are alive. For a more sophisticated approach, you'd
need a UI step where players mark dead stones, or use the score estimator
to guess. This is intentionally kept simple for the AI training loop.
"""

from typing import Tuple
from game.scoring.base import ScoringStrategy
from game.game_state import GameState
from game.board import BLACK, WHITE


class JapaneseScoring(ScoringStrategy):
    """
    Japanese territory scoring.
    
    black_score = black_territory + black_prisoners
    white_score = white_territory + white_prisoners + komi
    
    Prisoners = stones captured during the game (stored in game state).
    Territory = empty regions surrounded entirely by one color.
    """
    
    @property
    def name(self) -> str:
        return "Japanese (Territory)"
    
    def score(self, state: GameState) -> Tuple[float, float]:
        board = state.board
        
        # Count territory
        black_territory = 0
        white_territory = 0
        
        for region, border_colors in board.get_empty_regions():
            if border_colors == {BLACK}:
                black_territory += len(region)
            elif border_colors == {WHITE}:
                white_territory += len(region)
        
        # Add prisoners (stones captured BY each player during the game)
        black_prisoners = state.prisoners.get(BLACK, 0)
        white_prisoners = state.prisoners.get(WHITE, 0)
        
        black_score = float(black_territory + black_prisoners)
        white_score = float(white_territory + white_prisoners) + state.komi
        
        return black_score, white_score
