"""
base.py — Abstract scoring strategy interface.

Uses the Strategy pattern so you can swap Chinese ↔ Japanese rules
by changing a single config value. Both strategies share the same interface.

Usage:
    scorer = get_scorer("chinese")  # or "japanese"
    black_score, white_score = scorer.score(game_state)
    winner = scorer.determine_winner(game_state)
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
from game.game_state import GameState
from game.board import BLACK, WHITE


class ScoringStrategy(ABC):
    """
    Abstract base class for Go scoring rules.
    
    Subclasses must implement `score()` which returns raw scores
    (komi already applied to white).
    """
    
    @abstractmethod
    def score(self, state: GameState) -> Tuple[float, float]:
        """
        Calculate the final score for both players.
        
        Args:
            state: The final game state (game should be over).
        
        Returns:
            (black_score, white_score) — komi is added to white's score.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this scoring method."""
        pass
    
    def determine_winner(self, state: GameState) -> Tuple[Optional[int], float]:
        """
        Determine the winner and margin of victory.
        
        Returns:
            (winner_color, margin) — winner is BLACK or WHITE.
            If margin is 0, it's a draw (jigo), winner is None.
        """
        # If someone resigned, they lose
        if state.resign_color is not None:
            winner = WHITE if state.resign_color == BLACK else BLACK
            return winner, 0.0  # Margin is undefined for resignation
        
        black_score, white_score = self.score(state)
        margin = abs(black_score - white_score)
        
        if black_score > white_score:
            return BLACK, margin
        elif white_score > black_score:
            return WHITE, margin
        else:
            return None, 0.0  # Jigo (draw) — extremely rare with 0.5 komi


def get_scorer(method: str = "chinese") -> ScoringStrategy:
    """
    Factory function to get a scoring strategy by name.
    
    Args:
        method: "chinese" or "japanese"
    
    Returns:
        ScoringStrategy instance.
    """
    # Lazy imports to avoid circular dependencies
    if method == "chinese":
        from game.scoring.chinese import ChineseScoring
        return ChineseScoring()
    elif method == "japanese":
        from game.scoring.japanese import JapaneseScoring
        return JapaneseScoring()
    else:
        raise ValueError(f"Unknown scoring method: {method}. Use 'chinese' or 'japanese'.")
