"""
mercy_rule.py — when a bot should give up a game it is PLAYING.

Self-play has its own copy of this idea (`ai/self_play.py`), but that one is
built around protecting TRAINING DATA: it plays a share of games out to measure
wrong resignations, it gates on the whole board area so every sample-producing
position is recorded first, and it demands the winner agree before it stops.
None of that applies to a game a human is watching or playing. Here the only
question is the one the user actually asked: the game is decided, so stop
grinding out moves that change nothing.

The rule is the same shape as the model's own mercy settings, and takes its
defaults from them (`resign_threshold`, `resign_consecutive`,
`resign_min_move_factor`), so a bot resigns in play on the same evidence it
would resign on in training:

    * the bot's own search says the position is lost — root value <= -threshold
    * it has said so for `consecutive` of its own moves in a row, not once
    * the game is past `min_move_factor x board area`, so an early value spike
      in the opening can never end a game

`resign_both_sides` deliberately has no counterpart here. In training it guards
the outcome label by making the winner confirm; in play the loser's opinion is
the whole point, and against a human there is no second engine to ask.
"""

from typing import Optional


class MercyRule:
    """
    Tracks one player's own evaluations and says when it should resign.

    One instance per player per game: `observe()` is fed the root value of that
    player's own searches, from that player's point of view (+1 winning, -1
    lost), and returns True the first time the evidence is conclusive.
    """

    def __init__(self, enabled: bool = False, threshold: float = 0.90,
                 consecutive: int = 4, min_move_factor: float = 1.0,
                 board_size: int = 9):
        self.enabled = bool(enabled)
        self.threshold = float(threshold)
        self.consecutive = max(1, int(consecutive))
        self.min_move = int(round(float(min_move_factor) * board_size * board_size))
        self.streak = 0
        self.triggered_at: Optional[int] = None

    @classmethod
    def from_config(cls, config, board_size: int,
                    enabled: Optional[bool] = None) -> "MercyRule":
        """
        Build the rule from a model's `Config`.

        `enabled` overrides the model's own `resign_enabled`: that flag governs
        training, and a user who asks for shorter games on the Play page is not
        asking to change how the model trains.
        """
        training = config.training
        return cls(
            enabled=training.resign_enabled if enabled is None else enabled,
            threshold=training.resign_threshold,
            consecutive=training.resign_consecutive,
            min_move_factor=training.resign_min_move_factor,
            board_size=board_size,
        )

    def reset(self) -> None:
        """Start of a new game — evidence from the last one means nothing."""
        self.streak = 0
        self.triggered_at = None

    def observe(self, root_value: float, move_number: int) -> bool:
        """
        Feed one of this player's own root evaluations.

        Returns True when the player should resign now. Stays True-able only
        once per game: after it fires, the caller has ended the game.
        """
        if not self.enabled:
            return False

        if root_value is None:
            return False

        if float(root_value) <= -self.threshold:
            self.streak += 1
        else:
            self.streak = 0

        if move_number >= self.min_move and self.streak >= self.consecutive:
            self.triggered_at = move_number
            return True
        return False

    def describe(self) -> dict:
        """Settings summary, for API responses and game records."""
        return {
            'enabled': self.enabled,
            'threshold': self.threshold,
            'consecutive': self.consecutive,
            'min_move': self.min_move,
        }
