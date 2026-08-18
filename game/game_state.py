"""
game_state.py — Full game state with history tracking.

Wraps the Board and Rules into a single GameState object that tracks:
- Move history
- Pass counting (two consecutive passes → game over)
- Captured stones / prisoners
- Ko point
- Board hash history for superko
- Tensor encoding for neural network input

This is the object that gets passed around everywhere — to the AI, the web
interface, and the scoring system.
"""

import numpy as np
import torch
from typing import Optional, Tuple, List
from game.board import Board, EMPTY, BLACK, WHITE, opponent
from game import rules


# Move type constants
MOVE_PASS = (-1, -1)
MOVE_RESIGN = (-2, -2)


class GameState:
    """
    Complete game state for a Go game.
    
    Attributes:
        board: The current Board object.
        current_player: Who plays next (BLACK or WHITE).
        move_history: List of all moves played [(color, (row, col)), ...].
        ko_point: Current ko restriction, or None.
        passes: Number of consecutive passes (2 = game over).
        prisoners: Dict {BLACK: n, WHITE: n} — stones each player has captured.
        board_hash_history: Set of all superko keys seen — (board hash, player
            to move) pairs folded into one int by rules.situational_key(). The
            player matters: the same stones with the other side on move is a
            legal, different situation (see game/rules.py).
        is_over: Whether the game has ended.
        winner: Winner color or None (if not yet determined).
        resign_color: Color that resigned, or None.
        restrict_eye_fill: Optional bot-only restriction. When True,
            get_legal_moves() hides moves that would fill one of the mover's own
            two eyes (see game/eyes.py). It deliberately does NOT affect
            is_legal()/play_move(), so humans, replays and stored games behave
            identically whether it is set or not.
        restrict_self_atari: Optional bot-only restriction, same mechanism.
            When True, get_legal_moves() hides moves that capture nothing and
            leave the mover's group on one liberty (see game/self_atari.py).
        self_atari_max_stones: Sacrifice size that stays playable under that
            restriction.
    """

    def __init__(self, board_size: int = 9, komi: float = 6.5,
                 restrict_eye_fill: bool = False,
                 restrict_self_atari: bool = False,
                 self_atari_max_stones: int = 1):
        self.board = Board(board_size)
        self.board_size = board_size
        self.komi = komi
        self.current_player = BLACK  # Black always plays first in Go
        self.move_history: List[Tuple[int, Tuple[int, int]]] = []
        self.ko_point: Optional[Tuple[int, int]] = None
        self.passes = 0
        self.prisoners = {BLACK: 0, WHITE: 0}  # Stones captured BY each player
        self.board_hash_history: set = {
            rules.situational_key(self.board.board_hash, self.current_player)
        }
        self.is_over = False
        self.winner: Optional[int] = None
        self.resign_color: Optional[int] = None
        self.restrict_eye_fill = restrict_eye_fill
        self.restrict_self_atari = restrict_self_atari
        self.self_atari_max_stones = self_atari_max_stones

    def copy(self) -> 'GameState':
        """Deep copy the entire game state."""
        new = GameState.__new__(GameState)
        new.board = self.board.copy()
        new.board_size = self.board_size
        new.komi = self.komi
        new.current_player = self.current_player
        new.move_history = list(self.move_history)
        new.ko_point = self.ko_point
        new.passes = self.passes
        new.prisoners = dict(self.prisoners)
        new.board_hash_history = set(self.board_hash_history)
        new.is_over = self.is_over
        new.winner = self.winner
        new.resign_color = self.resign_color
        # Carried through every copy on purpose: MCTS builds its whole tree out
        # of copies, so this is what keeps the restrictions in force at every
        # depth of the search rather than only at the root.
        new.restrict_eye_fill = self.restrict_eye_fill
        new.restrict_self_atari = self.restrict_self_atari
        new.self_atari_max_stones = self.self_atari_max_stones
        return new
    
    @property
    def move_number(self) -> int:
        """Current move number (0-indexed)."""
        return len(self.move_history)
    
    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """
        Get all legal board moves for the current player.
        Does NOT include pass — that's always legal and handled separately.
        """
        return rules.get_legal_moves(
            self.board, self.current_player,
            self.ko_point, self.board_hash_history,
            restrict_eye_fill=self.restrict_eye_fill,
            restrict_self_atari=self.restrict_self_atari,
            self_atari_max_stones=self.self_atari_max_stones,
        )
    
    def is_legal(self, row: int, col: int) -> bool:
        """Check if a specific move is legal for the current player."""
        legal, _ = rules.is_legal_move(
            self.board, self.current_player, row, col,
            self.ko_point, self.board_hash_history
        )
        return legal
    
    def play_move(self, row: int, col: int) -> bool:
        """
        Play a stone at (row, col) for the current player.
        
        Returns True if the move was legal and applied, False otherwise.
        """
        if self.is_over:
            return False
        
        result = rules.apply_move(
            self.board, self.current_player, row, col,
            self.ko_point, self.board_hash_history
        )
        
        if not result.is_legal:
            return False
        
        # Record the move
        self.move_history.append((self.current_player, (row, col)))
        
        # Update prisoners — the CURRENT player captured these stones
        self.prisoners[self.current_player] += result.captured_count
        
        # Update ko point
        self.ko_point = result.ko_point
        
        # Reset pass counter (a stone move breaks the pass chain)
        self.passes = 0
        
        # Switch player
        self.current_player = opponent(self.current_player)

        # Record the superko key — the position AND whose turn it now is
        self.board_hash_history.add(
            rules.situational_key(self.board.board_hash, self.current_player)
        )
        
        return True
    
    def play_pass(self) -> None:
        """
        Current player passes their turn.
        Two consecutive passes end the game.
        """
        if self.is_over:
            return
        
        self.move_history.append((self.current_player, MOVE_PASS))
        self.passes += 1
        self.ko_point = None  # Ko is reset after a pass
        
        if self.passes >= 2:
            self.is_over = True
        
        self.current_player = opponent(self.current_player)
    
    def play_resign(self) -> None:
        """Current player resigns. Opponent wins."""
        if self.is_over:
            return
        
        self.resign_color = self.current_player
        self.winner = opponent(self.current_player)
        self.is_over = True
        self.move_history.append((self.current_player, MOVE_RESIGN))
    
    def undo_move(self) -> bool:
        """
        Undo the last move. This is expensive because we rebuild from scratch.
        Only used in the "easy mode" UI — never during training.
        
        Returns True if undo was successful, False if no moves to undo.
        """
        if not self.move_history:
            return False
        
        # Replay all moves except the last one
        history = self.move_history[:-1]
        komi = self.komi
        size = self.board_size
        restrict = self.restrict_eye_fill
        restrict_sa = self.restrict_self_atari
        sa_max = self.self_atari_max_stones

        # Reset to initial state
        self.__init__(board_size=size, komi=komi, restrict_eye_fill=restrict,
                      restrict_self_atari=restrict_sa,
                      self_atari_max_stones=sa_max)
        
        # Replay
        for color, move in history:
            if move == MOVE_PASS:
                self.play_pass()
            elif move == MOVE_RESIGN:
                self.play_resign()
            else:
                self.play_move(move[0], move[1])
        
        return True
    
    def encode_for_nn(self, features: Optional[str] = None) -> torch.Tensor:
        """
        Encode this position as network input, shape (planes, size, size).

        The plane layout is versioned — see game/features.py, which documents
        what each plane holds and why. `features` names the set; None means the
        project default (v1_10), which is what every model trained before the
        registry existed uses.

        Prefer `features.encode_for_network(state, network)` when a network is
        in hand: it reads the layout off the network, so a position can never be
        paired with the wrong encoding.
        """
        from game.features import encode_state
        return encode_state(self, features)

    def to_dict(self) -> dict:
        """
        Serialize game state to a JSON-compatible dict.
        Used for web API responses and game storage.
        """
        return {
            'board_size': self.board_size,
            'komi': self.komi,
            'grid': self.board.grid.tolist(),
            'current_player': self.current_player,
            'move_history': [
                {'color': c, 'move': list(m)} for c, m in self.move_history
            ],
            'ko_point': list(self.ko_point) if self.ko_point else None,
            'passes': self.passes,
            'prisoners': self.prisoners,
            'is_over': self.is_over,
            'winner': self.winner,
            'move_number': self.move_number,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'GameState':
        """
        Reconstruct a GameState from a dict (e.g., loaded from JSON).
        Replays all moves to rebuild internal state correctly.
        """
        state = cls(board_size=data['board_size'], komi=data['komi'])
        for entry in data['move_history']:
            move = tuple(entry['move'])
            if move == MOVE_PASS:
                state.play_pass()
            elif move == MOVE_RESIGN:
                state.play_resign()
            else:
                state.play_move(move[0], move[1])
        return state
    
    def __repr__(self) -> str:
        color_name = "Black" if self.current_player == BLACK else "White"
        status = "OVER" if self.is_over else f"Move {self.move_number + 1}"
        return (
            f"GoGame(size={self.board_size}, {status}, "
            f"to_play={color_name}, captures=B:{self.prisoners[BLACK]}/W:{self.prisoners[WHITE]})\n"
            f"{self.board}"
        )
