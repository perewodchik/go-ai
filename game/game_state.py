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
        board_hash_history: Set of all board hashes seen (for superko).
        is_over: Whether the game has ended.
        winner: Winner color or None (if not yet determined).
        resign_color: Color that resigned, or None.
        restrict_eye_fill: Optional bot-only restriction. When True,
            get_legal_moves() hides moves that would fill one of the mover's own
            two eyes (see game/eyes.py). It deliberately does NOT affect
            is_legal()/play_move(), so humans, replays and stored games behave
            identically whether it is set or not.
    """

    def __init__(self, board_size: int = 9, komi: float = 6.5,
                 restrict_eye_fill: bool = False):
        self.board = Board(board_size)
        self.board_size = board_size
        self.komi = komi
        self.current_player = BLACK  # Black always plays first in Go
        self.move_history: List[Tuple[int, Tuple[int, int]]] = []
        self.ko_point: Optional[Tuple[int, int]] = None
        self.passes = 0
        self.prisoners = {BLACK: 0, WHITE: 0}  # Stones captured BY each player
        self.board_hash_history: set = {self.board.board_hash}
        self.is_over = False
        self.winner: Optional[int] = None
        self.resign_color: Optional[int] = None
        self.restrict_eye_fill = restrict_eye_fill

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
        # of copies, so this is what keeps the restriction in force at every
        # depth of the search rather than only at the root.
        new.restrict_eye_fill = self.restrict_eye_fill
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
        
        # Record board hash for superko
        self.board_hash_history.add(self.board.board_hash)
        
        # Switch player
        self.current_player = opponent(self.current_player)
        
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

        # Reset to initial state
        self.__init__(board_size=size, komi=komi, restrict_eye_fill=restrict)
        
        # Replay
        for color, move in history:
            if move == MOVE_PASS:
                self.play_pass()
            elif move == MOVE_RESIGN:
                self.play_resign()
            else:
                self.play_move(move[0], move[1])
        
        return True
    
    def encode_for_nn(self) -> torch.Tensor:
        """
        Encode the current board state as a tensor for the neural network.
        
        Returns a tensor of shape (num_planes, board_size, board_size).
        
        Planes (9 total):
        0: Current player's stones (1 where current player has a stone)
        1: Opponent's stones
        2: Current player's liberties == 1 (atari)
        3: Current player's liberties == 2
        4: Current player's liberties >= 3
        5: Opponent's liberties == 1
        6: Opponent's liberties == 2
        7: Opponent's liberties >= 3
        8: Ko point (1 at the ko point, 0 elsewhere)
        
        NOTE: We don't include a "turn color" plane because the encoding is
        always from the perspective of the current player — the network sees
        "my stones" vs "opponent stones" regardless of color. This is a common
        AlphaZero trick that halves the effective state space.
        """
        size = self.board_size
        planes = np.zeros((10, size, size), dtype=np.float32)
        
        opp = opponent(self.current_player)
        
        # Plane 0-1: Stone positions (from current player's perspective)
        planes[0] = (self.board.grid == self.current_player).astype(np.float32)
        planes[1] = (self.board.grid == opp).astype(np.float32)
        
        # Planes 2-7: Liberty counts for each group
        # Process current player's groups
        for group in self.board.get_all_groups(self.current_player):
            lib_count = self.board.liberty_count(group)
            plane_idx = 2 if lib_count == 1 else (3 if lib_count == 2 else 4)
            for r, c in group:
                planes[plane_idx, r, c] = 1.0
        
        # Process opponent's groups
        for group in self.board.get_all_groups(opp):
            lib_count = self.board.liberty_count(group)
            plane_idx = 5 if lib_count == 1 else (6 if lib_count == 2 else 7)
            for r, c in group:
                planes[plane_idx, r, c] = 1.0
        
        # Plane 8: Ko point
        if self.ko_point is not None:
            planes[8, self.ko_point[0], self.ko_point[1]] = 1.0
            
        # Plane 9: Turn color (all 1s if Black to play, all 0s if White to play)
        # This is CRITICAL for Go because of Komi. Without knowing if we are Black
        # or White, the network cannot accurately predict who is winning close games!
        if self.current_player == BLACK:
            planes[9, :, :] = 1.0
        
        return torch.from_numpy(planes)
    
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
