"""
match.py — Run a series of games between two `Player`s and rate the result.

This is the engine behind "Bot vs Bot" on the Play page. It is deliberately
independent of Flask, of the Trainer, and of what kind of player is on either
side: it drives the `ai.players.Player` interface only, so a local model, the
random bot and (later) an OGS opponent all run through the same loop.

Compared with the promotion gate in `ai/evaluator.py`, which plays the same kind
of match, this runner is built to be WATCHED rather than to be fast:

* games run one move at a time on a single thread, publishing a snapshot after
  every move, so the browser can render the board live;
* an optional per-move delay makes a match watchable at human speed;
* it can be stopped between moves;
* both sides' ratings move after each game (`compute_pairwise_elo`), so a
  series tells you how the two bots actually compare.

The gate stays parallel and headless — it is a measurement inside the training
loop, and nobody watches it move by move.
"""

import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, List, Optional, Sequence, Tuple

from game.game_state import GameState, MOVE_PASS, MOVE_RESIGN
from game.board import BLACK, WHITE
from game.scoring.base import get_scorer
from ai.evaluator import compute_pairwise_elo
from ai.game_store import save_match_game
from ai.players import Player


# Match lifecycle states, as reported to the browser.
STATUS_PENDING = 'pending'
STATUS_RUNNING = 'running'
STATUS_FINISHED = 'finished'
STATUS_STOPPED = 'stopped'
STATUS_ERROR = 'error'


@dataclass
class MatchConfig:
    """Everything about a match that is not a player."""
    board_size: int = 9
    komi: float = 6.5
    scoring_method: str = 'chinese'
    num_games: int = 4
    # Swap colours every other game, so a bot that only plays one side well
    # cannot look stronger than it is.
    alternate_colors: bool = True
    # Seconds to wait after each move. 0 = as fast as the engines can play.
    move_delay: float = 0.0
    # Safety cap on game length; defaults to 3x the board area, matching the
    # gate and eval matches.
    max_moves: Optional[int] = None
    record_games: bool = True
    update_ratings: bool = True
    rating_k: float = 16.0
    # Evaluate every position with one of the players' networks so the live
    # win-rate curve (and the saved record) has the same series the review page
    # charts. One extra forward pass per move — negligible next to MCTS.
    compute_win_rates: bool = True

    def move_limit(self) -> int:
        return self.max_moves or (self.board_size * self.board_size * 3)


@dataclass
class GameOutcome:
    """Result of one game in the series."""
    index: int
    black_name: str
    white_name: str
    winner: int                    # BLACK, WHITE, or 0 for a draw
    margin: float = 0.0
    resigned_by: Optional[int] = None
    num_moves: int = 0
    black_score: float = 0.0
    white_score: float = 0.0
    elapsed: float = 0.0
    unfinished: bool = False
    saved_as: List[str] = field(default_factory=list)
    # Rating movement caused by this game, keyed by player slot ('a'/'b').
    rating_delta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'index': self.index,
            'black_name': self.black_name,
            'white_name': self.white_name,
            'winner': self.winner,
            'margin': round(self.margin, 1),
            'resigned_by': self.resigned_by,
            'num_moves': self.num_moves,
            'black_score': round(self.black_score, 1),
            'white_score': round(self.white_score, 1),
            'elapsed': round(self.elapsed, 1),
            'unfinished': self.unfinished,
            'saved_as': list(self.saved_as),
            'rating_delta': {k: round(v, 1) for k, v in self.rating_delta.items()},
        }


class MatchRunner:
    """
    Plays `config.num_games` games between two players, publishing a live
    snapshot as it goes.

    Player A and player B keep their identity across the series; which COLOUR
    each one has alternates per game when `alternate_colors` is set, so results
    are reported per player ('a'/'b'), never per colour.

    Args:
        match_id: Opaque id used by the web layer to address this match.
        player_a / player_b: the two contestants. A takes Black in game 0.
        config: MatchConfig.
        record_dirs: `games/` directories to write each finished game into —
            one per participating local model, so a match shows up in the
            review page of every model that took part. Empty = don't save.
        on_update: called with the fresh snapshot after every move and every
            game boundary. Must not block for long (it runs on the match thread).
    """

    def __init__(self, match_id: str, player_a: Player, player_b: Player,
                 config: MatchConfig, record_dirs: Optional[Sequence[str]] = None,
                 on_update: Optional[Callable[[dict], None]] = None,
                 name: Optional[str] = None):
        self.match_id = match_id
        self.player_a = player_a
        self.player_b = player_b
        self.config = config
        self.record_dirs = list(record_dirs or [])
        self.on_update = on_update
        self.name = name or f"{player_a.name} vs {player_b.name}"

        self.scorer = get_scorer(config.scoring_method)
        self._lock = threading.RLock()
        self._stop_requested = False
        self._paused = False

        self.status = STATUS_PENDING
        self.error: Optional[str] = None
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None

        self.outcomes: List[GameOutcome] = []
        self.wins = {'a': 0, 'b': 0, 'draw': 0}
        self.start_rating = {'a': player_a.rating, 'b': player_b.rating}

        # Live game state, replaced wholesale under the lock on every move.
        self._state: Optional[GameState] = None
        self._current_index = 0
        self._current_colors = {'a': BLACK, 'b': WHITE}
        self._last_move: Optional[dict] = None
        self._win_rates: List[float] = []
        self._scores = {'black': 0.0, 'white': 0.0}
        self._version = 0

        # Whichever side has a network provides the win-rate evaluation. Using
        # ONE network for the whole series keeps the curve on a single scale —
        # alternating between two networks would make it jump between two
        # different opinions of the same position.
        self._analysis_network = None
        for player in (player_a, player_b):
            net = getattr(player, 'network', None)
            if net is not None:
                self._analysis_network = net
                break

    # --- Control ----------------------------------------------------------

    def stop(self) -> None:
        """Request a stop. Takes effect before the next move."""
        with self._lock:
            self._stop_requested = True

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self._paused = bool(paused)

    @property
    def is_active(self) -> bool:
        return self.status in (STATUS_PENDING, STATUS_RUNNING)

    def current_state(self) -> Optional[GameState]:
        """A copy of the position on the board right now (for score estimation)."""
        with self._lock:
            return self._state.copy() if self._state is not None else None

    # --- Snapshot ---------------------------------------------------------

    def snapshot(self) -> dict:
        """The full live view of the match. Safe to call from any thread."""
        with self._lock:
            state_dict = self._state.to_dict() if self._state is not None else None
            elapsed = 0.0
            if self.started_at:
                elapsed = (self.finished_at or time.time()) - self.started_at

            return {
                'match_id': self.match_id,
                'name': self.name,
                'status': self.status,
                'error': self.error,
                'paused': self._paused,
                'version': self._version,
                'board_size': self.config.board_size,
                'komi': self.config.komi,
                'ruleset': self.config.scoring_method,
                'num_games': self.config.num_games,
                'games_completed': len(self.outcomes),
                'current_game': self._current_index + 1,
                'move_delay': self.config.move_delay,
                'elapsed': round(elapsed, 1),
                'state': state_dict,
                'last_move': self._last_move,
                'win_rates': list(self._win_rates),
                'scores': dict(self._scores),
                'colors': dict(self._current_colors),
                'series': {
                    'a': self.wins['a'],
                    'b': self.wins['b'],
                    'draw': self.wins['draw'],
                },
                'players': {
                    'a': self._player_view('a', self.player_a),
                    'b': self._player_view('b', self.player_b),
                },
                'rated': self._ratings_apply(),
                'results': [o.to_dict() for o in self.outcomes],
            }

    def _player_view(self, slot: str, player: Player) -> dict:
        info = player.describe()
        info.update({
            'slot': slot,
            'color': self._current_colors.get(slot),
            'start_rating': round(self.start_rating[slot], 1),
            'rating_delta': round(player.rating - self.start_rating[slot], 1),
        })
        return info

    def _ratings_apply(self) -> bool:
        """
        Whether this match moves any rating.

        A model playing ITSELF (same rating key both sides) is excluded: the
        result is decided by colour and luck, and feeding it into Elo would add
        noise to a number that is supposed to mean something. Matches where the
        opponent is a fixed anchor still count — only the anchor stays put.
        """
        if not self.config.update_ratings:
            return False
        if self.player_a.rating_key and self.player_a.rating_key == self.player_b.rating_key:
            return False
        return not (self.player_a.rating_is_fixed and self.player_b.rating_is_fixed)

    def _publish(self) -> None:
        with self._lock:
            self._version += 1
        if self.on_update:
            try:
                self.on_update(self.snapshot())
            except Exception:
                # A broken listener must never take the match down with it.
                pass

    # --- Running ----------------------------------------------------------

    def run(self) -> dict:
        """
        Play the whole series. Blocking — callers run it on a thread.

        Returns the final snapshot.
        """
        self.status = STATUS_RUNNING
        self.started_at = time.time()
        self._publish()

        try:
            for game_idx in range(self.config.num_games):
                with self._lock:
                    if self._stop_requested:
                        break
                    self._current_index = game_idx
                self._play_one_game(game_idx)

            with self._lock:
                self.status = STATUS_STOPPED if self._stop_requested else STATUS_FINISHED
        except Exception as exc:  # noqa: BLE001 — surfaced to the UI verbatim
            self.status = STATUS_ERROR
            self.error = str(exc)
        finally:
            self.finished_at = time.time()
            for player in (self.player_a, self.player_b):
                try:
                    player.close()
                except Exception:
                    pass
            self._publish()

        return self.snapshot()

    def _colors_for_game(self, game_idx: int) -> Tuple[int, int]:
        """(colour of A, colour of B) for this game."""
        if self.config.alternate_colors and game_idx % 2 == 1:
            return WHITE, BLACK
        return BLACK, WHITE

    def _play_one_game(self, game_idx: int) -> None:
        color_a, color_b = self._colors_for_game(game_idx)
        black = self.player_a if color_a == BLACK else self.player_b
        white = self.player_b if color_a == BLACK else self.player_a

        state = GameState(board_size=self.config.board_size, komi=self.config.komi)
        start = time.time()

        with self._lock:
            self._state = state
            self._current_colors = {'a': color_a, 'b': color_b}
            self._last_move = None
            self._win_rates = []
            self._scores = {'black': 0.0, 'white': 0.0}

        black.game_started(state, BLACK)
        white.game_started(state, WHITE)

        self._record_position(state)
        self._publish()

        move_list: List[dict] = []
        limit = self.config.move_limit()
        stopped_early = False

        for move_num in range(limit):
            if state.is_over:
                break
            if self._wait_while_paused():
                stopped_early = True
                break

            mover = state.current_player
            player = black if mover == BLACK else white
            action = player.select_move(state)

            if tuple(action) == MOVE_RESIGN:
                state.play_resign()
            elif tuple(action) == MOVE_PASS:
                state.play_pass()
            elif not state.play_move(action[0], action[1]):
                # An engine that offers an illegal move passes instead — the
                # same fallback the gate and eval matches use.
                action = MOVE_PASS
                state.play_pass()

            move_list.append({
                'color': int(mover),
                'move': list(action),
                'move_num': move_num,
            })

            # Both sides hear about every move — a remote player needs ours
            # pushed to it, and ignores the ones it produced itself.
            for observer in (black, white):
                try:
                    observer.observe_move(state, mover, tuple(action))
                except Exception:
                    pass

            with self._lock:
                self._last_move = {
                    'color': int(mover),
                    'move': list(action),
                    'move_num': move_num,
                    'player': 'a' if player is self.player_a else 'b',
                    'player_name': player.name,
                }
            self._record_position(state)
            self._publish()

            if self.config.move_delay > 0:
                time.sleep(self.config.move_delay)

        # Ran out of moves without either side ending it — close it out the way
        # the eval matches do, so the position still gets scored.
        if not state.is_over and not stopped_early:
            state.play_pass()
            if not state.is_over:
                state.play_pass()

        if stopped_early and not state.is_over:
            # Abandoned mid-game: record nothing, rate nothing.
            self._publish()
            return

        self._finish_game(game_idx, state, move_list, color_a, color_b,
                          black, white, time.time() - start)

    def _wait_while_paused(self) -> bool:
        """Block while paused. Returns True if the match should stop."""
        while True:
            with self._lock:
                if self._stop_requested:
                    return True
                if not self._paused:
                    return False
            time.sleep(0.1)

    def _record_position(self, state: GameState) -> None:
        """
        Update the per-position readouts the spectator panel shows: the running
        score and Black's win probability.
        """
        try:
            black_score, white_score = self.scorer.score(state)
            with self._lock:
                self._scores = {'black': float(black_score), 'white': float(white_score)}
        except Exception:
            pass
        self._record_win_rate(state)

    def _record_win_rate(self, state: GameState) -> None:
        """Append Black's win probability (%) for the current position."""
        if not self.config.compute_win_rates or self._analysis_network is None:
            return
        try:
            import torch
            with torch.no_grad():
                _, value = self._analysis_network.predict(state.encode_for_nn(), "cpu")
            value_black = value if state.current_player == BLACK else -value
            rate = round(50.0 + 50.0 * float(value_black), 1)
        except Exception:
            # Board-size mismatch or a network that failed to load — the curve
            # is an extra, never a reason to abort the match.
            return
        with self._lock:
            self._win_rates.append(rate)

    def _finish_game(self, game_idx: int, state: GameState, move_list: List[dict],
                     color_a: int, color_b: int, black: Player, white: Player,
                     elapsed: float) -> None:
        if state.resign_color:
            winner = state.winner
            margin = 0.0
            black_score, white_score = self.scorer.score(state)
        else:
            winner, margin = self.scorer.determine_winner(state)
            black_score, white_score = self.scorer.score(state)

        outcome = GameOutcome(
            index=game_idx,
            black_name=black.name,
            white_name=white.name,
            winner=int(winner or 0),
            margin=float(margin or 0.0),
            resigned_by=int(state.resign_color) if state.resign_color else None,
            num_moves=len(move_list),
            black_score=float(black_score),
            white_score=float(white_score),
            elapsed=elapsed,
        )

        # Tally by PLAYER, not colour — colours alternate.
        if outcome.winner == color_a:
            self.wins['a'] += 1
            score_a = 1.0
        elif outcome.winner == color_b:
            self.wins['b'] += 1
            score_a = 0.0
        else:
            self.wins['draw'] += 1
            score_a = 0.5

        if self._ratings_apply():
            before_a, before_b = self.player_a.rating, self.player_b.rating
            new_a, new_b = compute_pairwise_elo(
                before_a, before_b, score_a, k=self.config.rating_k
            )
            self.player_a.commit_rating(new_a)
            self.player_b.commit_rating(new_b)
            outcome.rating_delta = {
                'a': self.player_a.rating - before_a,
                'b': self.player_b.rating - before_b,
            }

        for observer in (black, white):
            try:
                observer.game_finished(state, outcome.winner or None)
            except Exception:
                pass

        if self.config.record_games and move_list:
            outcome.saved_as = self._save_game(game_idx, state, move_list,
                                               color_a, color_b, outcome)

        with self._lock:
            self.outcomes.append(outcome)
        self._publish()

    def _save_game(self, game_idx: int, state: GameState, move_list: List[dict],
                   color_a: int, color_b: int, outcome: GameOutcome) -> List[str]:
        """Write the finished game into every participating model's games dir."""
        black_slot = 'a' if color_a == BLACK else 'b'
        black_player = self.player_a if black_slot == 'a' else self.player_b
        white_player = self.player_b if black_slot == 'a' else self.player_a

        with self._lock:
            win_rates = list(self._win_rates)

        record = {
            'board_size': self.config.board_size,
            'komi': self.config.komi,
            'ruleset': self.config.scoring_method,
            'moves': move_list,
            'num_moves': len(move_list),
            'win_rates': win_rates,
            'winner': outcome.winner,
            'black_score': outcome.black_score,
            'white_score': outcome.white_score,
            'margin': outcome.margin,
            'resigned_by': outcome.resigned_by,
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': round(outcome.elapsed, 2),
            # Match specifics — what the review UI needs to name the two sides.
            'is_match': True,
            'match_id': self.match_id,
            'match_name': self.name,
            'game_index': game_idx,
            'num_games': self.config.num_games,
            'black_player': black_player.describe(),
            'white_player': white_player.describe(),
            'rating_delta': outcome.rating_delta,
        }

        saved = []
        for games_dir in self.record_dirs:
            try:
                saved.append(save_match_game(games_dir, dict(record)))
            except OSError:
                continue
        return saved
