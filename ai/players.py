"""
players.py — Agents that can sit on one side of a Go board.

Everything that plays a game — a trained model, the random baseline, and later
an online opponent on OGS — implements the same small `Player` interface, so
`ai/match.py` never needs to know what is on the other end of a move.

The interface is deliberately shaped for remote opponents, not just local ones:

  game_started(state, color)   told which colour it has before the first move
  observe_move(state, color, move)  told about EVERY move, including its own —
                               a remote server needs our moves pushed to it
  select_move(state)           produce a move (may block; may return MOVE_RESIGN)
  game_finished(state, winner) told the result
  close()                      release sockets/sessions

A local engine ignores the observe/finish hooks; a networked one needs them.
Because the match runner calls all of them for both sides, adding an OGS player
is a matter of implementing this class and registering it in PLAYER_FACTORIES —
no change to the match loop, the API, or the UI.

Ratings work the same way. A player exposes `rating_key` (its identity for
rating purposes), `rating`, and `commit_rating()`. The match runner updates the
two ratings after each game and lets the player decide how (or whether) to
persist. The random bot is the Elo ANCHOR, so its rating is fixed — see
`rating_is_fixed`.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, Callable

from game.game_state import GameState, MOVE_PASS, MOVE_RESIGN
from ai.mercy_rule import MercyRule
from ai.mcts import MCTS
from ai.random_bot import RandomBot


# Elo the random bot is pinned at — the same anchor training uses.
RANDOM_BOT_ELO = 500.0


class Player(ABC):
    """One side of a game. Subclasses supply the move-generation policy."""

    # Short machine-readable type, matching the keys of PLAYER_FACTORIES.
    kind: str = "abstract"

    def __init__(self, name: str, rating_key: Optional[str] = None,
                 rating: float = RANDOM_BOT_ELO, meta: Optional[dict] = None):
        self.name = name
        # Identity for rating purposes. Two players with the same key are the
        # same entity — a match between them says nothing about strength, so
        # the runner skips the rating update entirely.
        self.rating_key = rating_key
        self.rating = float(rating)
        self.meta: Dict[str, Any] = meta or {}

    # --- Rating -----------------------------------------------------------

    @property
    def rating_is_fixed(self) -> bool:
        """
        True for players whose rating must not move (rating anchors).

        The random bot anchors the whole Elo scale at 500; letting it drift
        would silently rescale every model's rating along with it.
        """
        return False

    def commit_rating(self, new_rating: float) -> None:
        """
        Adopt (and, for players that have somewhere to put it, persist) a new
        rating. Called by the match runner after each rated game.
        """
        if self.rating_is_fixed:
            return
        self.rating = float(new_rating)

    # --- Game lifecycle ---------------------------------------------------

    def game_started(self, state: GameState, color: int) -> None:
        """Called once per game, before any move, with the colour we play."""

    def observe_move(self, state: GameState, color: int, move: Tuple[int, int]) -> None:
        """
        Called after every move by either side, with the state AFTER the move.

        Local engines ignore this; a remote player uses it to push our moves to
        the server it is bridging to.
        """

    @abstractmethod
    def select_move(self, state: GameState) -> Tuple[int, int]:
        """Return (row, col), MOVE_PASS, or MOVE_RESIGN."""

    def game_finished(self, state: GameState, winner: Optional[int]) -> None:
        """Called once the game is over (or was abandoned)."""

    def close(self) -> None:
        """Release any resources (sockets, sessions). Safe to call twice."""

    # --- Description ------------------------------------------------------

    def describe(self) -> dict:
        """Serializable summary, used by the API and stored in game records."""
        return {
            'kind': self.kind,
            'name': self.name,
            'rating_key': self.rating_key,
            'rating': round(self.rating, 1),
            'rating_is_fixed': self.rating_is_fixed,
            **self.meta,
        }


class ModelPlayer(Player):
    """
    A trained model playing through MCTS.

    The network is supplied already loaded (see `ai.model_loader`), so the
    caller controls caching and device placement — this class only searches.
    """

    kind = "model"

    def __init__(self, model_id: str, name: str, network, board_size: int,
                 num_simulations: int = 200, c_puct: float = 1.5,
                 restrict_eye_fill: bool = False, device: str = "cpu",
                 rating: float = RANDOM_BOT_ELO, iteration: int = 0,
                 temperature: float = 0.1, meta: Optional[dict] = None,
                 rating_sink: Optional[Callable[[str, float], None]] = None,
                 mercy: Optional[MercyRule] = None):
        super().__init__(
            name=name,
            # A model's rating lives in its config.json, so the model id is the
            # natural rating identity.
            rating_key=f"model:{model_id}",
            rating=rating,
            meta={
                'model_id': model_id,
                'iteration': iteration,
                'num_simulations': num_simulations,
                'board_size': board_size,
                **(meta or {}),
            },
        )
        self.model_id = model_id
        self.network = network
        self.board_size = board_size
        self.temperature = temperature
        self._rating_sink = rating_sink
        # Mercy rule: gives up a game its own search has called lost for
        # several moves running, instead of playing out a decided endgame.
        self.mercy = mercy
        self.mcts = MCTS(
            network=network,
            num_simulations=num_simulations,
            c_puct=c_puct,
            device=device,
            restrict_eye_fill=restrict_eye_fill,
        )

    def game_started(self, state: GameState, color: int) -> None:
        if self.mercy:
            self.mercy.reset()

    def select_move(self, state: GameState) -> Tuple[int, int]:
        # add_noise=False: this is competitive play, not data generation.
        action, _ = self.mcts.search(state, temperature=self.temperature, add_noise=False)

        # The search that produced this move also produced the evidence for
        # giving up on it, so the check goes here rather than in the caller.
        if self.mercy and self.mercy.observe(self.mcts.root_value, state.move_number):
            return MOVE_RESIGN

        return action

    def commit_rating(self, new_rating: float) -> None:
        super().commit_rating(new_rating)
        if self._rating_sink:
            self._rating_sink(self.model_id, self.rating)


class RandomPlayer(Player):
    """The uniform-random baseline. Fixed at the Elo anchor."""

    kind = "random"

    def __init__(self, name: str = "Random Bot", pass_probability: float = 0.05):
        super().__init__(name=name, rating_key="random", rating=RANDOM_BOT_ELO)
        self.bot = RandomBot(pass_probability=pass_probability)

    @property
    def rating_is_fixed(self) -> bool:
        return True

    def select_move(self, state: GameState) -> Tuple[int, int]:
        return self.bot.select_move(state)


class RemotePlayer(Player):
    """
    Base for opponents that live on another server (OGS, KGS, a GTP bridge).

    Not usable on its own — it documents the contract a networked player has to
    satisfy so the match runner can drive it unchanged:

    * `game_started` is where the connection/challenge is established.
    * `observe_move` is how OUR moves reach the remote board. It is called for
      both colours, so implementations must ignore moves they themselves
      generated (compare `color` with the colour handed to `game_started`).
    * `select_move` blocks until the remote side plays, and should honour
      `self.move_timeout`, returning MOVE_PASS rather than hanging forever.
    * `game_finished` / `close` tear the connection down.

    Ratings for remote players come from the remote service, so `rating_is_fixed`
    is True by default: an OGS opponent's rank is theirs, not ours to update —
    only our own model's rating moves after such a game.
    """

    kind = "remote"

    def __init__(self, name: str, rating_key: Optional[str] = None,
                 rating: float = RANDOM_BOT_ELO, move_timeout: float = 120.0,
                 meta: Optional[dict] = None):
        super().__init__(name=name, rating_key=rating_key, rating=rating, meta=meta)
        self.move_timeout = move_timeout

    @property
    def rating_is_fixed(self) -> bool:
        return True

    def select_move(self, state: GameState) -> Tuple[int, int]:
        raise NotImplementedError("RemotePlayer subclasses must implement select_move")


# ---------------------------------------------------------------------------
# Spec -> Player construction
#
# A "spec" is the JSON the browser sends to describe one side of a match, e.g.
#   {"type": "model", "model_id": "hero-of-time", "num_simulations": 200}
#   {"type": "random"}
#   {"type": "ogs", "username": "...", ...}     (not implemented yet)
#
# Adding a new opponent kind means adding one factory here; nothing in the
# match loop or the API needs to change.
# ---------------------------------------------------------------------------

def _make_model_player(spec: dict, context: dict) -> Player:
    from ai.model_loader import build_model_player
    model_id = spec.get('model_id')
    if not model_id:
        raise ValueError("A model opponent needs a model_id")
    return build_model_player(
        model_id,
        num_simulations=spec.get('num_simulations'),
        board_size=context.get('board_size'),
        label=spec.get('name'),
        # None leaves the model's own resign_enabled in charge.
        mercy_resign=spec.get('mercy_resign', context.get('mercy_resign')),
    )


def _make_random_player(spec: dict, context: dict) -> Player:
    return RandomPlayer(
        name=spec.get('name') or "Random Bot",
        pass_probability=float(spec.get('pass_probability', 0.05)),
    )


def _make_ogs_player(spec: dict, context: dict) -> Player:
    from ai.online.ogs import OGSPlayer  # imported lazily: optional dependency
    return OGSPlayer.from_spec(spec, context)


PLAYER_FACTORIES: Dict[str, Callable[[dict, dict], Player]] = {
    'model': _make_model_player,
    'random': _make_random_player,
    'ogs': _make_ogs_player,
}


# Descriptors for the opponent picker in the UI. `available` False means the
# type is recognised but not implemented yet — the UI shows it greyed out
# instead of pretending it does not exist.
PLAYER_TYPES = [
    {'type': 'model', 'label': 'Trained model', 'available': True,
     'note': 'Any model in this workspace'},
    {'type': 'random', 'label': 'Random bot', 'available': True,
     'note': 'Elo anchor at 500 — its own rating never moves'},
    {'type': 'ogs', 'label': 'OGS opponent', 'available': False,
     'note': 'Online play against OGS — planned'},
]


def create_player(spec: dict, context: Optional[dict] = None) -> Player:
    """
    Build a Player from a spec dict. Raises ValueError for unknown types.

    `context` carries match-level facts a factory may need (board_size, komi),
    so a spec does not have to repeat them.
    """
    spec = spec or {}
    kind = spec.get('type', 'model')
    factory = PLAYER_FACTORIES.get(kind)
    if factory is None:
        raise ValueError(f"Unknown opponent type: {kind}")
    return factory(spec, context or {})
