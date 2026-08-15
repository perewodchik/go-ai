"""
ogs.py — Placeholder bridge to online-go.com (OGS).

NOT IMPLEMENTED YET. This file exists so the shape of the future feature is
fixed now: the match runner, the REST API, the opponent picker and the game
store already treat an OGS opponent as just another `Player`, so building it
means filling in the four methods below — nothing upstream changes.

What an implementation needs to do
----------------------------------
1. Auth: OGS uses OAuth2 (client id/secret -> bearer token) for its REST API,
   and the same token to authenticate the realtime socket.io connection.
   Credentials belong in the spec/config, never hard-coded here.
2. Get a game: either accept an open challenge, create one with
   `POST /api/v1/challenges`, or attach to an existing game id. `game_started`
   is the hook for this, and it must resolve which colour we were given.
3. Move exchange over the realtime socket:
     - `observe_move` is called for BOTH colours. Send only the moves whose
       `color` is OUR colour (`self.color`); the opponent's arrive from the
       server and must be ignored here or they will be echoed back.
     - `select_move` blocks on the socket until the opponent's move arrives,
       honouring `self.move_timeout`, and translates OGS coordinates
       (`"pd"`-style or {i, j}) into this project's (row, col), with
       MOVE_PASS for a pass and MOVE_RESIGN for a resignation.
4. `game_finished` / `close`: leave the game and drop the socket.

Rating: an OGS opponent's rank belongs to OGS, so `rating_is_fixed` stays True
(inherited from RemotePlayer) — only our own model's Elo moves after such a
game. Seed `rating` from the opponent's OGS rating converted to this project's
scale so the Elo update is meaningful.

Board settings: OGS decides board size, komi and ruleset for a real game. The
match layer treats those as match-level facts (`context`), so when this is
implemented, `from_spec` should fail loudly if the caller's board settings do
not match the OGS game's.
"""

from typing import Tuple

from game.game_state import GameState
from ai.players import RemotePlayer


class OGSPlayer(RemotePlayer):
    """An opponent playing through an OGS game. Not implemented yet."""

    kind = "ogs"

    def __init__(self, name: str = "OGS opponent", rating: float = 500.0,
                 move_timeout: float = 120.0, meta: dict = None):
        super().__init__(name=name, rating_key=None, rating=rating,
                         move_timeout=move_timeout, meta=meta)
        self.color = None

    @classmethod
    def from_spec(cls, spec: dict, context: dict) -> "OGSPlayer":
        raise NotImplementedError(
            "Online play against OGS is not implemented yet. The Player "
            "interface and the match runner already support it — see the notes "
            "at the top of ai/online/ogs.py."
        )

    def game_started(self, state: GameState, color: int) -> None:
        self.color = color
        raise NotImplementedError

    def observe_move(self, state: GameState, color: int, move: Tuple[int, int]) -> None:
        raise NotImplementedError

    def select_move(self, state: GameState) -> Tuple[int, int]:
        raise NotImplementedError
