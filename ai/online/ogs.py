"""
ogs.py — an OGS bot as one side of a game in this project.

`ai/match.py` drives a game by asking each `Player` for a move and telling both
of them about every move played. `OGSPlayer` is the side of that conversation
that happens to live on online-go.com:

    game_started()   challenge the bot, wait for it to accept, connect to the game
    observe_move()   called for BOTH colours — send the ones that are ours
    select_move()    block until the bot's move arrives, translate it
    game_finished()  leave the game
    close()          drop the socket

Nothing above this file changes: the match runner, the live board, the game
recorder and the Elo update all treat this as another opponent.

Who is authoritative
--------------------
OGS is. We keep our own `GameState` because everything in this project is built
around one, but OGS decides what is legal, whose turn it is and when the game
ends. Two guards keep that honest:

  * the colour we asked for is the colour the runner assigned, and
    `game_started` refuses to continue if OGS disagrees;
  * every move OGS reports is checked against our move count, so a game that
    has drifted out of step stops immediately rather than recording a board
    that never existed.

The bot's own rating is OGS's business, so `rating_is_fixed` stays True
(inherited): a game here moves our model's Elo and nothing else.
"""

import logging
import queue
import threading
import time
from typing import Any, Dict, Optional, Tuple

from game.game_state import GameState, MOVE_PASS, MOVE_RESIGN
from game.board import BLACK, WHITE
from ai.players import RemotePlayer
from ai.online import ogs_coords
from ai.online.ogs_bots import GameRequest, OGSBot, playability, registry
from ai.online.ogs_rest import OGSClient, OGSRequestError
from ai.online.ogs_socket import OGSSocket, OGSSocketError, authenticate_payload

logger = logging.getLogger(__name__)

# How long to wait for a bot to accept. Bots normally accept within a second or
# two; one that is busy or asleep never will, and we should say so rather than
# hang the match.
ACCEPT_TIMEOUT = 90.0
# The server cancels an unanswered live challenge unless we keep it alive.
KEEPALIVE_INTERVAL = 1.0
# A bot that has not moved in this long has stalled (or is thinking very hard
# about a correspondence game, which we do not play).
MOVE_TIMEOUT = 300.0
# Once a game exists, its state should arrive as soon as we connect to it.
GAMEDATA_TIMEOUT = 30.0
# How often to ask OGS whether a quiet game has actually ended. Cheap next to
# the cost of sitting out MOVE_TIMEOUT on a game that is already over.
END_CHECK_INTERVAL = 10.0

_OGS_COLOR = {"black": BLACK, "white": WHITE}


class OGSGameError(RuntimeError):
    """The OGS game could not be started, or diverged from ours."""


class _NewGameWatcher:
    """
    Waits for the game a challenge turns into.

    OGS announces every game we are in with an `active_game` event, including
    the ones already running when we connect — so games known before the
    challenge are remembered and ignored, and only a NEW game between us and
    this bot counts as the one we just created.
    """

    def __init__(self, socket: OGSSocket, our_id: int, bot_id: int):
        self._socket = socket
        self._our_id = int(our_id)
        self._bot_id = int(bot_id)
        self._known: set = set()
        self._found = threading.Event()
        self._game_id: Optional[int] = None
        socket.on("active_game", self._on_active_game)

    def _on_active_game(self, data: dict) -> None:
        try:
            game_id = int(data.get("id"))
            players = {int((data.get("black") or {}).get("id")),
                       int((data.get("white") or {}).get("id"))}
        except (TypeError, ValueError):
            return

        if players != {self._our_id, self._bot_id}:
            return
        if data.get("phase") != "play":
            return
        if game_id in self._known:
            return

        self._game_id = game_id
        self._found.set()

    def snapshot_known(self, game_ids) -> None:
        self._known.update(int(g) for g in game_ids)

    def wait(self, timeout: float) -> int:
        if not self._found.wait(timeout):
            raise OGSSocketError("no game appeared for this challenge")
        return self._game_id

    def cancel(self) -> None:
        self._socket.off("active_game", self._on_active_game)


class OGSPlayer(RemotePlayer):
    """An OGS bot, playing one side of a game in this project."""

    kind = "ogs"

    def __init__(self, bot: OGSBot, request: GameRequest,
                 client: Optional[OGSClient] = None,
                 socket: Optional[OGSSocket] = None,
                 game_name: str = "go-ai",
                 move_timeout: float = MOVE_TIMEOUT,
                 accept_timeout: float = ACCEPT_TIMEOUT):
        super().__init__(
            name=bot.username,
            # An OGS account is the identity; the rating is theirs to move.
            # The seed is the bot's RANK converted to our scale, not its glicko
            # number — the two scales share ranks, not values.
            rating_key=f"ogs:{bot.id}",
            rating=bot.elo,
            move_timeout=move_timeout,
            meta={
                "ogs_id": bot.id,
                "ogs_rank": bot.rank,
                "ogs_ranking": bot.ranking,
                "board_size": request.board_size,
                "ranked": request.ranked,
            },
        )
        self.bot = bot
        self.request = request
        self.client = client or OGSClient()
        self.game_name = game_name
        self.accept_timeout = accept_timeout

        self._socket = socket
        self._owns_socket = socket is None
        self._authenticated = False
        self._player_id: Optional[int] = None

        # Per-game state, reset by game_started().
        self.color: Optional[int] = None
        self.game_id: Optional[int] = None
        self.challenge_id: Optional[int] = None
        self._incoming: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self._our_moves_sent = 0
        self._moves_seen = 0
        self._phase = "play"
        self._keepalive: Optional[threading.Thread] = None
        self._stop_keepalive = threading.Event()
        # Set when a move of ours could not be delivered; raised on the next
        # select_move, since the runner ignores errors from observe_move.
        self._send_failure: Optional[str] = None
        self._next_end_check = 0.0

    # ---- Construction from a spec ----------------------------------------

    @classmethod
    def from_spec(cls, spec: dict, context: Optional[dict] = None) -> "OGSPlayer":
        """
        Build from the same JSON the browser sends for any other opponent:

            {"type": "ogs", "bot_id": 1195517, "ranked": false}

        Board size and the clock come from the match context, so an OGS
        opponent is specified exactly like a local one.
        """
        context = context or {}
        bot_id = spec.get("bot_id") or spec.get("model_id")
        if not bot_id:
            raise ValueError("An OGS opponent needs a bot_id")

        bot = registry.get(int(bot_id))
        if bot is None:
            raise ValueError(f"OGS has no bot with id {bot_id}")

        request = GameRequest(
            board_size=int(context.get("board_size") or spec.get("board_size") or 9),
            speed=spec.get("speed", "live"),
            system=spec.get("system", "byoyomi"),
            ranked=bool(spec.get("ranked", False)),
            handicap=bool(spec.get("handicap", False)),
        )

        playable, reason = playability(bot, request)
        if not playable:
            raise ValueError(f"{bot.username}: {reason}")

        return cls(bot=bot, request=request, game_name=spec.get("name") or "go-ai")

    # ---- Socket ----------------------------------------------------------

    def _connect(self) -> OGSSocket:
        """Connect and authenticate as us, once per player."""
        if self._socket is None:
            self._socket = OGSSocket()
        if not self._socket.connected:
            self._socket.connect()
        if not self._authenticated:
            jwt = self.client.user_jwt()
            who = self._socket.request("authenticate", authenticate_payload(jwt=jwt))
            if not who or not who.get("id"):
                raise OGSGameError("OGS did not accept our socket credentials")
            self._authenticated = True
            logger.info("OGS socket authenticated as %s", who.get("username"))
        return self._socket

    # ---- Game lifecycle --------------------------------------------------

    def game_started(self, state: GameState, color: int) -> None:
        """
        Challenge the bot for the colour the match runner assigned it, wait for
        it to accept, and connect to the resulting game.
        """
        self.color = color
        self._incoming = queue.Queue()
        self._our_moves_sent = 0
        self._moves_seen = 0
        self._phase = "play"
        self._send_failure = None
        self._next_end_check = 0.0

        if state.board_size != self.request.board_size:
            raise OGSGameError(
                f"This match is {state.board_size}×{state.board_size} but the OGS "
                f"challenge is set up for {self.request.board_size}×{self.request.board_size}"
            )

        sock = self._connect()

        # We are the challenger, so the colour we ask for is OUR model's — the
        # opposite of the colour the bot was given.
        our_color = "white" if color == BLACK else "black"

        # The id in the challenge response is NOT the id of the game that gets
        # created when the bot accepts (observed live: a challenge answered with
        # game 89756761 produced game 89783761). The real one arrives on the
        # `active_game` event, so watch for it before challenging.
        watcher = _NewGameWatcher(sock, our_id=self._our_id(), bot_id=self.bot.id)
        # Games already running against this bot are not the one we are about
        # to create — OGS re-announces them whenever we connect.
        watcher.snapshot_known(self._ongoing_game_ids())

        created = self.client.challenge_bot(
            self.bot.id, self.request, name=self.game_name, our_color=our_color
        )
        self.challenge_id = created["challenge_id"]

        self._start_keepalive(sock, created["game_id"])
        try:
            self.game_id = watcher.wait(self.accept_timeout)
        except OGSSocketError:
            self._withdraw()
            raise OGSGameError(
                f"{self.bot.username} did not accept the game within "
                f"{self.accept_timeout:.0f}s — it may be busy or offline"
            )
        finally:
            self._stop_keepalive_timer()
            watcher.cancel()

        # Now that the game is real, subscribe to it and read its full state.
        gamedata = sock.latch(f"game/{self.game_id}/gamedata")
        sock.on(f"game/{self.game_id}/move", self._on_move)
        sock.on(f"game/{self.game_id}/phase", self._on_phase)
        sock.send("game/connect", {"game_id": self.game_id, "chat": False})

        data = self._wait_for_gamedata(gamedata)
        self._verify_game(data, state)
        logger.info("OGS game %s started: %s plays %s",
                    self.game_id, self.bot.username,
                    "black" if color == BLACK else "white")

    def _our_id(self) -> int:
        """Our own OGS player id, used to recognise our games."""
        if self._player_id is None:
            self._player_id = int(self.client.me()["id"])
        return self._player_id

    def _ongoing_game_ids(self) -> list:
        """Games we are already in, which a new challenge is definitely not."""
        try:
            games = self.client.ongoing_games()
        except Exception:
            # Not worth failing a challenge over; the checks in _verify_game
            # still catch a game that is not the one we just started.
            return []
        return [g.get("id") for g in games if g.get("id")]

    def _wait_for_gamedata(self, gamedata_latch) -> dict:
        """Block for the game's full state after connecting to it."""
        try:
            return gamedata_latch.wait(timeout=GAMEDATA_TIMEOUT) or {}
        except OGSSocketError:
            gamedata_latch.cancel()
            raise OGSGameError(
                f"OGS started game {self.game_id} but never sent its state"
            )

    def _verify_game(self, gamedata: dict, state: GameState) -> None:
        """Refuse to play a game that is not the one the runner set up."""
        if not gamedata:
            raise OGSGameError("OGS never sent the game data")

        width = gamedata.get("width")
        height = gamedata.get("height")
        if width and height and (width != state.board_size or height != state.board_size):
            raise OGSGameError(
                f"OGS started a {width}×{height} game but this match is "
                f"{state.board_size}×{state.board_size}"
            )

        players = gamedata.get("players") or {}
        ogs_black = (players.get("black") or {}).get("id")
        ogs_white = (players.get("white") or {}).get("id")
        bot_color = BLACK if ogs_black == self.bot.id else (
            WHITE if ogs_white == self.bot.id else None)

        if bot_color is None:
            raise OGSGameError(f"{self.bot.username} is not a player in OGS game {self.game_id}")
        if bot_color != self.color:
            raise OGSGameError(
                f"OGS gave {self.bot.username} "
                f"{'black' if bot_color == BLACK else 'white'}, but this match "
                f"assigned it {'black' if self.color == BLACK else 'white'}"
            )

        handicap = int(gamedata.get("handicap") or 0)
        if handicap and not self.request.handicap:
            raise OGSGameError(
                f"OGS applied a {handicap} stone handicap we did not ask for"
            )

        # Any moves already in the record before we connected would mean we are
        # joining a game in progress, which the match runner cannot represent.
        existing = [m for m in (gamedata.get("moves") or [])]
        if existing:
            raise OGSGameError(
                f"OGS game {self.game_id} already has {len(existing)} moves"
            )

    # ---- Keepalive -------------------------------------------------------

    def _start_keepalive(self, sock: OGSSocket, game_id: int) -> None:
        """
        OGS cancels a live challenge that nobody is waiting on.

        `game_id` here is the provisional id from the challenge response, which
        is what the keepalive is keyed on — not the id of the game that will
        eventually be created.
        """
        self._stop_keepalive.clear()

        def beat() -> None:
            while not self._stop_keepalive.wait(KEEPALIVE_INTERVAL):
                try:
                    sock.send("challenge/keepalive", {
                        "challenge_id": self.challenge_id,
                        "game_id": game_id,
                    })
                except OGSSocketError:
                    return

        self._keepalive = threading.Thread(target=beat, daemon=True,
                                           name=f"ogs-keepalive-{self.game_id}")
        self._keepalive.start()

    def _stop_keepalive_timer(self) -> None:
        self._stop_keepalive.set()
        self._keepalive = None

    # ---- Moves -----------------------------------------------------------

    def _on_move(self, data: dict) -> None:
        """A move was played on our game — by either side."""
        try:
            move = ogs_coords.unpack_move(data.get("move"))
        except ValueError:
            logger.warning("Unreadable move from OGS: %r", data.get("move"))
            return
        self._incoming.put(("move", {
            "move": move,
            "move_number": data.get("move_number"),
        }))

    def _on_phase(self, phase: str) -> None:
        self._phase = phase
        if phase != "play":
            logger.info("OGS game %s entered phase %r", self.game_id, phase)
            self._incoming.put(("phase", phase))

    def observe_move(self, state: GameState, color: int, move: Tuple[int, int]) -> None:
        """
        Called for every move by either side. Only OUR model's moves get sent —
        the bot's arrive from OGS and echoing them back would be an illegal
        move for the wrong colour.
        """
        if color == self.color or self.game_id is None:
            return

        # The match runner swallows exceptions from this hook, so a failure
        # here cannot be raised — it is remembered and thrown from the next
        # select_move(). Playing on with a move OGS never received is the one
        # thing that must not happen quietly.
        try:
            if tuple(move) == MOVE_RESIGN:
                self._send("game/resign", {"game_id": self.game_id})
                return
            self._send("game/move", {
                "game_id": self.game_id,
                "move": ogs_coords.to_ogs(move),
            })
            self._our_moves_sent += 1
        except (OGSSocketError, OGSGameError, ValueError) as exc:
            self._send_failure = f"our move could not be sent to OGS: {exc}"
            logger.error("OGS game %s: %s", self.game_id, self._send_failure)

    def _send(self, command: str, data: dict) -> None:
        if self._socket is None:
            raise OGSGameError("Not connected to OGS")
        self._socket.send(command, data)

    def _ended_on_ogs(self):
        """
        Has the game ended without us hearing about it?

        Returns MOVE_RESIGN when the BOT is the one that lost (it resigned or
        ran out of time — from the runner's point of view that is the opponent
        resigning), None while the game is still going, and raises when the
        game ended some other way, because claiming a win our model did not get
        would put a false result into its record.
        """
        now = time.monotonic()
        if now < self._next_end_check:
            return None
        self._next_end_check = now + END_CHECK_INTERVAL

        try:
            record = self.client.game(self.game_id)
        except Exception as exc:
            logger.debug("Could not read OGS game %s: %s", self.game_id, exc)
            return None

        gamedata = record.get("gamedata") or {}
        if gamedata.get("phase") == "play" and not record.get("ended"):
            return None

        bot_lost = record.get("black_lost" if self.color == BLACK else "white_lost")
        outcome = record.get("outcome") or gamedata.get("outcome") or "over"
        if bot_lost:
            logger.info("OGS game %s: %s lost (%s)", self.game_id,
                        self.bot.username, outcome)
            return MOVE_RESIGN

        raise OGSGameError(
            f"the OGS game ended while it was {self.bot.username}'s turn "
            f"({outcome}) — nothing to record"
        )

    def select_move(self, state: GameState) -> Tuple[int, int]:
        """
        Wait for the bot's move.

        Both sides' moves come back on the same event, including the echo of
        the move we just sent, so anything that is not the bot's turn is
        discarded. A game that ends while we are waiting (the bot resigned, or
        timed out) surfaces as a resignation, which is what it is from the
        runner's point of view.
        """
        if self.game_id is None:
            raise OGSGameError("No OGS game is running")
        if self._send_failure:
            raise OGSGameError(self._send_failure)

        deadline = time.monotonic() + self.move_timeout
        expected_move_number = state.move_number

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OGSGameError(
                    f"{self.bot.username} did not move within "
                    f"{self.move_timeout:.0f}s"
                )
            try:
                kind, payload = self._incoming.get(timeout=min(remaining, 5.0))
            except queue.Empty:
                if self._phase != "play":
                    return MOVE_RESIGN
                # Not every ending announces itself on an event we listen for:
                # the first live win came from a bot resigning at move 73, and
                # no `phase` event arrived at all. OGS's own record is the
                # authority, so ask it rather than waiting out the timeout.
                ended = self._ended_on_ogs()
                if ended is not None:
                    return ended
                continue

            if kind == "phase":
                # The game ended while it was the bot's turn: it resigned, ran
                # out of time, or was cancelled. Either way it is not moving.
                if payload != "play":
                    return MOVE_RESIGN
                continue

            move = payload["move"]
            move_number = payload.get("move_number")

            # Every move on the game comes back on this event, including the
            # echo of the one we just sent, so anything already on our board is
            # not the move we are waiting for.
            #
            # OGS numbers a move by the board's move COUNT after it: the first
            # move of a game is 1, not 0. So a move worth waking up for is
            # strictly beyond what our own board already holds.
            if move_number is not None and move_number <= expected_move_number:
                logger.debug("OGS game %s: skipping the echo of move %s",
                             self.game_id, move_number)
                continue

            # OGS is the authority, so a move it reports that our board says is
            # illegal means the two boards have drifted apart. The runner would
            # quietly turn it into a pass; stop the match instead, because
            # every move after this one would be recorded from a position that
            # never existed on OGS.
            if move != MOVE_PASS and not state.is_legal(move[0], move[1]):
                raise OGSGameError(
                    f"OGS played {ogs_coords.to_ogs(move)} at move "
                    f"{move_number}, which is illegal on our board — the games "
                    f"have diverged"
                )

            self._moves_seen += 1
            return move

    def game_finished(self, state: GameState, winner: Optional[int]) -> None:
        self._stop_keepalive_timer()
        if self._socket is not None and self.game_id is not None:
            try:
                self._socket.off(f"game/{self.game_id}/move")
                self._socket.off(f"game/{self.game_id}/phase")
                self._socket.send("game/disconnect", {"game_id": self.game_id})
            except OGSSocketError:
                pass
        self.meta["ogs_game_id"] = self.game_id
        self.game_id = None

    def close(self) -> None:
        self._stop_keepalive_timer()
        self._withdraw()
        if self._socket is not None and self._owns_socket:
            self._socket.close()
            self._socket = None
            self._authenticated = False

    def _withdraw(self) -> None:
        """Take back a challenge that was never accepted."""
        if self.challenge_id is None:
            return
        try:
            self.client.cancel_challenge(self.challenge_id)
        except OGSRequestError as exc:
            logger.warning("Could not withdraw OGS challenge %s: %s",
                           self.challenge_id, exc)
        self.challenge_id = None

    # ---- Description -----------------------------------------------------

    def describe(self) -> dict:
        info = super().describe()
        info.update({
            "ogs_id": self.bot.id,
            "ogs_rank": self.bot.rank,
            "game_id": self.game_id,
        })
        return info
