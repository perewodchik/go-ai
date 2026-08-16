"""
test_ogs_player.py — the OGS bridge as the match runner sees it.

`ai/match.py` drives a game by calling `game_started`, then `select_move` and
`observe_move` per move. This covers what those must get right when the other
side of the board is a server we do not control:

  * only OUR moves are pushed to OGS — echoing the bot's back is an illegal
    move for the wrong colour;
  * the echo of our own move is not mistaken for the bot's reply;
  * a game that has drifted out of step with OGS stops the match instead of
    recording a board that never existed;
  * a failure to deliver a move is raised, not swallowed — the runner ignores
    exceptions from observe_move, so it has to surface on the next call.

No network: the socket and REST client are fakes.
"""

import os
import queue
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import BLACK, WHITE
from game.game_state import GameState, MOVE_PASS, MOVE_RESIGN
from ai.online.ogs import OGSGameError, OGSPlayer
from ai.online.ogs_bots import GameRequest, OGSBot
from ai.online.ogs_socket import OGSSocketError


class FakeLatch:
    def __init__(self, payload, fail=False):
        self._payload = payload
        self._fail = fail

    def wait(self, timeout=30.0):
        if self._fail:
            raise OGSSocketError("nobody accepted")
        return self._payload

    def cancel(self):
        pass


class FakeSocket:
    """Records what was sent, and lets a test push events back."""

    def __init__(self, gamedata=None, accept=True):
        self.sent = []
        self.connected = True
        self.handlers = {}
        self._gamedata = gamedata or {}
        self._accept = accept

    def connect(self, timeout=None):
        self.connected = True

    def close(self):
        self.connected = False

    def request(self, command, data=None, timeout=None):
        self.sent.append((command, data))
        return {"id": 99, "username": "us"}

    def send(self, command, data=None):
        self.sent.append((command, data))

    def on(self, command, handler):
        self.handlers.setdefault(command, []).append(handler)

    def off(self, command, handler=None):
        self.handlers.pop(command, None)

    def latch(self, command):
        return FakeLatch(self._gamedata, fail=not self._accept)

    def fire(self, command, data):
        for handler in list(self.handlers.get(command, ())):
            handler(data)

    def commands(self):
        return [c for c, _ in self.sent]


OUR_ID = 99
# The id OGS hands back with the challenge is NOT the id of the game that gets
# created when the bot accepts — observed live, and the reason the first real
# game hung waiting for events on a game that did not exist.
PROVISIONAL_GAME_ID = 777
REAL_GAME_ID = 555


class FakeClient:
    """
    Stands in for the REST API.

    Creating a challenge makes the fake OGS announce the resulting game on
    `active_game`, which is how the real server tells us what id it chose.
    """

    def __init__(self, socket=None, accept=True, bot_color=WHITE,
                 game_id=REAL_GAME_ID):
        self.socket = socket
        self.accept = accept
        self.bot_color = bot_color
        self.game_id = game_id
        self.challenges = []
        self.cancelled = []

    def user_jwt(self):
        return "jwt-token"

    def me(self):
        return {"id": OUR_ID, "username": "us"}

    def ongoing_games(self):
        return []

    def challenge_bot(self, bot_id, request, name="go-ai", our_color="automatic"):
        self.challenges.append({"bot_id": bot_id, "our_color": our_color,
                                "request": request})
        if self.accept and self.socket is not None:
            self.socket.fire("active_game", {
                "id": self.game_id,
                "phase": "play",
                "black": {"id": BOT.id if self.bot_color == BLACK else OUR_ID},
                "white": {"id": BOT.id if self.bot_color == WHITE else OUR_ID},
            })
        return {"challenge_id": 1234, "game_id": PROVISIONAL_GAME_ID}

    def cancel_challenge(self, challenge_id):
        self.cancelled.append(challenge_id)


BOT = OGSBot(id=1195517, username="Bergamot", ranking=5.0, rank="25k",
             rating=627.0, config={"_config_version": 2,
                                   "allowed_board_sizes": [9, 13, 19]})


def _gamedata(bot_color=WHITE, size=9, moves=None, handicap=0):
    black = {"id": BOT.id if bot_color == BLACK else 99}
    white = {"id": BOT.id if bot_color == WHITE else 99}
    return {
        "width": size, "height": size, "handicap": handicap,
        "players": {"black": black, "white": white},
        "moves": moves or [],
    }


def _pair(bot_color=WHITE, accept=True, gamedata=None, board_size=9,
          move_timeout=None):
    """A player wired to a fake socket and a fake OGS that answers on it."""
    socket = FakeSocket(gamedata=gamedata if gamedata is not None
                        else _gamedata(bot_color=bot_color))
    client = FakeClient(socket=socket, accept=accept, bot_color=bot_color)
    player = OGSPlayer(
        bot=BOT,
        request=GameRequest(board_size=board_size),
        client=client,
        socket=socket,
        accept_timeout=0.3,
        **({"move_timeout": move_timeout} if move_timeout else {}),
    )
    return player, socket, client


def _player(socket=None, client=None, board_size=9):
    player, _, _ = _pair(board_size=board_size)
    return player


class TestStartingAGame:
    def test_challenges_for_the_colour_the_runner_assigned(self):
        player, socket, client = _pair(bot_color=WHITE)
        player.game_started(GameState(board_size=9), WHITE)

        # The bot is White here, so we challenged asking to be Black.
        assert client.challenges[0]["our_color"] == "black"
        assert "game/connect" in socket.commands()

    def test_uses_the_game_id_ogs_created_not_the_one_it_promised(self):
        """
        Regression: the challenge response's game id is provisional. Trusting
        it left the first real game hanging, subscribed to events for a game
        that never existed while the bot waited for a move.
        """
        player, socket, client = _pair()
        player.game_started(GameState(board_size=9), WHITE)

        assert player.game_id == REAL_GAME_ID
        assert player.game_id != PROVISIONAL_GAME_ID
        connect = [d for c, d in socket.sent if c == "game/connect"][0]
        assert connect["game_id"] == REAL_GAME_ID

    def test_refuses_a_game_where_ogs_flipped_the_colours(self):
        player, socket, client = _pair(bot_color=BLACK)
        with pytest.raises(OGSGameError, match="gave Bergamot black"):
            player.game_started(GameState(board_size=9), WHITE)

    def test_refuses_a_board_of_the_wrong_size(self):
        player, _, _ = _pair(gamedata=_gamedata(size=19))
        with pytest.raises(OGSGameError, match="19×19"):
            player.game_started(GameState(board_size=9), WHITE)

    def test_refuses_an_unrequested_handicap(self):
        player, _, _ = _pair(gamedata=_gamedata(handicap=4))
        with pytest.raises(OGSGameError, match="handicap"):
            player.game_started(GameState(board_size=9), WHITE)

    def test_refuses_a_game_that_is_already_underway(self):
        player, _, _ = _pair(gamedata=_gamedata(moves=[[3, 3, 100]]))
        with pytest.raises(OGSGameError, match="already has 1 moves"):
            player.game_started(GameState(board_size=9), WHITE)

    def test_withdraws_the_challenge_when_nobody_accepts(self):
        player, socket, client = _pair(accept=False)
        with pytest.raises(OGSGameError, match="did not accept"):
            player.game_started(GameState(board_size=9), WHITE)
        assert client.cancelled == [1234]


class TestSendingOurMoves:
    def _started(self):
        player, socket, _ = _pair(bot_color=WHITE)
        player.game_started(GameState(board_size=9), WHITE)
        socket.sent.clear()
        return player, socket

    def test_our_move_is_pushed_to_ogs(self):
        player, socket = self._started()
        player.observe_move(GameState(board_size=9), BLACK, (3, 15))
        assert socket.sent == [("game/move", {"game_id": REAL_GAME_ID, "move": "pd"})]

    def test_the_bots_own_move_is_not_echoed_back(self):
        player, socket = self._started()
        player.observe_move(GameState(board_size=9), WHITE, (4, 4))
        assert socket.sent == []

    def test_our_pass_is_sent_as_a_pass(self):
        player, socket = self._started()
        player.observe_move(GameState(board_size=9), BLACK, MOVE_PASS)
        assert socket.sent[0][1]["move"] == ".."

    def test_our_resignation_resigns_the_ogs_game(self):
        player, socket = self._started()
        player.observe_move(GameState(board_size=9), BLACK, MOVE_RESIGN)
        assert socket.sent == [("game/resign", {"game_id": REAL_GAME_ID})]

    def test_an_undeliverable_move_surfaces_on_the_next_select(self):
        """The runner ignores errors from observe_move, so it must resurface."""
        player, socket = self._started()

        def explode(command, data=None):
            raise OGSSocketError("socket is down")

        socket.send = explode
        player.observe_move(GameState(board_size=9), BLACK, (3, 3))

        with pytest.raises(OGSGameError, match="could not be sent"):
            player.select_move(GameState(board_size=9))


class TestReceivingTheirMoves:
    def _started(self, bot_color=WHITE, move_timeout=None):
        player, socket, _ = _pair(bot_color=bot_color, move_timeout=move_timeout)
        player.game_started(GameState(board_size=9), bot_color)
        return player, socket

    def _fire(self, socket, move_number, move):
        socket.fire(f"game/{REAL_GAME_ID}/move",
                    {"game_id": REAL_GAME_ID, "move_number": move_number,
                     "move": move})

    def test_the_bots_move_is_translated(self):
        player, socket = self._started()
        state = GameState(board_size=9)
        state.play_move(4, 4)                       # our model played first

        self._fire(socket, 2, "gd")
        assert player.select_move(state) == (3, 6)

    def test_the_echo_of_our_own_move_is_ignored(self):
        """
        OGS numbers a move by the board's move count AFTER it, so the first
        move of a game echoes as 1 — observed live, and the reason an earlier
        version mistook our own opening move for the bot's reply and aborted
        the match as a divergence.
        """
        player, socket = self._started()
        state = GameState(board_size=9)
        state.play_move(4, 4)

        self._fire(socket, 1, ogs_move(4, 4))       # ours, echoed back
        self._fire(socket, 2, "gd")                 # the bot's reply
        assert player.select_move(state) == (3, 6)

    def test_the_bot_moving_first_is_the_games_move_one(self):
        player, socket = self._started(bot_color=BLACK)
        state = GameState(board_size=9)
        self._fire(socket, 1, "ee")
        assert player.select_move(state) == (4, 4)

    def test_a_move_our_board_calls_illegal_stops_the_match(self):
        """
        The runner would quietly turn an illegal move into a pass, and every
        move after that would be recorded from a position OGS never had.
        """
        player, socket = self._started()
        state = GameState(board_size=9)
        state.play_move(4, 4)

        self._fire(socket, 2, ogs_move(4, 4))       # an occupied point
        with pytest.raises(OGSGameError, match="diverged"):
            player.select_move(state)

    def test_the_game_ending_while_we_wait_reads_as_a_resignation(self):
        player, socket = self._started()
        state = GameState(board_size=9)
        state.play_move(4, 4)

        socket.fire(f"game/{REAL_GAME_ID}/phase", "finished")
        assert player.select_move(state) == MOVE_RESIGN

    def test_a_silent_bot_times_out_rather_than_hanging(self):
        player, socket = self._started(move_timeout=0.3)
        with pytest.raises(OGSGameError, match="did not move"):
            player.select_move(GameState(board_size=9))

    def test_a_resignation_ogs_never_announced_is_noticed_anyway(self):
        """
        Our first live win: the bot resigned at move 73 and no `phase` event
        arrived, so the bridge sat waiting for a move until it timed out. OGS's
        own record is the authority, so a quiet game gets checked against it.
        """
        player, socket, client = _pair(move_timeout=8)
        player.game_started(GameState(board_size=9), WHITE)
        client.game = lambda game_id: {
            "gamedata": {"phase": "finished"},
            "ended": "2026-08-16T03:35:15Z",
            "black_lost": False, "white_lost": True,     # the bot (White) lost
            "outcome": "Resignation",
        }
        assert player.select_move(GameState(board_size=9)) == MOVE_RESIGN

    def test_a_game_our_model_lost_is_not_claimed_as_a_win(self):
        """The same check, the other way round: never invent a win."""
        player, socket, client = _pair(move_timeout=8)
        player.game_started(GameState(board_size=9), WHITE)
        client.game = lambda game_id: {
            "gamedata": {"phase": "finished"},
            "ended": "2026-08-16T03:35:15Z",
            "black_lost": True, "white_lost": False,     # our model lost
            "outcome": "Timeout",
        }
        with pytest.raises(OGSGameError, match="Timeout"):
            player.select_move(GameState(board_size=9))


class TestRatingAndDescription:
    def test_the_bots_rating_never_moves(self):
        player = _player()
        assert player.rating_is_fixed is True
        player.commit_rating(2000)
        assert player.rating != 2000

    def test_seeded_from_the_rank_on_our_scale(self):
        # Bergamot is 25k on OGS; 25k is 1000 on this project's scale, not the
        # 627 glicko number OGS reports.
        assert _player().rating == 1000

    def test_describe_carries_what_the_ui_shows(self):
        info = _player().describe()
        assert info["kind"] == "ogs"
        assert info["ogs_rank"] == "25k"
        assert info["ogs_id"] == BOT.id


def ogs_move(row, col):
    from ai.online import ogs_coords
    return ogs_coords.to_ogs((row, col))
