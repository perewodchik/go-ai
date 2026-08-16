"""
test_ogs_protocol.py — the parts of the OGS bridge that can be wrong silently.

Nothing here touches the network: the socket tests run against a fake OGS on
localhost, and the roster tests use recorded `active-bots` payloads. Coordinate
translation and the compatibility filter are pure functions.

The two things being defended against:

  * a coordinate swap, which looks right on a symmetric position and puts our
    stones in the wrong place everywhere else;
  * accepting a game whose settings are not the ones the match was set up for,
    which would record a board that never existed on OGS.
"""

import asyncio
import json
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.game_state import MOVE_PASS
from ai.online import ogs_coords
from ai.online.ogs_bots import (
    GameRequest, OGSBot, OGSBotRegistry, playability, rank_string, ranking_to_elo,
)

websockets = pytest.importorskip("websockets")


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------

class TestCoordinates:
    def test_the_documented_example(self):
        # OGS packs x first, so (row 3, col 15) is "pd" — p is column 15.
        assert ogs_coords.to_ogs((3, 15)) == "pd"
        assert ogs_coords.from_ogs("pd") == (3, 15)

    @pytest.mark.parametrize("size", [9, 13, 19])
    def test_round_trip_over_every_intersection(self, size):
        for row in range(size):
            for col in range(size):
                encoded = ogs_coords.to_ogs((row, col))
                assert ogs_coords.from_ogs(encoded) == (row, col)

    def test_asymmetric_point_is_not_transposed(self):
        # The test a swap survives: (0, 1) and (1, 0) must not encode alike.
        assert ogs_coords.to_ogs((0, 1)) != ogs_coords.to_ogs((1, 0))
        assert ogs_coords.to_ogs((0, 1)) == "ba"
        assert ogs_coords.to_ogs((1, 0)) == "ab"

    def test_pass_both_ways(self):
        assert ogs_coords.to_ogs(MOVE_PASS) == ".."
        assert ogs_coords.from_ogs("..") == MOVE_PASS

    def test_game_record_entries(self):
        # gamedata stores [x, y, time_ms]; ours is (row, col).
        assert ogs_coords.unpack_move([5, 4, 2510]) == (4, 5)
        assert ogs_coords.unpack_move([-1, -1, 0]) == MOVE_PASS
        assert ogs_coords.unpack_move("pd") == (3, 15)

    def test_unreadable_input_is_rejected(self):
        with pytest.raises(ValueError):
            ogs_coords.from_ogs("2")
        with pytest.raises(ValueError):
            ogs_coords.unpack_move({"x": 1})


# ---------------------------------------------------------------------------
# Ranks and ratings
# ---------------------------------------------------------------------------

class TestRanks:
    @pytest.mark.parametrize("ranking,label", [
        (5, "25k"),                      # Amaranthus
        (7.686597192497425, "23k"),      # Bouvardia
        (12.556254663797683, "18k"),     # GnuGo
        (22.13262961290587, "8k"),       # noob_bot
        (31.805545521207453, "2d"),      # simplegobot
        (38, "9d"),                      # katago-micro
    ])
    def test_labels_match_what_ogs_displays(self, ranking, label):
        assert rank_string(ranking) == label

    def test_rank_converts_to_our_elo_scale_not_ogs_glicko(self):
        # Our scale: 500 Elo = 30k, 100 Elo per rank. A 25k bot is 1000 here,
        # whatever its glicko number happens to be.
        assert ranking_to_elo(5) == 1000
        assert ranking_to_elo(0) == 500
        # Off the top of our table, it clamps rather than running away.
        assert ranking_to_elo(38) == 3900


# ---------------------------------------------------------------------------
# Which bots can play what
# ---------------------------------------------------------------------------

def _bot(**config) -> OGSBot:
    base = {
        "_config_version": 2,
        "allowed_board_sizes": [9, 13, 19],
        "allowed_time_control_systems": ["byoyomi", "fischer", "simple"],
        "allow_ranked": True,
        "allow_unranked": True,
        "allow_ranked_handicap": True,
        "allow_unranked_handicap": True,
        "allowed_live_settings": {
            "byoyomi": {
                "main_time_range": [0, 3600],
                "period_time_range": [10, 300],
                "periods_range": [1, 10],
            },
        },
    }
    base.update(config)
    return OGSBot(id=1, username="testbot", ranking=12.0, rank="18k",
                  rating=900.0, config=base)


class TestPlayability:
    def test_a_normal_bot_accepts_our_default_game(self):
        assert playability(_bot(), GameRequest())[0] is True

    def test_board_size_it_does_not_play(self):
        playable, reason = playability(_bot(), GameRequest(board_size=19))
        assert playable is True
        playable, reason = playability(_bot(allowed_board_sizes=[19]), GameRequest(board_size=9))
        assert playable is False
        assert "9×9" in reason

    def test_ranked_only_bot(self):
        playable, reason = playability(_bot(allow_unranked=False), GameRequest(ranked=False))
        assert playable is False
        assert "ranked" in reason

    def test_handicap_refusal_names_the_kind_of_game(self):
        bot = _bot(allow_unranked_handicap=False)
        playable, reason = playability(bot, GameRequest(ranked=False, handicap=True))
        assert playable is False
        assert "unranked" in reason and "handicap" in reason

    def test_a_clock_outside_its_range(self):
        bot = _bot(allowed_live_settings={
            "byoyomi": {"main_time_range": [0, 60],
                        "period_time_range": [10, 20],
                        "periods_range": [1, 3]},
        })
        playable, reason = playability(bot, GameRequest(period_time=30))
        assert playable is False
        assert "speed" in reason

    def test_a_bot_that_has_stopped_accepting(self):
        playable, reason = playability(_bot(decline_new_challenges=True), GameRequest())
        assert playable is False
        assert "not accepting" in reason

    def test_bots_that_publish_no_config_are_still_offered(self):
        """OGS lists these (noob_bot, kata_noob); hiding them would diverge."""
        bot = OGSBot(id=2, username="noob_bot", ranking=22.0, rank="8k",
                     rating=1366.0, config={"_config_version": 0})
        assert bot.settings_published is False
        assert playability(bot, GameRequest())[0] is True


# ---------------------------------------------------------------------------
# The roster cache
# ---------------------------------------------------------------------------

ACTIVE_BOTS_FRAME = {
    "58441": {
        "id": 58441, "username": "GnuGo", "ranking": 12.556254663797683,
        "ratings": {"overall": {"rating": 903.0}},
        "config": {"_config_version": 2, "allowed_board_sizes": [9, 13, 19]},
    },
    "2007355": {
        "id": 2007355, "username": "cEmbryo", "ranking": 38,
        "ratings": {"overall": {"rating": 2757.0}},
        "config": {"_config_version": 2, "allowed_board_sizes": [19]},
    },
}


class TestRegistry:
    def test_parses_and_sorts_weakest_first(self, tmp_path, monkeypatch):
        registry = OGSBotRegistry(cache_path=str(tmp_path / "bots.json"))
        monkeypatch.setattr("ai.online.ogs_bots.fetch_active_bots",
                            lambda **kwargs: ACTIVE_BOTS_FRAME)
        bots = registry.refresh()

        assert [b.username for b in bots] == ["GnuGo", "cEmbryo"]
        assert bots[0].rank == "18k"
        assert bots[0].elo == pytest.approx(1755.6, abs=0.1)

    def test_a_written_cache_is_read_back(self, tmp_path, monkeypatch):
        path = str(tmp_path / "bots.json")
        monkeypatch.setattr("ai.online.ogs_bots.fetch_active_bots",
                            lambda **kwargs: ACTIVE_BOTS_FRAME)
        OGSBotRegistry(cache_path=path).refresh()

        # A second registry with no network at all still lists the bots.
        def unreachable(**kwargs):
            raise AssertionError("should not have hit the network")

        monkeypatch.setattr("ai.online.ogs_bots.fetch_active_bots", unreachable)
        reloaded = OGSBotRegistry(cache_path=path)
        assert [b.username for b in reloaded.list_bots()] == ["GnuGo", "cEmbryo"]

    def test_a_broken_entry_does_not_lose_the_rest(self, tmp_path, monkeypatch):
        frame = dict(ACTIVE_BOTS_FRAME)
        frame["999"] = {"username": "no id here"}
        monkeypatch.setattr("ai.online.ogs_bots.fetch_active_bots", lambda **kwargs: frame)
        registry = OGSBotRegistry(cache_path=str(tmp_path / "bots.json"))
        assert len(registry.refresh()) == 2


# ---------------------------------------------------------------------------
# The socket, against a fake OGS on localhost
# ---------------------------------------------------------------------------

class FakeOGS:
    """
    A stand-in for the OGS realtime server.

    Speaks the same array framing: replies to any message carrying a request id
    with a canned response, and can push events at will.
    """

    def __init__(self):
        self.received = []
        self.port = None
        self._server = None
        self._loop = None
        self._thread = None
        self._clients = set()
        self.ready = threading.Event()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        assert self.ready.wait(10), "fake OGS did not start"
        return self

    def stop(self):
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def url(self):
        return f"ws://127.0.0.1:{self.port}"

    def push(self, command, data):
        """Send an unsolicited event to every connected client."""
        message = json.dumps([command, data])
        for ws in list(self._clients):
            asyncio.run_coroutine_threadsafe(ws.send(message), self._loop)

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())
        self._loop.run_forever()

    async def _serve(self):
        self._server = await websockets.serve(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        self.ready.set()

    async def _handle(self, ws):
        self._clients.add(ws)
        try:
            async for raw in ws:
                message = json.loads(raw)
                self.received.append(message)
                command = message[0]
                if command == "net/ping":
                    continue
                if len(message) > 2:            # wants a reply
                    request_id = message[2]
                    reply = {"id": 42, "username": "tester"} \
                        if command == "authenticate" else {"ok": True}
                    await ws.send(json.dumps([request_id, reply, None]))
        except Exception:
            pass
        finally:
            self._clients.discard(ws)


@pytest.fixture
def fake_ogs():
    server = FakeOGS().start()
    yield server
    server.stop()


class TestSocket:
    def test_request_gets_its_reply(self, fake_ogs):
        from ai.online.ogs_socket import OGSSocket, authenticate_payload

        with OGSSocket(fake_ogs.url) as sock:
            who = sock.request("authenticate", authenticate_payload(), timeout=10)
            assert who == {"id": 42, "username": "tester"}

        sent = [m[0] for m in fake_ogs.received]
        assert sent == ["authenticate"]

    def test_a_latch_catches_an_event_sent_before_we_wait(self, fake_ogs):
        """
        active-bots arrives the moment we authenticate — a handler registered
        after the reply lands misses it, which is exactly the bug the latch
        exists for.
        """
        from ai.online.ogs_socket import OGSSocket

        with OGSSocket(fake_ogs.url) as sock:
            arrived = sock.latch("active-bots")
            fake_ogs.push("active-bots", ACTIVE_BOTS_FRAME)
            data = arrived.wait(timeout=10)

        assert set(data) == {"58441", "2007355"}

    def test_events_reach_registered_handlers(self, fake_ogs):
        from ai.online.ogs_socket import OGSSocket

        seen = []
        with OGSSocket(fake_ogs.url) as sock:
            sock.on("game/7/move", seen.append)
            fake_ogs.push("game/7/move", {"game_id": 7, "move_number": 3, "move": "pd"})
            for _ in range(100):
                if seen:
                    break
                threading.Event().wait(0.05)

        assert seen and seen[0]["move"] == "pd"

    def test_a_request_that_is_never_answered_times_out(self, fake_ogs):
        from ai.online.ogs_socket import OGSSocket, OGSSocketError

        with OGSSocket(fake_ogs.url) as sock:
            # The fake only replies to messages that carry a request id, and
            # never to net/ping — so a request for it times out rather than
            # hanging the caller forever.
            sock.on("never", lambda data: None)
            with pytest.raises(OGSSocketError, match="did not send"):
                sock.wait_for("never", timeout=0.5)
