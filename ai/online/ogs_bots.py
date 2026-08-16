"""
ogs_bots.py — who is available to play on OGS right now.

OGS does not expose its bot list over REST. The roster is pushed over the
realtime socket as an `active-bots` event the moment you connect — including
as a guest, so listing opponents needs no account and no credentials at all.

Each bot carries a `config` describing what it will accept: board sizes, time
control ranges per speed, ranked/handicap rules, and how many games it will
play against one person at a time. `playability()` below is a port of the same
check the OGS web client runs (`getAcceptableTimeSetting` in its `lib/bots.ts`),
which is why our picker can grey out a bot for exactly the reason theirs does —
"Bot cannot play at this speed", "Bot doesn't accept ranked games with
handicap".

The roster is cached on disk. It changes as bot owners restart their engines,
but not on the timescale of someone opening the Play page twice, and a cached
list is a much better failure mode than an empty picker when OGS is unreachable.
"""

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from ai.online.ogs_socket import OGSSocket, authenticate_payload, OGSSocketError

logger = logging.getLogger(__name__)

# Where the roster is cached, and for how long it is considered fresh.
DEFAULT_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "ogs_bots.json",
)
CACHE_TTL_SECONDS = 15 * 60

# OGS speed categories, in the order the site presents them.
SPEEDS = ("blitz", "rapid", "live", "correspondence")


def ranking_to_elo(ranking: float) -> float:
    """
    An OGS rank number on this project's Elo scale.

    The two scales are unrelated numbers: OGS's glicko rating has GnuGo at 903,
    while 903 on our scale is roughly 26 kyu — and GnuGo is 18 kyu. What the
    two DO share is the rank they express, so the conversion goes through the
    rank: our table is 500 Elo = 30k with 100 Elo per rank (see ELO_TO_KYU in
    config.py), and OGS's ranking counts the same ranks from the same place.

    Seeding an OGS opponent this way is what makes the Elo our model earns
    against it mean anything.
    """
    return max(500.0, min(3900.0, 500.0 + 100.0 * float(ranking or 0.0)))


def rank_string(ranking: float) -> str:
    """
    OGS's own rank label for a ranking number.

    Below 30 it counts down in kyu, above it counts up in dan — the exact
    rounding matters, since it is what the site displays next to each bot
    (`rankString` in the OGS frontend's rank_utils).
    """
    if ranking is None:
        return "?"
    if ranking < 30:
        return f"{math.ceil(30 - ranking)}k"
    return f"{math.floor(ranking - 29)}d"


@dataclass
class OGSBot:
    """One bot on OGS, as the picker needs it."""

    id: int                       # player id — the challenge is POSTed to this
    username: str
    ranking: float                # OGS rank number (30 = 1d)
    rank: str                     # display label, e.g. "18k"
    rating: float                 # glicko overall rating, on OGS's own scale
    icon: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)

    @property
    def elo(self) -> float:
        """The bot's strength on this project's Elo scale."""
        return ranking_to_elo(self.ranking)

    @property
    def allowed_board_sizes(self) -> Optional[List[int]]:
        """None when the bot publishes no config — then nothing is ruled out."""
        sizes = self.config.get("allowed_board_sizes")
        return list(sizes) if sizes else None

    @property
    def config_version(self) -> int:
        return int(self.config.get("_config_version", 0) or 0)

    @property
    def accepting_challenges(self) -> bool:
        return not self.config.get("decline_new_challenges", False)

    @property
    def settings_published(self) -> bool:
        """
        False for the handful of old bots that publish no config at all.

        OGS still lists them as playable — sending the challenge is the only
        way to find out what they accept — so we do too, but the picker can
        say the settings are unknown.
        """
        return bool(self.config) and self.config_version > 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_event(cls, entry: dict) -> "OGSBot":
        ratings = entry.get("ratings") or {}
        overall = ratings.get("overall") or {}
        ranking = entry.get("ranking")
        return cls(
            id=int(entry["id"]),
            username=entry.get("username") or f"bot {entry['id']}",
            ranking=float(ranking if ranking is not None else 0.0),
            rank=rank_string(ranking),
            rating=float(overall.get("rating") or 0.0),
            icon=entry.get("icon-url") or entry.get("icon"),
            config=entry.get("config") or {},
        )


@dataclass
class GameRequest:
    """What we want to play, checked against a bot's config before challenging."""

    board_size: int = 9
    speed: str = "live"
    system: str = "byoyomi"       # byoyomi | fischer | simple
    ranked: bool = False
    handicap: bool = False
    main_time: int = 300
    period_time: int = 30
    periods: int = 5

    def time_control_parameters(self) -> dict:
        """
        The `time_control_parameters` block of a challenge.

        OGS wants the system named twice — once as `system` and once as
        `time_control` — for backwards compatibility reasons of its own.
        """
        params = {
            "system": self.system,
            "speed": self.speed,
            "pause_on_weekends": False,
            "time_control": self.system,
        }
        if self.system == "byoyomi":
            params.update({
                "main_time": self.main_time,
                "period_time": self.period_time,
                "periods": self.periods,
            })
        elif self.system == "fischer":
            params.update({
                "initial_time": self.main_time,
                "time_increment": self.period_time,
                "max_time": self.main_time * 10,
            })
        elif self.system == "simple":
            params.update({"per_move_time": self.period_time})
        return params


def playability(bot: OGSBot, request: GameRequest) -> Tuple[bool, Optional[str]]:
    """
    Can this bot play the game we want? Returns (playable, reason_if_not).

    The reasons are worded as OGS words them, because a user comparing our
    picker with theirs should not have to work out that they mean the same
    thing.
    """
    config = bot.config

    if not bot.accepting_challenges:
        return False, "Bot is not accepting new challenges"

    # Bots that publish no config are still offered by OGS — there is nothing
    # to check them against, and the challenge is the only way to find out.
    # Ruling them out here would hide three bots their site happily lists.
    if not bot.settings_published:
        return True, None

    sizes = bot.allowed_board_sizes
    if sizes is not None and request.board_size not in sizes:
        return False, f"Bot does not play {request.board_size}×{request.board_size}"

    if request.ranked and not config.get("allow_ranked", True):
        return False, "Bot doesn't accept ranked games"
    if not request.ranked and not config.get("allow_unranked", True):
        return False, "Bot only accepts ranked games"

    if request.handicap:
        key = "allow_ranked_handicap" if request.ranked else "allow_unranked_handicap"
        if not config.get(key, True):
            kind = "ranked" if request.ranked else "unranked"
            return False, f"Bot doesn't accept {kind} games with handicap"

    systems = config.get("allowed_time_control_systems")
    if systems and request.system not in systems:
        return False, f"Bot does not play with {request.system} timing"

    if not _time_settings_fit(config, request, request.speed):
        # A v1 config has no rapid settings; OGS maps rapid onto live for those.
        if bot.config_version == 1 and request.speed == "rapid" and \
                _time_settings_fit(config, request, "live"):
            return True, None
        return False, "Bot cannot play at this speed"

    return True, None


def _time_settings_fit(config: dict, request: GameRequest, speed: str) -> bool:
    """Whether our clock falls inside the bot's allowed range for `speed`."""
    settings = config.get(f"allowed_{speed}_settings")
    if not settings:
        return False
    limits = settings.get(request.system)
    if not limits:
        return False

    def within(value, bounds) -> bool:
        if not bounds or len(bounds) != 2:
            return True
        return bounds[0] <= value <= bounds[1]

    if request.system == "byoyomi":
        return (within(request.main_time, limits.get("main_time_range"))
                and within(request.period_time, limits.get("period_time_range"))
                and within(request.periods, limits.get("periods_range")))
    if request.system == "fischer":
        return (within(request.main_time, limits.get("initial_time_range"))
                and within(request.period_time, limits.get("time_increment_range"))
                and within(request.main_time * 10, limits.get("max_time_range")))
    if request.system == "simple":
        return within(request.period_time, limits.get("per_move_time_range"))
    return False


class OGSBotRegistry:
    """
    The roster, fetched over the socket and cached on disk.

    Thread-safe: the web layer calls `list_bots()` from request threads while a
    match may be running on another.
    """

    def __init__(self, cache_path: str = DEFAULT_CACHE_PATH,
                 ttl: float = CACHE_TTL_SECONDS):
        self.cache_path = cache_path
        self.ttl = ttl
        self._bots: Dict[int, OGSBot] = {}
        self._fetched_at: float = 0.0
        self._lock = threading.Lock()

    # ---- Reading ---------------------------------------------------------

    def list_bots(self, refresh: bool = False,
                  timeout: float = 20.0) -> List[OGSBot]:
        """
        Every bot OGS reported, weakest first.

        Uses the cached roster while it is fresh; falls back to a stale cache
        if OGS cannot be reached, because a slightly old list beats none.
        `timeout` bounds the fetch — a page rendering an opponent picker should
        pass something short rather than wait out a dead connection.
        """
        with self._lock:
            fresh = self._bots and (time.time() - self._fetched_at) < self.ttl
        if fresh and not refresh:
            return self._sorted()

        if not self._bots:
            self._load_cache()
            with self._lock:
                fresh = self._bots and (time.time() - self._fetched_at) < self.ttl
            if fresh and not refresh:
                return self._sorted()

        try:
            self.refresh(timeout=timeout)
        except OGSSocketError as exc:
            if not self._bots:
                raise
            logger.warning("Using the cached OGS roster: %s", exc)
        return self._sorted()

    def get(self, bot_id: int) -> Optional[OGSBot]:
        with self._lock:
            bot = self._bots.get(int(bot_id))
        if bot is None:
            self.list_bots()
            with self._lock:
                bot = self._bots.get(int(bot_id))
        return bot

    @property
    def fetched_at(self) -> float:
        return self._fetched_at

    def _sorted(self) -> List[OGSBot]:
        with self._lock:
            return sorted(self._bots.values(), key=lambda b: b.ranking)

    # ---- Fetching --------------------------------------------------------

    def refresh(self, timeout: float = 20.0) -> List[OGSBot]:
        """Pull a fresh roster from OGS. Raises OGSSocketError on failure."""
        raw = fetch_active_bots(timeout=timeout)
        bots = {}
        for entry in raw.values():
            try:
                bot = OGSBot.from_event(entry)
            except (KeyError, TypeError, ValueError):
                logger.warning("Skipping an unreadable bot entry from OGS")
                continue
            bots[bot.id] = bot

        with self._lock:
            self._bots = bots
            self._fetched_at = time.time()
        self._save_cache()
        logger.info("OGS roster: %d bots", len(bots))
        return self._sorted()

    # ---- Cache -----------------------------------------------------------

    def _load_cache(self) -> None:
        try:
            with open(self.cache_path) as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            return
        bots = {}
        for entry in payload.get("bots", []):
            try:
                bots[int(entry["id"])] = OGSBot(**entry)
            except (KeyError, TypeError, ValueError):
                continue
        if bots:
            with self._lock:
                self._bots = bots
                self._fetched_at = float(payload.get("fetched_at") or 0.0)

    def _save_cache(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w") as handle:
                json.dump({
                    "fetched_at": self._fetched_at,
                    "bots": [bot.to_dict() for bot in self._sorted()],
                }, handle, indent=1)
        except OSError as exc:
            logger.warning("Could not cache the OGS roster: %s", exc)


def fetch_active_bots(timeout: float = 20.0, url: Optional[str] = None) -> dict:
    """
    Connect as a guest and return the raw `active-bots` payload.

    The event is pushed as soon as the connection is authenticated — before the
    `authenticate` reply lands — so the latch is armed first.
    """
    socket = OGSSocket(url) if url else OGSSocket()
    with socket as sock:
        arrived = sock.latch("active-bots")
        sock.request("authenticate", authenticate_payload(), timeout=timeout)
        return arrived.wait(timeout=timeout) or {}


# The registry every caller shares, so one roster fetch serves the whole app.
registry = OGSBotRegistry()
