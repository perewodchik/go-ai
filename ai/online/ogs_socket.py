"""
ogs_socket.py — OGS's realtime protocol, which is NOT socket.io.

Despite the name of the endpoint everyone expects, online-go.com speaks a plain
WebSocket with a tiny JSON-array framing of its own (see `GobanSocket.ts` in
online-go/goban):

    client -> server   ["command", data]                 fire and forget
                       ["command", data, request_id]     wants a reply
    server -> client   ["command", data]                 an event
                       [request_id, data, error]         the reply to a request

So there is no engine.io handshake, no packet-type prefixes, and no socket.io
dependency — just JSON over a socket. That is the whole protocol.

Threading model
---------------
The rest of this project is synchronous: `ai/match.py` drives a game by calling
`select_move()` and blocking until a move comes back. So this client owns a
background thread running an asyncio loop, and exposes a synchronous API:

    sock = OGSSocket()
    sock.connect()                          # blocks until connected
    sock.send("game/connect", {...})        # fire and forget
    reply = sock.request("authenticate", {...})   # blocks for the reply
    sock.on("game/123/move", handler)       # handler runs on the socket thread

Handlers run on the socket thread, so they must not block; the ones in this
package only push onto queues.

The connection is kept alive with a `net/ping` every 10 seconds — OGS drops
silent connections — and reconnects with backoff, replaying the subscriptions
the caller registered through `on_reconnect`.
"""

import asyncio
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

try:
    import websockets
except ImportError:  # pragma: no cover - dependency is declared in requirements
    websockets = None

logger = logging.getLogger(__name__)

OGS_WEBSOCKET_URL = "wss://online-go.com"

# OGS drops connections that go quiet. Its own client pings every 10s.
PING_INTERVAL = 10.0
CONNECT_TIMEOUT = 20.0
REQUEST_TIMEOUT = 30.0

# Reconnect backoff, in seconds, then the last value repeats.
BACKOFF = (1, 2, 5, 10, 20, 30)


class OGSSocketError(RuntimeError):
    """The socket could not be connected, or a request failed."""


class OGSSocket:
    """
    A synchronous client for the OGS realtime API.

    One instance owns one connection and one background thread. It is safe to
    call `send`, `request` and `on` from any thread.
    """

    def __init__(self, url: str = OGS_WEBSOCKET_URL, client_name: str = "go-ai"):
        if websockets is None:
            raise OGSSocketError(
                "The 'websockets' package is required to talk to OGS: pip install websockets"
            )
        self.url = url
        self.client_name = client_name

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ws = None
        self._connected = threading.Event()
        self._stopping = False

        self._request_id = 0
        self._pending: Dict[int, "_Pending"] = {}
        self._handlers: Dict[str, List[Callable[[Any], None]]] = {}
        self._lock = threading.Lock()

        # Called after every (re)connect, on the socket thread. This is where a
        # caller re-authenticates and re-subscribes to its games.
        self.on_reconnect: Optional[Callable[["OGSSocket"], None]] = None

    # ---- Lifecycle -------------------------------------------------------

    def connect(self, timeout: float = CONNECT_TIMEOUT) -> None:
        """Start the socket thread and block until the connection is up."""
        if self._thread is not None:
            if not self._connected.wait(timeout):
                raise OGSSocketError(f"Timed out connecting to {self.url}")
            return

        self._stopping = False
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="ogs-socket")
        self._thread.start()

        if not self._connected.wait(timeout):
            raise OGSSocketError(f"Timed out connecting to {self.url}")

    def close(self) -> None:
        """Stop pinging, close the connection and stop the thread."""
        self._stopping = True
        loop, ws = self._loop, self._ws
        if loop is not None and ws is not None:
            asyncio.run_coroutine_threadsafe(ws.close(), loop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None
        self._connected.clear()

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def __enter__(self) -> "OGSSocket":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- Sending ---------------------------------------------------------

    def send(self, command: str, data: Any = None) -> None:
        """Send a message without waiting for a reply."""
        payload = [command, data] if data is not None else [command]
        self._write(payload)

    def request(self, command: str, data: Any = None,
                timeout: float = REQUEST_TIMEOUT) -> Any:
        """
        Send a message and block for the server's reply.

        Raises OGSSocketError if the server replies with an error, or if no
        reply arrives within `timeout`.
        """
        with self._lock:
            self._request_id += 1
            request_id = self._request_id
            pending = _Pending()
            self._pending[request_id] = pending

        self._write([command, data, request_id])

        if not pending.event.wait(timeout):
            with self._lock:
                self._pending.pop(request_id, None)
            raise OGSSocketError(f"OGS did not answer {command} within {timeout:.0f}s")

        if pending.error is not None:
            raise OGSSocketError(f"OGS rejected {command}: {pending.error}")
        return pending.result

    def _write(self, payload: list) -> None:
        loop, ws = self._loop, self._ws
        if loop is None or ws is None or not self._connected.is_set():
            raise OGSSocketError("Not connected to OGS")
        message = json.dumps(payload)
        future = asyncio.run_coroutine_threadsafe(ws.send(message), loop)
        future.result(timeout=10)

    # ---- Receiving -------------------------------------------------------

    def on(self, command: str, handler: Callable[[Any], None]) -> None:
        """
        Register a handler for a server event, e.g. `game/123/move`.

        Handlers run on the socket thread and must not block.
        """
        with self._lock:
            self._handlers.setdefault(command, []).append(handler)

    def off(self, command: str, handler: Optional[Callable[[Any], None]] = None) -> None:
        """Remove one handler, or every handler for a command."""
        with self._lock:
            if handler is None:
                self._handlers.pop(command, None)
            elif command in self._handlers:
                self._handlers[command] = [
                    h for h in self._handlers[command] if h is not handler
                ]

    def latch(self, command: str) -> "_Latch":
        """
        Start catching `command` NOW, and wait for it later.

        Some OGS events are pushed the moment you connect or authenticate —
        `active-bots` is sent before the `authenticate` reply comes back. A
        handler registered after that point never sees it, so the caller arms
        a latch first, performs the action, then waits:

            arrived = sock.latch("active-bots")
            sock.request("authenticate", ...)
            bots = arrived.wait(timeout=15)
        """
        latch = _Latch(self, command)
        self.on(command, latch._capture)
        return latch

    def wait_for(self, command: str, timeout: float = 30.0) -> Any:
        """
        Block until `command` is received once, and return its payload.

        Only safe for events that arrive in response to something you do after
        calling this; for anything pushed on connect, use `latch()`.
        """
        return self.latch(command).wait(timeout)

    def _dispatch(self, message: list) -> None:
        head = message[0]
        data = message[1] if len(message) > 1 else None
        error = message[2] if len(message) > 2 else None

        # A reply carries the request id we sent, so an int head is a reply.
        if isinstance(head, int):
            with self._lock:
                pending = self._pending.pop(head, None)
            if pending is not None:
                pending.result = data
                pending.error = error
                pending.event.set()
            return

        with self._lock:
            handlers = list(self._handlers.get(head, ()))
        for handler in handlers:
            try:
                handler(data)
            except Exception:
                logger.exception("OGS handler for %s failed", head)

    # ---- The socket thread ----------------------------------------------

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connection_loop())
        finally:
            self._loop.close()
            self._loop = None

    async def _connection_loop(self) -> None:
        attempt = 0
        while not self._stopping:
            try:
                async with websockets.connect(self.url, open_timeout=CONNECT_TIMEOUT,
                                              max_size=None) as ws:
                    self._ws = ws
                    self._connected.set()
                    attempt = 0
                    logger.info("Connected to OGS at %s", self.url)

                    if self.on_reconnect:
                        try:
                            self.on_reconnect(self)
                        except Exception:
                            logger.exception("OGS reconnect hook failed")

                    ping = asyncio.ensure_future(self._ping_loop(ws))
                    try:
                        async for raw in ws:
                            try:
                                message = json.loads(raw)
                            except ValueError:
                                logger.warning("Unparseable frame from OGS: %.120s", raw)
                                continue
                            if isinstance(message, list) and message:
                                self._dispatch(message)
                    finally:
                        ping.cancel()
            except Exception as exc:
                if self._stopping:
                    break
                logger.warning("OGS connection lost (%s)", exc)
            finally:
                self._connected.clear()
                self._ws = None
                # Anyone blocked on a reply is never getting one now.
                self._fail_pending("connection to OGS was lost")

            if self._stopping:
                break
            delay = BACKOFF[min(attempt, len(BACKOFF) - 1)]
            attempt += 1
            logger.info("Reconnecting to OGS in %ss", delay)
            await asyncio.sleep(delay)

    async def _ping_loop(self, ws) -> None:
        while True:
            await asyncio.sleep(PING_INTERVAL)
            try:
                await ws.send(json.dumps([
                    "net/ping",
                    {"client": int(time.time() * 1000), "drift": 0, "latency": 0},
                ]))
            except Exception:
                return

    def _fail_pending(self, reason: str) -> None:
        with self._lock:
            pending, self._pending = self._pending, {}
        for item in pending.values():
            item.error = reason
            item.event.set()


class _Latch:
    """A one-shot catcher for a pushed event, armed before the event can fire."""

    __slots__ = ("_socket", "_command", "_event", "_data")

    def __init__(self, socket: "OGSSocket", command: str) -> None:
        self._socket = socket
        self._command = command
        self._event = threading.Event()
        self._data: Any = None

    def _capture(self, data: Any) -> None:
        if not self._event.is_set():
            self._data = data
            self._event.set()

    @property
    def fired(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float = 30.0) -> Any:
        try:
            if not self._event.wait(timeout):
                raise OGSSocketError(
                    f"OGS did not send '{self._command}' within {timeout:.0f}s"
                )
            return self._data
        finally:
            self.cancel()

    def cancel(self) -> None:
        self._socket.off(self._command, self._capture)


class _Pending:
    """One in-flight request waiting for its reply."""

    __slots__ = ("event", "result", "error")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.result: Any = None
        self.error: Any = None


def authenticate_payload(jwt: str = "", client_name: str = "go-ai",
                         bot_username: Optional[str] = None,
                         bot_apikey: Optional[str] = None) -> dict:
    """
    The `authenticate` message body.

    Three ways in, all through the same message:
      * guest       — jwt "" (enough to read the bot roster)
      * a person    — the `user_jwt` from GET /api/v1/ui/config/
      * a bot       — its username and API key, no password anywhere
    """
    payload = {
        "jwt": jwt or "",
        "client": client_name,
        "device_id": "go-ai",
    }
    if bot_username and bot_apikey:
        payload["bot_username"] = bot_username
        payload["bot_apikey"] = bot_apikey
    return payload
