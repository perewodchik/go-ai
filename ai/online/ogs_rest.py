"""
ogs_rest.py — signing in to OGS, and the handful of REST calls a game needs.

Gameplay happens over the realtime socket (see `ogs_socket.py`), but three
things can only be done over REST, so this module exists:

    * getting an access token, and from it the `user_jwt` the socket
      authenticates with
    * creating a challenge against a bot
    * withdrawing a challenge the bot never accepted

Credentials
-----------
OGS uses OAuth2 with the resource-owner-password grant: an application
registered at https://online-go.com/oauth2/applications/ gives a client id and
secret, which are exchanged along with the account's own username and password
for a bearer token.

They are read from the environment first, then from `ogs_credentials.json` in
the project root (gitignored). Nothing here logs them, and no credential is
ever put in a URL — tokens go in the Authorization header, where they do not
end up in server logs or browser history.
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from ai.online.ogs_bots import GameRequest

logger = logging.getLogger(__name__)

OGS_HOST = "https://online-go.com"
TOKEN_URL = f"{OGS_HOST}/oauth2/token/"
API = f"{OGS_HOST}/api/v1"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CREDENTIALS_PATH = os.path.join(PROJECT_ROOT, "ogs_credentials.json")

REQUEST_TIMEOUT = 20.0
# Refresh a little before the token actually dies, so a long game never has a
# call fail on a token that expired mid-request.
TOKEN_EXPIRY_MARGIN = 300.0


# OGS stores client secrets hashed, and shows the usable one ONCE — on the
# page right after the application is created. Copying it later from the
# application's own edit page yields the hash, which looks like a secret and is
# rejected as `invalid_client`. Worth naming exactly, because the fix is not
# "check your secret" but "make a new application".
HASHED_SECRET_HELP = (
    "The client secret in the credentials is the HASHED one OGS stores, not the "
    "usable one.\n"
    "OGS shows a client secret exactly once — on the confirmation page right "
    "after you create the application. The value on the application's edit page "
    "afterwards is a hash (it starts with 'pbkdf2_') and can never be used.\n"
    "Fix: open https://online-go.com/oauth2/applications/, delete the "
    "application, create it again (Confidential + Resource owner "
    "password-based), and copy the Client secret from that first page before "
    "navigating away."
)


def _looks_hashed(secret: str) -> bool:
    """Django hashes look like `pbkdf2_sha256$390000$salt$digest`."""
    return str(secret).startswith(("pbkdf2_", "argon2", "bcrypt"))


class OGSAuthError(RuntimeError):
    """Credentials are missing, malformed, or rejected by OGS."""


class OGSRequestError(RuntimeError):
    """A REST call failed. Carries the status code and OGS's own message."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


@dataclass
class OGSCredentials:
    """One OGS account, plus the OAuth2 application used to reach it."""

    username: str
    password: str
    client_id: str
    client_secret: str

    @classmethod
    def load(cls, path: str = CREDENTIALS_PATH) -> "OGSCredentials":
        """
        Environment first (OGS_USERNAME, OGS_PASSWORD, OGS_CLIENT_ID,
        OGS_CLIENT_SECRET), then the credentials file.
        """
        env = {
            "username": os.environ.get("OGS_USERNAME"),
            "password": os.environ.get("OGS_PASSWORD"),
            "client_id": os.environ.get("OGS_CLIENT_ID"),
            "client_secret": os.environ.get("OGS_CLIENT_SECRET"),
        }
        if all(env.values()):
            return cls(**env)  # type: ignore[arg-type]

        try:
            with open(path) as handle:
                data = json.load(handle)
        except FileNotFoundError:
            raise OGSAuthError(
                f"No OGS credentials. Set OGS_USERNAME / OGS_PASSWORD / "
                f"OGS_CLIENT_ID / OGS_CLIENT_SECRET, or create {path}."
            )
        except ValueError as exc:
            raise OGSAuthError(f"{path} is not valid JSON: {exc}")

        missing = [k for k in ("username", "password", "client_id", "client_secret")
                   if not str(data.get(k, "")).strip()]
        if missing:
            raise OGSAuthError(f"{path} is missing: {', '.join(missing)}")

        if _looks_hashed(data["client_secret"]):
            raise OGSAuthError(HASHED_SECRET_HELP)

        return cls(
            username=data["username"].strip(),
            password=data["password"],
            client_id=data["client_id"].strip(),
            client_secret=data["client_secret"].strip(),
        )

    def __repr__(self) -> str:  # keep secrets out of tracebacks and logs
        return f"OGSCredentials(username={self.username!r}, client_id=<set>, secret=<hidden>)"


class OGSClient:
    """
    Authenticated access to the OGS REST API.

    One instance per account. Thread-safe: the token is fetched once and
    refreshed under a lock, so several games can share a client.
    """

    def __init__(self, credentials: Optional[OGSCredentials] = None,
                 host: str = OGS_HOST):
        self.credentials = credentials or OGSCredentials.load()
        self.host = host.rstrip("/")
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "go-ai (self-play research bot)"

        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._lock = threading.Lock()
        self._me: Optional[dict] = None

    # ---- Auth ------------------------------------------------------------

    def token(self) -> str:
        """A valid access token, fetched or refreshed as needed."""
        with self._lock:
            if self._token and time.time() < self._token_expires_at:
                return self._token

            response = self.session.post(
                f"{self.host}/oauth2/token/",
                data={
                    "grant_type": "password",
                    "username": self.credentials.username,
                    "password": self.credentials.password,
                    "client_id": self.credentials.client_id,
                    "client_secret": self.credentials.client_secret,
                },
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code != 200:
                raise OGSAuthError(_auth_hint(response))

            payload = response.json()
            self._token = payload["access_token"]
            self._token_expires_at = time.time() + float(payload.get("expires_in", 3600)) \
                - TOKEN_EXPIRY_MARGIN
            logger.info("Signed in to OGS as %s", self.credentials.username)
            return self._token

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token()}"}

    # ---- Calls -----------------------------------------------------------

    def _call(self, method: str, path: str, **kwargs) -> Any:
        url = f"{self.host}{path}"
        response = self.session.request(
            method, url, headers=self._headers(), timeout=REQUEST_TIMEOUT, **kwargs
        )
        if response.status_code == 401:
            # The token may have been revoked; drop it so the next call retries.
            with self._lock:
                self._token = None
            raise OGSRequestError("OGS rejected our credentials", 401)
        if response.status_code >= 400:
            raise OGSRequestError(_error_message(response), response.status_code)
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError:
            return None

    def ui_config(self) -> dict:
        """
        The client config for our account — most importantly `user_jwt`, which
        is what the realtime socket authenticates with.
        """
        return self._call("GET", "/api/v1/ui/config/")

    def user_jwt(self) -> str:
        jwt = (self.ui_config() or {}).get("user_jwt")
        if not jwt:
            raise OGSAuthError("OGS did not return a socket token for this account")
        return jwt

    def me(self, refresh: bool = False) -> dict:
        """Our own account: id, username, ranking, and whether it is a bot."""
        if self._me is None or refresh:
            self._me = self._call("GET", "/api/v1/me/")
        return self._me

    def game(self, game_id: int) -> dict:
        """The authoritative record of a game, including its full move list."""
        return self._call("GET", f"/api/v1/games/{int(game_id)}/")

    def ongoing_games(self) -> list:
        """Games of ours that have not ended."""
        payload = self._call("GET", "/api/v1/me/games/?ended__isnull=true&page_size=50")
        return (payload or {}).get("results") or []

    def challenge_bot(self, bot_id: int, request: GameRequest,
                      name: str = "go-ai vs bot",
                      our_color: str = "automatic") -> Dict[str, Any]:
        """
        Challenge a bot to a game, and return {challenge_id, game_id}.

        The game id comes back immediately, before the bot has accepted — the
        game only starts once it does, which the caller waits for.

        The payload mirrors what the OGS web client sends when you pick a
        computer opponent; anything omitted here is a field it also omits.
        """
        payload = {
            "challenger_color": our_color,
            "invite_only": False,
            "min_ranking": -1000,
            "max_ranking": 1000,
            "game": {
                "name": name,
                "rules": "japanese",
                "ranked": bool(request.ranked),
                "width": int(request.board_size),
                "height": int(request.board_size),
                # -1 means "let OGS decide", 0 means none. We ask for none: a
                # negotiated handicap would change what the result tells us.
                "handicap": -1 if request.handicap else 0,
                "komi_auto": "automatic",
                "disable_analysis": False,
                "initial_state": None,
                "private": False,
                "pause_on_weekends": False,
                "time_control": request.system,
                "time_control_parameters": request.time_control_parameters(),
            },
        }

        result = self._call("POST", f"/api/v1/players/{int(bot_id)}/challenge",
                            json=payload) or {}

        game = result.get("game")
        game_id = game.get("id") if isinstance(game, dict) else game
        challenge_id = result.get("challenge")
        if not game_id or not challenge_id:
            raise OGSRequestError(f"OGS accepted the challenge but returned {result!r}")

        logger.info("Challenged bot %s: challenge %s, game %s",
                    bot_id, challenge_id, game_id)
        return {"challenge_id": int(challenge_id), "game_id": int(game_id)}

    def cancel_challenge(self, challenge_id: int) -> None:
        """Withdraw a challenge. Safe to call on one that is already gone."""
        try:
            self._call("DELETE", f"/api/v1/me/challenges/{int(challenge_id)}")
        except OGSRequestError as exc:
            if exc.status not in (404, 403):
                raise


def _error_message(response) -> str:
    """OGS's own words for a failure, when it gives any."""
    try:
        body = response.json()
    except ValueError:
        return f"OGS returned {response.status_code}"
    if isinstance(body, dict):
        for key in ("error", "detail", "message", "errors"):
            if body.get(key):
                return f"{body[key]} (HTTP {response.status_code})"
    return f"OGS returned {response.status_code}: {str(body)[:200]}"


def _auth_hint(response) -> str:
    """
    Turn an OAuth2 rejection into something actionable.

    The three ways this goes wrong all look the same in the raw response, and
    all have different fixes.
    """
    try:
        body = response.json()
    except ValueError:
        body = {}
    error = body.get("error", "")
    description = body.get("error_description", "")

    if error == "invalid_client":
        return ("OGS did not recognise the client id/secret.\n" + HASHED_SECRET_HELP)
    if error == "unsupported_grant_type":
        return ("The OAuth2 application is not set to 'Resource owner "
                "password-based'. Edit it at "
                "https://online-go.com/oauth2/applications/ and change the "
                "authorization grant type.")
    if error == "invalid_grant":
        return "OGS rejected the account username or password."
    return (f"OGS refused to sign us in (HTTP {response.status_code}"
            + (f": {error} {description}" if error else "") + ")")
