#!/usr/bin/env python3
"""
ogs_probe.py — check the OGS setup without playing anything.

Proves the whole credential path in one command: sign in, read the account,
authenticate the realtime socket with the token that sign-in produced, and pull
the bot roster. It creates no challenge and plays no game.

    python scripts/ogs_probe.py

Prints what it authenticated as, whether the account is registered as a bot,
and how many bots could play the default game. Any failure is reported with
what to do about it.
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.online.ogs_bots import GameRequest, playability, registry, rank_string
from ai.online.ogs_rest import OGSAuthError, OGSClient, OGSRequestError
from ai.online.ogs_socket import OGSSocket, OGSSocketError, authenticate_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the OGS connection")
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--ranked", action="store_true",
                        help="check which bots would accept a RANKED game")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    print("OGS connection check")
    print("=" * 52)

    # 1. The roster needs no account at all, so it is the first thing to fail
    #    if the network or OGS itself is the problem.
    try:
        bots = registry.list_bots(refresh=True)
    except OGSSocketError as exc:
        print(f"  roster    FAILED  {exc}")
        print("\nCould not reach the OGS realtime API. Check the network; no "
              "credentials are involved in this step.")
        return 1
    print(f"  roster    ok      {len(bots)} bots online")

    # 2. Sign in.
    try:
        client = OGSClient()
        client.token()
    except OGSAuthError as exc:
        print(f"  sign-in   FAILED  {exc}")
        return 1

    try:
        me = client.me()
    except OGSRequestError as exc:
        print(f"  account   FAILED  {exc}")
        return 1

    ranking = me.get("ranking")
    print(f"  sign-in   ok      {me.get('username')} "
          f"(id {me.get('id')}, {rank_string(ranking) if ranking is not None else '?'})")

    is_bot = bool(me.get("is_bot") or (me.get("ui_class") or "").find("bot") >= 0)
    print(f"  account   {'bot' if is_bot else 'human':7s} "
          + ("registered as a bot account — ranked games are fine"
             if is_bot else
             "not registered as a bot — keep games UNRANKED until it is"))

    # 3. The socket token, which is what gameplay actually authenticates with.
    try:
        jwt = client.user_jwt()
    except (OGSAuthError, OGSRequestError) as exc:
        print(f"  socket    FAILED  {exc}")
        return 1

    try:
        with OGSSocket() as sock:
            who = sock.request("authenticate", authenticate_payload(jwt=jwt))
    except OGSSocketError as exc:
        print(f"  socket    FAILED  {exc}")
        return 1

    if not who or not who.get("id"):
        print("  socket    FAILED  the socket authenticated as a guest, not as us")
        return 1
    print(f"  socket    ok      authenticated as {who.get('username')}")

    # 4. What we could actually play right now.
    request = GameRequest(board_size=args.board_size, ranked=args.ranked)
    playable = [b for b in bots if playability(b, request)[0]]
    kind = "ranked" if args.ranked else "unranked"
    print(f"  opponents ok      {len(playable)} of {len(bots)} can play a "
          f"{args.board_size}×{args.board_size} {kind} live game")

    if playable:
        print("\nWeakest few:")
        for bot in playable[:5]:
            print(f"    {bot.rank:>4s}  {bot.username}")

    print("\nEverything checks out. No challenge was created.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
