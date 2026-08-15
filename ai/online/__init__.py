"""
online/ — Bridges to Go servers (OGS today's target, others later).

Everything in here implements `ai.players.RemotePlayer`, which is all the match
runner in `ai/match.py` knows about. That keeps the network protocol of a given
server entirely inside this package.
"""
