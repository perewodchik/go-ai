"""
test_play_turn_guard.py — the human must not be able to play the bot's stones.

Regression test for the fast-double-click bug: two /api/game/move calls issued
before the bot replies would place a stone for the bot's color, letting the
player play both sides. The server now rejects a move made off the human's turn.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def _new_game(client, color="black", sims=10):
    resp = client.post('/api/game/new', json={
        'player_color': color, 'num_simulations': sims,
    })
    data = resp.get_json()
    assert 'game_id' in data, data
    return data['game_id']


def test_second_move_before_bot_reply_is_rejected(client):
    gid = _new_game(client, "black")

    first = client.post('/api/game/move', json={'game_id': gid, 'row': 0, 'col': 0})
    assert first.status_code == 200

    # Racing second click before /bot_move — it is now White's (bot's) turn.
    second = client.post('/api/game/move', json={'game_id': gid, 'row': 1, 'col': 1})
    assert second.status_code == 409
    assert second.get_json()['error'] == "Not your turn"


def test_human_can_move_again_after_bot_reply(client):
    gid = _new_game(client, "black")
    client.post('/api/game/move', json={'game_id': gid, 'row': 0, 'col': 0})
    assert client.post('/api/game/bot_move', json={'game_id': gid}).status_code == 200
    # Back to the human's turn.
    again = client.post('/api/game/move', json={'game_id': gid, 'row': 2, 'col': 2})
    assert again.status_code == 200


def test_pass_off_turn_is_rejected(client):
    gid = _new_game(client, "black")
    client.post('/api/game/move', json={'game_id': gid, 'row': 0, 'col': 0})
    # It's the bot's turn now; a pass must not go through as the bot.
    resp = client.post('/api/game/pass', json={'game_id': gid})
    assert resp.status_code == 409
    assert resp.get_json()['error'] == "Not your turn"


def test_white_player_moves_after_bot_opens(client):
    # Human is White; the bot (Black) opens inside /new, so it's the human's turn.
    gid = _new_game(client, "white")
    resp = client.post('/api/game/move', json={'game_id': gid, 'row': 4, 'col': 4})
    assert resp.status_code == 200
