"""
test_play_opponent_choice.py — a human game can be played against ANY model.

The Play page used to be hardwired to the Dashboard's active model: board size,
komi, ruleset and the network all came from it. Now the setup panel picks an
opponent (defaulting to the active model), so /api/game/new has to honour a
`model_id` and lock the game to THAT model's settings.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.app import create_app
from model_manager import ModelManager


@pytest.fixture
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


@pytest.fixture
def manager():
    return ModelManager()


def test_opponents_lists_models_and_the_active_default(client, manager):
    payload = client.get('/api/game/opponents').get_json()

    listed = {m['model_id'] for m in payload['models']}
    assert listed == {info.id for info in manager.list_models()}
    assert payload['active_model_id'] == manager.get_active_model_id()

    for model in payload['models']:
        # The picker shows what a game against this model would be played on.
        assert {'name', 'board_size', 'komi', 'ruleset', 'elo'} <= set(model)


def test_new_game_defaults_to_the_active_model(client, manager):
    active = manager.get_active_model()
    if active is None:
        pytest.skip("no active model in this workspace")

    data = client.post('/api/game/new', json={'num_simulations': 5}).get_json()
    assert data['model_id'] == active.id
    assert data['board_size'] == active.board_size


def test_new_game_locks_to_the_chosen_model(client, manager):
    active_id = manager.get_active_model_id()
    other = next((i for i in manager.list_models() if i.id != active_id), None)
    if other is None:
        pytest.skip("only one model in this workspace")

    data = client.post('/api/game/new', json={
        'model_id': other.id, 'num_simulations': 5,
    }).get_json()

    assert data['model_id'] == other.id
    assert data['model_name'] == other.name
    # Board settings come from the opponent, not from the active model.
    assert data['board_size'] == other.board_size
    assert data['komi'] == other.komi
    assert data['ruleset'] == other.ruleset
    assert data['state']['board_size'] == other.board_size


def test_bot_plays_on_the_chosen_models_board(client, manager):
    """A model of a different size than the active one must still be playable."""
    active = manager.get_active_model()
    other = next((i for i in manager.list_models()
                  if active is None or i.board_size != active.board_size), None)
    if other is None:
        pytest.skip("every model in this workspace shares one board size")

    gid = client.post('/api/game/new', json={
        'model_id': other.id, 'num_simulations': 5,
    }).get_json()['game_id']

    assert client.post('/api/game/move',
                       json={'game_id': gid, 'row': 0, 'col': 0}).status_code == 200
    reply = client.post('/api/game/bot_move', json={'game_id': gid})
    assert reply.status_code == 200
    assert reply.get_json()['state']['board_size'] == other.board_size


def test_unknown_model_is_rejected(client):
    resp = client.post('/api/game/new', json={'model_id': 'no-such-model'})
    assert resp.status_code == 404
    assert 'no-such-model' in resp.get_json()['error']
