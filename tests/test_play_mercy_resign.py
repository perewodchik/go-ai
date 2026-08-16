"""
test_play_mercy_resign.py — bots give up games they have already lost.

Self-play has had a mercy rule for a while; a bot PLAYING one (a human game on
the Play page, or a bot-vs-bot match) used to grind every decided endgame out
to the last neutral point. `ai/mercy_rule.py` is the version for played games,
and this covers the two things that can go wrong with it: firing when it should
not (early, or on a single value spike), and not firing when it should.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.mercy_rule import MercyRule
from ai.players import ModelPlayer
from game.game_state import GameState, MOVE_RESIGN


def _rule(**kwargs):
    defaults = dict(enabled=True, threshold=0.9, consecutive=3,
                    min_move_factor=1.0, board_size=9)
    defaults.update(kwargs)
    return MercyRule(**defaults)


class TestMercyRule:
    def test_disabled_never_fires(self):
        rule = _rule(enabled=False)
        assert not any(rule.observe(-1.0, move) for move in range(200))

    def test_needs_a_streak_not_one_bad_evaluation(self):
        rule = _rule()
        assert rule.observe(-1.0, 100) is False
        assert rule.observe(-1.0, 101) is False
        assert rule.observe(-1.0, 102) is True   # third in a row

    def test_a_single_recovery_resets_the_streak(self):
        rule = _rule()
        rule.observe(-1.0, 100)
        rule.observe(-1.0, 101)
        rule.observe(0.2, 102)                   # position is fine again
        assert rule.observe(-1.0, 103) is False
        assert rule.observe(-1.0, 104) is False
        assert rule.observe(-1.0, 105) is True

    def test_never_fires_before_the_minimum_move(self):
        rule = _rule()                            # min move = 81 on 9x9
        for move in range(0, 80):
            assert rule.observe(-1.0, move) is False
        assert rule.observe(-1.0, 81) is True     # streak was already long

    def test_a_merely_bad_position_is_not_hopeless(self):
        rule = _rule()
        assert not any(rule.observe(-0.85, move) for move in range(100, 130))

    def test_reset_clears_the_evidence(self):
        rule = _rule()
        rule.observe(-1.0, 100)
        rule.observe(-1.0, 101)
        rule.reset()
        assert rule.observe(-1.0, 102) is False


class _StubMCTS:
    """Stands in for the search: fixed move, scripted root values."""

    def __init__(self, values):
        self._values = list(values)
        self.root_value = 0.0

    def search(self, state, temperature=0.0, add_noise=False, **kwargs):
        self.root_value = self._values.pop(0) if self._values else 0.0
        return (0, 0), None


def _player(values, mercy):
    player = ModelPlayer.__new__(ModelPlayer)     # no network to load
    player.mcts = _StubMCTS(values)
    player.temperature = 0.1
    player.mercy = mercy
    return player


class _StateAtMove:
    """`move_number` is a property on GameState; the search here is stubbed, so
    the player only needs a position that claims to be late in the game."""

    def __init__(self, move_number):
        self.move_number = move_number


class TestModelPlayerResigns:
    def test_resigns_once_its_own_search_gives_up(self):
        player = _player([-1.0, -1.0, -1.0], _rule(consecutive=3))
        state = _StateAtMove(100)                 # past the 81-move gate

        assert player.select_move(state) == (0, 0)
        assert player.select_move(state) == (0, 0)
        assert player.select_move(state) == MOVE_RESIGN

    def test_plays_on_without_a_mercy_rule(self):
        player = _player([-1.0] * 5, None)
        state = _StateAtMove(100)
        for _ in range(5):
            assert player.select_move(state) == (0, 0)

    def test_a_new_game_starts_from_no_evidence(self):
        rule = _rule(consecutive=3)
        player = _player([-1.0] * 6, rule)
        state = _StateAtMove(100)
        player.select_move(state)
        player.select_move(state)
        player.game_started(GameState(board_size=9), 1)   # next game
        assert player.select_move(state) == (0, 0)
        assert player.select_move(state) == (0, 0)
        assert player.select_move(state) == MOVE_RESIGN


class TestHumanGameResignation:
    """The Play page's own bot: /api/game/bot_move can END the game."""

    @pytest.fixture
    def client(self):
        from web.app import create_app
        app = create_app()
        app.config.update(TESTING=True)
        return app.test_client()

    def test_bot_resigning_ends_the_game_and_says_so(self, client):
        from web.routes.game_routes import game_sessions

        gid = client.post('/api/game/new', json={
            'player_color': 'black', 'num_simulations': 5,
        }).get_json()['game_id']

        # A rule that fires on the bot's very first search, whatever it thinks:
        # what is under test is the route, not the threshold.
        game_sessions[gid]['mercy'] = MercyRule(
            enabled=True, threshold=-2.0, consecutive=1, min_move_factor=0.0,
        )

        client.post('/api/game/move', json={'game_id': gid, 'row': 0, 'col': 0})
        data = client.post('/api/game/bot_move', json={'game_id': gid}).get_json()

        assert data['bot_move']['type'] == 'resign'
        assert data['state']['is_over'] is True
        # The human played Black, so the bot resigning is a win for Black.
        assert data['result'] == {'winner': 'black', 'reason': 'resignation',
                                  'resigned_by': 'bot'}

    def test_the_flag_can_turn_it_off(self, client):
        from web.routes.game_routes import game_sessions

        gid = client.post('/api/game/new', json={
            'player_color': 'black', 'num_simulations': 5, 'mercy_resign': False,
        }).get_json()['game_id']
        assert game_sessions[gid]['mercy'].enabled is False

    def test_the_flag_can_turn_it_on(self, client):
        from web.routes.game_routes import game_sessions

        response = client.post('/api/game/new', json={
            'player_color': 'black', 'num_simulations': 5, 'mercy_resign': True,
        }).get_json()
        assert response['mercy_resign']['enabled'] is True
        assert game_sessions[response['game_id']]['mercy'].enabled is True


class TestBuildsFromModelConfig:
    def test_override_beats_the_models_training_setting(self):
        from config import Config
        from model_manager import ModelManager

        mgr = ModelManager()
        info = mgr.get_active_model()
        if info is None:
            pytest.skip("no active model in this workspace")
        config = Config.from_model(info, mgr.get_model_dir(info.id))

        assert MercyRule.from_config(config, info.board_size, enabled=True).enabled
        assert not MercyRule.from_config(config, info.board_size, enabled=False).enabled
        # None defers to the model's own (training) setting.
        deferred = MercyRule.from_config(config, info.board_size, enabled=None)
        assert deferred.enabled == config.training.resign_enabled
        assert deferred.min_move == round(config.training.resign_min_move_factor
                                          * info.board_size ** 2)
