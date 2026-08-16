"""
test_stored_games_filter.py — Tests for filtering recorded human games in the stored games API.

The endpoint returns {groups, pagination}; these tests only care about the
groups. Paging itself is covered in test_games_paging.py.
"""

import os
import sys
import shutil
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, BoardConfig, NetworkConfig, TrainingConfig, PathConfig, MCTSConfig
from ai.trainer import Trainer
from ai.game_store import save_game, save_human_game, PHASE_SELF_PLAY
from web.app import create_app
import web.app as app_module


class TestStoredGamesFilter(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.games_dir = os.path.join(self.tmp_dir, 'games')
        os.makedirs(self.games_dir, exist_ok=True)

        cfg = Config(
            board=BoardConfig(size=7),
            network=NetworkConfig(num_res_blocks=1, num_filters=16),
            training=TrainingConfig(num_self_play_games=1, eval_games=1, batch_size=4),
            mcts=MCTSConfig(num_simulations=2),
            paths=PathConfig(model_dir=self.tmp_dir)
        )
        self.config = cfg
        self.games_dir = cfg.paths.games_dir
        self.trainer = Trainer(config=cfg)
        self.app = create_app()
        app_module.trainer = self.trainer
        self.client = self.app.test_client()

        # Create 1 self-play game under iter_000001
        self_play_record = {
            'board_size': 7,
            'komi': 6.5,
            'moves': [{'move': [0, 0], 'player': 1, 'color': 1}],
            'num_moves': 1,
            'winner': 1,
            'margin': 2.5,
        }
        save_game(self.games_dir, iteration=1, phase=PHASE_SELF_PLAY, index=0, record=self_play_record)

        # Create 1 human vs bot recorded game under human/
        human_record = {
            'board_size': 7,
            'komi': 6.5,
            'moves': [{'move': [1, 1], 'player': 1, 'color': 1}],
            'num_moves': 1,
            'winner': 1,
            'margin': 5.5,
            'name': 'My Human Game',
            'human_color': 1,
            'bot_color': 2,
        }
        save_human_game(self.games_dir, record=human_record)

    def tearDown(self):
        app_module.trainer = None
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_default_api_games_includes_recorded(self):
        """Default GET /training/api/games includes recorded human games for backward compatibility / review."""
        res = self.client.get('/training/api/games')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()['groups']

        kinds = [group.get('kind') for group in data]
        self.assertIn('recorded', kinds)
        self.assertIn('iteration', kinds)

        recorded_group = next(g for g in data if g.get('kind') == 'recorded')
        self.assertEqual(recorded_group['label'], 'My Recorded Games')
        self.assertEqual(len(recorded_group['games']), 1)

    def test_include_recorded_true(self):
        """GET /training/api/games?include_recorded=1 explicitly includes recorded games."""
        res = self.client.get('/training/api/games?include_recorded=1')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()['groups']

        kinds = [group.get('kind') for group in data]
        self.assertIn('recorded', kinds)
        self.assertIn('iteration', kinds)

    def test_exclude_recorded_games_with_include_recorded_zero(self):
        """GET /training/api/games?include_recorded=0 excludes recorded human games (for training stored games browser)."""
        res = self.client.get('/training/api/games?include_recorded=0')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()['groups']

        kinds = [group.get('kind') for group in data]
        self.assertNotIn('recorded', kinds)
        self.assertIn('iteration', kinds)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['iteration'], 1)

    def test_exclude_recorded_games_with_include_recorded_false(self):
        """GET /training/api/games?include_recorded=false excludes recorded human games."""
        res = self.client.get('/training/api/games?include_recorded=false')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()['groups']

        kinds = [group.get('kind') for group in data]
        self.assertNotIn('recorded', kinds)
        self.assertIn('iteration', kinds)

    def test_exclude_recorded_games_with_include_human_zero(self):
        """GET /training/api/games?include_human=0 excludes recorded human games."""
        res = self.client.get('/training/api/games?include_human=0')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()['groups']

        kinds = [group.get('kind') for group in data]
        self.assertNotIn('recorded', kinds)
        self.assertIn('iteration', kinds)


if __name__ == '__main__':
    unittest.main()
