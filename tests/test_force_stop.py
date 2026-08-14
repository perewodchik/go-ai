"""
test_force_stop.py — Fast, reliable unit test for force stop functionality.
"""

import unittest
import tempfile
import os
import sys
import torch
import shutil
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, BoardConfig, NetworkConfig, TrainingConfig, PathConfig, MCTSConfig
from ai.trainer import Trainer


class TestForceStop(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        
        cfg = Config(
            board=BoardConfig(size=7),
            network=NetworkConfig(num_res_blocks=1, num_filters=16),
            training=TrainingConfig(num_self_play_games=1, eval_games=1, batch_size=4),
            mcts=MCTSConfig(num_simulations=2),
            paths=PathConfig(model_dir=self.tmp_dir)
        )
        self.config = cfg
        self.trainer = Trainer(config=cfg)
        
        # Save clean baseline iteration 1 checkpoint on disk
        self.trainer.iteration = 1
        self.trainer.elo = 550.0
        self.trainer.total_games = 10
        self.trainer._save_weights()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_force_stop_rolls_back_corrupted_state(self):
        """Verify force_stop() aborts training and rolls back in-memory state from disk."""
        # Mutate trainer state in memory to simulate uncommitted changes
        self.trainer.iteration = 2
        self.trainer.elo = 999.0
        self.trainer.total_games = 50
        
        # Mutate neural network parameters in memory
        with torch.no_grad():
            for p in self.trainer.network.parameters():
                p.add_(5.0)

        # Signal force stop
        self.trainer.force_stop()
        self.assertTrue(self.trainer._stop_requested)
        self.assertTrue(self.trainer._force_stop_requested)

        # Run train() — should halt immediately and restore iteration 1 weights
        self.trainer.train()

        # Assert trainer state was rolled back to iteration 1 saved on disk
        self.assertEqual(self.trainer.iteration, 1)
        self.assertEqual(self.trainer.elo, 550.0)
        self.assertEqual(self.trainer.total_games, 10)
        self.assertFalse(self.trainer.is_running)
        self.assertFalse(self.trainer._force_stop_requested)

    @patch('ai.trainer.run_self_play_batch')
    def test_force_stop_during_self_play_batch(self, mock_self_play):
        """Verify force_stop() mid-self-play aborts immediately without continuing iteration."""
        # Define mock self play to trigger force_stop when called
        def mock_self_play_func(*args, **kwargs):
            stop_checker = kwargs.get('stop_checker')
            # Trigger force stop
            self.trainer.force_stop()
            if stop_checker:
                self.assertTrue(stop_checker())
            return []

        mock_self_play.side_effect = mock_self_play_func

        # Mutate state
        self.trainer.iteration = 5
        self.trainer.elo = 1200.0

        # Run train
        self.trainer.train()

        # Check that state was rolled back to iteration 1 from disk
        self.assertEqual(self.trainer.iteration, 1)
        self.assertEqual(self.trainer.elo, 550.0)
        self.assertFalse(self.trainer.is_running)


if __name__ == '__main__':
    unittest.main()
