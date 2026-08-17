"""
test_parallel_workers.py — The parallel game phases: pool reuse, worker
independence, and what a dead worker is allowed to do to a measurement.

Three things are pinned here, each of which was wrong or unguarded before:

1. A gate worker that CRASHES must not be counted as a candidate loss. It is
   not evidence about a network, and letting it vote lets an unrelated failure
   reject a candidate that was never measured.
2. Workers must not share a random stream. Under `fork` a child inherits the
   parent's RNG state, so every worker in a batch draws the same Dirichlet
   noise and the same sampled moves — and against a deterministic network that
   makes N "parallel" self-play games N copies of one game.
3. The pool is REUSED across phases and iterations. Rebuilding it costs a
   torch import per worker on any spawn platform (all of Windows), which is the
   cost that grows with exactly the setting a fast machine wants to raise.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai import worker_pool
from ai.evaluator import evaluate_against_checkpoint


class TestGateFailureAccounting(unittest.TestCase):
    """A crashed worker is excluded from the win rate, not scored as a loss."""

    def _run_gate(self, results, num_games=4, workers=2):
        """
        Drive evaluate_against_checkpoint's parallel path with a fake pool.

        `results` is one entry per game: an int (the game's result) or an
        Exception instance (that worker died).
        """
        import concurrent.futures

        calls = {'n': 0}

        class FakeFuture(concurrent.futures.Future):
            pass

        class FakeExecutor:
            def submit(self, fn, kwargs):
                idx = calls['n']
                calls['n'] += 1
                fut = FakeFuture()
                outcome = results[idx]
                if isinstance(outcome, Exception):
                    fut.set_exception(outcome)
                else:
                    fut.set_result(outcome)
                return fut

        with patch.object(worker_pool, 'get_executor', return_value=FakeExecutor()):
            return evaluate_against_checkpoint(
                current_network=None, opponent_network=None,
                num_games=num_games, num_workers=workers, games_dir=None,
            )

    def test_a_dead_worker_is_not_a_loss(self):
        # Two wins, one crash, one loss. The crash leaves the denominator at 3,
        # so the candidate scores 2/3 — not the 2/4 that would reject it.
        wr = self._run_gate([1, RuntimeError('worker died'), 1, 0])
        self.assertAlmostEqual(wr, 2 / 3)

    def test_a_clean_sweep_is_still_a_sweep(self):
        self.assertEqual(self._run_gate([1, 1, 1, 1]), 1.0)

    def test_every_worker_dying_reads_as_a_rejection(self):
        # 0.0 is below any threshold, so the champion is kept — the safe
        # direction when nothing was actually measured.
        wr = self._run_gate([RuntimeError('x')] * 4)
        self.assertEqual(wr, 0.0)


class TestWorkerSeeding(unittest.TestCase):
    """Each worker process gets its own random stream."""

    def test_initializer_reseeds_all_three_generators(self):
        import random
        import numpy as np
        import torch

        # Seed everything identically, the way a forked child would inherit it.
        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)
        before = (random.random(), float(np.random.rand()), float(torch.rand(1)))

        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)
        worker_pool._init_worker()
        after = (random.random(), float(np.random.rand()), float(torch.rand(1)))

        # Each generator must have moved off the inherited stream. (The chance
        # of a collision from os.urandom is ~2^-32 per generator.)
        self.assertNotEqual(before[0], after[0])
        self.assertNotEqual(before[1], after[1])
        self.assertNotEqual(before[2], after[2])

    def test_initializer_pins_workers_to_one_thread(self):
        import torch
        worker_pool._init_worker()
        self.assertEqual(torch.get_num_threads(), 1)


class TestSharedPool(unittest.TestCase):

    def tearDown(self):
        worker_pool.shutdown(wait=False)

    def test_same_pool_is_returned_for_a_smaller_request(self):
        # A phase asking for fewer workers than the live pool keeps fewer tasks
        # in flight; it must not rebuild the pool to do that.
        first = worker_pool.get_executor(4)
        again = worker_pool.get_executor(2)
        self.assertIs(first, again)
        self.assertEqual(worker_pool.active_workers(), 4)

    def test_a_bigger_request_grows_the_pool(self):
        first = worker_pool.get_executor(2)
        bigger = worker_pool.get_executor(6)
        self.assertIsNot(first, bigger)
        self.assertEqual(worker_pool.active_workers(), 6)

    def test_shutdown_clears_the_pool(self):
        worker_pool.get_executor(2)
        worker_pool.shutdown(wait=False)
        self.assertEqual(worker_pool.active_workers(), 0)

    def test_a_broken_pool_is_replaced_rather_than_reused(self):
        ex = worker_pool.get_executor(2)
        ex._broken = 'a worker process died'
        replacement = worker_pool.get_executor(2)
        self.assertIsNot(ex, replacement)


class TestTrainerReleasesThePool(unittest.TestCase):

    def test_training_run_shuts_the_pool_down_when_it_ends(self):
        import shutil
        import tempfile
        from config import (Config, BoardConfig, NetworkConfig, TrainingConfig,
                            MCTSConfig, PathConfig)
        from ai.trainer import Trainer

        tmp = tempfile.mkdtemp()
        try:
            trainer = Trainer(config=Config(
                board=BoardConfig(size=7),
                network=NetworkConfig(num_res_blocks=1, num_filters=8),
                training=TrainingConfig(num_self_play_games=1, batch_size=4),
                mcts=MCTSConfig(num_simulations=2),
                paths=PathConfig(model_dir=tmp),
            ))
            worker_pool.get_executor(2)
            # Stop before the first iteration: this is about the teardown in
            # train()'s finally block, not about playing a game.
            trainer.force_stop()
            trainer.train()
            self.assertEqual(worker_pool.active_workers(), 0)
        finally:
            worker_pool.shutdown(wait=False)
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
