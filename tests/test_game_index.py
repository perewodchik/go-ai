"""
test_game_index.py — The games index, and the recording toggles it enables.

Two things are being held in place here.

FIRST, the index has to be a faithful summary. Every statistic the training
page draws now comes from `games/index.jsonl` instead of from the game records
themselves — that is what took `/training/api/learning_stats` from ~3.5s on an
11k-game model down to well under a tenth of a second. A summary that drifts
from the records would silently produce wrong charts, so the tests below
compare the endpoints' output against the records on disk rather than against
hardcoded numbers.

SECOND, `record_self_play_games` / `record_gate_games` are only safe to turn
off BECAUSE the index is separate: a game whose record was never written must
still appear in every count, win rate and series. The tests assert exactly
that — no file, but no missing data point either.

The remaining subtlety is reconciliation. A model that predates the index (or
whose index was deleted) has to build one from the records it already has, and
must not rebuild it on every subsequent read.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai import game_index
from ai.game_store import (
    PHASE_EVAL, PHASE_PROMOTION, PHASE_SELF_PLAY,
    delete_saved_game, save_game,
)


def record(winner=1, margin=2.5, moves=6, elapsed=1.5, **extra):
    rec = {
        'board_size': 7,
        'komi': 6.5,
        'moves': [{'move': [0, i % 7], 'color': 1, 'move_num': i} for i in range(moves)],
        'win_rates': [50.0] * moves,
        'num_moves': moves,
        'winner': winner,
        'margin': margin,
        'elapsed_seconds': elapsed,
    }
    rec.update(extra)
    return rec


class IndexTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.games_dir = os.path.join(self.tmp_dir, 'games')
        os.makedirs(self.games_dir, exist_ok=True)
        game_index.invalidate()

    def tearDown(self):
        game_index.invalidate()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def rows(self):
        return game_index.load(self.games_dir)


class TestIndexingOnSave(IndexTestCase):

    def test_a_saved_game_is_indexed_and_stored(self):
        rel = save_game(self.games_dir, 1, PHASE_SELF_PLAY, 0, record())

        self.assertEqual(rel, 'iter_000001/self-play/game_0000.json')
        self.assertTrue(os.path.isfile(os.path.join(self.games_dir, rel)))

        rows = self.rows()
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]['stored'])
        self.assertEqual(game_index.row_path(rows[0]), rel)

    def test_the_summary_carries_the_numbers_the_charts_need(self):
        save_game(self.games_dir, 3, PHASE_SELF_PLAY, 2,
                  record(winner=2, margin=7.5, moves=41, elapsed=9.25))

        row = self.rows()[0]
        self.assertEqual(row['iteration'], 3)
        self.assertEqual(row['phase'], PHASE_SELF_PLAY)
        self.assertEqual(row['game_index'], 2)
        self.assertEqual(row['winner'], 2)
        self.assertEqual(row['margin'], 7.5)
        self.assertEqual(row['num_moves'], 41)
        self.assertEqual(row['elapsed_seconds'], 9.25)

    def test_the_summary_does_not_carry_the_bulk(self):
        """
        The whole point is that the index is small. If moves or win rates ever
        start riding along, reading it stops being cheaper than reading the
        records it summarizes.
        """
        save_game(self.games_dir, 1, PHASE_SELF_PLAY, 0, record(moves=200))

        row = self.rows()[0]
        for key in ('moves', 'win_rates', 'states'):
            self.assertNotIn(key, row)

    def test_phase_specific_fields_survive(self):
        save_game(self.games_dir, 1, PHASE_PROMOTION, 0,
                  record(candidate_won=True, candidate_color=1))
        save_game(self.games_dir, 1, PHASE_EVAL, 0, record(network_color=2))

        by_phase = {r['phase']: r for r in self.rows()}
        self.assertIs(by_phase[PHASE_PROMOTION]['candidate_won'], True)
        self.assertEqual(by_phase[PHASE_EVAL]['network_color'], 2)


class TestRecordingToggle(IndexTestCase):

    def test_an_unrecorded_game_writes_no_file_but_is_still_counted(self):
        rel = save_game(self.games_dir, 1, PHASE_SELF_PLAY, 0, record(),
                        store_full=False)

        self.assertIsNone(rel)
        self.assertFalse(os.path.exists(os.path.join(self.games_dir, 'iter_000001')))

        rows = self.rows()
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]['stored'])
        self.assertIsNone(game_index.row_path(rows[0]))

    def test_an_unrecorded_game_keeps_every_number(self):
        """Recording off must cost the replay and nothing else."""
        save_game(self.games_dir, 1, PHASE_SELF_PLAY, 0,
                  record(winner=2, margin=4.5, moves=33, elapsed=2.5),
                  store_full=False)

        row = self.rows()[0]
        self.assertEqual((row['winner'], row['margin'], row['num_moves'],
                          row['elapsed_seconds']), (2, 4.5, 33, 2.5))

    def test_recorded_and_unrecorded_games_coexist(self):
        """Toggling mid-run must not confuse the index."""
        save_game(self.games_dir, 1, PHASE_SELF_PLAY, 0, record())
        save_game(self.games_dir, 1, PHASE_SELF_PLAY, 1, record(), store_full=False)
        save_game(self.games_dir, 1, PHASE_SELF_PLAY, 2, record())

        rows = self.rows()
        self.assertEqual([r['stored'] for r in rows], [True, False, True])


class TestReconciliation(IndexTestCase):

    def test_a_model_with_no_index_builds_one_from_its_records(self):
        """The migration path: every existing model arrives here."""
        for index in range(3):
            save_game(self.games_dir, 1, PHASE_SELF_PLAY, index, record())
        os.remove(game_index.index_path(self.games_dir))
        game_index.invalidate()

        rows = self.rows()
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r['stored'] for r in rows))
        self.assertTrue(os.path.isfile(game_index.index_path(self.games_dir)))

    def test_an_up_to_date_index_is_not_rewritten(self):
        """
        Reconciliation costs one directory walk. If it re-indexed games it had
        already seen it would also re-parse every record, which is the cost the
        index exists to avoid.
        """
        save_game(self.games_dir, 1, PHASE_SELF_PLAY, 0, record())
        self.rows()

        path = game_index.index_path(self.games_dir)
        before = os.stat(path).st_mtime_ns
        game_index.invalidate()
        self.rows()
        self.assertEqual(os.stat(path).st_mtime_ns, before)

    def test_records_written_behind_the_indexs_back_are_picked_up(self):
        """A record restored from a backup, or written by an older build."""
        save_game(self.games_dir, 1, PHASE_SELF_PLAY, 0, record())
        self.assertEqual(len(self.rows()), 1)

        stray = os.path.join(self.games_dir, 'iter_000002', PHASE_SELF_PLAY)
        os.makedirs(stray, exist_ok=True)
        with open(os.path.join(stray, 'game_0000.json'), 'w') as fh:
            json.dump(record(winner=2), fh)

        rows = self.rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]['iteration'], 2)

    def test_reconciliation_never_drops_an_unrecorded_game(self):
        """
        The one thing reconciliation must not do: a row with no file behind it
        is not a stale row, it is a game played with recording off.
        """
        save_game(self.games_dir, 1, PHASE_SELF_PLAY, 0, record(), store_full=False)
        game_index.invalidate()
        self.assertEqual(len(self.rows()), 1)
        game_index.invalidate()
        self.assertEqual(len(self.rows()), 1)

    def test_a_torn_line_costs_one_game_not_the_file(self):
        for index in range(3):
            save_game(self.games_dir, 1, PHASE_SELF_PLAY, index, record(),
                      store_full=False)
        path = game_index.index_path(self.games_dir)
        with open(path) as fh:
            lines = fh.readlines()
        lines[1] = lines[1][:len(lines[1]) // 2] + '\n'
        with open(path, 'w') as fh:
            fh.writelines(lines)
        game_index.invalidate()

        self.assertEqual(len(self.rows()), 2)


class TestDeletePrunesTheIndex(IndexTestCase):

    def test_deleting_a_game_drops_its_row(self):
        save_game(self.games_dir, 1, PHASE_SELF_PLAY, 0, record())
        save_game(self.games_dir, 1, PHASE_SELF_PLAY, 1, record())
        self.rows()

        delete_saved_game(self.games_dir, 'iter_000001/self-play/game_0000.json')
        rows = self.rows()
        self.assertEqual([r['game_index'] for r in rows], [1])

    def test_deleting_an_iteration_drops_all_of_its_rows(self):
        for phase in (PHASE_SELF_PLAY, PHASE_PROMOTION):
            save_game(self.games_dir, 1, phase, 0, record())
        save_game(self.games_dir, 2, PHASE_SELF_PLAY, 0, record())
        self.rows()

        delete_saved_game(self.games_dir, 'iter_000001')
        self.assertEqual([r['iteration'] for r in self.rows()], [2])

    def test_deleting_an_iteration_drops_its_unrecorded_rows_too(self):
        """
        Otherwise there would be no way at all to clear the statistics of a run
        made with recording off.
        """
        save_game(self.games_dir, 1, PHASE_SELF_PLAY, 0, record(), store_full=False)
        save_game(self.games_dir, 1, PHASE_SELF_PLAY, 1, record())
        self.rows()

        delete_saved_game(self.games_dir, 'iter_000001')
        self.assertEqual(self.rows(), [])


class TestStatisticsEndpoints(IndexTestCase):
    """
    The endpoints that used to open every record on disk.

    Asserted against what was saved rather than against constants, so a change
    to the summary that quietly loses a field shows up here.
    """

    def setUp(self):
        super().setUp()
        import shutil as _shutil
        from config import (Config, BoardConfig, NetworkConfig, TrainingConfig,
                            PathConfig, MCTSConfig)
        from ai.trainer import Trainer
        from web.app import create_app
        import web.app as app_module

        self._shutil = _shutil
        self.app_module = app_module

        cfg = Config(
            board=BoardConfig(size=7),
            network=NetworkConfig(num_res_blocks=1, num_filters=16),
            training=TrainingConfig(num_self_play_games=1, eval_games=1, batch_size=4),
            mcts=MCTSConfig(num_simulations=2),
            paths=PathConfig(model_dir=self.tmp_dir),
        )
        self.games_dir = cfg.paths.games_dir
        os.makedirs(self.games_dir, exist_ok=True)

        # Iteration 1 recorded, iteration 2 played with recording off.
        save_game(self.games_dir, 1, PHASE_SELF_PLAY, 0,
                  record(winner=1, margin=3.5, moves=20, elapsed=1.0))
        save_game(self.games_dir, 1, PHASE_SELF_PLAY, 1,
                  record(winner=2, margin=5.5, moves=30, elapsed=2.0))
        save_game(self.games_dir, 2, PHASE_SELF_PLAY, 0,
                  record(winner=1, margin=1.5, moves=40, elapsed=3.0),
                  store_full=False)
        save_game(self.games_dir, 2, PHASE_PROMOTION, 0,
                  record(candidate_won=True), store_full=False)

        # create_app() installs the ACTIVE model's trainer, so the override has
        # to come after it or the endpoints read the real models/ directory.
        self.client = create_app().test_client()
        app_module.trainer = Trainer(config=cfg)

    def tearDown(self):
        self.app_module.trainer = None
        super().tearDown()

    def test_learning_stats_counts_unrecorded_games(self):
        stats = self.client.get('/training/api/learning_stats').get_json()

        # Three self-play games: two black wins, one white.
        self.assertEqual(stats['self_play_black_wins'], 2)
        self.assertEqual(stats['self_play_white_wins'], 1)
        self.assertAlmostEqual(stats['self_play_wr_black'], round(2 / 3, 3))
        # Signed margins, chronological: +3.5, -5.5, +1.5.
        self.assertEqual(stats['self_play_series'], [3.5, -5.5, 1.5])
        self.assertEqual(stats['avg_game_length'], 30.0)
        self.assertEqual(stats['avg_time_per_game_total'], 2.0)

    def test_the_promotion_phase_never_enters_the_self_play_stats(self):
        stats = self.client.get('/training/api/learning_stats').get_json()
        self.assertEqual(len(stats['self_play_series']), 3)

    def test_the_games_list_reports_what_was_not_recorded(self):
        payload = self.client.get('/training/api/games').get_json()
        groups = {g['iteration']: g for g in payload['groups']
                  if g.get('kind') == 'iteration'}

        self.assertEqual(groups[1]['total_games'], 2)
        self.assertEqual(groups[1]['total_not_recorded'], 0)

        # An iteration with nothing on disk still appears, and says why.
        self.assertEqual(groups[2]['total_games'], 0)
        self.assertEqual(groups[2]['total_not_recorded'], 2)
        by_phase = {p['phase']: p for p in groups[2]['phases']}
        self.assertEqual(by_phase[PHASE_SELF_PLAY]['not_recorded'], 1)
        self.assertEqual(by_phase[PHASE_PROMOTION]['not_recorded'], 1)

    def test_resign_stats_sees_unrecorded_games(self):
        save_game(self.games_dir, 3, PHASE_SELF_PLAY, 0,
                  record(resigned=True), store_full=False)
        save_game(self.games_dir, 3, PHASE_SELF_PLAY, 1,
                  record(would_resign_move=10, false_resign=True, moves=25),
                  store_full=False)
        game_index.invalidate()

        summary = self.client.get('/training/api/resign_stats').get_json()['summary']
        self.assertEqual(summary['total_resigned'], 1)
        self.assertEqual(summary['checked_games'], 1)
        self.assertEqual(summary['false_resigns'], 1)
        # 25 moves played, the rule would have cut at move 10.
        self.assertEqual(summary['avg_moves_saved'], 15.0)


if __name__ == '__main__':
    unittest.main()
