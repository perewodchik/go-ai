"""
test_games_paging.py — Tests for paging the stored-games list.

A model that has run for a hundred iterations has thousands of stored games,
and the sidebar used to request every one of them: the response carried each
game's full move list and win-rate curve, none of which a list of games
displays. Two things fix that, and both need holding in place:

  * only the newest N iterations are returned, and older ones are fetched on
    demand with `before`;
  * the files for iterations OUTSIDE the page are never opened — the saving is
    in the parsing, not in the JSON that comes back. That is asserted by
    booby-trapping an old game with unreadable content: if the endpoint still
    parses it, the test fails.

Recorded and match games are deliberately NOT paged: there are few of them and
they are what the review page is usually opened for.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
import unittest.mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (Config, BoardConfig, NetworkConfig, TrainingConfig,
                    PathConfig, MCTSConfig)
from ai.trainer import Trainer
from ai.game_store import save_game, save_human_game, save_match_game, PHASE_SELF_PLAY
from web.app import create_app
import web.app as app_module


def game_record(index=0, moves=3):
    return {
        'board_size': 7,
        'komi': 6.5,
        'moves': [{'move': [0, i % 7], 'color': 1, 'move_num': i} for i in range(moves)],
        'win_rates': [50.0] * moves,
        'num_moves': moves,
        'winner': 1,
        'margin': 2.5,
        'game_index': index,
    }


class TestGamesPaging(unittest.TestCase):

    ITERATIONS = 12
    GAMES_PER_ITERATION = 2

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        cfg = Config(
            board=BoardConfig(size=7),
            network=NetworkConfig(num_res_blocks=1, num_filters=16),
            training=TrainingConfig(num_self_play_games=1, eval_games=1, batch_size=4),
            mcts=MCTSConfig(num_simulations=2),
            paths=PathConfig(model_dir=self.tmp_dir),
        )
        self.games_dir = cfg.paths.games_dir
        os.makedirs(self.games_dir, exist_ok=True)

        for iteration in range(1, self.ITERATIONS + 1):
            for index in range(self.GAMES_PER_ITERATION):
                save_game(self.games_dir, iteration=iteration, phase=PHASE_SELF_PLAY,
                          index=index, record=game_record(index))

        self.trainer = Trainer(config=cfg)
        self.app = create_app()
        app_module.trainer = self.trainer
        self.client = self.app.test_client()

    def tearDown(self):
        app_module.trainer = None
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def get(self, query=''):
        res = self.client.get('/training/api/games' + query)
        self.assertEqual(res.status_code, 200)
        return res.get_json()

    def iterations_in(self, payload):
        return [g['iteration'] for g in payload['groups'] if g.get('kind') == 'iteration']

    # ---- page size ----

    def test_default_page_returns_the_newest_five_iterations(self):
        payload = self.get()
        self.assertEqual(self.iterations_in(payload), [12, 11, 10, 9, 8])

    def test_iterations_param_sets_the_page_size(self):
        self.assertEqual(self.iterations_in(self.get('?iterations=3')), [12, 11, 10])

    def test_page_size_is_clamped_to_a_sane_range(self):
        """The value arrives from the browser, so it cannot be trusted."""
        self.assertEqual(len(self.iterations_in(self.get('?iterations=0'))), 1)
        self.assertEqual(len(self.iterations_in(self.get('?iterations=-4'))), 1)
        self.assertEqual(len(self.iterations_in(self.get('?iterations=nonsense'))), 5)
        # Above the maximum, capped rather than refused.
        self.assertEqual(len(self.iterations_in(self.get('?iterations=9999'))),
                         self.ITERATIONS)

    # ---- cursor ----

    def test_before_returns_the_next_older_page(self):
        first = self.get('?iterations=4')
        self.assertEqual(self.iterations_in(first), [12, 11, 10, 9])

        cursor = first['pagination']['oldest_iteration']
        second = self.get(f'?iterations=4&before={cursor}')
        self.assertEqual(self.iterations_in(second), [8, 7, 6, 5])

    def test_paging_to_the_end_reports_no_more(self):
        seen = []
        cursor = None
        for _ in range(10):
            query = '?iterations=5' + (f'&before={cursor}' if cursor else '')
            payload = self.get(query)
            seen.extend(self.iterations_in(payload))
            if not payload['pagination']['has_more']:
                break
            cursor = payload['pagination']['oldest_iteration']

        self.assertEqual(seen, list(range(self.ITERATIONS, 0, -1)))
        self.assertEqual(len(seen), len(set(seen)), 'an iteration was returned twice')

    def test_pagination_counts_describe_what_is_left(self):
        page = self.get('?iterations=5')['pagination']
        self.assertEqual(page['returned'], 5)
        self.assertEqual(page['newest_iteration'], 12)
        self.assertEqual(page['oldest_iteration'], 8)
        self.assertEqual(page['total_iterations'], self.ITERATIONS)
        self.assertEqual(page['remaining'], self.ITERATIONS - 5)
        self.assertTrue(page['has_more'])

    def test_a_bad_cursor_is_ignored_rather_than_failing(self):
        self.assertEqual(self.iterations_in(self.get('?iterations=2&before=nope')),
                         [12, 11])

    # ---- what the page costs ----

    def test_files_outside_the_page_are_never_opened(self):
        """
        The point of paging is the parsing it avoids. An unparseable game in an
        old iteration must not affect a request for the newest ones — if it
        does, that file was being read.
        """
        booby_trapped = os.path.join(self.games_dir, 'iter_000001',
                                     PHASE_SELF_PLAY, 'game_0000.json')
        with open(booby_trapped, 'w') as fh:
            fh.write('{ this is not json')

        payload = self.get('?iterations=3')
        self.assertEqual(self.iterations_in(payload), [12, 11, 10])

        # And when the page does reach it, the corrupt game is skipped, not fatal.
        deep = self.get('?iterations=5&before=2')
        self.assertEqual(self.iterations_in(deep), [1])
        first = [g for g in deep['groups'] if g['iteration'] == 1][0]
        self.assertEqual(first['total_games'], self.GAMES_PER_ITERATION - 1)

    def test_list_response_omits_the_bulk_fields(self):
        """
        Moves and win rates are 90% of the payload and are never read by a
        list — the client refetches the full record when a game is opened.
        """
        payload = self.get('?iterations=1')
        games = payload['groups'][0]['phases'][0]['games']
        self.assertTrue(games)
        for game in games:
            self.assertNotIn('moves', game)
            self.assertNotIn('win_rates', game)
            self.assertIn('num_moves', game)      # the summary stays
            self.assertIn('winner', game)
            self.assertIn('filename', game)

    def test_opening_a_game_still_returns_its_moves(self):
        payload = self.get('?iterations=1')
        rel_path = payload['groups'][0]['phases'][0]['games'][0]['filename']
        res = self.client.get('/training/api/games/' + rel_path)
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.get_json()['moves'])

    def test_a_game_can_be_opened_from_another_models_directory(self):
        """
        A game id only means something relative to a model directory. Deep
        links from the fleet dashboard — an Elo-curve point, a head-to-head
        example — name a model that is usually not the active one, and without
        `?model=` they would all resolve against the active model and 404.
        """
        import model_manager

        other_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, other_dir, True)
        model_id = 'other-model'
        save_game(os.path.join(other_dir, model_id, 'games'),
                  iteration=1, phase=PHASE_SELF_PLAY, index=0, record=game_record(0))
        with open(os.path.join(other_dir, model_id, 'config.json'), 'w') as fh:
            json.dump({'id': model_id, 'name': model_id}, fh)

        rel = 'iter_000001/self-play/game_0000.json'
        with unittest.mock.patch.object(model_manager, 'MODELS_ROOT', other_dir):
            # The active model has no such game...
            self.assertEqual(
                self.client.get(f'/training/api/games/{rel}?model=only-in-theory').status_code,
                404)
            # ...but naming the model that owns it opens it.
            res = self.client.get(f'/training/api/games/{rel}?model={model_id}')
            self.assertEqual(res.status_code, 200)
            self.assertTrue(res.get_json()['moves'])


class TestUniteratedGamesAreNotPaged(unittest.TestCase):
    """Recorded and match games load whole, on the first page only."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        cfg = Config(
            board=BoardConfig(size=7),
            network=NetworkConfig(num_res_blocks=1, num_filters=16),
            training=TrainingConfig(num_self_play_games=1, eval_games=1, batch_size=4),
            mcts=MCTSConfig(num_simulations=2),
            paths=PathConfig(model_dir=self.tmp_dir),
        )
        self.games_dir = cfg.paths.games_dir
        os.makedirs(self.games_dir, exist_ok=True)

        for iteration in range(1, 9):
            save_game(self.games_dir, iteration=iteration, phase=PHASE_SELF_PLAY,
                      index=0, record=game_record())

        for i in range(7):
            record = game_record()
            record.update(name=f'Human game {i}', human_color=1, bot_color=2)
            save_human_game(self.games_dir, record=record)

        for i in range(6):
            record = game_record(index=i)
            record.update(is_match=True, match_id='m1', match_name='A vs B',
                          black_player={'name': 'A'}, white_player={'name': 'B'})
            save_match_game(self.games_dir, record=record)

        self.trainer = Trainer(config=cfg)
        self.app = create_app()
        app_module.trainer = self.trainer
        self.client = self.app.test_client()

    def tearDown(self):
        app_module.trainer = None
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def get(self, query=''):
        return self.client.get('/training/api/games' + query).get_json()

    def test_every_recorded_and_match_game_comes_back_on_the_first_page(self):
        groups = {g['kind']: g for g in self.get('?iterations=2')['groups']
                  if g['kind'] in ('recorded', 'match')}
        self.assertEqual(groups['recorded']['total_games'], 7)
        self.assertEqual(groups['match']['total_games'], 6)
        self.assertEqual(len(groups['match']['series'][0]['games']), 6)

    def test_older_pages_do_not_repeat_them(self):
        """Otherwise every "load more" would duplicate the whole top section."""
        kinds = [g['kind'] for g in self.get('?iterations=2&before=7')['groups']]
        self.assertNotIn('recorded', kinds)
        self.assertNotIn('match', kinds)
        self.assertEqual(set(kinds), {'iteration'})

    def test_recorded_games_also_drop_their_move_lists(self):
        recorded = next(g for g in self.get()['groups'] if g['kind'] == 'recorded')
        for game in recorded['games']:
            self.assertNotIn('moves', game)
            self.assertIn('num_moves', game)


if __name__ == '__main__':
    unittest.main()
