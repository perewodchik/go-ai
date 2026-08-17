"""
test_elo_history.py — the Elo ledger, and the properties that make it worth
having instead of a float.

A rating stored as one number in config.json could only ever answer "what is
it now". Four things went wrong with that, and each has a test here:

  * a rating that moved 300 points looked identical to one that never moved;
  * there was no way back to the game that caused a change, so a suspicious
    jump could not be reviewed;
  * training and matches both wrote the field, so a training run started before
    a match series would write its stale number back over the earned one;
  * a model with an existing rating had to keep it — a ledger that started
    everyone at 500 would erase ninety iterations of history.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import elo_history
from elo_history import (
    BASELINE_NOTE,
    DEFAULT_ELO,
    EloEntry,
    outcome_from_score,
)


def read_lines(model_dir):
    with open(os.path.join(str(model_dir), 'elo_history.jsonl')) as fh:
        return [json.loads(line) for line in fh if line.strip()]


class TestLedgerBasics:

    def test_an_entry_round_trips_with_everything_it_needs(self, tmp_path):
        elo_history.append(str(tmp_path), EloEntry(
            new_elo=512.0, elo_delta=12.0,
            opponent_name='Random Bot', opponent_id='random',
            game_outcome='win',
            game_record_path='match/match_20260817_120000.json',
        ))
        (entry,) = read_lines(tmp_path)

        # Every field the UI needs to explain the point on the curve.
        assert entry['new_elo'] == 512.0
        assert entry['elo_delta'] == 12.0
        assert entry['opponent_name'] == 'Random Bot'
        assert entry['opponent_id'] == 'random'
        assert entry['game_outcome'] == 'win'
        assert entry['game_record_path'] == 'match/match_20260817_120000.json'
        assert entry['timestamp']

    def test_it_appends_rather_than_overwrites(self, tmp_path):
        for elo in (510, 520, 530):
            elo_history.append(str(tmp_path), EloEntry(new_elo=elo, elo_delta=10))
        assert [e['new_elo'] for e in read_lines(tmp_path)] == [510, 520, 530]
        assert elo_history.current_elo(str(tmp_path)) == 530

    def test_a_corrupt_line_does_not_lose_the_rest(self, tmp_path):
        """A ledger truncated by a crash still describes the games that landed."""
        path = tmp_path / 'elo_history.jsonl'
        path.write_text('{"new_elo": 510}\n{"new_elo": 5\n{"new_elo": 530}\n')
        assert [e['new_elo'] for e in elo_history.read_history(str(tmp_path))] == [510, 530]
        assert elo_history.current_elo(str(tmp_path)) == 530

    def test_a_missing_ledger_reads_as_empty(self, tmp_path):
        assert elo_history.read_history(str(tmp_path / 'nope')) == []

    def test_outcome_wording(self):
        assert outcome_from_score(1.0) == 'win'
        assert outcome_from_score(0.0) == 'loss'
        assert outcome_from_score(0.5) == 'draw'


class TestMigration:

    def test_an_existing_rating_becomes_the_baseline(self, tmp_path):
        """
        The point of the migration: a model rated 1240 after ninety iterations
        must not be reset to the default just because the ledger is new.
        """
        written = elo_history.ensure_baseline(str(tmp_path), 1240.0)
        assert written['new_elo'] == 1240.0
        assert written['note'] == BASELINE_NOTE
        assert written['timestamp']
        assert elo_history.current_elo(str(tmp_path)) == 1240.0

    def test_a_model_with_no_rating_starts_at_the_default(self, tmp_path):
        elo_history.ensure_baseline(str(tmp_path), None)
        assert elo_history.current_elo(str(tmp_path)) == DEFAULT_ELO

    def test_it_never_fires_twice(self, tmp_path):
        elo_history.ensure_baseline(str(tmp_path), 800.0)
        elo_history.append(str(tmp_path), EloEntry(new_elo=816.0, elo_delta=16.0,
                                                   game_outcome='win'))
        assert elo_history.ensure_baseline(str(tmp_path), 800.0) is None
        assert elo_history.current_elo(str(tmp_path)) == 816.0

    def test_a_baseline_is_not_counted_as_a_played_game(self, tmp_path):
        elo_history.ensure_baseline(str(tmp_path), 800.0)
        assert elo_history.rated_games(str(tmp_path)) == 0

        elo_history.append(str(tmp_path), EloEntry(new_elo=816.0, elo_delta=16.0,
                                                   game_outcome='win'))
        assert elo_history.rated_games(str(tmp_path)) == 1

    def test_recording_a_result_seeds_the_baseline_from_the_prior_rating(self, tmp_path):
        """
        The first rated game of a long-trained model must not make the curve
        start at the post-game number — the point before it is where it was.
        """
        elo_history.record(str(tmp_path), new_elo=1256.0, elo_delta=16.0,
                           opponent_name='Random Bot', game_outcome='win')
        entries = read_lines(tmp_path)
        assert len(entries) == 2
        assert entries[0]['new_elo'] == 1240.0     # 1256 - 16
        assert entries[0]['note'] == BASELINE_NOTE
        assert entries[1]['new_elo'] == 1256.0


class TestSeries:
    """The graph axis: games played, not training iterations."""

    def test_the_x_axis_is_the_sequential_game_index(self, tmp_path):
        elo_history.ensure_baseline(str(tmp_path), 500.0)
        for i in range(1, 4):
            elo_history.record(str(tmp_path), new_elo=500.0 + 10 * i,
                               elo_delta=10.0, game_outcome='win')

        data = elo_history.series(str(tmp_path))
        assert data['games'] == [0, 1, 2, 3]
        assert data['elo'] == [500.0, 510.0, 520.0, 530.0]
        assert data['rated_games'] == 3

    def test_it_downsamples_but_keeps_both_ends(self, tmp_path):
        for i in range(1, 201):
            elo_history.append(str(tmp_path), EloEntry(new_elo=500.0 + i,
                                                       elo_delta=1.0,
                                                       game_outcome='win'))
        data = elo_history.series(str(tmp_path), limit=24)
        assert len(data['games']) == 24
        assert data['elo'][0] == 501.0
        assert data['elo'][-1] == 700.0

    def test_each_point_carries_its_game_for_review(self, tmp_path):
        elo_history.record(str(tmp_path), new_elo=516.0, elo_delta=16.0,
                           game_outcome='win',
                           game_record_path='match/match_0001.json')
        last = elo_history.series(str(tmp_path))['entries'][-1]
        assert last['game_record_path'] == 'match/match_0001.json'


class TestReconcile:

    def test_the_ledger_wins_over_a_stale_cache(self, tmp_path):
        elo_history.append(str(tmp_path), EloEntry(new_elo=642.0, elo_delta=16.0))
        assert elo_history.reconcile(str(tmp_path), 500.0) == 642.0

    def test_an_agreeing_cache_needs_no_correction(self, tmp_path):
        elo_history.append(str(tmp_path), EloEntry(new_elo=642.0, elo_delta=16.0))
        assert elo_history.reconcile(str(tmp_path), 642.0) is None

    def test_nothing_to_reconcile_without_a_ledger(self, tmp_path):
        assert elo_history.reconcile(str(tmp_path), 500.0) is None


class TestModelManagerIntegration:

    @pytest.fixture
    def manager(self, tmp_path, monkeypatch):
        import model_manager
        root = tmp_path / 'models'
        root.mkdir()
        monkeypatch.setattr(model_manager, 'MODELS_ROOT', str(root))
        monkeypatch.setattr(model_manager, 'ACTIVE_FILE', str(root / '.active'))
        return model_manager.ModelManager()

    def test_a_new_model_opens_its_ledger_at_the_default(self, manager):
        info = manager.create_model('Fresh')
        assert info.elo == DEFAULT_ELO
        entries = manager.read_elo_history(info.id)
        assert len(entries) == 1
        assert entries[0]['note'] == BASELINE_NOTE
        assert entries[0]['new_elo'] == DEFAULT_ELO

    def test_recording_a_result_appends_and_refreshes_the_cache(self, manager):
        info = manager.create_model('Player')
        manager.record_elo_result(info.id, EloEntry(
            new_elo=516.0, elo_delta=16.0, opponent_name='Random Bot',
            opponent_id='random', game_outcome='win',
            game_record_path='match/match_0001.json',
        ))

        entries = manager.read_elo_history(info.id)
        assert [e.get('new_elo') for e in entries] == [DEFAULT_ELO, 516.0]
        # The float in config.json is kept as a cache of the ledger's tail.
        assert manager.get_model(info.id).elo == 516.0

    def test_reading_migrates_a_model_that_predates_the_ledger(self, manager):
        info = manager.create_model('Veteran')
        os.remove(manager.elo_history_path(info.id))
        manager.update_model_state(info.id, elo=1240.0, kyu_rank='5k',
                                   iteration=90, total_games=450)

        entries = manager.read_elo_history(info.id)
        assert len(entries) == 1
        assert entries[0]['new_elo'] == 1240.0
        assert entries[0]['note'] == BASELINE_NOTE

    def test_training_state_updates_leave_the_rating_alone(self, manager):
        """
        The clobber this refactor exists to prevent: a training run holds a
        stale rating in memory and used to write it back over every point the
        model earned in matches while it trained.
        """
        info = manager.create_model('Busy')
        manager.record_elo_result(info.id, EloEntry(new_elo=700.0, elo_delta=200.0,
                                                    game_outcome='win'))

        manager.update_training_state(info.id, iteration=91, total_games=460)

        after = manager.get_model(info.id)
        assert after.elo == 700.0
        assert after.iteration == 91
        assert after.total_games == 460

    def test_a_drifted_cache_is_repaired_from_the_ledger(self, manager):
        info = manager.create_model('Drifted')
        manager.record_elo_result(info.id, EloEntry(new_elo=700.0, elo_delta=200.0,
                                                    game_outcome='win'))
        # Simulate a crash between the ledger append and the config write.
        manager.update_model_state(info.id, elo=500.0, kyu_rank='30k',
                                   iteration=1, total_games=1)

        assert manager.sync_elo_from_history(info.id) == 700.0
        assert manager.get_model(info.id).elo == 700.0

    def test_a_forked_model_inherits_the_rating_history(self, manager):
        info = manager.create_model('Trunk')
        manager.record_elo_result(info.id, EloEntry(new_elo=612.0, elo_delta=112.0,
                                                    game_outcome='win'))
        fork = manager.copy_model(info.id, 'Fork')

        assert [e['new_elo'] for e in manager.read_elo_history(fork.id)] \
            == [DEFAULT_ELO, 612.0]
