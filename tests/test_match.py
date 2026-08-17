"""
test_match.py — Bot vs bot matches: the runner, the rating maths, and the
storage layout for the games it produces.

The tests use a scripted Player rather than a network, so they exercise the
match loop itself (colour alternation, tallying, ratings, recording, stopping)
without paying for MCTS.
"""

import json
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import BLACK, WHITE
from game.game_state import GameState, MOVE_PASS
from ai.evaluator import compute_pairwise_elo
from ai.match import MatchConfig, MatchRunner, STATUS_FINISHED, STATUS_STOPPED
from ai.players import Player, RandomPlayer, create_player
from ai.game_store import (
    PHASE_MATCH,
    delete_saved_game,
    load_match_game_files,
    save_match_game,
)


class ScriptedPlayer(Player):
    """
    Plays the first legal move in row-major order, or passes after `pass_after`
    of its own moves. Deterministic, so a game's winner is fixed.
    """

    kind = 'scripted'

    def __init__(self, name, rating=500.0, pass_after=None, rating_key=None,
                 fixed=False):
        super().__init__(name=name, rating_key=rating_key or f"scripted:{name}",
                         rating=rating)
        self.pass_after = pass_after
        self.moves_made = 0
        self.started_as = []
        self.observed = []
        self._fixed = fixed

    @property
    def rating_is_fixed(self):
        return self._fixed

    def game_started(self, state, color):
        self.started_as.append(color)

    def observe_move(self, state, color, move):
        self.observed.append((color, tuple(move)))

    def select_move(self, state):
        self.moves_made += 1
        if self.pass_after is not None and self.moves_made > self.pass_after:
            return MOVE_PASS
        legal = state.get_legal_moves()
        return legal[0] if legal else MOVE_PASS


def _config(**kwargs):
    defaults = dict(board_size=7, komi=0.5, num_games=2, compute_win_rates=False,
                    record_games=False, move_delay=0.0)
    defaults.update(kwargs)
    return MatchConfig(**defaults)


class TestPairwiseElo:
    def test_win_moves_both_ratings_in_opposite_directions(self):
        new_a, new_b = compute_pairwise_elo(1500, 1500, score_a=1.0, k=16)
        assert new_a == pytest.approx(1508)
        assert new_b == pytest.approx(1492)
        # Points come from the opponent — the total is conserved.
        assert new_a + new_b == pytest.approx(3000)

    def test_draw_between_equals_changes_nothing(self):
        new_a, new_b = compute_pairwise_elo(1200, 1200, score_a=0.5)
        assert new_a == pytest.approx(1200)
        assert new_b == pytest.approx(1200)

    def test_beating_a_stronger_opponent_gains_more(self):
        underdog_gain = compute_pairwise_elo(1000, 1600, 1.0, k=16)[0] - 1000
        favourite_gain = compute_pairwise_elo(1600, 1000, 1.0, k=16)[0] - 1600
        assert underdog_gain > favourite_gain

    def test_ratings_never_go_negative(self):
        new_a, _ = compute_pairwise_elo(2, 3000, score_a=0.0, k=32)
        assert new_a >= 0


class TestMatchRunner:
    def test_plays_the_requested_number_of_games(self):
        runner = MatchRunner('m1', ScriptedPlayer('A', pass_after=6),
                             ScriptedPlayer('B', pass_after=6), _config(num_games=3))
        snapshot = runner.run()

        assert snapshot['status'] == STATUS_FINISHED
        assert snapshot['games_completed'] == 3
        assert len(snapshot['results']) == 3

    def test_colors_alternate_between_games(self):
        player_a = ScriptedPlayer('A', pass_after=4)
        player_b = ScriptedPlayer('B', pass_after=4)
        MatchRunner('m2', player_a, player_b, _config(num_games=2)).run()

        assert player_a.started_as == [BLACK, WHITE]
        assert player_b.started_as == [WHITE, BLACK]

    def test_colors_stay_fixed_when_alternation_is_off(self):
        player_a = ScriptedPlayer('A', pass_after=4)
        player_b = ScriptedPlayer('B', pass_after=4)
        MatchRunner('m3', player_a, player_b,
                    _config(num_games=2, alternate_colors=False)).run()

        assert player_a.started_as == [BLACK, BLACK]

    def test_series_is_tallied_per_player_not_per_color(self):
        # A fills the board, B passes immediately: A wins both games despite
        # switching colour between them.
        player_a = ScriptedPlayer('A')
        player_b = ScriptedPlayer('B', pass_after=0)
        snapshot = MatchRunner('m4', player_a, player_b, _config(num_games=2)).run()

        assert snapshot['series']['a'] == 2
        assert snapshot['series']['b'] == 0

    def test_both_sides_observe_every_move(self):
        player_a = ScriptedPlayer('A', pass_after=3)
        player_b = ScriptedPlayer('B', pass_after=3)
        MatchRunner('m5', player_a, player_b, _config(num_games=1)).run()

        # Every move reaches both players, including their own — that is what
        # a remote (OGS) player needs in order to push our moves upstream.
        assert player_a.observed == player_b.observed
        assert len(player_a.observed) > 0

    def test_ratings_move_after_each_game(self):
        player_a = ScriptedPlayer('A', rating=1000.0)
        player_b = ScriptedPlayer('B', rating=1000.0, pass_after=0)
        snapshot = MatchRunner('m6', player_a, player_b, _config(num_games=2)).run()

        assert snapshot['rated'] is True
        assert player_a.rating > 1000.0
        assert player_b.rating < 1000.0
        assert snapshot['players']['a']['rating_delta'] > 0

    def test_mirror_match_does_not_touch_ratings(self):
        # Same rating key on both sides: the result is decided by colour and
        # luck, so feeding it into Elo would be noise.
        player_a = ScriptedPlayer('Self', rating=1400.0, rating_key='model:x')
        player_b = ScriptedPlayer('Self', rating=1400.0, rating_key='model:x',
                                  pass_after=0)
        snapshot = MatchRunner('m7', player_a, player_b, _config(num_games=1)).run()

        assert snapshot['rated'] is False
        assert player_a.rating == 1400.0
        assert player_b.rating == 1400.0

    def test_fixed_rating_player_is_an_anchor(self):
        model = ScriptedPlayer('Model', rating=900.0)
        anchor = ScriptedPlayer('Anchor', rating=500.0, pass_after=0, fixed=True)
        MatchRunner('m8', model, anchor, _config(num_games=1)).run()

        assert anchor.rating == 500.0
        assert model.rating != 900.0

    def test_update_ratings_disabled(self):
        player_a = ScriptedPlayer('A', rating=1000.0)
        player_b = ScriptedPlayer('B', rating=1000.0, pass_after=0)
        snapshot = MatchRunner('m9', player_a, player_b,
                               _config(num_games=1, update_ratings=False)).run()

        assert snapshot['rated'] is False
        assert player_a.rating == 1000.0

    def test_the_rating_event_names_the_opponent_and_the_outcome(self):
        """
        `commit_rating` used to receive a bare float, so persisting a rating
        destroyed the previous one and left nothing to explain the change.
        """
        events = {}

        class LedgerPlayer(ScriptedPlayer):
            def commit_rating(self, new_rating, event=None):
                super().commit_rating(new_rating, event)
                events[self.name] = event

        winner = LedgerPlayer('Winner', rating=1000.0)
        loser = LedgerPlayer('Loser', rating=1000.0, pass_after=0)
        MatchRunner('m15', winner, loser, _config(num_games=1)).run()

        assert events['Winner'].game_outcome == 'win'
        assert events['Loser'].game_outcome == 'loss'
        assert events['Winner'].opponent_name == 'Loser'
        assert events['Loser'].opponent_id == winner.rating_key
        assert events['Winner'].match_id == 'm15'
        assert events['Winner'].game_index == 0
        # Symmetric: the points one side gains are the points the other loses.
        assert events['Winner'].elo_delta == pytest.approx(-events['Loser'].elo_delta)
        assert events['Winner'].new_elo == winner.rating

    def test_each_side_is_linked_to_its_own_copy_of_the_game(self, tmp_path):
        """
        A match is written into every participating model's games dir as a
        SEPARATE file, so "the game record" is per model. A ledger entry that
        pointed at the other model's copy would open nothing.
        """
        events = {}
        dir_a = tmp_path / 'model-a' / 'games'
        dir_b = tmp_path / 'model-b' / 'games'

        class LocatedPlayer(ScriptedPlayer):
            def __init__(self, name, games_dir, **kw):
                super().__init__(name, **kw)
                self.games_dir = str(games_dir)

            def commit_rating(self, new_rating, event=None):
                super().commit_rating(new_rating, event)
                events[self.name] = event

        player_a = LocatedPlayer('A', dir_a, pass_after=3)
        player_b = LocatedPlayer('B', dir_b, pass_after=3)
        MatchRunner('m16', player_a, player_b,
                    _config(num_games=1, record_games=True),
                    record_dirs=[str(dir_a), str(dir_b)]).run()

        for name, games_dir in (('A', dir_a), ('B', dir_b)):
            rel = events[name].game_record_path
            assert rel, f"{name} has no game to link to"
            # The path is the same id the review page takes: relative to that
            # model's own games dir, and it must actually resolve there.
            assert os.path.isfile(os.path.join(str(games_dir), rel))

    def test_a_player_with_nowhere_to_store_games_links_to_nothing(self, tmp_path):
        """The random bot and OGS opponents have no games dir — and no ledger."""
        events = {}
        dir_a = tmp_path / 'model-a' / 'games'

        class Recording(ScriptedPlayer):
            def commit_rating(self, new_rating, event=None):
                super().commit_rating(new_rating, event)
                events[self.name] = event

        local = Recording('Local', pass_after=3)
        local.games_dir = str(dir_a)
        away = Recording('Away', pass_after=3)

        MatchRunner('m17', local, away, _config(num_games=1, record_games=True),
                    record_dirs=[str(dir_a)]).run()

        assert events['Local'].game_record_path
        assert events['Away'].game_record_path is None

    def test_stop_ends_the_match_between_moves(self):
        class StoppingPlayer(ScriptedPlayer):
            """Stops the match from inside its own first move."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.runner = None

            def select_move(self, state):
                if self.runner and self.moves_made == 2:
                    self.runner.stop()
                return super().select_move(state)

        player_a = StoppingPlayer('A')
        player_b = ScriptedPlayer('B')
        runner = MatchRunner('m10', player_a, player_b, _config(num_games=3))
        player_a.runner = runner
        snapshot = runner.run()

        assert snapshot['status'] == STATUS_STOPPED
        # The interrupted game is abandoned rather than scored.
        assert snapshot['games_completed'] == 0

    def test_snapshot_is_readable_while_the_match_runs(self):
        player_a = ScriptedPlayer('A', pass_after=8)
        player_b = ScriptedPlayer('B', pass_after=8)
        runner = MatchRunner('m11', player_a, player_b,
                             _config(num_games=1, move_delay=0.01))

        thread = threading.Thread(target=runner.run)
        thread.start()
        try:
            snapshot = runner.snapshot()
            assert snapshot['match_id'] == 'm11'
            assert snapshot['players']['a']['name'] == 'A'
        finally:
            thread.join(timeout=30)

        assert not thread.is_alive()

    def test_illegal_move_is_treated_as_a_pass(self):
        class IllegalPlayer(ScriptedPlayer):
            def select_move(self, state):
                return (0, 0)   # occupied from move 2 onwards

        runner = MatchRunner('m12', IllegalPlayer('A'), IllegalPlayer('B'),
                             _config(num_games=1))
        snapshot = runner.run()

        # Two "illegal" moves in a row become two passes, which ends the game.
        assert snapshot['status'] == STATUS_FINISHED
        assert snapshot['games_completed'] == 1


class TestMatchRecording:
    def test_records_one_file_per_participating_model(self, tmp_path):
        dir_a = tmp_path / 'model-a' / 'games'
        dir_b = tmp_path / 'model-b' / 'games'
        runner = MatchRunner(
            'm13', ScriptedPlayer('A', pass_after=3), ScriptedPlayer('B', pass_after=3),
            _config(num_games=1, record_games=True),
            record_dirs=[str(dir_a), str(dir_b)],
        )
        snapshot = runner.run()

        saved = snapshot['results'][0]['saved_as']
        assert len(saved) == 2
        assert all(path.startswith('match' + os.sep) for path in saved)

        for games_dir in (dir_a, dir_b):
            stored = list(load_match_game_files(str(games_dir)))
            assert len(stored) == 1
            _, record = stored[0]
            assert record['phase'] == PHASE_MATCH
            assert record['is_match'] is True
            assert record['match_id'] == 'm13'
            assert record['black_player']['name'] == 'A'
            assert record['white_player']['name'] == 'B'
            assert record['moves']

    def test_the_stored_record_carries_both_sides_of_the_rating_move(self, tmp_path):
        """
        The record is written BEFORE the ratings are committed (the ledger
        entries need its path), so the ratings have to be stamped explicitly —
        otherwise the file would silently show the pre-game numbers.
        """
        games_dir = tmp_path / 'model-a' / 'games'
        runner = MatchRunner(
            'm14', ScriptedPlayer('A', pass_after=3), ScriptedPlayer('B', pass_after=3),
            _config(num_games=1, record_games=True),
            record_dirs=[str(games_dir)],
        )
        runner.run()

        (_, record), = load_match_game_files(str(games_dir))
        for side in (record['black_player'], record['white_player']):
            assert side['rating_before'] == 500.0
            assert side['rating_after'] == side['rating']
            assert side['rating_after'] != 500.0     # the game moved both

    def test_match_games_live_outside_the_iteration_tree(self, tmp_path):
        games_dir = str(tmp_path / 'games')
        rel = save_match_game(games_dir, {'moves': [], 'match_id': 'x'})

        assert rel.startswith('match' + os.sep)
        assert os.path.isfile(os.path.join(games_dir, rel))

    def test_match_games_are_deletable_from_the_review_ui(self, tmp_path):
        games_dir = str(tmp_path / 'games')
        rel = save_match_game(games_dir, {'moves': [], 'match_id': 'x'})

        assert delete_saved_game(games_dir, rel) is True
        assert not os.path.exists(os.path.join(games_dir, rel))

    def test_training_games_and_folders_are_deletable(self, tmp_path):
        games_dir = str(tmp_path / 'games')
        target = os.path.join(games_dir, 'iter_000001', 'self-play', 'game_0000.json')
        os.makedirs(os.path.dirname(target))
        with open(target, 'w') as fh:
            json.dump({'moves': []}, fh)

        assert delete_saved_game(games_dir, 'iter_000001/self-play/game_0000.json') is True
        assert not os.path.exists(target)

    def test_traversal_paths_cannot_be_deleted(self, tmp_path):
        games_dir = str(tmp_path / 'games')
        os.makedirs(games_dir, exist_ok=True)
        outside_file = str(tmp_path / 'outside.txt')
        with open(outside_file, 'w') as fh:
            fh.write('secret')

        assert delete_saved_game(games_dir, '../outside.txt') is False
        assert os.path.exists(outside_file)


class TestPlayerSpecs:
    def test_random_spec_builds_the_anchor(self):
        player = create_player({'type': 'random'}, {'board_size': 9})
        assert isinstance(player, RandomPlayer)
        assert player.rating_is_fixed is True
        assert player.rating == 500.0

    def test_unknown_type_is_rejected(self):
        with pytest.raises(ValueError):
            create_player({'type': 'telepathy'}, {})

    def test_ogs_needs_a_bot_to_play(self):
        # Online play is implemented (see tests/test_ogs_player.py); a spec
        # that names no bot fails on that, not as an unknown opponent type.
        # Offline: the failure happens before anything reaches the network.
        with pytest.raises(ValueError, match="bot_id"):
            create_player({'type': 'ogs'}, {'board_size': 9})

    def test_random_player_produces_legal_moves(self):
        state = GameState(board_size=7, komi=0.5)
        player = RandomPlayer()
        move = player.select_move(state)
        assert move == MOVE_PASS or state.is_legal(move[0], move[1])
