"""
test_model_stats.py — Tests for the read-only fleet statistics layer.

`model_stats` derives four things nobody stores explicitly, and each has a way
of being subtly wrong:

  * **lineage** — sharing a log prefix is SYMMETRIC, so the naive version makes
    every fork pair parent each other. Creation order is what breaks it.
  * **head-to-head** — every match game is stored in both participants'
    directories, so a naive tally doubles every score.
  * **health** — a verdict is only useful if it fires on the conditions that
    actually happened (a recorded collapse, a stalled gate) and stays quiet
    otherwise.
  * **cost** — games and bytes are what a delete destroys, so they have to be
    counted, not estimated.

Everything runs against a synthetic `models/` tree so the assertions do not
depend on whatever the user has trained.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model_stats
from model_manager import ModelInfo


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    """
    A models/ directory builder.

    `add(id, rows, created=..., games=...)` writes a config and a metrics log;
    the returned helper also exposes `infos()` for functions that take
    ModelInfo objects.
    """
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(model_stats, "MODELS_ROOT", str(root))
    model_stats.invalidate_caches()

    created_models = []

    def add(model_id, rows=(), created="2026-01-01T00:00:00", games=0,
            match_games=(), **config):
        model_dir = root / model_id
        (model_dir / "logs").mkdir(parents=True)
        cfg = {"id": model_id, "name": model_id, "created_at": created}
        cfg.update(config)
        (model_dir / "config.json").write_text(json.dumps(cfg))

        with open(model_dir / "logs" / "training_log.jsonl", "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

        if games:
            games_dir = model_dir / "games" / "iter_000001" / "self-play"
            games_dir.mkdir(parents=True)
            for i in range(games):
                (games_dir / f"game_{i:04d}.json").write_text('{"moves": []}')

        if match_games:
            match_dir = model_dir / "games" / "match"
            match_dir.mkdir(parents=True, exist_ok=True)
            for name, payload in match_games:
                (match_dir / name).write_text(json.dumps(payload))

        created_models.append(ModelInfo.from_dict(dict(cfg)))
        return model_dir

    add.root = root
    add.infos = lambda: list(created_models)
    return add


def run(n, start=1, stamp="2026-01-01T00:%02d:00", **extra):
    """`n` log rows with distinct, reproducible timestamps."""
    rows = []
    for i in range(start, start + n):
        row = {"iteration": i, "timestamp": stamp % i, "elo": 500 + i}
        row.update(extra)
        rows.append(row)
    return rows


class TestReadLog:

    def test_missing_log_is_empty_not_an_error(self, fleet):
        fleet("fresh")
        assert model_stats.read_log("fresh") == []

    def test_corrupt_lines_are_skipped(self, fleet):
        model_dir = fleet("bent", run(2))
        with open(model_dir / "logs" / "training_log.jsonl", "a") as fh:
            fh.write("{not json\n")
            fh.write(json.dumps({"iteration": 3, "timestamp": "t3"}) + "\n")
        rows = model_stats.read_log("bent")
        assert [r["iteration"] for r in rows] == [1, 2, 3]

    def test_rows_come_back_in_iteration_order(self, fleet):
        model_dir = fleet("shuffled")
        with open(model_dir / "logs" / "training_log.jsonl", "w") as fh:
            for i in (3, 1, 2):
                fh.write(json.dumps({"iteration": i, "timestamp": f"t{i}"}) + "\n")
        assert [r["iteration"] for r in model_stats.read_log("shuffled")] == [1, 2, 3]

    def test_cache_is_invalidated_when_the_log_grows(self, fleet):
        model_dir = fleet("growing", run(2))
        assert len(model_stats.read_log("growing")) == 2
        with open(model_dir / "logs" / "training_log.jsonl", "a") as fh:
            fh.write(json.dumps({"iteration": 3, "timestamp": "t3"}) + "\n")
        assert len(model_stats.read_log("growing")) == 3


class TestLineage:

    def test_a_fork_points_at_the_model_it_was_copied_from(self, fleet):
        fleet("trunk", run(20), created="2026-01-01T00:00:00")
        fleet("snap", run(7), created="2026-01-02T00:00:00")
        nodes = model_stats.lineage(fleet.infos())

        assert nodes["snap"]["parent"] == "trunk"
        assert nodes["snap"]["fork_iteration"] == 7
        assert nodes["trunk"]["parent"] is None
        assert nodes["trunk"]["children"] == ["snap"]

    def test_two_forks_of_one_trunk_do_not_parent_each_other(self, fleet):
        """
        The early snapshot's log is a prefix of the later snapshot's too, so
        without creation order it would be parented by its own sibling.
        """
        fleet("trunk", run(40), created="2026-01-01T00:00:00")
        fleet("early", run(7), created="2026-01-02T00:00:00")
        fleet("late", run(19), created="2026-01-03T00:00:00")
        nodes = model_stats.lineage(fleet.infos())

        assert nodes["early"]["parent"] == "trunk"
        assert nodes["late"]["parent"] == "trunk"
        assert sorted(nodes["trunk"]["children"]) == ["early", "late"]

    def test_a_fork_that_kept_training_is_never_its_parent_s_parent(self, fleet):
        """
        Both sides have rows past the fork, so both qualify structurally. Only
        creation order says which one existed to be copied.
        """
        fleet("origin", run(48), created="2026-01-01T00:00:00")
        fleet("branch",
              run(27) + run(2, start=28, stamp="2026-06-01T00:%02d:00"),
              created="2026-01-01T06:00:00")
        nodes = model_stats.lineage(fleet.infos())

        assert nodes["branch"]["parent"] == "origin"
        assert nodes["branch"]["fork_iteration"] == 27
        assert nodes["branch"]["own_iterations"] == 2
        assert nodes["origin"]["parent"] is None, "cycle: parents point at each other"

    def test_unrelated_runs_have_no_parent(self, fleet):
        fleet("a", run(10, stamp="2026-01-01T00:%02d:00"))
        fleet("b", run(10, stamp="2026-02-02T00:%02d:00"))
        nodes = model_stats.lineage(fleet.infos())
        assert nodes["a"]["parent"] is None
        assert nodes["b"]["parent"] is None

    def test_a_chain_of_forks_resolves_to_one_root(self, fleet):
        gen2_log = run(20) + run(5, start=21, stamp="2026-03-01T00:%02d:00")
        fleet("gen1", run(30), created="2026-01-01T00:00:00")
        fleet("gen2", gen2_log, created="2026-01-02T00:00:00")
        fleet("gen3", gen2_log + run(3, start=26, stamp="2026-04-01T00:%02d:00"),
              created="2026-01-03T00:00:00")
        nodes = model_stats.lineage(fleet.infos())

        assert nodes["gen3"]["parent"] == "gen2"
        assert nodes["gen3"]["root"] == "gen1"

    def test_a_snapshot_of_a_snapshot_keeps_the_nearer_parent(self, fleet):
        """
        gen3 is copied from gen2 and never trained, so their logs are identical
        — the pair a "parent must have trained further" rule gets wrong.
        """
        gen2_log = run(20) + run(5, start=21, stamp="2026-03-01T00:%02d:00")
        fleet("gen1", run(30), created="2026-01-01T00:00:00")
        fleet("gen2", gen2_log, created="2026-01-02T00:00:00")
        fleet("gen3", list(gen2_log), created="2026-01-03T00:00:00")
        nodes = model_stats.lineage(fleet.infos())

        assert nodes["gen3"]["parent"] == "gen2"
        assert nodes["gen3"]["own_iterations"] == 0
        assert nodes["gen3"]["root"] == "gen1"

    def test_an_untrained_model_is_its_own_root(self, fleet):
        fleet("empty")
        nodes = model_stats.lineage(fleet.infos())
        assert nodes["empty"] == {
            "parent": None, "fork_iteration": None, "own_iterations": 0,
            "children": [], "root": "empty",
        }


class TestHeadToHead:

    def match(self, match_id, index, black, white, winner):
        return (f"match_{match_id}_{index}.json", {
            "match_id": match_id,
            "game_index": index,
            "winner": winner,
            "timestamp": f"2026-01-0{index + 1}T00:00:00",
            "black_player": {"rating_key": black},
            "white_player": {"rating_key": white},
        })

    def test_a_game_stored_in_both_directories_is_counted_once(self, fleet):
        """The match runner writes every game into both models' dirs."""
        game = self.match("m1", 0, "model:alpha", "model:beta", 1)
        fleet("alpha", match_games=[game])
        fleet("beta", match_games=[game])

        table = model_stats.head_to_head()
        assert table["alpha"]["beta"]["games"] == 1
        assert table["alpha"]["beta"]["wins"] == 1
        assert table["beta"]["alpha"]["losses"] == 1

    def test_results_are_recorded_from_both_sides(self, fleet):
        fleet("alpha", match_games=[
            self.match("m1", 0, "model:alpha", "model:beta", 1),
            self.match("m1", 1, "model:beta", "model:alpha", 1),
        ])
        table = model_stats.head_to_head()
        assert (table["alpha"]["beta"]["wins"], table["alpha"]["beta"]["losses"]) == (1, 1)
        assert (table["beta"]["alpha"]["wins"], table["beta"]["alpha"]["losses"]) == (1, 1)

    def test_draws_are_neither_wins_nor_losses(self, fleet):
        fleet("alpha", match_games=[self.match("m1", 0, "model:alpha", "model:beta", 0)])
        cell = model_stats.head_to_head()["alpha"]["beta"]
        assert (cell["wins"], cell["losses"], cell["draws"]) == (0, 0, 1)

    def test_the_random_bot_is_an_opponent_like_any_other(self, fleet):
        fleet("alpha", match_games=[self.match("m1", 0, "model:alpha", "random", 1)])
        assert model_stats.head_to_head()["alpha"]["random"]["wins"] == 1

    def test_ogs_bot_opponent_name_and_kind_are_preserved(self, fleet):
        game = ("match_m1_0.json", {
            "match_id": "m1",
            "game_index": 0,
            "winner": 2,
            "timestamp": "2026-01-01T00:00:00",
            "black_player": {"rating_key": "model:alpha", "name": "alpha", "kind": "model"},
            "white_player": {"rating_key": "ogs:1195518", "name": "Carnation", "kind": "ogs"},
        })
        fleet("alpha", match_games=[game])
        cell = model_stats.head_to_head()["alpha"]["ogs:1195518"]
        assert cell["opponent_name"] == "Carnation"
        assert cell["opponent_kind"] == "ogs"
        assert cell["losses"] == 1

    def test_no_matches_is_an_empty_table(self, fleet):
        fleet("alpha", run(3))
        assert model_stats.head_to_head() == {}


class TestHealth:

    def test_an_untrained_model_is_idle_not_unhealthy(self, fleet):
        fleet("fresh")
        verdict = model_stats.health("fresh")
        assert verdict["level"] == "idle"
        assert verdict["headline"] == "Never trained"

    def test_a_recorded_collapse_is_critical(self, fleet):
        rows = run(3, gate_win_rate=0.8, gate_promoted=True)
        rows[-1]["collapse_warning"] = "White is passing 25% of its moves"
        fleet("collapsed", rows)
        verdict = model_stats.health("collapsed")
        assert verdict["level"] == "critical"
        assert "passing 25%" in verdict["reasons"][0]["text"]

    def test_a_stalled_gate_warns(self, fleet):
        rows = run(8, gate_win_rate=0.4, gate_promoted=False)
        fleet("stalled", rows)
        verdict = model_stats.health("stalled")
        assert verdict["level"] == "warn"
        assert "rejected 8 candidates in a row" in verdict["reasons"][0]["text"]

    def test_a_promoting_gate_is_healthy(self, fleet):
        fleet("healthy", run(8, gate_win_rate=0.7, gate_promoted=True))
        verdict = model_stats.health("healthy")
        assert verdict["level"] == "ok"
        assert "promoted 8 of 8" in verdict["reasons"][0]["text"]

    def test_no_gate_evidence_is_called_out(self, fleet):
        """The Elo of an ungated run rests on a measure that saturates."""
        fleet("ungated", run(20))
        verdict = model_stats.health("ungated")
        assert verdict["level"] == "warn"
        assert "No promotion-gate matches" in verdict["reasons"][0]["text"]

    def test_a_wrong_resignation_rate_warns_only_with_enough_checks(self, fleet):
        noisy = run(3, gate_win_rate=0.7, gate_promoted=True)
        noisy[-1].update(false_resign_rate=0.5, resign_checked_games=2)
        fleet("noisy", noisy)
        assert model_stats.health("noisy")["level"] == "ok"

        evidenced = run(3, gate_win_rate=0.7, gate_promoted=True)
        evidenced[-1].update(false_resign_rate=0.5, resign_checked_games=40)
        fleet("evidenced", evidenced)
        verdict = model_stats.health("evidenced")
        assert verdict["level"] == "warn"
        assert any("mislabelling" in r["text"] for r in verdict["reasons"])


class TestSummary:

    def test_counts_games_and_bytes_actually_on_disk(self, fleet):
        fleet("counted", run(3), games=12)
        info = fleet.infos()[0]
        summary = model_stats.summarize(info)
        assert summary["games_on_disk"] == 12
        assert summary["bytes_on_disk"] > 0

    def test_a_model_with_no_games_directory_still_summarizes(self, fleet):
        """The legacy default model has a config and nothing else."""
        fleet("bare")
        summary = model_stats.summarize(fleet.infos()[0])
        assert summary["games_on_disk"] == 0
        assert summary["iterations_logged"] == 0
        assert summary["elo_series"] == []

    def test_gate_record_is_counted_from_the_log(self, fleet):
        rows = run(4, gate_win_rate=0.7, gate_promoted=True)
        rows[1]["gate_promoted"] = False
        fleet("gated", rows)
        summary = model_stats.summarize(fleet.infos()[0])
        assert summary["gate_matches"] == 4
        assert summary["gate_promotions"] == 3

    def test_elo_series_is_downsampled_but_keeps_the_last_point(self, fleet):
        fleet("long", run(200))
        summary = model_stats.summarize(fleet.infos()[0])
        assert len(summary["elo_series"]) <= 24
        assert summary["elo_series"][-1] == 700   # 500 + iteration 200

    def test_elo_delta_covers_the_last_ten_iterations(self, fleet):
        fleet("moving", run(30))    # elo climbs by 1 per iteration
        summary = model_stats.summarize(fleet.infos()[0])
        assert summary["elo_delta_10"] == 10

    def test_history_returns_only_requested_fields(self, fleet):
        fleet("charted", run(5, total_loss=1.5))
        data = model_stats.history("charted", ["elo", "total_loss"])
        assert data["iterations"] == [1, 2, 3, 4, 5]
        assert set(data["series"]) == {"elo", "total_loss"}
        assert data["series"]["total_loss"] == [1.5] * 5

    def test_history_reports_missing_fields_as_gaps(self, fleet):
        """Older rows predate fields added later; a gap is not a zero."""
        rows = run(3)
        rows[-1]["gate_win_rate"] = 0.6
        fleet("partial", rows)
        data = model_stats.history("partial", ["gate_win_rate"])
        assert data["series"]["gate_win_rate"] == [None, None, 0.6]
