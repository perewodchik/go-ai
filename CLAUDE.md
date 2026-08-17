# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the web server (dashboard, play UI, training UI)
python run_server.py [--port PORT] [--debug]

# Run training headless from the CLI
python run_training.py [--iterations N] [--board-size 9] [--simulations 200]

# Check the OGS connection (no game is played)
python scripts/ogs_probe.py

# Tests (suite lives under tests/, 500+ tests)
python -m pytest tests/ -v
python -m pytest tests/test_rules.py -v          # single file
python -m pytest tests/test_rules.py::test_name -v  # single test
```

There is no linter/formatter configured in this repo (no flake8/black/ruff config).

Note: `test_mp.py`, `test_mp2.py`, and `test_socket.py` at the project root are ad-hoc debug scripts (multiprocessing experiments, a socket.io smoke-test client), not part of the pytest suite in `tests/`. Don't confuse them with real tests.

## Architecture

Three independent layers, each importable without the others pulling in unrelated deps:

- **`game/`** — pure Go rules engine, no ML or web dependencies. `board.py` (grid, groups, liberties, Zobrist hashing), `rules.py` (legality, captures, ko/superko), `game_state.py` (`GameState`: move history, undo, NN state encoding via `encode_for_nn()`), `scoring/` (pluggable `ScoringStrategy` via `get_scorer(method)` — chinese/japanese, plus a display-only `estimator.py` using Benson's algorithm + flood-fill that bots never see).
- **`ai/`** — AlphaZero-style pipeline built on top of `game/`. `network.py` (`GoNetwork`: small ResNet, policy + value heads), `mcts.py` (`MCTS`/`MCTSNode`, PUCT selection, Dirichlet root noise), `self_play.py` (plays games with MCTS + network, produces `TrainingSample = (state_tensor, mcts_policy, outcome)`), `trainer.py` (`Trainer`: orchestrates self-play → replay buffer → train → promotion gate → checkpoint loop; runs in a background thread when driven from the web UI, emits progress via a callback), `evaluator.py` (the promotion gate: candidate vs champion, plus the pairwise Elo maths matches use), `checkpoint.py` (save/load `weights.pt`).
- **`web/`** — Flask + SocketIO app (`web/app.py` factory). REST blueprints: `game_routes.py` (`/play`, `/api/game/*` — new/move/bot_move/pass/resign/undo/suggest/estimate), `training_routes.py` (`/training/*`), `model_routes.py` (`/models/api/*`), `api.py` (`/api/health`, `/api/config`). Real-time training progress goes over SocketIO (`start_training`/`stop_training`/`request_status` events, `training_update` emissions), not REST polling.

### Config

`config.py` defines dataclasses (`BoardConfig`, `NetworkConfig`, `MCTSConfig`, `TrainingConfig`, `PathConfig`, bundled into `Config`) as the single source of truth for tunables. `_get_device()` auto-picks MPS > CUDA > CPU. `Config.from_model(model_info, model_dir)` builds a per-model config, routing all paths into that model's directory.

### Multi-model architecture

`model_manager.py`'s `ModelManager` owns everything under `models/`: each model is a directory (`models/<slug>/`) with its own `config.json` (board size/komi/ruleset/training hyperparams/live Elo & iteration state), `weights.pt`, `elo_history.jsonl`, `games/`, and `logs/`.

Recorded games live under `games/iter_<NNNNNN>/<phase>/`, where phase is `self-play`, `promotion` (candidate vs champion gate matches), or `eval` (champion vs random bot). `ai/game_store.py` owns that layout — writing (`save_game`, which also indexes every game — see below), listing (`iter_game_files` / `load_game_files`), resolving a client-supplied id (`resolve_game_path`, which rejects paths escaping `games/`), and migrating the old flat `iter_000001_game_0000.json` naming (`migrate_legacy_layout`, run from `Trainer.__init__` and the games API). A game's id everywhere — API, review URLs — is its path relative to `games/`. The currently active model id is stored in `models/.active`. `web/app.py`'s `switch_model(model_id)` builds a `Config` from the model's `ModelInfo` and creates a new `Trainer` bound to that model's directory — so switching models swaps the entire training context (weights, replay data, hyperparameters) at once.

### The games index — `games/index.jsonl`

Alongside the records, every model keeps ONE LINE PER GAME in
`games/index.jsonl`, owned by `ai/game_index.py`: the ~dozen scalars the
charts need (`iteration`, `phase`, `game_index`, `winner`, `margin`,
`num_moves`, `elapsed_seconds`, `network_color`, `candidate_won`, and the
mercy-rule fields), and none of the bulk.

**Every statistics endpoint reads the index, never the records.**
`/training/api/learning_stats` and `/training/api/resign_stats` used to open
and parse every game file on disk — 3.5s and growing on an 11k-game model,
paid twice on every training-page load for the same numbers. Off the index
both are under 0.1s. The full records are read to REPLAY one game, and never
in bulk.

Two invariants make it safe to rely on:

- **A row is identified by `(iteration, phase, game_index)`, not by a path** —
  because a row exists for games whose record was never written.
- **`load()` reconciles against the directory on every read**, appending rows
  for game files it has not seen. That is the migration path (a model that
  predates the index builds one on first read) and the repair path, and it
  costs one directory walk with zero JSON parses when the index is current.
  Reconciliation only ever ADDS; `delete_saved_game` is what prunes, so
  deleting games clears both the bytes and the statistics.

### Recording toggles — `record_self_play_games` / `record_gate_games`

Because the index is separate, writing the records is optional.
`save_game(..., store_full=False)` indexes the game and skips the file, so a
run with recording off keeps **every point on every chart** and loses only the
ability to replay those games. That is the whole trade: ~12 KB per 9x9 game,
and the gate is the bigger offender (`gate_games`, 20 by default, against
`num_self_play_games`).

Both are live-tunable `TrainingConfig` / `TrainingParams` booleans plumbed
through `run_self_play_batch(record_games=)` and
`evaluate_against_checkpoint(record_games=)`. They appear in the UI
automatically — the Create/Edit/Live-Tune panels are generated from
`param_bounds.PARAM_BOUNDS` (category `storage`), so adding the entries there
was the only edit needed. The games list reports `not_recorded` per phase and
`total_not_recorded` per iteration, so an iteration with nothing on disk says
why instead of looking empty.

### Playing on OGS (online-go.com)

`ai/online/` bridges to online-go.com so a model can play the live bots there.
It plugs in as a `Player` (`ai/players.py`), so `ai/match.py`, the live board,
the game recorder and the Elo update are all unchanged — the spec
`{"type": "ogs", "bot_id": 1195517}` is just another opponent.

- **`ogs_socket.py`** — OGS's realtime API is NOT socket.io: it is a plain
  WebSocket at `wss://online-go.com` framing messages as JSON arrays
  (`["command", data, request_id?]` out, `[command_or_id, data, error]` back).
  `OGSSocket` wraps that with a synchronous API for this project's threaded
  match loop. Events pushed on connect (`active-bots`) need `latch()`, armed
  before the triggering action — a handler registered afterwards misses them.
- **`ogs_bots.py`** — the roster, which OGS pushes as `active-bots` to any
  client, including an unauthenticated one, so listing opponents needs no
  credentials. Cached at `data/ogs_bots.json` (15 min TTL, stale cache used if
  OGS is unreachable). `playability()` ports OGS's own compatibility check, so
  a bot is greyed out here for the same reason as on their site.
  `ranking_to_elo()` converts an OGS rank to this project's Elo scale (500 =
  30k, 100 per rank) — the raw glicko numbers are a different scale entirely.
- **`ogs_rest.py`** — OAuth2 sign-in and the three calls that cannot go over
  the socket: challenge a bot, withdraw a challenge, read a game.
- **`ogs.py`** — `OGSPlayer`. Challenges for the colour the match runner
  assigned, waits for the bot to accept (with `challenge/keepalive`), then
  exchanges moves. OGS is authoritative, so it refuses to start a game whose
  size/colour/handicap differ from the match, and aborts if a move OGS reports
  is illegal on our board rather than letting the runner turn it into a pass.
- **`ogs_coords.py`** — OGS packs coordinates as two letters, x first
  (`(3, 15)` → `"pd"`, pass `".."`); game records use `[x, y, time_ms]`.

Credentials live in `ogs_credentials.json` (gitignored) or `OGS_USERNAME` /
`OGS_PASSWORD` / `OGS_CLIENT_ID` / `OGS_CLIENT_SECRET`, from an application
registered at https://online-go.com/oauth2/applications/ (Confidential +
Resource owner password-based). **OGS shows the client secret once, on the page
right after the application is created** — the value on the edit page
afterwards is a `pbkdf2_` hash and never works. `python scripts/ogs_probe.py`
checks the whole path (sign-in, socket token, roster) without playing a game.

OGS asks that engine-driven accounts be registered as bot accounts; matches
default to unranked and are capped at `MAX_OGS_GAMES` per series.

Three things about OGS that are not written down anywhere and each cost a live
game to find — all now covered by `tests/test_ogs_player.py`:

- **The game id in the challenge response is provisional.** The real one
  arrives on the `active_game` event; `_NewGameWatcher` waits for it.
- **Moves are numbered from 1**, by the board's move count *after* the move, so
  our own move echoes back at the number our board already holds.
- **A resignation is not announced on `game/<id>/phase`.** Any game that goes
  quiet is checked against OGS's own record (`_ended_on_ogs`), which is also
  read for *who* lost so a loss can never be recorded as a win.

Most bots set `max_games_per_player: 1`, so the next game of a series cannot be
challenged until OGS has finished the previous one — our runner ends a game as
soon as OUR board is over, which is earlier. `_wait_for_previous_game_to_clear`
bridges that gap; without it a 5 game series stops at 4 with a challenge nobody
answers. `OGSPlayer.status()` reports what the bridge is waiting on (it reaches
the live match panel via the snapshot), `cancel()` makes Stop Match immediate
instead of sitting out `ACCEPT_TIMEOUT`, and both `cancel()` and `close()` give
back any game on OGS we would otherwise abandon.

### Concurrency

Both game-playing training phases run their games across a `ProcessPoolExecutor`, sized by the single `config.training.num_parallel_workers` setting (capped at the CPU count and at that phase's game count): self-play (`run_self_play_batch`) and the promotion gate (`evaluate_against_checkpoint`). `num_workers=1` just means a pool of one.

MPS is not a constraint here despite the Apple-Silicon target: `Trainer` keeps a separate CPU-resident `eval_network` (the gated champion) and hands *that* to both phases with `device="cpu"`, so the MPS training network is never pickled into a worker. Only `_train_network` uses the MPS device.

### Data flow for a training iteration

`Trainer.train()` (`ai/trainer.py`): run self-play games with the current `GoNetwork` + `MCTS` → push `TrainingSample`s into a fixed-size FIFO `ReplayBuffer` → sample mini-batches and train → put the candidate through the promotion gate against the champion → persist `weights.pt` and update the model's `config.json` training state (iteration, total_games) → emit progress events (used by both the CLI printer in `run_training.py` and the SocketIO relay in `web/app.py`).

Four stages, not five: the random-bot evaluation phase that used to sit between the gate and the checkpoint is gone. **Training does not rate a model.** It writes `iteration` and `total_games` (via `ModelManager.update_training_state`) and never touches `elo` — a long run overlapping a match series would otherwise write its stale in-memory rating back over every point the model earned. The only rating training maintains is `gate_elo`, its own self-referential ladder, which moves via `performance_elo_gap` when and only when a candidate beats the champion.

### Competitive Elo — the `elo_history.jsonl` ledger

A model's rating is the last line of `models/<slug>/elo_history.jsonl`, an append-only ledger owned by `elo_history.py` (a root peer of `model_manager.py`/`model_stats.py`). `config.json`'s `elo` is a **cache** of that tail, kept in sync so a fleet listing of forty models does not have to open forty ledgers; `elo_history.reconcile` / `ModelManager.sync_elo_from_history` repair it if the two drift.

Each line records `timestamp`, `opponent_name`/`opponent_id` (the opponent's `rating_key`), `game_outcome`, `elo_delta`, `new_elo` and `game_record_path` — the same id `save_match_game` returns, so every point on the Elo curve links straight to its game in the review page. The path is stored **per model** because a match is written into both participants' `games/` directories as two separate files; `MatchRunner._saved_path_for` matches a player's `games_dir` against the record dirs to pick the right one.

`ai/match.py` is the only thing that moves a competitive rating, and the order in `_finish_game` matters: ratings are COMPUTED (`compute_pairwise_elo`), the game is WRITTEN (it carries the rating delta and explicit `rating_before`/`rating_after` per side), then the ratings are COMMITTED — `commit_rating(new_rating, event)` with an `EloEntry` that now has a path to point at. `ModelPlayer`'s sink is `ai/model_loader.persist_model_rating` → `ModelManager.record_elo_result`.

**Migration:** a model with no ledger but an existing `elo` gets one baseline entry (`note: "Baseline initialization"`) carrying that rating, so a model rated 1240 after ninety iterations is not reset to the default. It fires lazily on first read (`ModelManager.read_elo_history`, `model_stats.elo_curve`) and eagerly in `create_model`. Baseline entries have no `game_outcome`, so `rated_games` does not count them as earned.

The dashboard plots this curve against **games played**, not training iteration (`model_stats.elo_curve` → `/models/api/<id>/elo_history`): a rating that only moves when a game is played draws a flat line through every iteration nobody played in, and collapses a forty-game evening into a single point. `/models/api/<id>/history` stays per-iteration for losses and the gate.
