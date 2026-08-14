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

# Tests (suite lives under tests/, 97+ tests)
python -m pytest tests/ -v
python -m pytest tests/test_rules.py -v          # single file
python -m pytest tests/test_rules.py::test_name -v  # single test
```

There is no linter/formatter configured in this repo (no flake8/black/ruff config).

Note: `test_mp.py`, `test_mp2.py`, and `test_socket.py` at the project root are ad-hoc debug scripts (multiprocessing experiments, a socket.io smoke-test client), not part of the pytest suite in `tests/`. Don't confuse them with real tests.

## Architecture

Three independent layers, each importable without the others pulling in unrelated deps:

- **`game/`** — pure Go rules engine, no ML or web dependencies. `board.py` (grid, groups, liberties, Zobrist hashing), `rules.py` (legality, captures, ko/superko), `game_state.py` (`GameState`: move history, undo, NN state encoding via `encode_for_nn()`), `scoring/` (pluggable `ScoringStrategy` via `get_scorer(method)` — chinese/japanese, plus a display-only `estimator.py` using Benson's algorithm + flood-fill that bots never see).
- **`ai/`** — AlphaZero-style pipeline built on top of `game/`. `network.py` (`GoNetwork`: small ResNet, policy + value heads), `mcts.py` (`MCTS`/`MCTSNode`, PUCT selection, Dirichlet root noise), `self_play.py` (plays games with MCTS + network, produces `TrainingSample = (state_tensor, mcts_policy, outcome)`), `trainer.py` (`Trainer`: orchestrates self-play → replay buffer → train → evaluate → checkpoint loop; runs in a background thread when driven from the web UI, emits progress via a callback), `evaluator.py` (Elo tracking, eval vs random bot / previous checkpoint), `checkpoint.py` (save/load `weights.pt`).
- **`web/`** — Flask + SocketIO app (`web/app.py` factory). REST blueprints: `game_routes.py` (`/play`, `/api/game/*` — new/move/bot_move/pass/resign/undo/suggest/estimate), `training_routes.py` (`/training/*`), `model_routes.py` (`/models/api/*`), `api.py` (`/api/health`, `/api/config`). Real-time training progress goes over SocketIO (`start_training`/`stop_training`/`request_status` events, `training_update` emissions), not REST polling.

### Config

`config.py` defines dataclasses (`BoardConfig`, `NetworkConfig`, `MCTSConfig`, `TrainingConfig`, `PathConfig`, bundled into `Config`) as the single source of truth for tunables. `_get_device()` auto-picks MPS > CUDA > CPU. `Config.from_model(model_info, model_dir)` builds a per-model config, routing all paths into that model's directory.

### Multi-model architecture

`model_manager.py`'s `ModelManager` owns everything under `models/`: each model is a directory (`models/<slug>/`) with its own `config.json` (board size/komi/ruleset/training hyperparams/live Elo & iteration state), `weights.pt`, `games/`, and `logs/`. The currently active model id is stored in `models/.active`. `web/app.py`'s `switch_model(model_id)` builds a `Config` from the model's `ModelInfo` and creates a new `Trainer` bound to that model's directory — so switching models swaps the entire training context (weights, replay data, hyperparameters) at once.

### Self-play concurrency

Self-play intentionally runs single-process on the target hardware (M2 MacBook Air) because MPS (Apple GPU) doesn't share well across processes and sequential MPS inference is already fast enough (~5-10 games/min on 9×9). A `ProcessPoolExecutor`-based parallel path exists in `self_play.py` but is meant for CPU-only setups — don't assume it's the active path without checking how it's invoked.

### Data flow for a training iteration

`Trainer.train()` (`ai/trainer.py`): run self-play games with the current `GoNetwork` + `MCTS` → push `TrainingSample`s into a fixed-size FIFO `ReplayBuffer` → sample mini-batches and train → evaluate new weights against the random bot / previous checkpoint to update Elo → persist `weights.pt` and update the model's `config.json` state (elo, kyu_rank, iteration, total_games) → emit progress events (used by both the CLI printer in `run_training.py` and the SocketIO relay in `web/app.py`).
