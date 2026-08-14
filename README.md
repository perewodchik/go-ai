# Go AI — AlphaZero-style Self-Play Training

A complete Go game engine with machine learning self-play training, built in Python.
Features a web UI for monitoring training progress and playing against the bot.

## Quick Start

```bash
# 1. Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Start the web server
python run_server.py

# 3. Open in browser
# Dashboard: http://localhost:5000/
# Play:      http://localhost:5000/play
# Training:  http://localhost:5000/training/
```

## Training from Command Line

```bash
# Train with defaults (9x9, 200 MCTS simulations)
python run_training.py

# Custom settings
python run_training.py --board-size 7 --simulations 100 --iterations 50
```

## Project Structure

```
go-ai/
├── config.py              # All configuration in one place
├── run_server.py           # Web server entry point
├── run_training.py         # CLI training entry point
│
├── game/                   # Go game engine (pure logic)
│   ├── board.py            # Board state, groups, liberties
│   ├── rules.py            # Move validation, captures, ko, superko
│   ├── game_state.py       # Full game state with history
│   └── scoring/            # Scoring strategies
│       ├── base.py         # Abstract interface + factory
│       ├── chinese.py      # Area scoring (stones + territory)
│       ├── japanese.py     # Territory scoring (territory + captures)
│       └── estimator.py    # Display-only score estimation
│
├── ai/                     # Machine learning
│   ├── network.py          # ResNet with policy + value heads
│   ├── mcts.py             # Monte Carlo Tree Search
│   ├── self_play.py        # Self-play game generation
│   ├── trainer.py          # Training loop orchestrator
│   ├── evaluator.py        # Elo rating & strength estimation
│   ├── random_bot.py       # Random move baseline bot
│   └── checkpoint.py       # Save/load model checkpoints
│
├── web/                    # Flask web application
│   ├── app.py              # Flask + SocketIO factory
│   ├── routes/             # API endpoints
│   ├── templates/          # HTML pages
│   └── static/             # CSS + JavaScript
│
├── tests/                  # Test suite (97 tests)
│   ├── test_board.py       # Board operations
│   ├── test_rules.py       # Rules enforcement
│   ├── test_scoring.py     # Scoring systems
│   └── test_game.py        # Full game integration
│
└── data/                   # Runtime data (auto-created)
    ├── checkpoints/        # Model snapshots
    ├── games/              # Stored self-play games
    └── logs/               # Training metrics
```

## Features

### Game Engine
- Full Go rules: captures, ko, superko, suicide prevention
- Configurable board sizes: 7×7, 9×9, 13×13, 15×15, 19×19
- Chinese scoring (default) with easy swap to Japanese scoring
- Configurable komi (default 6.5)

### AI / Training
- **AlphaZero architecture**: ResNet + MCTS self-play
- Starts with random moves, gradually improves
- CPU/MPS optimized (small network: 4 res blocks, 64 filters)
- Elo rating with kyu/dan rank estimation
- Automatic checkpointing and training logs
- Progress reflection reports every N games

### Web Interface
- **Dashboard**: Current bot strength, total games, quick actions
- **Play vs Bot**: Two modes:
  - **Easy**: Undo, move suggestions, analysis overlay
  - **Hard**: Pure play, no assistance
- **Training Dashboard**: Real-time charts (Elo, loss, win rate),
  game browser with replay, checkpoint management, milestone notifications
- **Score Estimation**: Toggleable territory overlay using Benson's algorithm
  and flood-fill (display only — bots never see this)

### Hardware Requirements
- **MacBook Air M2**: ~5-10 games/min on 9×9 with MPS
- **CPU only**: Slower but functional
- Checkpoints saved every 3-5 minutes

## Running Tests

```bash
python -m pytest tests/ -v
```

## Configuration

All settings live in `config.py`. Key tunables:
- `BoardConfig.size`: Board dimension (default 9)
- `BoardConfig.komi`: Compensation for white (default 6.5)
- `MCTSConfig.num_simulations`: Search depth (default 200)
- `TrainingConfig.num_self_play_games`: Games per iteration (default 25)
- `TrainingConfig.game_store_every_n`: Store 1/N games (default 5)
