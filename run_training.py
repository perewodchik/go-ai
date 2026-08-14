#!/usr/bin/env python3
"""
run_training.py — Start training from the command line (without web UI).

Usage:
    python run_training.py [--iterations N] [--board-size 9]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from ai.trainer import Trainer


def print_progress(event: dict):
    """Simple console progress printer."""
    msg = event.get('message', '')
    elo = event.get('elo', 0)
    rank = event.get('kyu_rank', '')
    if msg:
        print(f"[{event.get('type', ''):20s}] {msg}")
    if event.get('type') == 'iteration_done':
        print(f"{'':23s} Elo: {elo:.0f} ({rank})")
        print()


def main():
    parser = argparse.ArgumentParser(description="Go AI Training (CLI)")
    parser.add_argument('--iterations', type=int, default=None,
                        help='Max iterations (None = run forever)')
    parser.add_argument('--board-size', type=int, default=9,
                        help='Board size (7, 9, 13, 15, 19)')
    parser.add_argument('--simulations', type=int, default=200,
                        help='MCTS simulations per move')
    args = parser.parse_args()

    config = Config()
    config.board.size = args.board_size
    config.mcts.num_simulations = args.simulations

    print(f"🧠 Starting Go AI training")
    print(f"   Board size: {config.board.size}x{config.board.size}")
    print(f"   Device: {config.training.device}")
    print(f"   MCTS simulations: {config.mcts.num_simulations}")
    print(f"   Games per iteration: {config.training.num_self_play_games}")
    print()

    trainer = Trainer(config=config, progress_callback=print_progress)
    try:
        trainer.train(max_iterations=args.iterations)
    except KeyboardInterrupt:
        print("\n⏹ Training stopped by user")


if __name__ == '__main__':
    main()
