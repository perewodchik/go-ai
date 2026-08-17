#!/usr/bin/env python3
"""
migrate_features.py — move a trained model onto a wider input encoding.

The input encoding is frozen per model because it decides how many channels the
first convolution accepts. This script is the one supported way across: it
widens that convolution with zero-initialised weights, which leaves the network
computing the identical function on the day it runs, and lets it learn to use
the new planes from there. See ai/feature_migration.py for the full argument.

    # See what a model is on
    python scripts/migrate_features.py --list

    # Dry run, then do it
    python scripts/migrate_features.py hot-boi --to v2_12 --dry-run
    python scripts/migrate_features.py hot-boi --to v2_12

Stop training before migrating. The original weights.pt is copied aside first.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.feature_migration import migrate_model
from game.features import FEATURE_SETS, FEATURE_SET_ORDER, resolve
from model_manager import ModelManager


def _list_models(manager):
    print(f"{'model':<36}{'encoding':<12}{'planes':>7}{'iter':>7}{'elo':>8}")
    for info in manager.list_models():
        features = getattr(getattr(info, 'network', None),
                           'input_features', None) or 'v1_10'
        spec = resolve(features)
        print(f"{info.id:<36}{features:<12}{spec.num_planes:>7}"
              f"{info.iteration:>7}{info.elo:>8.0f}")
    print()
    print("available encodings:")
    for key in FEATURE_SET_ORDER:
        spec = FEATURE_SETS[key]
        print(f"  {key:<10} {spec.num_planes:>2} planes  {spec.label}")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate a model to a different input encoding")
    parser.add_argument('model_id', nargs='?', help='Model to migrate')
    parser.add_argument('--to', dest='target', default='v2_12',
                        help='Target feature set (default: v2_12)')
    parser.add_argument('--list', action='store_true',
                        help='List models and their current encodings')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report what would change without writing')
    parser.add_argument('--no-backup', action='store_true',
                        help='Skip copying weights.pt aside first')
    args = parser.parse_args()

    manager = ModelManager()

    if args.list or not args.model_id:
        _list_models(manager)
        return 0

    info = manager.get_model(args.model_id)
    if info is None:
        print(f"No such model: {args.model_id}")
        return 1

    model_dir = manager.get_model_dir(args.model_id)
    current = getattr(getattr(info, 'network', None),
                      'input_features', None) or 'v1_10'

    if args.target not in FEATURE_SETS:
        print(f"Unknown feature set: {args.target}")
        return 1

    print(f"model      : {info.id}  (iteration {info.iteration}, Elo {info.elo:.0f})")
    print(f"encoding   : {current} -> {args.target}")
    print(f"planes     : {resolve(current).num_planes} -> "
          f"{resolve(args.target).num_planes}")

    if current == args.target:
        print("\nAlready on that encoding; nothing to do.")
        return 0

    print("\nWhat this does:")
    print("  * widens input_conv with ZERO weights — the model plays identically")
    print("    today, then learns to use the new planes as training continues")
    print("  * preserves iteration, Elo, gate Elo, total games and the champion")
    print("  * DROPS the optimizer state (Adam re-warms in a few hundred steps)")
    print("  * INVALIDATES the replay buffer — its positions were encoded with")
    print("    the old planes and cannot be reconstructed; it refills from the")
    print("    next iteration's self-play")

    if args.dry_run:
        print("\n(dry run — nothing written)")
        return 0

    reply = input(f"\nMigrate {info.id}? Stop training first. [y/N] ").strip().lower()
    if reply != 'y':
        print("Cancelled.")
        return 1

    result = migrate_model(model_dir, args.target,
                           from_features=current, backup=not args.no_backup)

    if not result.get('migrated'):
        print(f"Nothing to do: {result.get('reason')}")
        return 0

    print("\nDone.")
    if result.get('backup_path'):
        print(f"  backup: {result['backup_path']}")
    if result.get('replay_buffer_invalidated'):
        print("  replay buffer will be rejected on next load (expected)")
    print("\nRestart the server so the model is rebuilt with the new encoding.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
