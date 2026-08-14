#!/usr/bin/env python3
"""
run_server.py — Start the Go AI web server.

Usage:
    python run_server.py [--port PORT] [--debug]
"""

import argparse
import sys
import os
import json
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def migrate_legacy_data():
    """Migrate legacy `data/` directory to the new `models/default_9x9/` structure."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    legacy_data = os.path.join(project_root, "data")
    models_dir = os.path.join(project_root, "models")
    
    if os.path.exists(legacy_data):
        default_model = os.path.join(models_dir, "default_9x9")
        if not os.path.exists(default_model):
            print("Migrating legacy data to new models architecture...")
            os.makedirs(models_dir, exist_ok=True)
            os.rename(legacy_data, default_model)
        
        config = {
            "id": "default_9x9",
            "name": "Legacy Default (9x9)",
            "board_size": 9,
            "komi": 6.5,
            "ruleset": "chinese",
            "training": {
                "num_self_play_games": 5,
                "eval_games": 4,
                "num_simulations": 50,
                "c_puct": 1.5,
                "learning_rate": 0.002,
                "batch_size": 32,
                "num_epochs_per_iteration": 3,
                "replay_buffer_size": 50000,
                "reflection_interval_games": 50
            },
            "elo": 500.0,
            "kyu_rank": "30k",
            "iteration": 0,
            "total_games": 0,
        }
        
        # Try to migrate the latest checkpoint to the new single weights.pt file
        ckpt_dir = os.path.join(default_model, "checkpoints")
        if os.path.exists(ckpt_dir):
            checkpoints = glob.glob(os.path.join(ckpt_dir, "checkpoint_*.pt"))
            if checkpoints:
                latest = max(checkpoints)
                print(f"Migrating checkpoint: {latest}")
                os.rename(latest, os.path.join(default_model, "weights.pt"))
                
                meta_json = latest.replace('.pt', '.json')
                if os.path.exists(meta_json):
                    try:
                        with open(meta_json) as mf:
                            meta = json.load(mf)
                            config["iteration"] = meta.get("iteration", 0)
                            config["elo"] = meta.get("elo", 500.0)
                            config["total_games"] = meta.get("total_games", 0)
                            config["kyu_rank"] = meta.get("kyu_rank", "30k")
                    except Exception as e:
                        print(f"Warning: could not read checkpoint metadata: {e}")
        
        with open(os.path.join(default_model, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
            
        with open(os.path.join(models_dir, "active_model.txt"), "w") as f:
            f.write("default_9x9")
            
        print("Migration complete!")


def main():
    migrate_legacy_data()
    
    from web.app import create_app, socketio
    
    parser = argparse.ArgumentParser(description="Go AI Web Server")
    parser.add_argument('--port', type=int, default=5000, help='Port to run on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    app = create_app()
    print(f"\n🎯 Go AI server starting on http://localhost:{args.port}")
    print("   → Dashboard: http://localhost:{}/".format(args.port))
    print("   → Play:      http://localhost:{}/play".format(args.port))
    print("   → Training:  http://localhost:{}/training/\n".format(args.port))

    socketio.run(app, host='0.0.0.0', port=args.port, debug=args.debug,
                 allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()
