#!/usr/bin/env python3
"""
run_server.py — Start the Go AI web server.

Usage:
    python run_server.py [--port PORT] [--debug]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from console import use_utf8_console


def main():
    from web.app import create_app, socketio
    
    parser = argparse.ArgumentParser(description="Go AI Web Server")
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 5000)),
                        help='Port to run on (defaults to $PORT, else 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    # Before the first print: the banner below contains an emoji, and on a
    # cp1252 Windows console printing it raises rather than degrading — which
    # would kill the process before socketio.run() ever binds the port.
    use_utf8_console()

    app = create_app()
    print(f"\n🎯 Go AI server starting on http://localhost:{args.port}")
    print("   → Dashboard: http://localhost:{}/".format(args.port))
    print("   → Play:      http://localhost:{}/play".format(args.port))
    print("   → Training:  http://localhost:{}/training/\n".format(args.port))

    socketio.run(app, host='0.0.0.0', port=args.port, debug=args.debug,
                 allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()
