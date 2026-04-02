#!/usr/bin/env python3
"""Compatibility shim for legacy ``visualizer.py`` usage."""

from whisper_tuner.visualizer import *  # noqa: F401,F403

from whisper_tuner.visualizer import app, socketio

if __name__ == "__main__":
    print("Starting Whisper Training Visualizer in test mode...")
    print("Open http://localhost:8080 in your browser")
    socketio.run(app, host="0.0.0.0", port=8080, debug=True, allow_unsafe_werkzeug=True)
