#!/usr/bin/env python3
"""Compatibility shim for legacy ``visualizer.py`` usage."""

from gemma_tuner.visualizer import *  # noqa: F401,F403
from gemma_tuner.visualizer import _get_app

if __name__ == "__main__":
    print("Starting Gemma Training Visualizer in test mode...")
    print("Open http://localhost:8080 in your browser")
    # host="127.0.0.1" instead of "0.0.0.0": only accept connections from localhost.
    # debug=False: never expose Werkzeug's interactive debugger, even in test mode.
    _app, _sio = _get_app(cors_origin="http://127.0.0.1:8080")
    _sio.run(_app, host="127.0.0.1", port=8080, debug=False, allow_unsafe_werkzeug=True)
