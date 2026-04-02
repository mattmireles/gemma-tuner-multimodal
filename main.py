#!/usr/bin/env python3
"""Compatibility shim for legacy ``main.py`` usage.

Canonical entrypoint is now `whisper-tuner` (`whisper_tuner.cli_typer`).
This file keeps `python main.py ...` functional by delegating to
`whisper_tuner.main`.
"""

from whisper_tuner.main import main

if __name__ == "__main__":
    main()
