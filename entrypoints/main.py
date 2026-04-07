#!/usr/bin/env python3
"""Compatibility shim for legacy ``main.py`` usage.

Canonical entrypoint is now `gemma-macos-tuner` (`gemma_tuner.cli_typer`).
This file keeps `python entrypoints/main.py ...` functional by delegating to
`gemma_tuner.main`.
"""

from gemma_tuner.main import main

if __name__ == "__main__":
    main()
