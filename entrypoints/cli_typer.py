#!/usr/bin/env python3
"""Compatibility shim for legacy ``cli_typer.py`` at repo root (now ``entrypoints/cli_typer.py``)."""

from gemma_tuner.cli_typer import app

if __name__ == "__main__":
    app()
