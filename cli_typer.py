#!/usr/bin/env python3
"""Compatibility shim for legacy ``cli_typer.py`` entrypoint."""

from whisper_tuner.cli_typer import app

if __name__ == "__main__":
    app()
