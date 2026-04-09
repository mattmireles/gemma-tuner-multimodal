#!/usr/bin/env python3
"""
Compatibility CLI entrypoint — delegates to the canonical Typer CLI
(``gemma-macos-tuner`` / ``python -m gemma_tuner.cli_typer``).

Kept for:
- wizard/runner.py subprocess invocation
- CI smoke tests
- Users with ``python -m gemma_tuner.main`` in scripts or history
"""


def main():
    """Delegate to ``cli_typer``."""
    from gemma_tuner.cli_typer import app

    app()


if __name__ == "__main__":
    main()
