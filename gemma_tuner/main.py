#!/usr/bin/env python3
"""
Legacy CLI entrypoint — delegates to the canonical Typer CLI.

This module is DEPRECATED and will be removed in a future release.
Use ``gemma-macos-tuner <command>`` or ``python -m gemma_tuner.cli_typer`` instead.

Kept only for backward compatibility with:
- wizard/runner.py subprocess invocation
- CI smoke tests
- Users who have ``python main.py`` in their shell history
"""

import sys


def main():
    """Thin shim that prints a deprecation notice and delegates to cli_typer."""
    print(
        "[DEPRECATION] main.py is legacy and will be removed in a future release.\n"
        "Use the modern Typer CLI instead:\n"
        "  - gemma-macos-tuner <command>\n"
        "  - or python -m gemma_tuner.cli_typer <command>\n"
        "See docs/TROUBLESHOOTING.md if you need help migrating.",
        file=sys.stderr,
    )

    from gemma_tuner.cli_typer import app

    app()


if __name__ == "__main__":
    main()
