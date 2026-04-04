"""Core orchestration package for gemma-macos-tuner.

Provides the backbone utilities used by the CLI and scripts:
- config: hierarchical profile/model/dataset configuration resolution
- runs: run directory creation, metadata, metrics, and discovery
- logging: unified logging with human and JSON formatters
- ops: operation dispatch that defers heavy imports until needed

Called by:
- main.py (primary orchestrator)
- cli_typer.py (optional Typer CLI)
"""

__all__ = [
    "config",
    "runs",
    "logging",
    "ops",
]
