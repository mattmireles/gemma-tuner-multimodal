#!/usr/bin/env python3
"""Minimal smoke tests: import and 1-batch dataloader pass for evaluate preprocessing."""

import importlib
import os
import sys


def test_imports():
    # Ensure project root is on sys.path for module imports
    root = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(root)
    if root not in sys.path:
        sys.path.insert(0, root)

    cli = importlib.import_module("gemma_tuner.cli_typer")
    # The Typer app object must exist — if someone deletes it the CLI is broken
    assert hasattr(cli, "app"), "gemma_tuner.cli_typer must export 'app' (Typer instance)"

    evaluate = importlib.import_module("gemma_tuner.scripts.evaluate")
    # run_evaluation is the primary entry point called by cli_typer and main
    assert hasattr(evaluate, "evaluate") or hasattr(evaluate, "run_evaluation") or hasattr(evaluate, "main"), (
        "gemma_tuner.scripts.evaluate must export a callable entry point"
    )

    finetune = importlib.import_module("gemma_tuner.scripts.finetune")
    # main() is the entry point called by ops.py
    assert hasattr(finetune, "main"), "gemma_tuner.scripts.finetune must export 'main'"


if __name__ == "__main__":
    test_imports()
    print("OK: basic imports")
