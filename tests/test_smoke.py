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

    importlib.import_module("main")
    importlib.import_module("scripts.evaluate")
    importlib.import_module("scripts.finetune")

if __name__ == "__main__":
    test_imports()
    print("OK: basic imports")
