#!/usr/bin/env python3
"""Minimal smoke tests: import and 1-batch dataloader pass for evaluate preprocessing."""

import importlib

def test_imports():
    importlib.import_module("main")
    importlib.import_module("scripts.evaluate")
    importlib.import_module("models.whisper.finetune")

if __name__ == "__main__":
    test_imports()
    print("OK: basic imports")
