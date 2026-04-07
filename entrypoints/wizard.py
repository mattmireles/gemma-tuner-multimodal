"""Backward-compatible entrypoint. See wizard/ package for implementation.

This file is kept for compatibility with documented invocations:
  python entrypoints/wizard.py

The actual implementation lives in the wizard/ package.
"""

from gemma_tuner.wizard import wizard_main

if __name__ == "__main__":
    wizard_main()
