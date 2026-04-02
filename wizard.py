"""Backward-compatible entrypoint. See wizard/ package for implementation.

This file is kept for compatibility with documented invocations:
  python wizard.py

The actual implementation lives in the wizard/ package.
"""
from wizard import wizard_main

if __name__ == "__main__":
    wizard_main()
