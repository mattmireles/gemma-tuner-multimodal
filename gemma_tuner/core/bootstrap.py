#!/usr/bin/env python3
"""
Early Apple Silicon (MPS) environment bootstrap.

This module MUST be imported before any library that may import PyTorch.
It standardizes MPS memory watermark configuration so Metal's unified
memory pressure is controlled consistently across all entrypoints.

Called by:
- main.py (legacy argparse CLI) as the very first import
- cli_typer.py (modern Typer CLI) as the very first import

Effects:
- Sets PYTORCH_MPS_HIGH_WATERMARK_RATIO (default 0.80)
- Sets PYTORCH_MPS_LOW_WATERMARK_RATIO  (default 0.70) and ensures low < high

Notes:
- Only applies on macOS arm64 (Apple Silicon). Safe no-op elsewhere.
- This must happen BEFORE importing torch to take effect.
"""

from __future__ import annotations

import os
import platform


def _clamp_ratio(var_name: str, default_value: float) -> float:
    """Clamp env var to (0.0, 1.0); set default if missing/invalid."""
    current = os.environ.get(var_name)
    if current is None:
        os.environ[var_name] = str(default_value)
        return default_value
    try:
        value = float(current)
        if not (0.0 < value < 1.0):
            os.environ[var_name] = str(default_value)
            return default_value
        return value
    except Exception:
        os.environ[var_name] = str(default_value)
        return default_value


def _bootstrap_mps_env() -> None:
    if platform.system() != "Darwin" or platform.machine().lower() != "arm64":
        return  # Not Apple Silicon; nothing to do

    # Use the canonical constant from constants (0.80) as the safe default.
    # 0.90 is aggressive and risks disk swapping on memory-constrained machines.
    try:
        from gemma_tuner.constants import MemoryLimits

        _default_high = MemoryLimits.MPS_DEFAULT_FRACTION
    except ImportError:
        _default_high = 0.80
    high = _clamp_ratio("PYTORCH_MPS_HIGH_WATERMARK_RATIO", _default_high)
    low = _clamp_ratio("PYTORCH_MPS_LOW_WATERMARK_RATIO", 0.70)

    # Ensure ordering: low < high (provide a safe margin if misconfigured)
    if not (low < high):
        safe_low = max(min(high - 0.10, 0.85), 0.10)
        os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = f"{safe_low:.2f}"


# Execute immediately on import
_bootstrap_mps_env()
