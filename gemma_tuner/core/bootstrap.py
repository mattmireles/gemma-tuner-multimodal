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
- Loads `.env` from the current directory tree (nearest ancestor) or repo root when present;
  does not override existing environment variables.
- Only applies on macOS arm64 (Apple Silicon). Safe no-op elsewhere.
- This must happen BEFORE importing torch to take effect.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path


def _load_dotenv_from_file(env_path: Path) -> None:
    """
    Load KEY=value pairs from a .env file into os.environ.
    Does not override variables already present (same rule as python-dotenv).
    """
    try:
        raw = env_path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, rest = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = rest.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ[key] = value


def _find_dotenv_path() -> Path | None:
    """Resolve .env: walk up from cwd, else repo root next to gemma_tuner/ (editable install)."""
    try:
        cwd = Path.cwd()
        for p in [cwd, *list(cwd.parents)[:12]]:
            cand = p / ".env"
            if cand.is_file():
                return cand
    except OSError:
        pass
    try:
        root = Path(__file__).resolve().parents[2]
        if (root / "pyproject.toml").is_file():
            cand = root / ".env"
            if cand.is_file():
                return cand
    except (OSError, IndexError):
        pass
    return None


def _load_repo_dotenv() -> None:
    path = _find_dotenv_path()
    if path is not None:
        _load_dotenv_from_file(path)


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


# Execute immediately on import (dotenv first so HF_TOKEN etc. are visible everywhere)
_load_repo_dotenv()
_bootstrap_mps_env()
