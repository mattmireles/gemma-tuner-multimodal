"""Training results persistence utilities.

Shared across all model finetune paths.

Called by:
- models/gemma/finetune.py after trainer.train()
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any

logger = logging.getLogger(__name__)


def _to_safe(obj: Any) -> Any:
    """Recursively convert values to JSON-safe types."""
    if isinstance(obj, dict):
        return {k: _to_safe(v) for k, v in obj.items()}
    try:
        return float(obj)
    except Exception:
        return obj


def persist_training_results(
    output_dir: str,
    trainer=None,
    train_result=None,
    modality: str = "audio",
) -> None:
    """Write train_results.json for orchestrator compatibility.

    Tries sources in order of richness:
    1. train_result.metrics (from HF Trainer.train() return value)
    2. Last log_history entry with 'loss' or 'train_runtime' (from trainer state)
    3. Empty dict (safe fallback — file is always written)

    Args:
        output_dir: Directory to write train_results.json into.
        trainer: Optional HF Trainer instance (used for log_history fallback).
        train_result: Optional return value from trainer.train().
    """
    try:
        metrics: dict = {}

        if train_result is not None and hasattr(train_result, "metrics"):
            metrics = train_result.metrics or {}
        elif trainer is not None and hasattr(trainer, "state"):
            log_history = getattr(trainer.state, "log_history", None)
            if isinstance(log_history, list):
                for entry in reversed(log_history):
                    if isinstance(entry, dict) and ("loss" in entry or "train_runtime" in entry):
                        metrics = entry
                        break

        metrics = dict(metrics)
        if str(modality).lower() == "text":
            el = metrics.get("eval_loss")
            if el is not None:
                try:
                    metrics["perplexity"] = math.exp(min(float(el), 20.0))
                except (TypeError, ValueError):
                    pass

        results_path = os.path.join(output_dir, "train_results.json")
        with open(results_path, "w") as wf:
            json.dump(_to_safe(metrics), wf, indent=2)
    except Exception:
        logger.warning("Failed to write train_results.json to %s", output_dir, exc_info=True)


def load_training_results(output_dir: str) -> dict:
    """Load train_results.json from output_dir. Returns empty dict on any failure."""
    results_path = os.path.join(output_dir, "train_results.json")
    if not os.path.exists(results_path):
        return {}
    try:
        with open(results_path, "r") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}
