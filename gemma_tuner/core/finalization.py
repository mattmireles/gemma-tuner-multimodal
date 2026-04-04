from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from gemma_tuner.core.runs import (
    mark_run_as_completed,
    update_experiments_csv,
    update_experiments_sqlite,
    update_run_metadata,
    write_metrics,
)


@dataclass(frozen=True)
class TrainingFinalizationResult:
    train_metrics: dict[str, Any]


@dataclass(frozen=True)
class EvaluationFinalizationResult:
    metrics: dict[str, Any]


def finalize_training_run(
    run_dir: str,
    output_dir: str,
    *,
    profile_config: Optional[dict[str, Any]] = None,
    training_result: Optional[dict[str, Any]] = None,
    duration_sec: Optional[float] = None,
) -> TrainingFinalizationResult:
    train_metrics = _resolve_train_metrics(run_dir, training_result=training_result, duration_sec=duration_sec)
    if train_metrics:
        write_metrics(run_dir, {"train": train_metrics})

    end_time = now_str()
    mark_run_as_completed(run_dir)
    update_run_metadata(run_dir, end_time=end_time)
    _update_experiment_indexes(output_dir, run_dir)
    return TrainingFinalizationResult(train_metrics=train_metrics)


def finalize_evaluation_run(run_dir: str, output_dir: str, metrics: Optional[dict[str, Any]]) -> EvaluationFinalizationResult:
    metrics = metrics or {}
    if metrics:
        update_run_metadata(run_dir, metrics=metrics)
        write_metrics(run_dir, metrics)
    mark_run_as_completed(run_dir)
    update_run_metadata(run_dir, end_time=now_str())
    _update_experiment_indexes(output_dir, run_dir)
    return EvaluationFinalizationResult(metrics=metrics)


def _resolve_train_metrics(
    run_dir: str,
    *,
    training_result: Optional[dict[str, Any]] = None,
    duration_sec: Optional[float] = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if training_result:
        metrics.update(training_result.get("train_metrics") or {})
    if not metrics:
        metrics.update(_load_train_results(run_dir))
    if duration_sec is not None:
        metrics["duration_sec"] = round(float(duration_sec), 3)
    return metrics


def _load_train_results(run_dir: str) -> dict[str, Any]:
    results_path = os.path.join(run_dir, "train_results.json")
    if not os.path.exists(results_path):
        return {}
    with open(results_path, "r") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _update_experiment_indexes(output_dir: str, run_dir: str) -> None:
    import logging as _logging
    try:
        update_experiments_csv(output_dir, run_dir)
        update_experiments_sqlite(output_dir, run_dir)
    except Exception:
        _logging.getLogger(__name__).warning(
            "Failed to update experiment indexes for run %s — CSV/SQLite index may be stale",
            run_dir,
            exc_info=True,
        )


def now_str() -> str:
    """Return current local timestamp in standard metadata format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
