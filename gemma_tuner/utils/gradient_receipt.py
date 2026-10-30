"""Redacted gradient evidence for bounded multimodal learning proofs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import TrainerCallback

SUBSYSTEMS = ("vision", "projector", "decoder", "output_head", "audio", "other")
REDACTED_METRIC_FIELDS = (
    "loss",
    "eval_loss",
    "grad_norm",
    "learning_rate",
    "train_runtime",
    "train_samples_per_second",
    "train_steps_per_second",
)


def parameter_subsystem(name: str) -> str:
    if ".lm_head." in f".{name}." or name.startswith("lm_head."):
        return "output_head"
    if "vision_tower" in name:
        return "vision"
    if "embed_vision" in name:
        return "projector"
    if "language_model" in name:
        return "decoder"
    if "audio_tower" in name:
        return "audio"
    return "other"


class GradientSubsystemReceiptCallback(TrainerCallback):
    """Measure post-clipping gradients and fail if required subsystems are silent."""

    def __init__(self, output_dir: str | Path, required: Iterable[str]) -> None:
        self.output_dir = Path(output_dir)
        self.required = tuple(dict.fromkeys(str(value).strip() for value in required if str(value).strip()))
        unknown = sorted(set(self.required) - set(SUBSYSTEMS))
        if unknown:
            raise ValueError(f"unknown required gradient subsystems: {unknown}")
        self.observations = 0
        self.maximum_norm = {name: 0.0 for name in SUBSYSTEMS}
        self.maximum_tensors = {name: 0 for name in SUBSYSTEMS}

    @property
    def path(self) -> Path:
        return self.output_dir / "gradient_subsystems.json"

    def _write(self) -> None:
        receipt = {
            "schema_version": "gemma_gradient_subsystems_v1",
            "measurement": "post_gradient_clipping_pre_optimizer_step_l2_norm",
            "observations": self.observations,
            "required": list(self.required),
            "by_subsystem": {
                name: {
                    "maximum_l2_norm": self.maximum_norm[name],
                    "maximum_gradient_tensors": self.maximum_tensors[name],
                }
                for name in SUBSYSTEMS
            },
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            raise RuntimeError("gradient receipt callback did not receive the model")
        squared = {name: 0.0 for name in SUBSYSTEMS}
        tensors = {name: 0 for name in SUBSYSTEMS}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            subsystem = parameter_subsystem(name)
            norm = float(torch.linalg.vector_norm(parameter.grad.detach().float()).cpu())
            if not math.isfinite(norm):
                raise RuntimeError(f"non-finite gradient in {subsystem}")
            squared[subsystem] += norm * norm
            tensors[subsystem] += 1
        self.observations += 1
        for name in SUBSYSTEMS:
            aggregate = math.sqrt(squared[name])
            self.maximum_norm[name] = max(self.maximum_norm[name], aggregate)
            self.maximum_tensors[name] = max(self.maximum_tensors[name], tensors[name])
        self._write()

    def on_train_end(self, args, state, control, **kwargs):
        self._write()
        if self.observations < 1:
            raise RuntimeError("no optimizer-step gradient observation was recorded")
        missing = [name for name in self.required if self.maximum_norm[name] <= 0.0]
        if missing:
            raise RuntimeError(f"required LoRA subsystems have zero gradients: {missing}")


class RedactedTrainingMetricsCallback(TrainerCallback):
    """Append finite scalar training metrics without prompts, paths, or row data."""

    def __init__(self, output_dir: str | Path) -> None:
        self.path = Path(output_dir) / "redacted_metrics.jsonl"
        self.rows: dict[tuple[int, float | None, tuple[str, ...]], dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key = self._key(row)
                if key in self.rows:
                    raise ValueError(f"duplicate redacted metric key: {key}")
                self.rows[key] = row

    @staticmethod
    def _key(row: dict[str, Any]) -> tuple[int, float | None, tuple[str, ...]]:
        fields = tuple(field for field in REDACTED_METRIC_FIELDS if field in row)
        return int(row["step"]), row.get("epoch"), fields

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        row: dict[str, Any] = {
            "schema_version": "gemma_redacted_training_metric_v1",
            "step": int(state.global_step),
        }
        if state.epoch is not None:
            epoch = float(state.epoch)
            if not math.isfinite(epoch):
                raise RuntimeError("non-finite epoch in training metrics")
            row["epoch"] = epoch
        for field in REDACTED_METRIC_FIELDS:
            if field not in logs:
                continue
            value = float(logs[field])
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite {field} in training metrics")
            row[field] = value
        if not any(field in row for field in REDACTED_METRIC_FIELDS):
            return
        key = self._key(row)
        previous = self.rows.get(key)
        if previous is not None:
            if previous != row:
                raise RuntimeError(f"redacted metric changed on resume at {key}")
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
            handle.flush()
        self.rows[key] = row
