"""Tests for redacted multimodal gradient evidence."""

from __future__ import annotations

import json

import pytest
import torch.nn as nn

from gemma_tuner.utils.gradient_receipt import (
    GradientSubsystemReceiptCallback,
    RedactedTrainingMetricsCallback,
)


class ToyMultimodal(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_tower = nn.Linear(2, 2, bias=False)
        self.embed_vision = nn.Module()
        self.embed_vision.embedding_projection = nn.Linear(2, 2, bias=False)
        self.language_model = nn.Linear(2, 2, bias=False)


def test_gradient_receipt_records_all_required_subsystems(tmp_path) -> None:
    model = ToyMultimodal()
    sum(parameter.sum() for parameter in model.parameters()).backward()
    callback = GradientSubsystemReceiptCallback(tmp_path, ["vision", "projector", "decoder"])
    callback.on_pre_optimizer_step(None, None, None, model=model)
    callback.on_train_end(None, None, None)
    receipt = json.loads((tmp_path / "gradient_subsystems.json").read_text())
    assert receipt["observations"] == 1
    for subsystem in ("vision", "projector", "decoder"):
        assert receipt["by_subsystem"][subsystem]["maximum_l2_norm"] > 0
        assert receipt["by_subsystem"][subsystem]["maximum_gradient_tensors"] == 1


def test_gradient_receipt_rejects_silent_required_subsystem(tmp_path) -> None:
    model = ToyMultimodal()
    model.language_model.weight.sum().backward()
    callback = GradientSubsystemReceiptCallback(tmp_path, ["vision", "decoder"])
    callback.on_pre_optimizer_step(None, None, None, model=model)
    with pytest.raises(RuntimeError, match="zero gradients.*vision"):
        callback.on_train_end(None, None, None)


def test_gradient_receipt_rejects_unknown_subsystem(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown required"):
        GradientSubsystemReceiptCallback(tmp_path, ["bogus"])


class State:
    global_step = 3
    epoch = 0.5


def test_redacted_metrics_are_append_only_finite_and_resume_safe(tmp_path) -> None:
    callback = RedactedTrainingMetricsCallback(tmp_path)
    callback.on_log(None, State(), None, logs={"loss": 1.25, "grad_norm": 2.5, "ignored": "private"})
    callback.on_log(None, State(), None, logs={"loss": 1.25, "grad_norm": 2.5, "ignored": "private"})
    rows = (tmp_path / "redacted_metrics.jsonl").read_text().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0]) == {
        "epoch": 0.5,
        "grad_norm": 2.5,
        "loss": 1.25,
        "schema_version": "gemma_redacted_training_metric_v1",
        "step": 3,
    }
    resumed = RedactedTrainingMetricsCallback(tmp_path)
    resumed.on_log(None, State(), None, logs={"loss": 1.25, "grad_norm": 2.5})
    resumed.on_log(None, State(), None, logs={"train_runtime": 4.0})
    assert len((tmp_path / "redacted_metrics.jsonl").read_text().splitlines()) == 2
    with pytest.raises(RuntimeError, match="changed on resume"):
        resumed.on_log(None, State(), None, logs={"loss": 9.0, "grad_norm": 2.5})


def test_redacted_metrics_reject_non_finite_values(tmp_path) -> None:
    callback = RedactedTrainingMetricsCallback(tmp_path)
    with pytest.raises(RuntimeError, match="non-finite loss"):
        callback.on_log(None, State(), None, logs={"loss": float("nan")})
