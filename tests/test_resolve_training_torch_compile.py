"""Tests for resolve_training_torch_compile (MPS guard for TrainingArguments.torch_compile)."""

import torch

from gemma_tuner.models.gemma.finetune import resolve_training_torch_compile


def test_mps_forces_false_even_when_requested():
    assert resolve_training_torch_compile(torch.device("mps"), {"torch_compile": True}) is False


def test_non_mps_honors_true():
    assert resolve_training_torch_compile(torch.device("cpu"), {"torch_compile": True}) is True


def test_non_mps_defaults_false():
    assert resolve_training_torch_compile(torch.device("cpu"), {}) is False
