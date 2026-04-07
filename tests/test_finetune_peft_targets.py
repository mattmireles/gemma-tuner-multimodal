"""Tests for Gemma finetune LoRA / PEFT target validation."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from gemma_tuner.models.gemma.finetune import _raise_if_lora_targets_use_peft_incompatible_linears


class Gemma4ClippableLinear(nn.Module):
    """Stand-in for transformers' wrapper (not nn.Linear); PEFT rejects it."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 2))


class ToyWithClip(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Module()
        self.block.q_proj = Gemma4ClippableLinear()


class ToyLinearOnly(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Module()
        self.layers.q_proj = nn.Linear(4, 4, bias=False)


def test_raises_when_target_suffix_matches_clippable_wrapper() -> None:
    m = ToyWithClip()
    with pytest.raises(RuntimeError, match="Gemma4ClippableLinear|PEFT cannot"):
        _raise_if_lora_targets_use_peft_incompatible_linears(m, ["q_proj"])


def test_ok_when_plain_linear() -> None:
    m = ToyLinearOnly()
    _raise_if_lora_targets_use_peft_incompatible_linears(m, ["q_proj"])


def test_no_targets_no_op() -> None:
    m = ToyWithClip()
    _raise_if_lora_targets_use_peft_incompatible_linears(m, [])
