"""Gemma 4 post-load PEFT compatibility conversion."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from gemma_tuner.models.gemma.base_model_loader import _convert_gemma4_linears_after_load
from gemma_tuner.models.gemma.family import GemmaFamily
from gemma_tuner.models.gemma.gemma4_patches import (
    PeftCompatibleGemma4ClippableLinear,
    apply_clippable_linear_patch,
    convert_loaded_clippable_linears,
)


class Gemma4ClippableLinear(nn.Module):
    def __init__(self, *, clipped: bool = False) -> None:
        super().__init__()
        self.use_clipped_linears = clipped
        self.linear = nn.Linear(4, 3, bias=False)
        if clipped:
            self.register_buffer("input_min", torch.tensor(-1.0))
            self.register_buffer("input_max", torch.tensor(1.0))
            self.register_buffer("output_min", torch.tensor(-2.0))
            self.register_buffer("output_max", torch.tensor(2.0))

    def forward(self, value):
        if self.use_clipped_linears:
            value = torch.clamp(value, self.input_min, self.input_max)
        value = self.linear(value)
        if self.use_clipped_linears:
            value = torch.clamp(value, self.output_min, self.output_max)
        return value


def test_import_gemma_tuner_does_not_eager_load_transformers_gemma4():
    repo_root = Path(__file__).resolve().parents[1]
    code = (
        "import sys\n"
        "import gemma_tuner\n"
        "import gemma_tuner.models.gemma.finetune\n"
        "assert 'transformers.models.gemma4' not in sys.modules\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=repo_root, capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.parametrize("clipped", [False, True])
def test_post_load_conversion_preserves_parameter_identity_and_forward(clipped: bool) -> None:
    source = Gemma4ClippableLinear(clipped=clipped)
    with torch.no_grad():
        source.linear.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10)
    original_parameter = source.linear.weight
    value = torch.tensor([[2.0, -2.0, 0.5, -0.5]])
    expected = source(value)
    model = nn.Sequential(source)

    assert convert_loaded_clippable_linears(model) == 1
    replacement = model[0]
    assert isinstance(replacement, PeftCompatibleGemma4ClippableLinear)
    assert isinstance(replacement, nn.Linear)
    assert replacement.weight is original_parameter
    assert replacement.linear is replacement
    assert torch.equal(replacement(value), expected)
    assert "0.linear.weight" not in model.state_dict()
    assert torch.equal(model.state_dict()["0.weight"], original_parameter)


def test_post_load_conversion_recurses_and_is_idempotent() -> None:
    model = nn.Sequential(nn.Sequential(Gemma4ClippableLinear()), nn.Linear(3, 2))
    assert convert_loaded_clippable_linears(model) == 1
    assert convert_loaded_clippable_linears(model) == 0


def test_preload_patch_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="pre-load.*unsafe"):
        apply_clippable_linear_patch()


def test_multimodal_gemma4_load_requires_native_clippable_linears() -> None:
    with pytest.raises(RuntimeError, match="no native clippable linears"):
        _convert_gemma4_linears_after_load(
            nn.Sequential(nn.Linear(4, 3)),
            GemmaFamily.GEMMA_4,
            required=True,
        )


def test_text_only_gemma4_family_fixture_allows_no_clippable_linears() -> None:
    model = nn.Sequential(nn.Linear(4, 3))
    assert (
        _convert_gemma4_linears_after_load(
            model,
            GemmaFamily.GEMMA_4,
            required=False,
        )
        is model
    )
