"""Tests for Gemma 4 monkey-patches (requires ``transformers>=5.5`` for modeling import)."""

from __future__ import annotations

from importlib import metadata

import pytest
import torch.nn as nn
from packaging.version import Version

from gemma_tuner.models.gemma.gemma4_patches import apply_clippable_linear_patch


def _transformers_version() -> Version:
    return Version(metadata.version("transformers"))


@pytest.mark.skipif(
    _transformers_version() < Version("5.5.0"),
    reason="Gemma 4 modeling module requires transformers>=5.5.0",
)
def test_apply_clippable_linear_patch_makes_gemma4_linear_subclass_of_nn_linear():
    from transformers.models.gemma4 import modeling_gemma4 as m

    assert not issubclass(m.Gemma4ClippableLinear, nn.Linear)
    apply_clippable_linear_patch()
    assert issubclass(m.Gemma4ClippableLinear, nn.Linear)
    apply_clippable_linear_patch()
    assert issubclass(m.Gemma4ClippableLinear, nn.Linear)
