"""Tests for Gemma 4 monkey-patches (requires ``transformers>=5.5`` for modeling import)."""

from __future__ import annotations

import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest
import torch.nn as nn
from packaging.version import Version


def _transformers_version() -> Version:
    return Version(metadata.version("transformers"))


def test_import_gemma_tuner_does_not_eager_load_transformers_gemma4():
    """Base install must not pull ``transformers.models.gemma4`` until finetune applies the patch."""
    repo_root = Path(__file__).resolve().parents[1]
    code = (
        "import sys\n"
        "import gemma_tuner\n"
        "import gemma_tuner.models.gemma.finetune\n"
        "assert 'transformers.models.gemma4' not in sys.modules\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.skipif(
    _transformers_version() < Version("5.5.0"),
    reason="Gemma 4 modeling module requires transformers>=5.5.0",
)
def test_apply_clippable_linear_patch_makes_gemma4_linear_subclass_of_nn_linear():
    from transformers.models.gemma4 import modeling_gemma4 as m

    from gemma_tuner.models.gemma.gemma4_patches import apply_clippable_linear_patch

    assert not issubclass(m.Gemma4ClippableLinear, nn.Linear)
    apply_clippable_linear_patch()
    assert issubclass(m.Gemma4ClippableLinear, nn.Linear)
    apply_clippable_linear_patch()
    assert issubclass(m.Gemma4ClippableLinear, nn.Linear)
