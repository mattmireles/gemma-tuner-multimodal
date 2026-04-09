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


@pytest.mark.skipif(
    _transformers_version() < Version("5.5.0"),
    reason="Gemma 4 modeling module requires transformers>=5.5.0",
)
def test_patched_clippable_linear_has_linear_property_returning_self():
    """Upstream ``modeling_gemma4`` reads ``self.ffw_layer_1.linear.weight.dtype``
    (and similar patterns in ``lconv1d`` / ``self_attn.post``) on the assumption
    that ``Gemma4ClippableLinear`` still wraps an inner ``nn.Linear``. Without
    the compatibility shim the forward pass ``AttributeError``s. This test
    pins the shim: ``linear`` must resolve to the module itself so that
    ``module.linear.weight is module.weight``."""
    from transformers.models.gemma4 import modeling_gemma4 as m

    from gemma_tuner.models.gemma.gemma4_patches import apply_clippable_linear_patch

    apply_clippable_linear_patch()

    class _Cfg:
        use_clipped_linears = False

    mod = m.Gemma4ClippableLinear(_Cfg(), in_features=4, out_features=3)
    # The shim makes ``mod.linear`` the module itself, so ``mod.linear.weight``
    # is the *same* parameter tensor as ``mod.weight``.
    assert mod.linear is mod
    assert mod.linear.weight is mod.weight
    assert mod.linear.weight.dtype == mod.weight.dtype

    # The shim is a ``@property``, not a real submodule, so iterating
    # ``named_modules`` must NOT yield a self-referential ``mod.linear`` entry
    # (which would break state-dict round-tripping and any recursive module
    # walk).
    submodule_names = {name for name, _ in mod.named_modules()}
    assert "linear" not in submodule_names
