"""CI guard for the unit-gemma4 job: fail if the Gemma 4 stack is missing (avoids silent skips)."""

from __future__ import annotations

import os
from importlib import metadata

import pytest
from packaging.version import Version

_MIN_TF = Version("5.5.0")


def _transformers_version() -> Version:
    return Version(metadata.version("transformers"))


@pytest.mark.skipif(
    os.environ.get("GEMMA_MACOS_EXPECT_GEMMA4_STACK") != "1",
    reason="Set only in .github/workflows/ci.yml unit-gemma4 job",
)
def test_ci_gemma4_stack_has_transformers_at_least_5_5():
    """If this fails, pip did not install requirements/requirements-gemma4.txt; integration smoke may skip."""
    got = _transformers_version()
    assert got >= _MIN_TF, (
        f"unit-gemma4 must install transformers>={_MIN_TF}; got {got}. "
        "Check requirements/requirements-gemma4.txt and the pip resolver (see plan Risks: always-skipping)."
    )
