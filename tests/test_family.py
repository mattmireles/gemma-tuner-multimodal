"""Tests for gemma_tuner.models.gemma.family."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from gemma_tuner.models.gemma.family import (
    GEMMA4_UNSUPPORTED_ENTRYPOINTS,
    GemmaFamily,
    assert_entrypoint_support,
    assert_family_supported,
    detect_family,
    family_capabilities,
    gate_gemma_model,
)


def test_detect_family_variants():
    assert detect_family("google/gemma-3n-E2B-it") == GemmaFamily.GEMMA_3N
    assert detect_family("GOOGLE/GEMMA-3N-E4B-IT") == GemmaFamily.GEMMA_3N
    assert detect_family("google/gemma-4-E4B") == GemmaFamily.GEMMA_4
    assert detect_family("/models/gemma-4-E2B-it") == GemmaFamily.GEMMA_4
    assert detect_family("fxmarty/tiny-random-GemmaForCausalLM") == GemmaFamily.GEMMA_3N


def test_detect_family_unknown():
    with pytest.raises(ValueError, match="Unsupported Gemma model_id"):
        detect_family("google/gemma-2-2b")


def test_family_capabilities_keys():
    c3 = family_capabilities(GemmaFamily.GEMMA_3N)
    assert c3["control_token"] == "<start_of_turn>"
    assert c3["needs_clippable_patch"] is False
    assert c3["needs_mm_token_type_ids_injection"] is False
    c4 = family_capabilities(GemmaFamily.GEMMA_4)
    assert c4["control_token"] == "<|turn|>"
    assert c4["needs_clippable_patch"] is True


def test_assert_family_supported_gemma4_too_old():
    with patch("gemma_tuner.models.gemma.family._installed_transformers_version", return_value="4.57.6"):
        with pytest.raises(RuntimeError, match="Gemma 4 requires transformers"):
            assert_family_supported(GemmaFamily.GEMMA_4)


def test_assert_family_supported_gemma4_ok():
    with patch("gemma_tuner.models.gemma.family._installed_transformers_version", return_value="5.5.0"):
        assert_family_supported(GemmaFamily.GEMMA_4)


def test_assert_entrypoint_support_blocks_gemma4_on_export():
    with pytest.raises(RuntimeError, match="not implemented"):
        assert_entrypoint_support("export", GemmaFamily.GEMMA_4)


def test_assert_entrypoint_support_allows_finetune():
    assert_entrypoint_support("finetune", GemmaFamily.GEMMA_4) is None


def test_gate_gemma_model_export_gemma4():
    with patch("gemma_tuner.models.gemma.family._installed_transformers_version", return_value="5.5.0"):
        with pytest.raises(RuntimeError, match="not implemented"):
            gate_gemma_model("google/gemma-4-E2B-it", entrypoint="export")


def test_gate_gemma_model_finetune_gemma4():
    with patch("gemma_tuner.models.gemma.family._installed_transformers_version", return_value="5.5.0"):
        assert gate_gemma_model("google/gemma-4-E2B-it", entrypoint="finetune") == GemmaFamily.GEMMA_4


def test_unsupported_entrypoints_cover_plan():
    assert "gemma_generate" in GEMMA4_UNSUPPORTED_ENTRYPOINTS
    assert "export" in GEMMA4_UNSUPPORTED_ENTRYPOINTS
    assert "finetune" not in GEMMA4_UNSUPPORTED_ENTRYPOINTS
