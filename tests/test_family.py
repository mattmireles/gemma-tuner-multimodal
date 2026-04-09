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
    assert detect_family("fxmarty/tiny-random-Gemma4ForCausalLM") == GemmaFamily.GEMMA_4


def test_detect_family_unknown():
    with pytest.raises(RuntimeError, match="Unsupported Gemma model_id"):
        detect_family("google/gemma-2-2b")


def test_family_capabilities_keys():
    c3 = family_capabilities(GemmaFamily.GEMMA_3N)
    assert c3["control_token"] == "<start_of_turn>"
    assert c3["supports_assistant_mask"] is True
    assert c3["needs_clippable_patch"] is False
    assert c3["needs_mm_token_type_ids_injection"] is True
    c4 = family_capabilities(GemmaFamily.GEMMA_4)
    # Gemma 4's real tokenizer uses <|turn> (id 105) as the opener; do NOT use
    # <|turn|> with bars on both sides — that string tokenizes to a 4-subword
    # sequence that never matches anything in the rendered chat. See family.py
    # family_capabilities docstring.
    assert c4["control_token"] == "<|turn>"
    # Gemma 4's shipped chat template lacks the Jinja `{% generation %}` block,
    # so HF's assistant_masks primary path returns all-zeros. Collators must skip it.
    assert c4["supports_assistant_mask"] is False
    assert c4["needs_clippable_patch"] is True


def test_assert_family_supported_gemma4_too_old():
    with patch("gemma_tuner.models.gemma.family._installed_transformers_version", return_value="4.57.6"):
        with pytest.raises(RuntimeError, match="Gemma 4 requires transformers"):
            assert_family_supported(GemmaFamily.GEMMA_4)


def test_assert_family_supported_gemma4_ok():
    with patch("gemma_tuner.models.gemma.family._installed_transformers_version", return_value="5.5.0"):
        assert_family_supported(GemmaFamily.GEMMA_4)


def test_assert_entrypoint_support_allows_export_gemma4():
    assert_entrypoint_support("export", GemmaFamily.GEMMA_4) is None


def test_assert_entrypoint_support_allows_finetune():
    assert_entrypoint_support("finetune", GemmaFamily.GEMMA_4) is None


def test_assert_entrypoint_support_allows_eval_gemma_asr():
    assert_entrypoint_support("eval_gemma_asr", GemmaFamily.GEMMA_4) is None


def test_gate_gemma_model_export_gemma4():
    with patch("gemma_tuner.models.gemma.family._installed_transformers_version", return_value="5.5.0"):
        assert gate_gemma_model("google/gemma-4-E2B-it", entrypoint="export") == GemmaFamily.GEMMA_4


def test_gate_gemma_model_eval_gemma_asr_gemma4():
    with patch("gemma_tuner.models.gemma.family._installed_transformers_version", return_value="5.5.0"):
        assert gate_gemma_model("google/gemma-4-E2B-it", entrypoint="eval_gemma_asr") == GemmaFamily.GEMMA_4


def test_gate_gemma_model_finetune_gemma4():
    with patch("gemma_tuner.models.gemma.family._installed_transformers_version", return_value="5.5.0"):
        assert gate_gemma_model("google/gemma-4-E2B-it", entrypoint="finetune") == GemmaFamily.GEMMA_4


def test_unsupported_entrypoints_cover_plan():
    assert "gemma_generate" in GEMMA4_UNSUPPORTED_ENTRYPOINTS
    assert "export" not in GEMMA4_UNSUPPORTED_ENTRYPOINTS
    assert "eval_gemma_asr" not in GEMMA4_UNSUPPORTED_ENTRYPOINTS
    assert "finetune" not in GEMMA4_UNSUPPORTED_ENTRYPOINTS
