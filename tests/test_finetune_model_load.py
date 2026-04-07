"""Tests for Gemma finetune base model loader heuristics."""

from __future__ import annotations

from types import SimpleNamespace

from gemma_tuner.models.gemma.base_model_loader import config_is_multimodal_gemma_like


def test_config_multimodal_from_architecture_conditional():
    cfg = SimpleNamespace(architectures=["Gemma4ForConditionalGeneration"], model_type="gemma4")
    assert config_is_multimodal_gemma_like(cfg) is True


def test_config_gemma4_model_type_only():
    cfg = SimpleNamespace(architectures=[], model_type="gemma4")
    assert config_is_multimodal_gemma_like(cfg) is True


def test_config_text_only_causal():
    cfg = SimpleNamespace(architectures=["Gemma2ForCausalLM"], model_type="gemma2")
    assert config_is_multimodal_gemma_like(cfg) is False


def test_config_gemma3n_model_type():
    cfg = SimpleNamespace(architectures=None, model_type="gemma3n")
    assert config_is_multimodal_gemma_like(cfg) is True
