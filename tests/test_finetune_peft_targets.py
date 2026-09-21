"""Tests for Gemma finetune LoRA / PEFT target validation."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from gemma_tuner.models.gemma.finetune import (
    _raise_if_lora_targets_use_peft_incompatible_linears,
    _trainable_parameter_receipt,
    _validate_conditioned_prompt_file,
    _validate_lora_target_regex,
    completion_only_causal_loss,
)


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


def test_anchored_regex_includes_vision_projector_decoder_and_excludes_audio() -> None:
    class ToyMultimodal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.language_model = nn.Module()
            self.model.language_model.q_proj = nn.Linear(4, 4, bias=False)
            self.model.vision_tower = nn.Module()
            self.model.vision_tower.q_proj = nn.Linear(4, 4, bias=False)
            self.model.embed_vision = nn.Module()
            self.model.embed_vision.embedding_projection = nn.Linear(4, 4, bias=False)
            self.model.audio_tower = nn.Module()
            self.model.audio_tower.q_proj = nn.Linear(4, 4, bias=False)

    model = ToyMultimodal()
    regex = (
        r"^(?:model\.language_model\.q_proj|model\.vision_tower\.q_proj|"
        r"model\.embed_vision\.embedding_projection)$"
    )
    assert _validate_lora_target_regex(model, regex) == [
        "model.language_model.q_proj",
        "model.vision_tower.q_proj",
        "model.embed_vision.embedding_projection",
    ]


def test_frozen_regex_targets_patched_vision_linears_not_inner_linear_property() -> None:
    model = nn.Module()
    model.model = nn.Module()
    model.model.vision_tower = nn.Module()
    model.model.vision_tower.encoder = nn.Module()
    model.model.vision_tower.encoder.layers = nn.ModuleList([nn.Module()])
    layer = model.model.vision_tower.encoder.layers[0]
    layer.self_attn = nn.Module()
    layer.self_attn.q_proj = nn.Linear(4, 4, bias=False)
    regex = (
        r"^model\.vision_tower\.encoder\.layers\.\d+\."
        r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
        r"mlp\.(?:gate_proj|up_proj|down_proj))$"
    )
    assert _validate_lora_target_regex(model, regex) == [
        "model.vision_tower.encoder.layers.0.self_attn.q_proj"
    ]


def test_trainable_parameter_receipt_groups_subsystems() -> None:
    model = nn.Module()
    model.vision_tower = nn.Linear(2, 3, bias=False)
    model.embed_vision = nn.Linear(3, 4, bias=False)
    model.language_model = nn.Linear(4, 5, bias=False)
    model.language_model.lm_head = nn.Linear(5, 2, bias=False)
    model.audio_tower = nn.Linear(5, 6, bias=False)
    receipt = _trainable_parameter_receipt(model)
    assert receipt["by_subsystem"] == {
        "vision": 6,
        "projector": 12,
        "decoder": 20,
        "output_head": 10,
        "audio": 30,
        "other": 0,
    }
    assert receipt["total"] == 78


def test_conditioned_prompt_hash_is_fail_closed(tmp_path) -> None:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("For {USER_FULL_NAME} in {APPLICATION_NAME}.")
    import hashlib

    config = {
        "system_prompt_column": "system_prompt",
        "conditioned_system_prompt_template": str(prompt),
        "conditioned_system_prompt_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest(),
    }
    _validate_conditioned_prompt_file(config)
    config["conditioned_system_prompt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        _validate_conditioned_prompt_file(config)


def test_content_bound_per_row_system_prompts_need_no_template_file() -> None:
    _validate_conditioned_prompt_file({"system_prompt_column": "system_prompt"})


def test_completion_only_loss_matches_full_masked_causal_loss() -> None:
    torch.manual_seed(7)
    full_logits = torch.randn(1, 8, 13)
    labels = torch.tensor([[-100, -100, -100, -100, 4, 5, 6, 7]])
    attention = torch.ones(1, 8, dtype=torch.long)
    full = torch.nn.functional.cross_entropy(
        full_logits[:, :-1, :].reshape(-1, 13),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    # Five kept logits: one preceding position plus four causal positions needed
    # to predict labels 4..7 after shifting.
    suffix = completion_only_causal_loss(full_logits[:, -5:, :], labels, attention)
    assert torch.equal(full, suffix)
