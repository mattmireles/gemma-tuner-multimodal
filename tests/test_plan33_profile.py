"""Plan 33 stock-E2B profiles freeze the E4B recipe over three literal v3 epochs."""

import configparser
import re
from pathlib import Path

import pytest

from gemma_tuner.core.config import load_profile_config
from gemma_tuner.models.gemma import finetune

PATH = Path(__file__).parents[1] / "config/plan33-e2b-literal-three-epochs.ini"
PLAN32 = Path(__file__).parents[1] / "config/plan32-literal-two-epochs.ini"
SEGMENTS = [(epoch, 352 * (epoch - 1) + 88 * quarter) for epoch in (1, 2, 3) for quarter in range(4)]
SCHEDULES = {
    1: "86281e43044aa1d5f97b1a3f072c67fbef319c8ad6bbfec0d1384e43848f081b",
    2: "197f4972219e5b28721deec1fc303058ddeb0b94b037dd835eeb98431672175e",
    3: "63795b66cedda7b88fefc2a189458b8f68a8c9dd235b8a60ae9a4252d362b1d0",
}


def _config(path=PATH):
    config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    assert config.read(path)
    return config


def _profile(name, path=PATH):
    return load_profile_config(_config(path), name)


def test_twelve_segments_chain_from_stock_e2b_without_warm_start():
    names = [f"plan33-epoch{epoch}-step{start + 88}" for epoch, start in SEGMENTS]
    assert sorted(names) == sorted(s.split(":", 1)[1] for s in _config().sections()
                                   if s.startswith("profile:plan33-epoch"))
    for (epoch, start), name in zip(SEGMENTS, names):
        profile = _profile(name)
        assert profile["base_model"] == "google/gemma-4-E2B-it"
        assert profile["model_revision"] == "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
        assert profile.get("initial_adapter_path") is None
        assert (profile["telemetry_start_step"], profile["stop_after_step"]) == (start, start + 88)
        assert int(profile["plan32_epoch"]) == epoch and int(profile["literal_epochs"]) == 3
        assert profile["plan32_schedule_sha256"] == SCHEDULES[epoch]
        assert profile["source"] == f"tt-screenshot-plan33-literal-v3-deploy/epoch-{epoch}"
        assert profile["telemetry_full_validation_at_stop"] in ((start + 88) % 352 == 0, str((start + 88) % 352 == 0).lower())
        if start == 0:
            assert profile.get("resume_from_checkpoint") is None
        else:
            assert profile["resume_from_checkpoint"].endswith(f"-step{start}/checkpoint-{start}")
            assert "smoke" not in profile["resume_from_checkpoint"]


def test_recipe_matches_plan32_except_model_epochs_and_start():
    ours, theirs = _profile("plan33-epoch1-step88"), _profile("plan32-epoch1-step88", PLAN32)
    for key in ("lora_r", "lora_alpha", "lora_dropout", "learning_rate", "weight_decay", "lr_scheduler_type",
                "warmup_ratio", "gradient_accumulation_steps", "seed", "image_token_budget",
                "image_view_policy", "completion_only_logits", "max_seq_length", "gradient_checkpointing",
                "telemetry_interval_steps", "telemetry_train_rows", "telemetry_validation_rows",
                "lora_target_modules_regex", "required_transformers_version", "require_gradient_subsystems"):
        assert ours[key] == theirs[key], key
    assert theirs["initial_adapter_path"] and not ours.get("initial_adapter_path")


def test_target_regex_selects_decoder_vision_and_projector_names():
    pattern = re.compile(_profile("plan33-epoch1-step88")["lora_target_modules_regex"])
    assert pattern.match("model.language_model.layers.34.mlp.down_proj")
    assert pattern.match("model.vision_tower.encoder.layers.15.self_attn.v_proj")
    assert pattern.match("model.embed_vision.embedding_projection")
    assert not pattern.match("model.audio_tower.layers.0.self_attn.q_proj")
    assert not pattern.match("lm_head")


@pytest.mark.parametrize("name", ["plan33-smoke-step1", "plan33-smoke-step2", "plan33-smoke-continuous-step2",
                                  *(f"plan33-epoch{e}-step{s + 88}" for e, s in SEGMENTS)])
def test_every_profile_passes_strict_telemetry_gate(name, monkeypatch):
    import transformers

    monkeypatch.setattr(transformers, "__version__", "5.5.2")
    assert finetune._validate_strict_telemetry_config(_profile(name)) is True


@pytest.mark.parametrize("changes", [
    {"plan32_epoch": 4},
    {"literal_epochs": 4},
    {"telemetry_start_step": 44},
    {"segment_steps": 1},
    {"stop_after_step": 100},
])
def test_misaligned_or_out_of_lineage_segments_fail_closed(changes, monkeypatch):
    import transformers

    monkeypatch.setattr(transformers, "__version__", "5.5.2")
    profile = _profile("plan33-epoch2-step528")
    for key, value in changes.items():
        profile[key] = value
    with pytest.raises(ValueError):
        finetune._validate_strict_telemetry_config(profile)


def test_plan32_two_epoch_profiles_cannot_reach_a_third_epoch(monkeypatch):
    import transformers

    monkeypatch.setattr(transformers, "__version__", "5.5.2")
    profile = _profile("plan32-epoch2-step704", PLAN32)
    assert finetune._validate_strict_telemetry_config(profile) is True
    profile["plan32_epoch"] = 3
    profile["telemetry_start_step"], profile["stop_after_step"] = 704, 792
    with pytest.raises(ValueError):
        finetune._validate_strict_telemetry_config(profile)


def test_wrong_transformers_version_is_rejected(monkeypatch):
    import transformers

    monkeypatch.setattr(transformers, "__version__", "5.5.0")
    with pytest.raises(RuntimeError):
        finetune._validate_strict_telemetry_config(_profile("plan33-epoch1-step88"))
