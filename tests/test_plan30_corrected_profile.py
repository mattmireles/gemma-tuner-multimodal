"""Plan 30 smoke configuration is isolated and targets only active E4B modules."""

from __future__ import annotations

import configparser
from pathlib import Path

import pytest

from gemma_tuner.core.config import load_profile_config

ROOT = Path(__file__).parents[1]
PROFILE_PATH = ROOT / "config" / "plan30-corrected-telepathic-smoke.ini"
LOCK_PATH = ROOT / "requirements" / "requirements-gemma4-telepathic-corrected.lock"
NAME = "telepathic-plan30-r64-corrected-smoke"
REVISION = "fee6332c1abaafb77f6f9624236c63aa2f1d0187"


def _profile():
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    assert cfg.read(PROFILE_PATH)
    return load_profile_config(cfg, NAME)


def test_plan30_profile_is_bounded_and_validation_enabled() -> None:
    profile = _profile()
    assert "transformers==5.5.2" in LOCK_PATH.read_text(encoding="utf-8").splitlines()
    assert profile["model_revision"] == REVISION
    assert profile["source"] == "tt-screenshot-plan28-sft-v1/full"
    assert profile["train_split"] == "train"
    assert profile["validation_split"] == "validation"
    assert profile["system_prompt_column"] == "system_prompt"
    assert profile["image_view_policy"] == "global_plus_four_nonoverlapping_quadrants"
    assert profile["dtype"] == "bfloat16"
    assert profile["optim"] == "adamw_torch"
    assert profile["seed"] == 42
    assert profile["lora_r"] == 64
    assert profile["gradient_accumulation_steps"] == 8
    assert profile["stop_after_step"] == 88
    assert profile["load_validation"] is True
    assert profile["eval_strategy"] == "steps"
    assert profile["eval_steps"] == "88"
    assert profile["require_validation_telemetry"] == "true"
    assert profile["telemetry_interval_steps"] == "22"
    assert profile["telemetry_train_rows"] == "20"
    assert profile["telemetry_validation_rows"] == "20"
    assert profile["telemetry_full_validation_at_stop"] == "true"
    assert profile["required_transformers_version"] == "5.5.2"
    assert profile["output_dir"] == "output-plan30-corrected"
    assert "resume_from_checkpoint" not in profile


def test_plan30_regex_attaches_only_active_552_modules() -> None:
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "5.5.2":
        pytest.skip("active-target inventory requires the isolated Transformers 5.5.2 runtime")

    from accelerate import init_empty_weights
    from huggingface_hub import try_to_load_from_cache
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, Gemma4ForConditionalGeneration

    from gemma_tuner.models.gemma.finetune import _validate_lora_target_regex
    from gemma_tuner.models.gemma.gemma4_patches import convert_loaded_clippable_linears

    config_path = try_to_load_from_cache("google/gemma-4-E4B-it", "config.json", revision=REVISION)
    if not isinstance(config_path, str):
        pytest.skip("pinned E4B public model config is not cached locally")

    profile = _profile()
    config = AutoConfig.from_pretrained(Path(config_path).parent, local_files_only=True)
    with init_empty_weights():
        model = Gemma4ForConditionalGeneration(config)
    convert_loaded_clippable_linears(model)
    targets = _validate_lora_target_regex(model, profile["lora_target_modules_regex"])
    assert len(targets) == 371
    assert sum(name.startswith("model.language_model.") for name in targets) == 258
    assert sum(name.startswith("model.vision_tower.") for name in targets) == 112
    assert sum(name.startswith("model.embed_vision.") for name in targets) == 1
    assert all(
        not name.endswith(("k_proj", "v_proj"))
        for name in targets
        if name.startswith("model.language_model.layers.")
        and int(name.split(".")[3]) >= 24
    )

    with init_empty_weights():
        adapted = get_peft_model(
            model,
            LoraConfig(
                r=64,
                lora_alpha=128,
                lora_dropout=0.05,
                target_modules=profile["lora_target_modules_regex"],
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    attached = [name[: -len(".lora_A.default")] for name, _ in adapted.named_modules() if name.endswith(".lora_A.default")]
    assert len(attached) == len(targets)
    assert all(any(name.endswith(target) for name in attached) for target in targets)
