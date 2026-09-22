"""Portable GCP package helpers preserve IDs, images, and relative paths."""

from __future__ import annotations

import configparser
import importlib.util
from pathlib import Path

from PIL import Image

MODULE_PATH = Path(__file__).parents[1] / "tools" / "package_tt_telepathic_gcp.py"
SPEC = importlib.util.spec_from_file_location("package_tt_telepathic_gcp", MODULE_PATH)
assert SPEC and SPEC.loader
package = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package)


def test_portable_rows_copy_once_and_use_relative_shared_image(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (2, 2), "red").save(source)
    rows = [{"id": "7", "image_path": str(source), "prompt": "p"}]
    images = tmp_path / "bundle" / "images"
    first = package.portable_rows(rows, selected_ids={"7"}, images=images)
    second = package.portable_rows(rows, selected_ids={"7"}, images=images)
    assert first == second
    assert first[0]["image_path"] == "../images/7.png"
    assert (images / "7.png").read_bytes() == source.read_bytes()


def test_portable_rows_reject_missing_selected_id(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError, match="missing"):
        package.portable_rows([], selected_ids={"missing"}, images=tmp_path)


def test_rank64_followup_is_conditioned_constant_lr_and_checkpointed() -> None:
    profiles = configparser.ConfigParser(interpolation=None)
    profiles["profile:telepathic-conditioned"] = {"model": "gemma4-e4b"}

    package.profile_for_rank64_followup(profiles)

    profile = profiles["profile:telepathic-conditioned-overfit-r64-lr0.0001"]
    assert profile["lora_r"] == "64"
    assert profile["lora_alpha"] == "128"
    assert profile["learning_rate"] == "0.0001"
    assert profile["num_train_epochs"] == "20"
    assert profile["gradient_accumulation_steps"] == "1"
    assert profile["lr_scheduler_type"] == "constant"
    assert profile["warmup_steps"] == "0"
    assert profile["warmup_ratio"] == "0"
    assert profile["save_strategy"] == "steps"
    assert profile["save_steps"] == "160"
    assert profile["save_total_limit"] == "3"


def test_full_prompt_rank64_epoch_uses_frozen_one_epoch_schedule() -> None:
    profiles = configparser.ConfigParser(interpolation=None)
    profiles["profile:telepathic-full"] = {"model": "gemma4-e4b"}

    package.profile_for_full_prompt_epoch(profiles)

    profile = profiles["profile:telepathic-full-r64-one-epoch"]
    assert profile["lora_r"] == "64"
    assert profile["lora_alpha"] == "128"
    assert profile["learning_rate"] == "0.0001"
    assert profile["num_train_epochs"] == "1"
    assert profile["gradient_accumulation_steps"] == "8"
    assert profile["lr_scheduler_type"] == "constant"
    assert profile["warmup_steps"] == "0"
    assert profile["warmup_ratio"] == "0"
    assert profile["save_steps"] == "78"
    assert profile["save_total_limit"] == "2"
    assert profiles["profile:telepathic-full-r64-half-epoch"]["stop_after_step"] == "78"


def test_full_prompt_bundle_prunes_legacy_experiment_profiles() -> None:
    profiles = configparser.ConfigParser(interpolation=None)
    profiles["dataset_defaults"] = {"text_column": "response"}
    profiles["group:gemma"] = {"dtype": "bfloat16"}
    profiles["model:gemma-4-e4b-it-pinned"] = {"group": "gemma"}
    profiles["dataset:tt-screenshot-telepathic-v3-sft/full"] = {"source": "full"}
    profiles["profile:telepathic-full-r64-one-epoch"] = {"dataset": "full"}
    profiles["profile:telepathic-full-r64-half-epoch"] = {"dataset": "full", "stop_after_step": "78"}
    profiles["profile:telepathic-conditioned-overfit-r64-lr0.0001"] = {"dataset": "overfit"}

    package.prune_to_full_prompt_epoch(profiles)

    assert set(profiles.sections()) == {
        "dataset_defaults",
        "group:gemma",
        "model:gemma-4-e4b-it-pinned",
        "dataset:tt-screenshot-telepathic-v3-sft/full",
        "profile:telepathic-full-r64-one-epoch",
        "profile:telepathic-full-r64-half-epoch",
    }


def test_plan29_profile_freezes_one_epoch_quarter_checkpoints() -> None:
    profiles = configparser.ConfigParser(interpolation=None)
    profiles["profile:telepathic-plan29-r64-one-epoch"] = {
        "lora_r": "64", "lora_alpha": "128", "lora_dropout": "0.05",
        "learning_rate": "0.0001", "num_train_epochs": "1",
        "gradient_accumulation_steps": "8", "lr_scheduler_type": "constant",
            "save_steps": "88", "save_total_limit": "4", "max_seq_length": "24576",
            "completion_only_logits": "true", "image_view_policy": "global_plus_four_nonoverlapping_quadrants",
            "record_exposures": "true",
        }
    package.verify_plan29_profile(profiles)


def test_plan29_bundle_preserves_only_plan29_profile(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    staging = root / "data" / "datasets" / "tt-screenshot-plan28-sft-v1"
    if not staging.is_dir():
        import pytest
        pytest.skip("private Plan 29 staging not built")
    result = package.build_plan29_bundle(staging, tmp_path / "bundle")
    assert result["written"] == {"train": 2812, "validation": 252}
    assert result["checkpoint_steps"] == [88, 176, 264, 352]
    assert result["sealed_test_staged"] is False
    profiles = configparser.ConfigParser(interpolation=None)
    profiles.read(tmp_path / "bundle" / "profiles.ini")
    assert "profile:telepathic-plan29-r64-one-epoch" in profiles
    assert "profile:telepathic-full-r64-one-epoch" not in profiles
