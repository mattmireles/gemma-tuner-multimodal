"""First Plan 31 segment preserves the corrected training recipe."""

import configparser
from pathlib import Path

import pytest
from datasets import Dataset as HFDataset

from gemma_tuner.core.config import load_profile_config
from gemma_tuner.models.gemma.finetune import _resolve_dataset_image_paths, _validate_strict_telemetry_config
import gemma_tuner.utils.dataset_utils as dataset_utils


PATH = Path(__file__).parents[1] / "config/plan31-mixed-telepathic-v1.ini"


def test_plan31_first_segment_is_bounded_and_uses_frozen_mix(monkeypatch):
    config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    assert config.read(PATH)
    profile = load_profile_config(config, "telepathic-plan31-r64-mixed-step440")
    assert profile["source"] == "tt-screenshot-plan31-mixed-v4/full"
    assert profile["input_mode_column"] == "input_mode"
    assert profile["plan31_schedule_sha256"] == "ad89766c00958bce014c48fcc561d2e2fec34f3d95a98392344e3e0d4eb81ca2"
    assert (profile["telemetry_start_step"], profile["stop_after_step"], int(profile["max_steps"])) == (352, 440, 440)
    assert (profile["lora_r"], profile["gradient_accumulation_steps"], profile["seed"]) == (64, 8, 42)
    assert profile["lr_scheduler_type"] == "constant"
    assert profile["required_transformers_version"] == "5.5.2"
    assert profile["resume_from_checkpoint"].endswith("checkpoint-352")
    import transformers

    monkeypatch.setattr(transformers, "__version__", "5.5.0")
    with pytest.raises(RuntimeError, match="Transformers 5.5.2"):
        _validate_strict_telemetry_config(profile)
    monkeypatch.setattr(transformers, "__version__", "5.5.2")
    assert _validate_strict_telemetry_config(profile)


def test_mode_panel_relative_images_resolve_like_main_validation(tmp_path):
    dataset_dir = tmp_path / "full"
    dataset_dir.mkdir()
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image = image_dir / "row.png"
    image.write_bytes(b"screenshot")
    panel = HFDataset.from_list([{"id": "row", "image_path": "../images/row.png"}])
    resolved = _resolve_dataset_image_paths(panel, str(dataset_dir), "image_path")
    assert Path(resolved[0]["image_path"]).resolve() == image
    assert Path(resolved[0]["image_path"]).is_file()


def test_plan31_launch_uses_same_explicit_config_for_dataset_loader(monkeypatch):
    monkeypatch.setenv("GEMMA_TUNER_CONFIG", str(PATH))
    monkeypatch.setattr(dataset_utils, "_config", None)
    config = dataset_utils._get_config()
    assert config.has_section("dataset:tt-screenshot-plan31-mixed-v4/full")
    profile = load_profile_config(config, "telepathic-plan31-r64-mixed-step440")
    assert profile["dataset"] == "tt-screenshot-plan31-mixed-v4/full"


@pytest.mark.parametrize(("epoch", "start", "stop"), [(1, 0, 88), (1, 264, 352), (2, 352, 440), (2, 616, 704)])
def test_plan32_accepts_only_epoch_aligned_segments(epoch, start, stop, monkeypatch):
    import transformers

    profile = {
        "required_transformers_version": "5.5.2",
        "require_validation_telemetry": True,
        "modality": "image",
        "load_validation": True,
        "completion_only_logits": True,
        "record_exposures": True,
        "eval_strategy": "steps",
        "logging_steps": 1,
        "telemetry_interval_steps": 22,
        "telemetry_train_rows": 20,
        "telemetry_validation_rows": 20,
        "stop_after_step": stop,
        "telemetry_start_step": start,
        "input_mode_column": "input_mode",
        "plan32_epoch": epoch,
        "plan32_schedule_path": "schedule.jsonl",
        "plan32_schedule_sha256": "a" * 64,
        "plan32_projection_receipt_sha256": "b" * 64,
    }
    monkeypatch.setattr(transformers, "__version__", "5.5.2")
    assert _validate_strict_telemetry_config(profile)


@pytest.mark.parametrize(("epoch", "start", "stop"), [(1, 352, 440), (2, 264, 352), (2, 0, 88)])
def test_plan32_rejects_segments_crossing_epoch_contract(epoch, start, stop, monkeypatch):
    import transformers

    monkeypatch.setattr(transformers, "__version__", "5.5.2")
    profile = {
        "required_transformers_version": "5.5.2",
        "require_validation_telemetry": True,
        "modality": "image",
        "load_validation": True,
        "completion_only_logits": True,
        "record_exposures": True,
        "eval_strategy": "steps",
        "logging_steps": 1,
        "telemetry_interval_steps": 22,
        "telemetry_train_rows": 20,
        "telemetry_validation_rows": 20,
        "stop_after_step": stop,
        "telemetry_start_step": start,
        "input_mode_column": "input_mode",
        "plan32_epoch": epoch,
        "plan32_schedule_path": "schedule.jsonl",
        "plan32_schedule_sha256": "a" * 64,
        "plan32_projection_receipt_sha256": "b" * 64,
    }
    with pytest.raises(ValueError, match="Plan 32 telemetry"):
        _validate_strict_telemetry_config(profile)


def test_plan32_frozen_config_has_eight_epoch_aligned_profiles(monkeypatch):
    import transformers

    path = Path(__file__).parents[1] / "config/plan32-literal-two-epochs.ini"
    config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    assert config.read(path)
    monkeypatch.setattr(transformers, "__version__", "5.5.2")
    profile_names = [
        section.split(":", 1)[1]
        for section in config.sections()
        if section.startswith("profile:") and section != "profile:plan32-warmstart-smoke"
    ]
    assert len(profile_names) == 8
    for name in profile_names:
        profile = load_profile_config(config, name)
        assert _validate_strict_telemetry_config(profile)
        assert profile["num_train_epochs"] == 2
        assert profile["gradient_accumulation_steps"] == 8
        assert profile["learning_rate"] == 1e-4
        assert profile["required_transformers_version"] == "5.5.2"
        assert profile["plan32_schedule_path"].endswith(f"epoch-{profile['plan32_epoch']}/epoch-modes.jsonl")


def test_plan32_resume_paths_match_literal_segment_output_directories():
    path = Path(__file__).parents[1] / "config/plan32-literal-two-epochs.ini"
    config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    assert config.read(path)
    expected = {
        "plan32-epoch1-step176": "output/plan32-literal-epoch1-step88/checkpoint-88",
        "plan32-epoch1-step264": "output/plan32-literal-epoch1-step176/checkpoint-176",
        "plan32-epoch1-step352": "output/plan32-literal-epoch1-step264/checkpoint-264",
        "plan32-epoch2-step440": "output/plan32-literal-epoch1-step352/checkpoint-352",
        "plan32-epoch2-step528": "output/plan32-literal-epoch2-step440/checkpoint-440",
        "plan32-epoch2-step616": "output/plan32-literal-epoch2-step528/checkpoint-528",
        "plan32-epoch2-step704": "output/plan32-literal-epoch2-step616/checkpoint-616",
    }
    for name, checkpoint in expected.items():
        profile = load_profile_config(config, name)
        assert profile["resume_from_checkpoint"] == checkpoint
