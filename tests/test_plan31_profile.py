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
