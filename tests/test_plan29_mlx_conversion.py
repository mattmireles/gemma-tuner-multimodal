from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

import pytest

MODULE = Path(__file__).parents[1] / "tools" / "convert_plan29_checkpoint_mlx.py"
SPEC = importlib.util.spec_from_file_location("convert_plan29_checkpoint_mlx", MODULE)
assert SPEC and SPEC.loader
conversion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(conversion)


@pytest.mark.parametrize(
    ("module", "hf_key", "mlx_key"),
    [
        (
            "base_model.model.model.language_model.layers.0.self_attn.q_proj",
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "language_model.model.layers.0.self_attn.q_proj.weight",
        ),
        (
            "base_model.model.model.vision_tower.encoder.layers.0.self_attn.q_proj",
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight",
            "vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight",
        ),
        (
            "base_model.model.model.embed_vision.embedding_projection",
            "model.embed_vision.embedding_projection.weight",
            "embed_vision.embedding_projection.weight",
        ),
    ],
)
def test_weight_mapping_covers_all_target_families(module, hf_key, mlx_key) -> None:
    assert conversion.mapped_weight_keys(module) == (hf_key, mlx_key)


def test_adapter_pairs_must_be_complete_and_unique() -> None:
    keys = ["x.lora_A.weight", "x.lora_B.weight", "y.lora_A.weight", "y.lora_B.weight"]
    assert conversion.adapter_module_names(keys) == ["x", "y"]
    with pytest.raises(ValueError, match="exactly one"):
        conversion.adapter_module_names(["x.lora_A.weight"])


def test_adapter_config_is_fail_closed() -> None:
    config = {
        "peft_type": "LORA", "r": 64, "lora_alpha": 128, "bias": "none",
        "use_dora": False, "use_rslora": False,
        "base_model_name_or_path": conversion.HF_REPO_ID,
    }
    assert conversion.validate_adapter_config(config) == 2.0
    config["r"] = 32
    with pytest.raises(ValueError, match="config mismatch"):
        conversion.validate_adapter_config(config)


def test_target_classification_rejects_unknown_modules() -> None:
    assert conversion.classify_module("x.language_model.y") == "language"
    assert conversion.classify_module("x.vision_tower.y") == "vision"
    assert conversion.classify_module("x.embed_vision.embedding_projection") == "projection"
    with pytest.raises(ValueError, match="unexpected LoRA target"):
        conversion.classify_module("x.audio_tower.y")


def test_only_final_shared_layer_kv_targets_are_inference_inactive() -> None:
    config = {"text_config": {"num_hidden_layers": 4, "num_kv_shared_layers": 2}}
    prefix = "base_model.model.model.language_model.layers"
    modules = [
        f"{prefix}.{layer}.self_attn.{projection}"
        for layer in range(4)
        for projection in ("k_proj", "v_proj")
    ]
    assert conversion.inference_inactive_shared_kv_modules(config, modules) == {
        f"{prefix}.{layer}.self_attn.{projection}"
        for layer in (2, 3)
        for projection in ("k_proj", "v_proj")
    }


def test_shared_kv_target_count_is_fail_closed() -> None:
    config = {"text_config": {"num_hidden_layers": 4, "num_kv_shared_layers": 2}}
    with pytest.raises(ValueError, match="inactive target mismatch"):
        conversion.inference_inactive_shared_kv_modules(config, [])


def test_corrected_profile_requires_no_adapted_shared_kv_targets() -> None:
    config = {"text_config": {"num_hidden_layers": 4, "num_kv_shared_layers": 2}}
    prefix = "base_model.model.model.language_model.layers"
    active = [f"{prefix}.0.self_attn.k_proj"]
    assert conversion.inference_inactive_shared_kv_modules(
        config, active, profile="plan30-corrected"
    ) == set()
    with pytest.raises(ValueError, match="inactive target mismatch"):
        conversion.inference_inactive_shared_kv_modules(
            config, active + [f"{prefix}.2.self_attn.v_proj"],
            profile="plan30-corrected",
        )


def test_conversion_profiles_have_distinct_exact_target_counts() -> None:
    assert conversion.EXPECTED_TARGETS["plan29"] == {
        "language": 294, "vision": 112, "projection": 1,
    }
    assert conversion.EXPECTED_TARGETS["plan30-corrected"] == {
        "language": 258, "vision": 112, "projection": 1,
    }


def test_safetensor_ranges_are_absolute_and_skip_metadata(tmp_path: Path) -> None:
    header = json.dumps(
        {
            "x": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
            "__metadata__": {"format": "mlx"},
        },
        separators=(",", ":"),
    ).encode()
    path = tmp_path / "tiny.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 4)
    start = 8 + len(header)
    assert conversion.safetensor_byte_ranges(path) == {
        "x": (start, start + 4, "BF16")
    }
