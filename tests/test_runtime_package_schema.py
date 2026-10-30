"""Unit tests for the fail-closed Gemma 4 E4B package schema."""

from __future__ import annotations

import copy

import pytest

from gemma_tuner.runtime.package_schema import (
    EXPECTED_GEOMETRY,
    PackageValidationError,
    validate_gemma4_e4b_config,
)


def synthetic_config(*, quantized: bool = True) -> dict[str, object]:
    config: dict[str, object] = {}
    for dotted, value in EXPECTED_GEOMETRY.items():
        target = config
        parts = dotted.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})  # type: ignore[assignment]
        target[parts[-1]] = copy.deepcopy(value)
    config["text_config"]["layer_types"] = [  # type: ignore[index]
        "full_attention" if (index + 1) % 6 == 0 else "sliding_attention" for index in range(42)
    ]
    if quantized:
        config["quantization_config"] = {"mode": "affine", "bits": 6, "group_size": 64}
    return config


def test_exact_e4b_w6_geometry_is_accepted() -> None:
    assert validate_gemma4_e4b_config(synthetic_config(), "stock") == {
        "mode": "affine",
        "bits": 6,
        "group_size": 64,
    }


def test_geometry_change_fails_before_packaging() -> None:
    config = synthetic_config()
    config["text_config"]["hidden_size"] = 2048  # type: ignore[index]
    with pytest.raises(PackageValidationError, match="hidden_size"):
        validate_gemma4_e4b_config(config, "stock")


def test_stock_package_requires_exact_w6_recipe() -> None:
    config = synthetic_config()
    config["quantization_config"]["bits"] = 4  # type: ignore[index]
    with pytest.raises(PackageValidationError, match="quantization recipe"):
        validate_gemma4_e4b_config(config, "stock")


def test_merged_sft_may_remain_unquantized_for_reference_phase() -> None:
    assert validate_gemma4_e4b_config(synthetic_config(quantized=False), "merged_sft") is None
