"""Guard the five-view MLX batch to fixed-batch Core ML handoff."""

import json
from collections import OrderedDict

import mlx.core as mx
import numpy as np
import pytest

from tools import plan32_ane_vision
from tools.plan32_ane_vision import (
    ANEVisionTower,
    VerifiedANEPackage,
    parse_free_memory_percent,
    parse_swap_used_mb,
    tree_sha256,
)


class FakeCoreMLModel:
    def predict(self, inputs):
        image = inputs["pixels"]
        assert image.shape == (1, 3, 48, 48)
        value = image[0, 0, 0, 0]
        return {"hidden_states": np.full((1, 2, 768), value, dtype=np.float16)}


def test_five_view_batch_preserves_mlx_feature_order():
    tower = ANEVisionTower.__new__(ANEVisionTower)
    tower.models = {(48, 48): FakeCoreMLModel()}
    tower.checked_shapes = {(48, 48)}
    pixels = mx.array(np.arange(5, dtype=np.float32)[:, None, None, None] * np.ones((5, 3, 48, 48)))

    output = np.asarray(tower(pixels))

    assert output.shape == (1, 10, 768)
    assert output[0, :, 0].tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_constructor_verifies_but_does_not_eagerly_load_packages(tmp_path, monkeypatch):
    conversion = tmp_path / "conversion.json"
    conversion.write_text(
        json.dumps(
            {
                "conversion_identity_sha256": "conversion",
                "adapter": {"weights_sha256": "adapter"},
            }
        )
    )
    directory = tmp_path / "plan32-cp616-ane-48x48"
    package = directory / "Vision.mlpackage"
    package.mkdir(parents=True)
    (package / "model.bin").write_bytes(b"verified")
    (directory / "export-receipt.json").write_text(
        json.dumps(
            {
                "model_revision": "conversion",
                "adapter_weights_sha256": "adapter",
                "coreml_vs_torch_cosine": 0.999,
                "mlx_vs_torch_cosine": 0.9999,
                "input_shape": [1, 3, 48, 48],
                "package": "Vision.mlpackage",
                "package_tree_sha256": tree_sha256(package),
            }
        )
    )

    def fail_if_loaded(*_args, **_kwargs):
        raise AssertionError("constructor eagerly loaded a Core ML model")

    monkeypatch.setattr(plan32_ane_vision.ct.models, "MLModel", fail_if_loaded)
    tower = ANEVisionTower(tmp_path, conversion, None, tmp_path / "parity.jsonl")

    assert list(tower.packages) == [(48, 48)]
    assert tower.models == {}


def test_shape_loader_keeps_only_one_resident_model(tmp_path, monkeypatch):
    loaded = []

    def fake_load(path, **_kwargs):
        loaded.append(path)
        return object()

    monkeypatch.setattr(plan32_ane_vision, "assert_safe_host_pressure", lambda **_kwargs: {})
    monkeypatch.setattr(plan32_ane_vision.ct.models, "MLModel", fake_load)
    tower = ANEVisionTower.__new__(ANEVisionTower)
    tower.packages = {
        shape: VerifiedANEPackage(shape, tmp_path, tmp_path / name, tmp_path / "receipt", {})
        for shape, name in [((48, 48), "one.mlpackage"), ((96, 96), "two.mlpackage")]
    }
    tower.models = OrderedDict()
    tower.compiled_root = None
    tower.max_loaded_models = 1
    tower.max_swap_used_mb = 1
    tower.min_free_memory_percent = 1

    first = tower._model_for_shape((48, 48))
    assert tower._model_for_shape((48, 48)) is first
    tower._model_for_shape((96, 96))

    assert loaded == [str(tmp_path / "one.mlpackage"), str(tmp_path / "two.mlpackage")]
    assert list(tower.models) == [(96, 96)]


def test_target_compiled_artifact_is_required_when_configured(tmp_path, monkeypatch):
    monkeypatch.setattr(plan32_ane_vision, "assert_safe_host_pressure", lambda **_kwargs: {})
    tower = ANEVisionTower.__new__(ANEVisionTower)
    shape = (48, 48)
    tower.packages = {
        shape: VerifiedANEPackage(
            shape,
            tmp_path,
            tmp_path / "source.mlpackage",
            tmp_path / "receipt.json",
            {"package_tree_sha256": "0" * 64},
        )
    }
    tower.models = OrderedDict()
    tower.compiled_root = tmp_path / "compiled"
    tower.max_loaded_models = 1
    tower.max_swap_used_mb = 1
    tower.min_free_memory_percent = 1

    with pytest.raises(ValueError, match="missing target-local compiled"):
        tower._model_for_shape(shape)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("total = 4096.00M  used = 512.25M  free = 3583.75M", 512.25),
        ("total = 8.00G  used = 1.50G  free = 6.50G", 1536.0),
        ("total = 4.00G  used = 1024.00K  free = 3.99G", 1.0),
    ],
)
def test_parse_swap_used_mb(raw, expected):
    assert parse_swap_used_mb(raw) == expected


def test_parse_free_memory_percent():
    assert parse_free_memory_percent("System-wide memory free percentage: 42%") == 42


def test_pressure_guard_fails_closed_before_ane_load(monkeypatch):
    monkeypatch.setattr(
        plan32_ane_vision,
        "host_pressure_snapshot",
        lambda: {"swap_used_mb": 2048.0, "free_memory_percent": 50.0},
    )
    with pytest.raises(RuntimeError, match="swap 2048.0 MiB exceeds 512.0 MiB"):
        plan32_ane_vision.assert_safe_host_pressure(
            max_swap_used_mb=512.0, min_free_memory_percent=35.0
        )


def test_pressure_violation_detects_growth_from_a_nonzero_baseline():
    baseline = {"swap_used_mb": 1800.0, "free_memory_percent": 60.0}
    safe = {"swap_used_mb": 2300.0, "free_memory_percent": 10.0}
    unsafe = {"swap_used_mb": 2900.0, "free_memory_percent": 10.0}

    assert (
        plan32_ane_vision.pressure_violation(
            safe,
            baseline,
            max_swap_used_mb=4096.0,
            max_swap_growth_mb=1024.0,
            min_free_memory_percent=3.0,
        )
        is None
    )
    assert "swap grew 1100.0 MiB" in plan32_ane_vision.pressure_violation(
        unsafe,
        baseline,
        max_swap_used_mb=4096.0,
        max_swap_growth_mb=1024.0,
        min_free_memory_percent=3.0,
    )
