"""Determinism, layout, and corruption tests for runtime packaging."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from gemma_tuner.runtime.package_schema import EXPECTED_GEOMETRY, PackageValidationError, canonical_json_bytes
from gemma_tuner.runtime.packer import pack_runtime, verify_runtime_package


def _config(*, quantized: bool) -> dict[str, object]:
    result: dict[str, object] = {}
    for dotted, value in EXPECTED_GEOMETRY.items():
        target = result
        parts = dotted.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})  # type: ignore[assignment]
        target[parts[-1]] = value
    result["text_config"]["layer_types"] = [  # type: ignore[index]
        "full_attention" if (index + 1) % 6 == 0 else "sliding_attention" for index in range(42)
    ]
    if quantized:
        result["quantization_config"] = {"bits": 6, "group_size": 64, "mode": "affine"}
    return result


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header: dict[str, object] = {}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(payload)]}
    encoded = canonical_json_bytes(header)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _model(root: Path, *, quantized: bool = True) -> Path:
    root.mkdir()
    (root / "config.json").write_text(json.dumps(_config(quantized=quantized)), encoding="utf-8")
    (root / "processor_config.json").write_text('{"processor":"gemma4"}', encoding="utf-8")
    (root / "tokenizer.json").write_text('{"version":"1"}', encoding="utf-8")
    (root / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    _write_safetensors(root / "model.safetensors", {"model.weight": ("BF16", [2], b"\x00\x00\x01\x00")})
    return root


def test_stock_package_is_byte_deterministic(tmp_path: Path) -> None:
    source = _model(tmp_path / "model")
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest_a = pack_runtime(source, first, model_id="test/gemma", revision="abc")
    manifest_b = pack_runtime(source, second, model_id="test/gemma", revision="abc")
    assert manifest_a == manifest_b
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    for entry in manifest_a["payload"]["files"]:
        assert (first / entry["path"]).read_bytes() == (second / entry["path"]).read_bytes()
    assert verify_runtime_package(first) == manifest_a


def test_merged_sft_binds_export_manifest(tmp_path: Path) -> None:
    source = _model(tmp_path / "model", quantized=False)
    export_manifest = tmp_path / "export.json"
    export_manifest.write_text('{"base_revision":"abc","merged":true}', encoding="utf-8")
    manifest = pack_runtime(
        source,
        tmp_path / "package",
        model_id="test/sft",
        revision="sft-1",
        source_kind="merged_sft",
        upstream_export_manifest=export_manifest,
    )
    assert manifest["model"]["quantization"] is None
    assert manifest["source"]["upstream_export_manifest_sha256"]
    assert verify_runtime_package(tmp_path / "package") == manifest


def test_coreml_directory_is_content_addressed(tmp_path: Path) -> None:
    source = _model(tmp_path / "model")
    coreml = tmp_path / "vision.mlpackage"
    (coreml / "Data").mkdir(parents=True)
    (coreml / "Manifest.json").write_text('{"model":"vision"}', encoding="utf-8")
    (coreml / "Data" / "weights.bin").write_bytes(b"coreml-weights")
    manifest = pack_runtime(
        source,
        tmp_path / "package",
        model_id="test/gemma",
        revision="abc",
        coreml_package=coreml,
    )
    assert manifest["coreml"]["files"] == 2
    assert len(manifest["coreml"]["content_sha256"]) == 64
    assert verify_runtime_package(tmp_path / "package") == manifest


def test_corrupt_payload_fails_verification(tmp_path: Path) -> None:
    source = _model(tmp_path / "model")
    package = tmp_path / "package"
    pack_runtime(source, package, model_id="test/gemma", revision="abc")
    (package / "model" / "tokenizer.json").write_text("corrupt", encoding="utf-8")
    with pytest.raises(PackageValidationError, match="checksums"):
        verify_runtime_package(package)


def test_duplicate_tensor_names_across_shards_fail(tmp_path: Path) -> None:
    source = _model(tmp_path / "model")
    (source / "model.safetensors").unlink()
    _write_safetensors(source / "model-00001-of-00002.safetensors", {"duplicate": ("U32", [1], b"\x00" * 4)})
    _write_safetensors(source / "model-00002-of-00002.safetensors", {"duplicate": ("U32", [1], b"\x00" * 4)})
    with pytest.raises(PackageValidationError, match="duplicate tensor"):
        pack_runtime(source, tmp_path / "package", model_id="test/gemma", revision="abc")


def test_wrong_tensor_byte_layout_fails(tmp_path: Path) -> None:
    source = _model(tmp_path / "model")
    header = {"broken": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}
    encoded = canonical_json_bytes(header)
    (source / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\x00" * 4)
    with pytest.raises(PackageValidationError, match="byte size mismatch"):
        pack_runtime(source, tmp_path / "package", model_id="test/gemma", revision="abc")


def test_geometry_mismatch_fails_before_output_creation(tmp_path: Path) -> None:
    source = _model(tmp_path / "model")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config["vision_config"]["patch_size"] = 14
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "package"
    with pytest.raises(PackageValidationError, match="patch_size"):
        pack_runtime(source, output, model_id="test/gemma", revision="abc")
    assert not output.exists()
