"""Build and verify content-addressed Gemma 4 E4B runtime packages."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .package_schema import (
    ENGINE_ABI_MAJOR,
    EXPECTED_GEOMETRY,
    SCHEMA_VERSION,
    SOURCE_KINDS,
    PackageValidationError,
    canonical_json_bytes,
    safe_package_path,
    sha256_entries,
    sha256_file,
    tensor_index_bytes,
    tensor_inventory,
    validate_gemma4_e4b_config,
    validate_manifest,
    validate_w6_tensor_layout,
)

_MODEL_SUFFIXES = frozenset({".json", ".safetensors", ".jinja", ".model"})


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackageValidationError(f"invalid JSON file {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise PackageValidationError(f"JSON root must be an object: {path.name}")
    return value


def _source_files(model_root: Path) -> list[Path]:
    if not model_root.is_dir():
        raise PackageValidationError(f"model source is not a directory: {model_root}")
    files = sorted(path for path in model_root.iterdir() if path.is_file() and path.suffix in _MODEL_SUFFIXES)
    names = {path.name for path in files}
    missing = {"config.json", "tokenizer.json", "processor_config.json"} - names
    if missing:
        raise PackageValidationError(f"model export is missing required files: {', '.join(sorted(missing))}")
    return files


def _copy_tree(source: Path, destination: Path) -> None:
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return
    if not source.is_dir():
        raise PackageValidationError(f"package input does not exist: {source}")
    for path in sorted(source.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(source)
        if any(part in {"", ".", ".."} for part in relative.parts):
            raise PackageValidationError(f"unsafe package source path: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def _payload_entries(package_root: Path) -> list[dict[str, Any]]:
    entries = []
    for path in sorted(package_root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = safe_package_path(path.relative_to(package_root).as_posix())
        entries.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return entries


def _hash_if_present(root: Path, name: str) -> str | None:
    path = root / name
    return sha256_file(path) if path.is_file() else None


def pack_runtime(
    model_source: str | Path,
    output: str | Path,
    *,
    model_id: str,
    revision: str,
    source_kind: str = "stock",
    upstream_export_manifest: str | Path | None = None,
    assistant_revision: str | None = None,
    coreml_package: str | Path | None = None,
) -> dict[str, Any]:
    """Validate an export, copy it atomically, and return its canonical manifest."""

    source = Path(model_source).resolve()
    destination = Path(output).resolve()
    if source_kind not in SOURCE_KINDS:
        raise PackageValidationError(f"unsupported source kind: {source_kind}")
    if not model_id.strip() or not revision.strip():
        raise PackageValidationError("model_id and revision must be non-empty")
    if destination.exists():
        raise PackageValidationError(f"output already exists: {destination}")
    files = _source_files(source)
    config = _json_object(source / "config.json")
    quantization = validate_gemma4_e4b_config(config, source_kind)
    tensors = tensor_inventory(source)
    w6_layout = validate_w6_tensor_layout(tensors["entries"]) if quantization is not None else None

    upstream = Path(upstream_export_manifest).resolve() if upstream_export_manifest else None
    if source_kind == "merged_sft" and (upstream is None or not upstream.is_file()):
        raise PackageValidationError("merged SFT input requires --upstream-export-manifest")
    if upstream is not None and not upstream.is_file():
        raise PackageValidationError(f"upstream export manifest does not exist: {upstream}")
    coreml = Path(coreml_package).resolve() if coreml_package else None
    if coreml is not None and not coreml.exists():
        raise PackageValidationError(f"Core ML package does not exist: {coreml}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        model_destination = temporary / "model"
        model_destination.mkdir()
        for path in files:
            shutil.copyfile(path, model_destination / path.name)
        metadata_destination = temporary / "metadata"
        metadata_destination.mkdir()
        (metadata_destination / "tensor-index.tsv").write_bytes(tensor_index_bytes(tensors["entries"]))
        if upstream is not None:
            _copy_tree(upstream, metadata_destination / "upstream-export-manifest.json")
        if coreml is not None:
            suffix = coreml.suffix if coreml.is_file() else ".mlpackage"
            _copy_tree(coreml, temporary / f"vision{suffix}")

        payload = _payload_entries(temporary)
        coreml_entries = [entry for entry in payload if entry["path"].startswith("vision")]
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "engine_abi": {"major": ENGINE_ABI_MAJOR},
            "source": {
                "kind": source_kind,
                "model_id": model_id,
                "revision": revision,
                "upstream_export_manifest_sha256": sha256_file(upstream) if upstream else None,
            },
            "model": {
                "geometry": EXPECTED_GEOMETRY,
                "config_sha256": sha256_file(source / "config.json"),
                "quantization": quantization,
                "w6_layout": w6_layout,
                "tensors": tensors,
            },
            "processor": {
                "processor_config_sha256": _hash_if_present(source, "processor_config.json"),
                "tokenizer_sha256": _hash_if_present(source, "tokenizer.json"),
                "tokenizer_config_sha256": _hash_if_present(source, "tokenizer_config.json"),
                "chat_template_sha256": _hash_if_present(source, "chat_template.jinja"),
            },
            "assistant": {"revision": assistant_revision} if assistant_revision else None,
            "coreml": (
                {
                    "content_sha256": sha256_entries(coreml_entries),
                    "files": len(coreml_entries),
                }
                if coreml
                else None
            ),
            "payload": {"files": payload, "content_sha256": sha256_entries(payload)},
        }
        validate_manifest(manifest)
        (temporary / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def verify_runtime_package(package: str | Path) -> dict[str, Any]:
    """Fail closed if package metadata, inventory, or payload bytes drifted."""

    root = Path(package).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise PackageValidationError("runtime package manifest is missing")
    manifest = _json_object(manifest_path)
    validate_manifest(manifest)
    expected = manifest["payload"]["files"]
    actual = _payload_entries(root)
    if actual != expected:
        raise PackageValidationError("runtime package payload files or checksums do not match manifest")
    if sha256_entries(actual) != manifest["payload"]["content_sha256"]:
        raise PackageValidationError("runtime package content digest does not match payload")
    return manifest
