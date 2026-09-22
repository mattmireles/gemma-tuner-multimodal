"""Schema and validation for deterministic Gemma 4 E4B runtime packages.

The native engine must reject incompatible model geometry before it maps a
weight buffer.  This module deliberately depends only on the Python standard
library so packaging can run without importing PyTorch, Transformers, or MLX.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = "gemma4-e4b-runtime-package-v1"
ENGINE_ABI_MAJOR = 1
TENSOR_INDEX_VERSION = "gemma4-tensor-index-v1"
SOURCE_KINDS = frozenset({"stock", "merged_sft"})
W6_RECIPE = {"bits": 6, "group_size": 64, "mode": "affine"}

EXPECTED_GEOMETRY: dict[str, Any] = {
    "model_type": "gemma4",
    "architectures": ["Gemma4ForConditionalGeneration"],
    "vision_soft_tokens_per_image": 280,
    "text_config.model_type": "gemma4_text",
    "text_config.num_hidden_layers": 42,
    "text_config.hidden_size": 2560,
    "text_config.intermediate_size": 10240,
    "text_config.vocab_size": 262144,
    "text_config.num_attention_heads": 8,
    "text_config.num_key_value_heads": 2,
    "text_config.num_kv_shared_layers": 18,
    "text_config.head_dim": 256,
    "text_config.global_head_dim": 512,
    "text_config.sliding_window": 512,
    "text_config.hidden_size_per_layer_input": 256,
    "text_config.vocab_size_per_layer_input": 262144,
    "text_config.hidden_activation": "gelu_pytorch_tanh",
    "text_config.tie_word_embeddings": True,
    "vision_config.model_type": "gemma4_vision",
    "vision_config.num_hidden_layers": 16,
    "vision_config.hidden_size": 768,
    "vision_config.intermediate_size": 3072,
    "vision_config.num_attention_heads": 12,
    "vision_config.num_key_value_heads": 12,
    "vision_config.head_dim": 64,
    "vision_config.global_head_dim": 64,
    "vision_config.patch_size": 16,
    "vision_config.default_output_length": 280,
}

_DTYPE_BYTES: dict[str, float] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
    "I4": 0.5,
    "U4": 0.5,
}


class PackageValidationError(ValueError):
    """Raised when package input is incompatible, ambiguous, or corrupt."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical JSON encoding used in manifests and hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_entries(entries: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_json_bytes(entries)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackageValidationError(f"invalid JSON file {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise PackageValidationError(f"JSON root must be an object: {path.name}")
    return value


def _nested(config: dict[str, Any], dotted: str) -> Any:
    value: Any = config
    for component in dotted.split("."):
        if not isinstance(value, dict) or component not in value:
            raise PackageValidationError(f"missing Gemma geometry field: {dotted}")
        value = value[component]
    return value


def validate_gemma4_e4b_config(config: dict[str, Any], source_kind: str) -> dict[str, Any] | None:
    """Validate exact text/vision geometry and return normalized quantization."""

    if source_kind not in SOURCE_KINDS:
        raise PackageValidationError(f"unsupported source kind: {source_kind}")
    mismatches = []
    for field, expected in EXPECTED_GEOMETRY.items():
        actual = _nested(config, field)
        if actual != expected:
            mismatches.append(f"{field}={actual!r}, expected {expected!r}")
    layer_types = _nested(config, "text_config.layer_types")
    expected_layers = ["full_attention" if (index + 1) % 6 == 0 else "sliding_attention" for index in range(42)]
    if layer_types != expected_layers:
        mismatches.append("text_config.layer_types does not match the E4B five-sliding/one-full schedule")
    if mismatches:
        raise PackageValidationError("incompatible Gemma 4 E4B geometry: " + "; ".join(mismatches))

    quantization = config.get("quantization_config", config.get("quantization"))
    if quantization is None:
        if source_kind == "stock":
            raise PackageValidationError("stock runtime input must declare the frozen W6 quantization recipe")
        return None
    normalized = {key: quantization.get(key) for key in W6_RECIPE}
    if normalized != W6_RECIPE:
        raise PackageValidationError(f"unsupported quantization recipe: {normalized!r}; expected {W6_RECIPE!r}")
    return normalized


def safe_package_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise PackageValidationError(f"unsafe package path: {value!r}")
    return path.as_posix()


def inspect_safetensors(path: Path) -> list[dict[str, Any]]:
    """Inspect a SafeTensors header without loading tensor payloads."""

    size = path.stat().st_size
    if size < 10:
        raise PackageValidationError(f"truncated SafeTensors file: {path.name}")
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length > 128 * 1024 * 1024 or header_length > size - 8:
            raise PackageValidationError(f"invalid SafeTensors header length: {path.name}")
        try:
            header = json.loads(handle.read(header_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PackageValidationError(f"invalid SafeTensors header: {path.name}") from exc
    if not isinstance(header, dict):
        raise PackageValidationError(f"SafeTensors header must be an object: {path.name}")

    data_start = 8 + header_length
    data_bytes = size - data_start
    intervals: list[tuple[int, int, str]] = []
    tensors: list[dict[str, Any]] = []
    for name, descriptor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not name or not isinstance(descriptor, dict):
            raise PackageValidationError(f"invalid tensor descriptor in {path.name}")
        dtype = descriptor.get("dtype")
        shape = descriptor.get("shape")
        offsets = descriptor.get("data_offsets")
        if dtype not in _DTYPE_BYTES:
            raise PackageValidationError(f"unsupported dtype {dtype!r} for tensor {name}")
        if not isinstance(shape, list) or any(not isinstance(dim, int) or dim < 0 for dim in shape):
            raise PackageValidationError(f"invalid shape for tensor {name}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(offset, int) for offset in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or offsets[1] > data_bytes
        ):
            raise PackageValidationError(f"invalid data offsets for tensor {name}")
        elements = 1
        for dimension in shape:
            elements *= dimension
        expected_bytes = int(elements * _DTYPE_BYTES[dtype])
        if offsets[1] - offsets[0] != expected_bytes:
            raise PackageValidationError(
                f"tensor byte size mismatch for {name}: {offsets[1] - offsets[0]} != {expected_bytes}"
            )
        intervals.append((offsets[0], offsets[1], name))
        tensors.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": shape,
                "shard": path.name,
                "file_offset": data_start + offsets[0],
                "byte_length": expected_bytes,
            }
        )
    for previous, current in zip(sorted(intervals), sorted(intervals)[1:]):
        if current[0] < previous[1]:
            raise PackageValidationError(f"overlapping tensor data in {path.name}: {previous[2]} and {current[2]}")
    return sorted(tensors, key=lambda tensor: tensor["name"])


def tensor_index_bytes(entries: list[dict[str, Any]], *, path_prefix: str = "model") -> bytes:
    """Encode the native tensor index without requiring a JSON parser in C++.

    Runtime packages keep shards below ``model/``. Reference tooling may pass
    an empty prefix so the same native reader can map an existing SafeTensors
    directory without copying multi-gigabyte checkpoints.
    """

    lines = [TENSOR_INDEX_VERSION]
    for tensor in sorted(entries, key=lambda value: value["name"]):
        name = tensor.get("name")
        dtype = tensor.get("dtype")
        shard = tensor.get("shard")
        shape = tensor.get("shape")
        file_offset = tensor.get("file_offset")
        byte_length = tensor.get("byte_length")
        if not isinstance(name, str) or not name or any(character in name for character in "\t\r\n"):
            raise PackageValidationError("tensor index contains an unsafe tensor name")
        if not isinstance(dtype, str) or dtype not in _DTYPE_BYTES:
            raise PackageValidationError(f"tensor index contains an invalid dtype for {name}")
        if not isinstance(shard, str) or any(character in shard for character in "\t\r\n"):
            raise PackageValidationError(f"tensor index contains an unsafe shard for {name}")
        relative_path = safe_package_path(f"{path_prefix}/{shard}" if path_prefix else shard)
        if not isinstance(shape, list) or any(not isinstance(dimension, int) or dimension < 0 for dimension in shape):
            raise PackageValidationError(f"tensor index contains an invalid shape for {name}")
        if not isinstance(file_offset, int) or file_offset < 0:
            raise PackageValidationError(f"tensor index contains an invalid file offset for {name}")
        if not isinstance(byte_length, int) or byte_length < 0:
            raise PackageValidationError(f"tensor index contains an invalid byte length for {name}")
        dimensions = ",".join(str(dimension) for dimension in shape)
        lines.append(
            "\t".join(
                (
                    name,
                    dtype,
                    relative_path,
                    str(file_offset),
                    str(byte_length),
                    str(len(shape)),
                    dimensions,
                )
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def tensor_inventory(model_root: Path) -> dict[str, Any]:
    """Return the full deterministic tensor inventory and verify the shard index."""

    shards = sorted(model_root.glob("*.safetensors"))
    if not shards:
        raise PackageValidationError("model export contains no SafeTensors files")
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for shard in shards:
        for tensor in inspect_safetensors(shard):
            if tensor["name"] in seen:
                raise PackageValidationError(f"duplicate tensor name across shards: {tensor['name']}")
            seen.add(tensor["name"])
            entries.append(tensor)
    entries.sort(key=lambda tensor: tensor["name"])

    index_path = model_root / "model.safetensors.index.json"
    if index_path.exists():
        index = _read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or set(weight_map) != seen:
            raise PackageValidationError("SafeTensors index tensor inventory does not match shard headers")
        for tensor in entries:
            if weight_map[tensor["name"]] != tensor["shard"]:
                raise PackageValidationError(f"SafeTensors index maps {tensor['name']} to the wrong shard")

    dtypes = Counter(tensor["dtype"] for tensor in entries)
    return {
        "count": len(entries),
        "dtypes": dict(sorted(dtypes.items())),
        "entries": entries,
    }


def validate_w6_tensor_layout(entries: list[dict[str, Any]]) -> dict[str, int]:
    """Validate MLX affine W6 triples without loading multi-gigabyte payloads."""

    by_name = {str(entry["name"]): entry for entry in entries}
    quantized = 0
    for name, weight in by_name.items():
        if weight["dtype"] != "U32":
            continue
        if not name.endswith(".weight"):
            raise PackageValidationError(f"W6 U32 tensor is not a weight: {name}")
        shape = weight["shape"]
        if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
            raise PackageValidationError(f"W6 packed weight must be rank two: {name}")
        packed_columns = shape[1]
        packed_bits = packed_columns * 32
        if packed_bits % W6_RECIPE["bits"]:
            raise PackageValidationError(f"W6 packed width is not divisible by six bits: {name}")
        columns = packed_bits // W6_RECIPE["bits"]
        if columns % W6_RECIPE["group_size"]:
            raise PackageValidationError(f"W6 logical width is not group-aligned: {name}")
        expected_aux_shape = [shape[0], columns // W6_RECIPE["group_size"]]
        base = name[: -len(".weight")]
        for suffix in ("scales", "biases"):
            auxiliary_name = f"{base}.{suffix}"
            auxiliary = by_name.get(auxiliary_name)
            if auxiliary is None:
                raise PackageValidationError(f"W6 tensor is missing {auxiliary_name}")
            if auxiliary["dtype"] != "BF16" or auxiliary["shape"] != expected_aux_shape:
                raise PackageValidationError(
                    f"W6 auxiliary tensor has the wrong dtype or shape: {auxiliary_name}"
                )
        quantized += 1
    return {"quantized_tensors": quantized, "bits": 6, "group_size": 64}


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise PackageValidationError("unsupported runtime package schema")
    if manifest.get("engine_abi") != {"major": ENGINE_ABI_MAJOR}:
        raise PackageValidationError("incompatible native engine ABI")
    source = manifest.get("source")
    if not isinstance(source, dict) or source.get("kind") not in SOURCE_KINDS:
        raise PackageValidationError("invalid package source metadata")
    if source["kind"] == "merged_sft" and not source.get("upstream_export_manifest_sha256"):
        raise PackageValidationError("merged SFT package requires an upstream export manifest hash")
    payload = manifest.get("payload")
    if not isinstance(payload, dict) or not isinstance(payload.get("files"), list):
        raise PackageValidationError("package payload file inventory is missing")
    for entry in payload["files"]:
        if not isinstance(entry, dict):
            raise PackageValidationError("invalid payload entry")
        safe_package_path(str(entry.get("path", "")))
        if not isinstance(entry.get("bytes"), int) or entry["bytes"] < 0:
            raise PackageValidationError("invalid payload byte count")
        digest = entry.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise PackageValidationError("invalid payload SHA-256")
    if payload.get("content_sha256") != sha256_entries(payload["files"]):
        raise PackageValidationError("payload content digest does not match file inventory")
