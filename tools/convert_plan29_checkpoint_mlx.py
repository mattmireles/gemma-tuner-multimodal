#!/usr/bin/env python3
"""Fuse a sealed Plan 29 or corrected Plan 30 PEFT checkpoint into MLX BF16.

The conversion is deliberately streaming.  The pinned MLX snapshot is a
bit-identical BF16 conversion of the pinned HF source for every
cached-generation-active LoRA target.  Each source shard is loaded, its active
targeted linear weights are replaced by the PEFT merge result, and the complete
shard is written once. Plan 29's 36 final-layer K/V adapters that cached HF and
MLX generation both bypass are accounted for explicitly; corrected Plan 30
does not train those adapters. This matches the HF cached-inference merge while
avoiding a second 30-GB intermediate model on the target Mac.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import struct
import subprocess
from pathlib import Path
from typing import Any

HF_REPO_ID = "google/gemma-4-E4B-it"
HF_REVISION = "fee6332c1abaafb77f6f9624236c63aa2f1d0187"
MLX_REPO_ID = "mlx-community/gemma-4-e4b-it-bf16"
MLX_REVISION = "eec12d0899edea9b738ab1009af9159cdfd70d71"
EXPECTED_TARGETS = {
    "plan29": {"language": 294, "vision": 112, "projection": 1},
    "plan30-corrected": {"language": 258, "vision": 112, "projection": 1},
    # Plan 33 stock-E2B lineage: 35 layers, 20 shared-K/V layers own no K/V.
    "plan33-e2b": {"language": 205, "vision": 112, "projection": 1},
}
# Pinned HF source and its bit-identical MLX BF16 conversion, per profile.
SOURCES = {
    "plan29": (HF_REPO_ID, HF_REVISION, MLX_REPO_ID, MLX_REVISION),
    "plan30-corrected": (HF_REPO_ID, HF_REVISION, MLX_REPO_ID, MLX_REVISION),
    "plan33-e2b": ("google/gemma-4-E2B-it", "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
                   "mlx-community/gemma-4-e2b-it-bf16", "fb0b166bbb9a0eb4b37915bfc515a197c9122f39"),
}
SHARED_KV_PATTERN = re.compile(
    r"^base_model\.model\.model\.language_model\.layers\.(\d+)\.self_attn\.([kv]_proj)$"
)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_module_names(adapter_keys: list[str]) -> list[str]:
    suffix = ".lora_A.weight"
    modules = sorted(key[: -len(suffix)] for key in adapter_keys if key.endswith(suffix))
    expected = {f"{module}.lora_B.weight" for module in modules}
    actual = {key for key in adapter_keys if key.endswith(".lora_B.weight")}
    if not modules or expected != actual or len(adapter_keys) != 2 * len(modules):
        raise ValueError("adapter must contain exactly one LoRA A/B pair per target")
    return modules


def classify_module(module: str) -> str:
    if ".language_model." in module:
        return "language"
    if ".vision_tower." in module:
        return "vision"
    if module.endswith(".embed_vision.embedding_projection"):
        return "projection"
    raise ValueError(f"unexpected LoRA target: {module}")


def mapped_weight_keys(module: str) -> tuple[str, str]:
    prefix = "base_model.model.model."
    if not module.startswith(prefix):
        raise ValueError(f"unexpected adapter prefix: {module}")
    core = module.removeprefix(prefix)
    if core.startswith("language_model."):
        hf_key = f"model.{core}.weight"
        mlx_key = f"{core.replace('language_model.', 'language_model.model.', 1)}.weight"
    elif core.startswith("vision_tower."):
        hf_key = f"model.{core}.linear.weight"
        mlx_key = f"{core}.linear.weight"
    elif core == "embed_vision.embedding_projection":
        hf_key = f"model.{core}.weight"
        mlx_key = f"{core}.weight"
    else:
        raise ValueError(f"unexpected adapter target: {module}")
    return hf_key, mlx_key


def inference_inactive_shared_kv_modules(
    base_config: dict[str, Any], modules: list[str], *, profile: str = "plan29"
) -> set[str]:
    """Return adapted K/V modules omitted by Gemma 4 cached inference.

    Plan 29 trained the final ``num_kv_shared_layers`` K/V modules but cached
    inference reuses earlier K/V states; those adapters are inference-inactive.
    The corrected Plan 30 trainer never adapted these modules. Both exact
    profiles are checked rather than silently dropping targets.
    """
    if profile not in EXPECTED_TARGETS:
        raise ValueError(f"unknown conversion profile: {profile}")
    text_config = base_config.get("text_config", {})
    layer_count = int(text_config.get("num_hidden_layers", 0))
    shared_count = int(text_config.get("num_kv_shared_layers", 0))
    if layer_count <= 0 or shared_count <= 0 or shared_count >= layer_count:
        raise ValueError("invalid Gemma 4 shared-KV configuration")
    first_shared = layer_count - shared_count
    inactive = set()
    for module in modules:
        match = SHARED_KV_PATTERN.fullmatch(module)
        if match and int(match.group(1)) >= first_shared:
            inactive.add(module)
    expected = shared_count * 2 if profile == "plan29" else 0
    if len(inactive) != expected:
        raise ValueError(
            f"shared-KV inactive target mismatch: {len(inactive)} != {expected}"
        )
    return inactive


def validate_adapter_config(config: dict[str, Any], *, profile: str = "plan29") -> float:
    required = {
        "peft_type": "LORA",
        "r": 64,
        "lora_alpha": 128,
        "bias": "none",
        "use_dora": False,
        "use_rslora": False,
        "base_model_name_or_path": SOURCES[profile][0],
    }
    mismatches = {key: (config.get(key), value) for key, value in required.items()
                  if config.get(key) != value}
    if mismatches:
        raise ValueError(f"adapter config mismatch: {mismatches}")
    return float(config["lora_alpha"]) / int(config["r"])


def read_index(model_path: Path) -> dict[str, str]:
    index_path = model_path / "model.safetensors.index.json"
    data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("invalid MLX weight index")
    return {str(key): str(value) for key, value in weight_map.items()}


def safetensor_byte_ranges(path: Path) -> dict[str, tuple[int, int, str]]:
    """Return absolute data byte ranges without materializing tensor payloads."""
    with path.open("rb") as handle:
        header_size_bytes = handle.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError(f"invalid safetensors header: {path}")
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        header = json.loads(handle.read(header_size))
    data_start = 8 + header_size
    ranges = {}
    for key, value in header.items():
        if key == "__metadata__":
            continue
        start, end = value["data_offsets"]
        ranges[key] = (data_start + int(start), data_start + int(end), value["dtype"])
    return ranges


def clone_file(source: Path, destination: Path) -> None:
    """Create an APFS copy-on-write clone, failing rather than doing a full copy."""
    result = subprocess.run(
        ["cp", "-c", str(source.resolve()), str(destination)],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"APFS clone failed for {source.name}: {result.stderr.strip()}")
    destination.chmod(destination.stat().st_mode | stat.S_IWUSR)


def runtime_versions() -> dict[str, str]:
    import importlib.metadata
    import platform
    import sys

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "mlx": importlib.metadata.version("mlx"),
        "mlx_vlm": importlib.metadata.version("mlx-vlm"),
        "safetensors": importlib.metadata.version("safetensors"),
    }


def copy_model_metadata(source: Path, destination: Path) -> None:
    for path in source.iterdir():
        if path.name.endswith(".safetensors") or path.name == "model.safetensors.index.json":
            continue
        if path.is_file():
            shutil.copy2(path, destination / path.name)


def conversion_identity(*, base: Path, hf_base: Path, adapter: Path,
                        profile: str = "plan29") -> dict[str, Any]:
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    scale = validate_adapter_config(config, profile=profile)
    hf_repo, hf_revision, mlx_repo, mlx_revision = SOURCES[profile]
    return {
        "schema_version": f"{profile}_mlx_conversion_v1",
        "profile": profile,
        "converter_sha256": sha256_file(Path(__file__)),
        "source_hf": {"repo_id": hf_repo, "revision": hf_revision,
                      "weights_sha256": sha256_file(hf_base / "model.safetensors")},
        "source_mlx": {"repo_id": mlx_repo, "revision": mlx_revision,
                       "config_sha256": sha256_file(base / "config.json"),
                       "index_sha256": sha256_file(base / "model.safetensors.index.json")},
        "adapter": {
            "config_sha256": sha256_file(adapter / "adapter_config.json"),
            "weights_sha256": sha256_file(adapter / "adapter_model.safetensors"),
            "complete_sha256": sha256_file(adapter / ".complete.json"),
            "scale": scale,
        },
        "precision": "bfloat16",
        "quantization": None,
        "runtime": runtime_versions(),
    }


def convert(*, base: Path, hf_base: Path, adapter: Path, output: Path,
            profile: str = "plan29") -> dict[str, Any]:
    import mlx.core as mx

    if profile not in EXPECTED_TARGETS:
        raise ValueError(f"unknown conversion profile: {profile}")

    for required in (base / "config.json", base / "model.safetensors.index.json",
                     hf_base / "model.safetensors", adapter / "adapter_config.json",
                     adapter / "adapter_model.safetensors", adapter / ".complete.json"):
        if not required.is_file():
            raise FileNotFoundError(required)
    base_config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    if base_config.get("quantization") is not None:
        raise ValueError("Plan 29 requires an unquantized MLX base")

    identity = conversion_identity(base=base, hf_base=hf_base, adapter=adapter,
                                   profile=profile)
    identity_sha = hashlib.sha256(canonical(identity).encode()).hexdigest()
    receipt_path = output / "conversion-receipt.json"
    if output.exists():
        if receipt_path.is_file():
            prior = json.loads(receipt_path.read_text(encoding="utf-8"))
            if prior.get("conversion_identity_sha256") == identity_sha:
                for name, digest in prior["output"]["files_sha256"].items():
                    if sha256_file(output / name) != digest:
                        raise ValueError(f"completed conversion hash mismatch: {name}")
                return prior
        raise FileExistsError(f"conflicting conversion output: {output}")

    partial = output.with_name(output.name + ".partial")
    state_path = partial / "conversion-state.json"
    if partial.exists():
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
        if state.get("conversion_identity_sha256") != identity_sha:
            raise FileExistsError(f"conflicting partial conversion: {partial}")
    else:
        partial.mkdir(parents=True)
        state_path.write_text(json.dumps({"conversion_identity_sha256": identity_sha,
                                          "completed_shards": []}, indent=2) + "\n")
    copy_model_metadata(base, partial)

    adapter_weights = mx.load(str(adapter / "adapter_model.safetensors"))
    modules = adapter_module_names(list(adapter_weights))
    counts = {kind: 0 for kind in EXPECTED_TARGETS[profile]}
    all_mappings: dict[str, dict[str, str]] = {}
    for module in modules:
        kind = classify_module(module)
        counts[kind] += 1
        hf_key, mlx_key = mapped_weight_keys(module)
        all_mappings[mlx_key] = {"module": module, "hf_key": hf_key}
    if counts != EXPECTED_TARGETS[profile] or len(all_mappings) != sum(EXPECTED_TARGETS[profile].values()):
        raise ValueError(f"LoRA target coverage mismatch: {counts}")

    weight_map = read_index(base)
    inactive_modules = inference_inactive_shared_kv_modules(base_config, modules,
                                                             profile=profile)
    missing_mlx = {key for key in all_mappings if key not in weight_map}
    expected_missing = {
        mapped_weight_keys(module)[1] for module in inactive_modules
    }
    if missing_mlx != expected_missing:
        raise ValueError(
            "MLX target absence does not exactly match cached-inference shared K/V: "
            f"missing={len(missing_mlx)} expected={len(expected_missing)}"
        )
    mappings = {
        key: info for key, info in all_mappings.items() if key not in expected_missing
    }
    hf_weights = mx.load(str(hf_base / "model.safetensors"))
    missing_hf = sorted(info["hf_key"] for info in mappings.values()
                        if info["hf_key"] not in hf_weights)
    if missing_hf:
        raise ValueError(f"{len(missing_hf)} LoRA targets absent from HF base")

    completed = set(json.loads(state_path.read_text())["completed_shards"])
    shard_names = sorted(set(weight_map.values()))
    completed_targets = sum(
        1 for mlx_key in mappings if weight_map[mlx_key] in completed
    )
    base_matches_hf = completed_targets
    merged_targets = completed_targets
    scale = identity["adapter"]["scale"]
    for shard_index, shard_name in enumerate(shard_names, 1):
        if shard_name in completed:
            continue
        source_shard = base / shard_name
        shard_targets = {
            key for key in mappings if weight_map[key] == shard_name
        }
        destination = partial / shard_name
        if not shard_targets:
            # Preserve a standalone normal file while sharing immutable APFS
            # storage with the pinned source blob. This is required on the
            # target Mac, where a second 15-GB copy would exceed free space.
            os.link(source_shard.resolve(), destination)
            completed.add(shard_name)
            state_path.write_text(json.dumps({"conversion_identity_sha256": identity_sha,
                                              "completed_shards": sorted(completed)}, indent=2) + "\n")
            print(canonical({"shard": shard_index, "shards": len(shard_names),
                             "name": shard_name, "storage": "hardlink-untouched"}),
                  flush=True)
            continue
        if destination.exists():
            # A shard absent from completed_shards is an interrupted clone or
            # patch and is never safe to resume in place.
            destination.unlink()
        clone_file(source_shard, destination)
        weights = mx.load(str(source_shard))
        byte_ranges = safetensor_byte_ranges(destination)
        import numpy as np

        with destination.open("r+b", buffering=0) as output_handle:
            for mlx_key in sorted(shard_targets):
                info = mappings[mlx_key]
                module = info["module"]
                hf_weight = hf_weights[info["hf_key"]]
                base_weight = weights[mlx_key]
                if base_weight.dtype != mx.bfloat16 or hf_weight.dtype != mx.bfloat16:
                    raise ValueError(f"non-BF16 base target: {mlx_key}")
                if not bool(mx.array_equal(base_weight, hf_weight)):
                    raise ValueError(f"MLX/HF source target mismatch: {mlx_key}")
                base_matches_hf += 1
                a = adapter_weights[f"{module}.lora_A.weight"]
                b = adapter_weights[f"{module}.lora_B.weight"]
                if tuple(b.shape[:-1] + a.shape[-1:]) != tuple(base_weight.shape):
                    raise ValueError(f"LoRA shape mismatch: {module}")
                merged = (base_weight.astype(mx.float32) + scale * (b @ a)).astype(mx.bfloat16)
                mx.eval(merged)
                start, end, dtype = byte_ranges[mlx_key]
                if dtype != "BF16":
                    raise ValueError(f"non-BF16 target layout: {mlx_key}={dtype}")
                raw = np.asarray(merged.view(mx.uint16)).tobytes(order="C")
                if len(raw) != end - start:
                    raise ValueError(f"merged byte-size mismatch: {mlx_key}")
                output_handle.seek(start)
                if output_handle.write(raw) != len(raw):
                    raise OSError(f"short write while patching {mlx_key}")
                merged_targets += 1
                del raw, merged, base_weight, hf_weight
                mx.clear_cache()
        completed.add(shard_name)
        state_path.write_text(json.dumps({"conversion_identity_sha256": identity_sha,
                                          "completed_shards": sorted(completed)}, indent=2) + "\n")
        print(canonical({"shard": shard_index, "shards": len(shard_names),
                         "name": shard_name, "storage": "apfs-clone-patched"}),
              flush=True)
        mx.clear_cache()

    if merged_targets != len(mappings) or base_matches_hf != len(mappings):
        raise ValueError("conversion did not merge every LoRA target exactly once")

    shutil.copy2(base / "model.safetensors.index.json",
                 partial / "model.safetensors.index.json")
    files = sorted(path for path in partial.iterdir()
                   if path.is_file() and path.name != "conversion-state.json")
    files_sha256 = {path.name: sha256_file(path) for path in files}
    dtype_counts: dict[str, int] = {}
    output_weight_count = 0
    for shard_name in shard_names:
        shard = mx.load(str(partial / shard_name))
        for value in shard.values():
            dtype_counts[str(value.dtype)] = dtype_counts.get(str(value.dtype), 0) + 1
            output_weight_count += 1
    if set(dtype_counts) - {"mlx.core.bfloat16", "mlx.core.float32"}:
        raise ValueError(f"unexpected output dtypes: {dtype_counts}")
    output_config = json.loads((partial / "config.json").read_text())
    if output_config.get("quantization") is not None:
        raise ValueError("converted output is quantized")

    receipt = {
        **identity,
        "conversion_identity_sha256": identity_sha,
        "method": "streaming BF16 LoRA merge into bit-identical pinned MLX conversion",
        "coverage": {
            "counts": counts,
            "adapter_targets": len(all_mappings),
            "mapped_inference_active_targets": len(mappings),
            "mlx_hf_base_matches": len(mappings),
            "inference_inactive_shared_kv_targets": len(inactive_modules),
            "inference_inactive_reason": (
                "HF and MLX cached generation reuse earlier shared K/V states"
                if profile == "plan29" else
                "corrected trainer did not adapt shared K/V projections bypassed by cached inference"
            ),
        },
        "output": {"weight_count": output_weight_count, "dtype_counts": dtype_counts,
                   "files_sha256": files_sha256},
    }
    (partial / "conversion-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    state_path.unlink()
    os.replace(partial, output)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--hf-base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=sorted(EXPECTED_TARGETS), default="plan29")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = convert(base=args.base.resolve(), hf_base=args.hf_base.resolve(),
                      adapter=args.adapter.resolve(), output=args.output.resolve(),
                      profile=args.profile)
    print(canonical({"output": str(args.output.resolve()),
                     "conversion_identity_sha256": receipt["conversion_identity_sha256"],
                     "coverage": receipt["coverage"]}))


if __name__ == "__main__":
    main()
