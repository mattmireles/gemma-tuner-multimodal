"""Exact-shape Core ML vision tower for the Plan 32 MLX decoder comparison."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import coremltools as ct
import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class VerifiedANEPackage:
    shape: tuple[int, int]
    directory: Path
    package: Path
    receipt_path: Path
    receipt: dict[str, Any]


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path for path in root.rglob("*") if path.is_file()):
        digest.update(str(item.relative_to(root)).encode())
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_ane_packages(root: Path, bf16_conversion: Path) -> dict[tuple[int, int], VerifiedANEPackage]:
    conversion = json.loads(bf16_conversion.read_text(encoding="utf-8"))
    expected_revision = conversion["conversion_identity_sha256"]
    expected_adapter = conversion["adapter"]["weights_sha256"]
    packages: dict[tuple[int, int], VerifiedANEPackage] = {}
    for directory in sorted(root.glob("plan32-cp616-ane-*")):
        receipt_path = directory / "export-receipt.json"
        if not receipt_path.is_file():
            continue
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt["model_revision"] != expected_revision
            or receipt["adapter_weights_sha256"] != expected_adapter
            or receipt["coreml_vs_torch_cosine"] < 0.995
            or receipt["mlx_vs_torch_cosine"] < 0.999
        ):
            raise ValueError(f"ANE export provenance or parity failed: {directory}")
        shape = tuple(receipt["input_shape"][-2:])
        package = directory / receipt["package"]
        if tree_sha256(package) != receipt["package_tree_sha256"]:
            raise ValueError(f"ANE package hash mismatch: {package}")
        if shape in packages:
            raise ValueError(f"duplicate ANE shape {shape}")
        packages[shape] = VerifiedANEPackage(shape, directory, package, receipt_path, receipt)
    if not packages:
        raise ValueError("no verified fine-tuned ANE vision packages")
    return packages


def parse_swap_used_mb(value: str) -> float:
    match = re.search(r"used\s*=\s*([0-9.]+)([KMG])", value)
    if not match:
        raise ValueError(f"unrecognized vm.swapusage output: {value.strip()}")
    amount = float(match.group(1))
    return amount * {"K": 1 / 1024, "M": 1, "G": 1024}[match.group(2)]


def parse_free_memory_percent(value: str) -> float:
    match = re.search(r"System-wide memory free percentage:\s*([0-9.]+)%", value)
    if not match:
        raise ValueError(f"unrecognized memory_pressure output: {value.strip()}")
    return float(match.group(1))


def host_pressure_snapshot() -> dict[str, float]:
    if platform.system() != "Darwin":
        raise RuntimeError("ANE host-pressure checks require macOS")
    swap = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=True
    )
    memory = subprocess.run(
        ["/usr/bin/memory_pressure", "-Q"], capture_output=True, text=True, check=True
    )
    return {
        "swap_used_mb": parse_swap_used_mb(swap.stdout),
        "free_memory_percent": parse_free_memory_percent(memory.stdout),
    }


def assert_safe_host_pressure(
    *, max_swap_used_mb: float, min_free_memory_percent: float
) -> dict[str, float]:
    snapshot = host_pressure_snapshot()
    if snapshot["swap_used_mb"] > max_swap_used_mb:
        raise RuntimeError(
            f"unsafe ANE host pressure: swap {snapshot['swap_used_mb']:.1f} MiB exceeds "
            f"{max_swap_used_mb:.1f} MiB"
        )
    if snapshot["free_memory_percent"] < min_free_memory_percent:
        raise RuntimeError(
            f"unsafe ANE host pressure: free memory {snapshot['free_memory_percent']:.1f}% is below "
            f"{min_free_memory_percent:.1f}%"
        )
    return snapshot


def pressure_violation(
    snapshot: dict[str, float],
    baseline: dict[str, float],
    *,
    max_swap_used_mb: float,
    max_swap_growth_mb: float,
    min_free_memory_percent: float,
) -> str | None:
    if snapshot["swap_used_mb"] > max_swap_used_mb:
        return (
            f"swap {snapshot['swap_used_mb']:.1f} MiB exceeds the absolute "
            f"{max_swap_used_mb:.1f} MiB ceiling"
        )
    growth = snapshot["swap_used_mb"] - baseline["swap_used_mb"]
    if growth > max_swap_growth_mb:
        return f"swap grew {growth:.1f} MiB beyond the {max_swap_growth_mb:.1f} MiB run allowance"
    if snapshot["free_memory_percent"] < min_free_memory_percent:
        return (
            f"free memory {snapshot['free_memory_percent']:.1f}% is below "
            f"{min_free_memory_percent:.1f}%"
        )
    return None


def compiled_artifact_paths(
    compiled_root: Path, source: VerifiedANEPackage
) -> tuple[Path, Path]:
    identity = source.receipt["package_tree_sha256"][:16]
    stem = f"{source.directory.name}-{identity}"
    return compiled_root / f"{stem}.mlmodelc", compiled_root / f"{stem}.json"


def current_host_identity() -> dict[str, str]:
    hardware = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "hw.model"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return {
        "hardware_model": hardware,
        "machine": platform.machine(),
        "os_version": platform.mac_ver()[0],
        "python_executable": str(Path(sys.executable).resolve()),
    }


def verify_compiled_artifact(compiled_root: Path, source: VerifiedANEPackage) -> Path:
    package, receipt_path = compiled_artifact_paths(compiled_root, source)
    if not package.is_dir() or not receipt_path.is_file():
        raise ValueError(f"missing target-local compiled ANE artifact for shape {source.shape}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "plan32_target_compiled_ane_v1",
        "shape": list(source.shape),
        "source_export_receipt_sha256": file_sha256(source.receipt_path),
        "source_package_tree_sha256": source.receipt["package_tree_sha256"],
        "compiled_tree_sha256": tree_sha256(package),
        "target": current_host_identity(),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"compiled ANE artifact identity mismatch for {source.shape}: {key}")
    return package


class ANEVisionTower:
    def __init__(
        self,
        root: Path,
        bf16_conversion: Path,
        reference_tower,
        parity_out: Path,
        *,
        compiled_root: Path | None = None,
        max_loaded_models: int = 1,
        max_swap_used_mb: float = 4096,
        min_free_memory_percent: float = 20,
    ) -> None:
        if max_loaded_models != 1:
            raise ValueError("Plan 32 M1 qualification permits exactly one resident ANE model")
        self.packages = discover_ane_packages(root, bf16_conversion)
        self.models: OrderedDict[tuple[int, int], Any] = OrderedDict()
        self.compiled_root = compiled_root
        self.max_loaded_models = max_loaded_models
        self.max_swap_used_mb = max_swap_used_mb
        self.min_free_memory_percent = min_free_memory_percent
        self.reference_tower = reference_tower
        self.parity_out = parity_out
        self.checked_shapes = set()

    def _model_for_shape(self, shape: tuple[int, int]):
        if shape in self.models:
            model = self.models.pop(shape)
            self.models[shape] = model
            return model
        if shape not in self.packages:
            raise ValueError(f"missing exact-shape fine-tuned ANE package {shape}")
        assert_safe_host_pressure(
            max_swap_used_mb=self.max_swap_used_mb,
            min_free_memory_percent=self.min_free_memory_percent,
        )
        while len(self.models) >= self.max_loaded_models:
            _, evicted = self.models.popitem(last=False)
            del evicted
            gc.collect()
        source = self.packages[shape]
        if self.compiled_root is None:
            model = ct.models.MLModel(
                str(source.package), compute_units=ct.ComputeUnit.CPU_AND_NE
            )
        else:
            compiled = verify_compiled_artifact(self.compiled_root, source)
            model = ct.models.CompiledMLModel(
                str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE
            )
        self.models[shape] = model
        return model

    def __call__(self, pixel_values, pixel_position_ids=None):
        if pixel_position_ids is not None:
            raise ValueError("ANE vision only supports complete screenshot pixel tensors")
        if isinstance(pixel_values, list):
            pieces = [self(image)[0] for image in pixel_values]
            return mx.concatenate(pieces, axis=0)[None]
        reference_pixels = pixel_values if isinstance(pixel_values, mx.array) else mx.array(pixel_values)
        pixels = np.asarray(pixel_values).astype(np.float16)
        if pixels.ndim == 3:
            pixels = pixels[None]
            reference_pixels = reference_pixels[None]
        if pixels.ndim != 4 or pixels.shape[1] != 3:
            raise ValueError(f"unsupported ANE vision input shape {pixels.shape}")
        shape = tuple(pixels.shape[-2:])
        model = self._model_for_shape(shape)
        pieces = [
            model.predict({"pixels": pixels[index : index + 1]})["hidden_states"][0]
            for index in range(pixels.shape[0])
        ]
        output = np.concatenate(pieces, axis=0)[None]
        if not np.isfinite(output).all():
            raise ValueError("non-finite ANE vision output")
        if shape not in self.checked_shapes:
            reference = np.asarray(
                self.reference_tower(reference_pixels).astype(mx.float32)
            )
            left = output.astype(np.float64).ravel()
            right = reference.astype(np.float64).ravel()
            similarity = float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
            if similarity < 0.995 or reference.shape != output.shape:
                raise ValueError(f"real-image ANE/MLX vision parity failed: {shape}, {similarity}")
            self.parity_out.parent.mkdir(parents=True, exist_ok=True)
            with self.parity_out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"shape": shape, "cosine": similarity}) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.checked_shapes.add(shape)
        return mx.array(output)
