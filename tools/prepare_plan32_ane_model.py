"""Compile and prime exactly one Plan 32 ANE vision shape on the target Mac.

Run this before loading the MLX decoder. The one-shape process is intentional:
it prevents Core ML compilation from competing with decoder residency and makes
every completed target-local artifact independently resumable.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
import time
from pathlib import Path

import coremltools as ct
import numpy as np

from tools.plan32_ane_vision import (
    assert_safe_host_pressure,
    compiled_artifact_paths,
    current_host_identity,
    discover_ane_packages,
    file_sha256,
    tree_sha256,
    verify_compiled_artifact,
)


def canonical(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def parse_shape(value: str) -> tuple[int, int]:
    pieces = value.lower().split("x")
    if len(pieces) != 2 or not all(piece.isdigit() for piece in pieces):
        raise argparse.ArgumentTypeError("shape must be HEIGHTxWIDTH")
    shape = tuple(int(piece) for piece in pieces)
    if min(shape) <= 0:
        raise argparse.ArgumentTypeError("shape axes must be positive")
    return shape


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(canonical(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ane-vision-root", type=Path, required=True)
    parser.add_argument("--ane-source-conversion", type=Path, required=True)
    parser.add_argument("--compiled-root", type=Path, required=True)
    parser.add_argument("--shape", type=parse_shape, required=True)
    parser.add_argument("--max-swap-used-mb", type=float, default=512)
    parser.add_argument("--min-free-memory-percent", type=float, default=35)
    args = parser.parse_args()

    pressure_before = assert_safe_host_pressure(
        max_swap_used_mb=args.max_swap_used_mb,
        min_free_memory_percent=args.min_free_memory_percent,
    )
    packages = discover_ane_packages(
        args.ane_vision_root.resolve(), args.ane_source_conversion.resolve()
    )
    if args.shape not in packages:
        raise ValueError(f"no verified ANE source package for shape {args.shape}")
    source = packages[args.shape]
    compiled_root = args.compiled_root.resolve()
    compiled_root.mkdir(parents=True, exist_ok=True)
    compiled, receipt_path = compiled_artifact_paths(compiled_root, source)

    if compiled.exists() and receipt_path.exists():
        verify_compiled_artifact(compiled_root, source)
        print(canonical({"event": "already_prepared", "shape": args.shape, "path": str(compiled)}))
        return
    if receipt_path.exists() and not compiled.exists():
        raise ValueError(f"compiled receipt exists without its model: {receipt_path}")

    compile_seconds = 0.0
    if not compiled.exists():
        with tempfile.TemporaryDirectory(prefix="plan32-ane-compile-", dir=compiled_root) as temp:
            staged = Path(temp) / "model.mlmodelc"
            started = time.monotonic()
            ct.models.utils.compile_model(str(source.package), destination_path=str(staged))
            compile_seconds = time.monotonic() - started
            if not staged.is_dir():
                raise RuntimeError("Core ML compile did not produce a compiled model directory")
            os.replace(staged, compiled)

    pressure_before_prime = assert_safe_host_pressure(
        max_swap_used_mb=args.max_swap_used_mb,
        min_free_memory_percent=args.min_free_memory_percent,
    )
    started = time.monotonic()
    model = ct.models.CompiledMLModel(
        str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    output = model.predict(
        {"pixels": np.zeros((1, 3, *args.shape), dtype=np.float16)}
    )["hidden_states"]
    prime_seconds = time.monotonic() - started
    if tuple(output.shape) != tuple(source.receipt["output_shape"]) or not np.isfinite(output).all():
        raise ValueError("target-local compiled ANE smoke output is invalid")
    del output, model
    gc.collect()
    pressure_after = assert_safe_host_pressure(
        max_swap_used_mb=args.max_swap_used_mb,
        min_free_memory_percent=args.min_free_memory_percent,
    )

    receipt = {
        "schema_version": "plan32_target_compiled_ane_v1",
        "shape": list(args.shape),
        "source_export_receipt_sha256": file_sha256(source.receipt_path),
        "source_package_tree_sha256": source.receipt["package_tree_sha256"],
        "compiled_tree_sha256": tree_sha256(compiled),
        "target": current_host_identity(),
        "coremltools_version": ct.__version__,
        "compute_units": "CPU_AND_NE",
        "compile_seconds": compile_seconds,
        "prime_seconds": prime_seconds,
        "pressure_before": pressure_before,
        "pressure_before_prime": pressure_before_prime,
        "pressure_after": pressure_after,
    }
    atomic_json(receipt_path, receipt)
    verify_compiled_artifact(compiled_root, source)
    print(canonical({"event": "prepared", "shape": args.shape, "receipt": receipt}))


if __name__ == "__main__":
    main()
