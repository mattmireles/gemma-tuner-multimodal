#!/usr/bin/env python3
"""Run reproducible E4B W6 matvec shape benchmarks.

The output is a development receipt, not a product-latency or idle-qualified
promotion receipt.  It deliberately records both command-buffer wall time and
Metal's GPU interval so dispatch overhead is not mistaken for kernel work.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any

SHAPES = (
    "language_model.model.layers.0.self_attn.q_proj",
    "language_model.model.layers.0.self_attn.o_proj",
    "language_model.model.layers.0.mlp.gate_proj",
    "language_model.model.layers.0.mlp.down_proj",
    "language_model.model.layers.0.per_layer_projection",
    "language_model.model.layers.0.per_layer_input_gate",
    "language_model.model.embed_tokens",
)

GATHERS = (
    ("token_embedding", "language_model.model.embed_tokens", 6210, 0, 2560, 2560**0.5),
    ("per_layer_embedding_41", "language_model.model.embed_tokens_per_layer", 6210, 41 * 256, 256, 16.0),
)

MATMULS = (
    ("prefill_q_proj", "language_model.model.layers.0.self_attn.q_proj", 6210),
    ("prefill_mlp_gate", "language_model.model.layers.0.mlp.gate_proj", 6210),
    ("prefill_mlp_down", "language_model.model.layers.0.mlp.down_proj", 6210),
)


def run_json(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    if not lines:
        raise RuntimeError(f"command emitted no JSON object: {' '.join(command)}")
    return json.loads(lines[-1])


def build_receipt(args: argparse.Namespace) -> dict[str, Any]:
    host = run_json([str(args.metal_smoke), str(args.metallib)])
    benchmarks = []
    for tensor in SHAPES:
        result = run_json(
            [
                str(args.benchmark),
                str(args.metallib),
                str(args.index),
                str(args.payload_root),
                tensor,
                str(args.warmups),
                str(args.runs),
            ]
        )
        benchmarks.append({"tensor": tensor, **result})
    gathers = []
    for label, tensor, tokens, offset, columns, scale in GATHERS:
        result = run_json(
            [
                str(args.gather_benchmark),
                str(args.metallib),
                str(args.index),
                str(args.payload_root),
                tensor,
                str(tokens),
                str(offset),
                str(columns),
                str(scale),
                str(args.warmups),
                str(args.runs),
            ]
        )
        gathers.append({"operation": label, "tensor": tensor, **result})
    matmuls = []
    for label, tensor, matrix_rows in MATMULS:
        result = run_json(
            [
                str(args.matmul_benchmark),
                str(args.metallib),
                str(args.index),
                str(args.payload_root),
                tensor,
                str(matrix_rows),
                str(args.matmul_warmups),
                str(args.matmul_runs),
            ]
        )
        matmuls.append({"operation": label, "tensor": tensor, **result})
    return {
        "schema_version": "gemma4-e4b-w6-kernel-tuning-v1",
        "recorded_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "status": "development-not-promotion",
        "host": host,
        "kernel": {
            "format": "affine-w6-group64-bf16",
            "values_per_lane": 8,
            "rows_per_simdgroup": 4,
            "simdgroups_per_threadgroup": 2,
            "metal_language": "3.1",
        },
        "protocol": {
            "warmups": args.warmups,
            "runs": args.runs,
            "matmul_warmups": args.matmul_warmups,
            "matmul_runs": args.matmul_runs,
            "comparison": "same-source weights materialized as dense BF16",
            "gpu_time": "MTLCommandBuffer GPUStartTime to GPUEndTime",
            "wall_time": "one command buffer, one dispatch, commit, and completion wait",
            "idle_qualified": False,
        },
        "benchmarks": benchmarks,
        "gathers": gathers,
        "matmuls": matmuls,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=Path("runtime/build/benchmark-w6-matvec"))
    parser.add_argument(
        "--gather-benchmark", type=Path, default=Path("runtime/build/benchmark-w6-gather")
    )
    parser.add_argument(
        "--matmul-benchmark", type=Path, default=Path("runtime/build/benchmark-w6-matmul")
    )
    parser.add_argument("--metal-smoke", type=Path, default=Path("runtime/build/metal_smoke"))
    parser.add_argument("--metallib", type=Path, default=Path("runtime/build/metal_smoke.metallib"))
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--matmul-warmups", type=int, default=3)
    parser.add_argument("--matmul-runs", type=int, default=10)
    args = parser.parse_args()
    if args.warmups < 0 or args.runs <= 0 or args.matmul_warmups < 0 or args.matmul_runs <= 0:
        parser.error("warmups must be non-negative and runs must be positive")
    return args


def main() -> None:
    args = parse_args()
    receipt = build_receipt(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "benchmarks": len(receipt["benchmarks"]),
                "gathers": len(receipt["gathers"]),
                "matmuls": len(receipt["matmuls"]),
            }
        )
    )


if __name__ == "__main__":
    main()
