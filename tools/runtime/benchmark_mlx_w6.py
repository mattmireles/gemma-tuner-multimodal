#!/usr/bin/env python3
"""Benchmark the pinned Gemma 4 E4B W6 MLX artifact on frozen fixtures.

The checked receipt contains hashes and aggregates only.  Generated text is
written to a caller-selected private JSONL ledger, which must remain ignored.
MLX-VLM exposes combined multimodal-encoder plus prompt-prefill time; this tool
reports that boundary honestly instead of inventing a separate encoder time.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from reference_common import (
    canonical_json,
    load_manifest,
    load_private_fixture,
    normalized_error_rates,
    percentile,
    private_root_from,
    sha256_file,
    sha256_text,
    stable_manifest_hash,
)

EXPECTED_QUANTIZATION = {"group_size": 64, "bits": 6, "mode": "affine"}


def runtime_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def weight_set_digest(model_path: Path) -> tuple[str, int]:
    import hashlib

    weights = sorted(model_path.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"no safetensors files in {model_path}")
    digest = hashlib.sha256()
    total_bytes = 0
    for path in weights:
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
        total_bytes += path.stat().st_size
    return digest.hexdigest(), total_bytes


def result_timings_ms(result: Any, elapsed_seconds: float) -> dict[str, float]:
    if result.prompt_tps <= 0:
        raise ValueError("MLX result has no positive prompt throughput")
    if result.generation_tokens and result.generation_tps <= 0:
        raise ValueError("MLX result has no positive generation throughput")
    first_token = float(result.prompt_tokens) / float(result.prompt_tps)
    decode = (
        float(result.generation_tokens) / float(result.generation_tps)
        if result.generation_tokens
        else 0.0
    )
    other = max(0.0, elapsed_seconds - first_token - decode)
    return {
        "other_prepost_ms": other * 1000,
        "encoder_and_prefill_ms": first_token * 1000,
        "first_token_ms": first_token * 1000,
        "decode_ms": decode * 1000,
        "end_to_end_ms": elapsed_seconds * 1000,
    }


def summarize_timings(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not rows:
        raise ValueError("cannot summarize an empty timing set")
    summary: dict[str, dict[str, float]] = {}
    for name in rows[0]:
        values = [row[name] for row in rows]
        summary[name] = {
            "p50_ms": percentile(values, 0.5),
            "p95_ms": percentile(values, 0.95),
            "min_ms": min(values),
            "max_ms": max(values),
        }
    return summary


def build_prompt_and_media(processor: Any, model: Any, fixture: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    from mlx_vlm.prompt_utils import apply_chat_template

    messages = copy.deepcopy(fixture["request"].get("messages"))
    if not isinstance(messages, list):
        raise ValueError("private request.messages must be a list")
    media = fixture["_resolved_media"]
    mode = fixture["mode"]
    kwargs: dict[str, Any] = {}
    if mode == "image_to_text":
        image_keys = sorted(media)
        if not image_keys:
            raise ValueError("image_to_text fixture has no images")
        kwargs["image"] = [str(media[key]) for key in image_keys]
        media_counts = {"num_images": len(image_keys), "num_audios": 0}
    elif mode == "audio_to_text":
        if set(media) != {"audio"}:
            raise ValueError("audio_to_text requires exactly one media entry named audio")
        kwargs["audio"] = [str(media["audio"])]
        media_counts = {"num_images": 0, "num_audios": 1}
    elif mode == "text_to_text":
        if media:
            raise ValueError("text_to_text fixtures must not declare media")
        media_counts = {"num_images": 0, "num_audios": 0}
    else:
        raise ValueError(f"unsupported mode: {mode}")
    enable_thinking = bool(fixture["generation"].get("thinking", False))
    prompt = apply_chat_template(
        processor,
        model.config,
        messages,
        enable_thinking=enable_thinking,
        **media_counts,
    )
    return prompt, kwargs


def run_once(model: Any, processor: Any, fixture: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    import mlx.core as mx
    from mlx_vlm import generate

    generation = fixture["generation"]
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    started = time.perf_counter()
    prompt, media_kwargs = build_prompt_and_media(processor, model, fixture)
    result = generate(
        model=model,
        processor=processor,
        prompt=prompt,
        max_tokens=int(generation["max_new_tokens"]),
        temperature=float(generation.get("temperature", 0.0)),
        enable_thinking=bool(generation.get("thinking", False)),
        verbose=False,
        **media_kwargs,
    )
    elapsed = time.perf_counter() - started
    text = result.text.strip()
    timings = result_timings_ms(result, elapsed)
    metrics: dict[str, Any] = {
        "output_sha256": sha256_text(text),
        "generated_tokens": int(result.generation_tokens),
        "prompt_tokens": int(result.prompt_tokens),
        "peak_memory_gb": float(result.peak_memory),
        "finish_reason": result.finish_reason,
        "matches_expected_output": sha256_text(text) == fixture["expected_output_sha256"],
        "text": text,
    }
    expected_output = fixture.get("expected_output")
    if fixture["mode"] == "audio_to_text" and isinstance(expected_output, str):
        metrics.update(normalized_error_rates(expected_output, text))
    return timings, metrics


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    from mlx_vlm import load

    source_rows = load_manifest(args.manifest)
    rows = source_rows
    if args.mode:
        rows = [row for row in rows if row["mode"] == args.mode]
    if args.fixture_id:
        rows = [row for row in rows if row["fixture_id"] == args.fixture_id]
    if not rows:
        raise ValueError("no fixtures selected")

    private_root = private_root_from(args.private_root)
    fixtures: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        payload = load_private_fixture(row, private_root)
        payload["generation"] = row["generation"]
        payload["expected_output_sha256"] = row["expected"]["output_sha256"]
        fixtures.append((row, payload))

    model_path = args.model.resolve()
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("quantization") != EXPECTED_QUANTIZATION:
        raise ValueError(f"unexpected W6 quantization: {config.get('quantization')}")
    weights_sha256, weight_bytes = weight_set_digest(model_path)

    load_started = time.perf_counter()
    model, processor = load(str(model_path), revision=args.revision)
    load_ms = (time.perf_counter() - load_started) * 1000

    cold_timings, cold_metrics = run_once(model, processor, fixtures[0][1])
    for index in range(args.warmups):
        _, fixture = fixtures[index % len(fixtures)]
        run_once(model, processor, fixture)

    samples: list[dict[str, Any]] = []
    timing_rows: list[dict[str, float]] = []
    private_rows: list[dict[str, Any]] = []
    for run_index in range(args.runs):
        for row, fixture in fixtures:
            timings, metrics = run_once(model, processor, fixture)
            timing_rows.append(timings)
            text = metrics.pop("text")
            sample = {"fixture_id": row["fixture_id"], "run": run_index, "timings": timings, **metrics}
            samples.append(sample)
            private_rows.append({**sample, "generated_text": text})

    args.private_ledger.parent.mkdir(parents=True, exist_ok=True)
    args.private_ledger.write_text(
        "".join(canonical_json(row) + "\n" for row in private_rows), encoding="utf-8"
    )
    public_samples = [
        {
            key: value
            for key, value in sample.items()
            if key not in {"fixture_id"}
        }
        for sample in samples
    ]
    receipt = {
        "schema_version": "gemma4-e4b-mlx-w6-benchmark-v1",
        "model": {
            "id": args.model_id,
            "revision": args.revision,
            "config_sha256": sha256_file(config_path),
            "weights_sha256": weights_sha256,
            "weight_bytes": weight_bytes,
            "quantization": EXPECTED_QUANTIZATION,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "mlx": runtime_version("mlx"),
            "mlx_vlm": runtime_version("mlx-vlm"),
        },
        "contract": {
            "source_manifest_sha256": stable_manifest_hash(source_rows),
            "selection_sha256": stable_manifest_hash(rows),
            "fixtures": len(rows),
            "warmups": args.warmups,
            "runs": args.runs,
            "timing_boundary": "encoder and prompt prefill are combined by the MLX-VLM API",
            "private_ledger_sha256": sha256_file(args.private_ledger),
        },
        "cold": {
            "load_ms": load_ms,
            "first_inference_timings": cold_timings,
            **{key: value for key, value in cold_metrics.items() if key != "text"},
        },
        "warmed": {
            "timing_summary": summarize_timings(timing_rows),
            "mean_generated_tokens": statistics.fmean(sample["generated_tokens"] for sample in samples),
            "peak_memory_gb": max(sample["peak_memory_gb"] for sample in samples),
            "exact_output_hash_matches": sum(sample["matches_expected_output"] for sample in samples),
            "mean_wer": (
                statistics.fmean(sample["wer"] for sample in samples if "wer" in sample)
                if any("wer" in sample for sample in samples)
                else None
            ),
            "mean_cer": (
                statistics.fmean(sample["cer"] for sample in samples if "cer" in sample)
                if any("cer" in sample for sample in samples)
                else None
            ),
            "samples": public_samples,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--private-root")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", default="mlx-community/gemma-4-e4b-it-6bit")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--mode", choices=("image_to_text", "audio_to_text", "text_to_text"))
    parser.add_argument("--fixture-id")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/runtime-receipts/mlx-w6-benchmark.json")
    )
    parser.add_argument(
        "--private-ledger",
        type=Path,
        default=Path("artifacts/runtime-receipts/mlx-w6-private.jsonl"),
    )
    return parser.parse_args()


def main() -> int:
    print(canonical_json(benchmark(parse_args())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
