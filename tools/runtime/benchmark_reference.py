#!/usr/bin/env python3
"""Benchmark a pinned Hugging Face Gemma 4 reference by product stage.

Cold model/processor load and the first inference are reported separately from
warmed runs.  The benchmark never writes prompt, media, or generated text into
its machine-readable receipt.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from capture_reference import move_inputs, prepare_inputs, resolve_device, resolve_dtype, synchronize
from reference_common import (
    canonical_json,
    load_manifest,
    load_private_fixture,
    normalized_error_rates,
    private_root_from,
    sha256_text,
    stable_manifest_hash,
    summarize_stage_rows,
)


class TokenClock:
    """Transformers stopping criterion that records each completed decode step."""

    def __init__(self) -> None:
        self.started = 0.0
        self.timestamps: list[float] = []

    def __call__(self, input_ids: Any, scores: Any, **_: Any) -> bool:
        self.timestamps.append(time.perf_counter())
        return False


class EncoderClock:
    def __init__(self, torch_module: Any, device: str) -> None:
        self.torch = torch_module
        self.device = device
        self.started: float | None = None
        self.elapsed = 0.0

    def pre(self, *_: Any) -> None:
        synchronize(self.torch, self.device)
        self.started = time.perf_counter()

    def post(self, *_: Any) -> None:
        synchronize(self.torch, self.device)
        if self.started is not None:
            self.elapsed += time.perf_counter() - self.started


def load_runtime(args: argparse.Namespace) -> tuple[Any, Any, Any, str]:
    import torch
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration

    device = resolve_device(torch, args.device)
    dtype = resolve_dtype(torch, args.dtype)
    processor = AutoProcessor.from_pretrained(
        args.model, revision=args.revision, local_files_only=args.local_files_only
    )
    model = Gemma4ForConditionalGeneration.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=dtype,
        attn_implementation="eager",
        local_files_only=args.local_files_only,
    ).eval()
    model.to(device)
    synchronize(torch, device)
    return torch, processor, model, device


def run_once(
    torch_module: Any,
    processor: Any,
    model: Any,
    device: str,
    fixture: dict[str, Any],
    generation: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    from transformers import StoppingCriteriaList

    end_to_end_started = time.perf_counter()
    preprocess_started = time.perf_counter()
    inputs, _ = prepare_inputs(processor, fixture)
    inputs = move_inputs(inputs, device)
    synchronize(torch_module, device)
    preprocess_seconds = time.perf_counter() - preprocess_started

    encoder_clock = EncoderClock(torch_module, device)
    handles = []
    if fixture["mode"] == "image_to_text":
        handles.extend(
            [
                model.model.vision_tower.register_forward_pre_hook(encoder_clock.pre),
                model.model.vision_tower.register_forward_hook(encoder_clock.post),
            ]
        )
    elif fixture["mode"] == "audio_to_text":
        handles.extend(
            [
                model.model.audio_tower.register_forward_pre_hook(encoder_clock.pre),
                model.model.audio_tower.register_forward_hook(encoder_clock.post),
            ]
        )

    clock = TokenClock()
    generation_started = time.perf_counter()
    try:
        with torch_module.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=int(generation["max_new_tokens"]),
                stopping_criteria=StoppingCriteriaList([clock]),
                return_dict_in_generate=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    synchronize(torch_module, device)
    generation_finished = time.perf_counter()
    if not clock.timestamps:
        raise RuntimeError("generation produced no token timestamps")
    first_token_seconds = clock.timestamps[0] - generation_started
    encoder_seconds = encoder_clock.elapsed
    prefill_seconds = max(0.0, first_token_seconds - encoder_seconds)
    decode_seconds = max(0.0, generation_finished - clock.timestamps[0])

    postprocess_started = time.perf_counter()
    input_length = int(inputs["input_ids"].shape[-1])
    new_tokens = output.sequences[:, input_length:]
    text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    postprocess_seconds = time.perf_counter() - postprocess_started
    end_to_end_seconds = time.perf_counter() - end_to_end_started
    stages = {
        "preprocess": preprocess_seconds * 1000,
        "encoder": encoder_seconds * 1000,
        # The reference path has no separately observable device handoff.  Its
        # synchronization cost remains in encoder/prefill instead of being
        # guessed or double-counted.
        "handoff": 0.0,
        "prefill": prefill_seconds * 1000,
        "first_token": first_token_seconds * 1000,
        "decode": decode_seconds * 1000,
        "postprocess": postprocess_seconds * 1000,
        "end_to_end": end_to_end_seconds * 1000,
    }
    metrics: dict[str, Any] = {
        "output_sha256": sha256_text(text),
        "generated_tokens": int(new_tokens.shape[-1]),
        "matches_expected_output": sha256_text(text) == fixture["expected_output_sha256"],
    }
    expected_output = fixture.get("expected_output")
    if fixture["mode"] == "audio_to_text" and isinstance(expected_output, str):
        metrics.update(normalized_error_rates(expected_output, text))
    return stages, metrics


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    source_rows = load_manifest(args.manifest)
    rows = source_rows
    if args.mode:
        rows = [row for row in rows if row["mode"] == args.mode]
    if args.fixture_id:
        rows = [row for row in rows if row["fixture_id"] == args.fixture_id]
    if not rows:
        raise ValueError("no fixtures selected")
    private_root = private_root_from(args.private_root)
    fixtures = []
    for row in rows:
        payload = load_private_fixture(row, private_root)
        payload["expected_output_sha256"] = row["expected"]["output_sha256"]
        fixtures.append((row, payload))

    load_started = time.perf_counter()
    torch_module, processor, model, device = load_runtime(args)
    load_ms = (time.perf_counter() - load_started) * 1000

    cold_stages, cold_output = run_once(
        torch_module, processor, model, device, fixtures[0][1], fixtures[0][0]["generation"]
    )
    for index in range(args.warmups):
        row, fixture = fixtures[index % len(fixtures)]
        run_once(torch_module, processor, model, device, fixture, row["generation"])

    samples: list[dict[str, Any]] = []
    stage_rows: list[dict[str, float]] = []
    for run_index in range(args.runs):
        for row, fixture in fixtures:
            stages, output = run_once(torch_module, processor, model, device, fixture, row["generation"])
            stage_rows.append(stages)
            samples.append(
                {
                    "fixture_id": row["fixture_id"],
                    "run": run_index,
                    "stages_ms": stages,
                    **output,
                }
            )

    receipt = {
        "schema_version": "gemma4-e4b-reference-benchmark-v1",
        "model": {"id": args.model, "revision": args.revision, "dtype": args.dtype},
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "device": device,
            "torch": torch_module.__version__,
        },
        "contract": {
            "source_manifest_sha256": stable_manifest_hash(source_rows),
            "selection_sha256": stable_manifest_hash(rows),
            "fixtures": len(rows),
            "warmups": args.warmups,
            "runs": args.runs,
            "handoff_accounting": "reference synchronization is included in encoder/prefill",
        },
        "cold": {"load_ms": load_ms, "first_inference_stages_ms": cold_stages, **cold_output},
        "warmed": {
            "stage_summary": summarize_stage_rows(stage_rows),
            "mean_generated_tokens": statistics.fmean(sample["generated_tokens"] for sample in samples),
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
            "samples": samples,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--private-root")
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--mode", choices=("image_to_text", "audio_to_text", "text_to_text"))
    parser.add_argument("--fixture-id")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("artifacts/runtime-receipts/reference-benchmark.json"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_manifest(args.manifest)
    if args.mode:
        rows = [row for row in rows if row["mode"] == args.mode]
    if args.fixture_id:
        rows = [row for row in rows if row["fixture_id"] == args.fixture_id]
    if args.validate_only:
        root = private_root_from(args.private_root)
        for row in rows:
            load_private_fixture(row, root)
        print(canonical_json({"fixtures": len(rows), "manifest_sha256": stable_manifest_hash(rows)}))
        return 0
    print(canonical_json(benchmark(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
