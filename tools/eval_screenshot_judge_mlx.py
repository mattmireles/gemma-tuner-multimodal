#!/usr/bin/env python3
"""Run a frozen image-plus-prompt judge set through a local MLX VLM.

The input and output ledgers may contain private customer material and must stay
outside Git.  Checked receipts should be derived separately from this private
ledger and contain aggregates and hashes only.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = "mlx_screenshot_judge_v1"
GRADES = ("Excellent", "VeryGood", "Good", "NeedsImprovement", "Fail")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_weights(model_path: Path) -> str:
    weights = sorted(model_path.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"no safetensors in {model_path}")
    digest = hashlib.sha256()
    for path in weights:
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def split_thinking_output(text: str) -> tuple[str, str | None]:
    """Separate Gemma 4 private thought from the gradeable final answer."""
    thought_marker = "<|channel>thought\n"
    channel_end = "<channel|>"
    if not text.startswith(thought_marker):
        return text.strip(), None
    body = text[len(thought_marker) :]
    if channel_end not in body:
        return "", sha256_text(body)
    thought, final = body.split(channel_end, 1)
    return final.lstrip(), sha256_text(thought)


def parse_grade(text: str) -> str | None:
    value = text.strip()
    if value in GRADES[:-1]:
        return value
    if re.fullmatch(r"Fail\s+-\s+\S(?:.|\n)*", value):
        return "Fail"
    return None


def build_messages(item: dict[str, Any]) -> list[dict[str, str]]:
    return [{"role": "user", "content": str(item["rendered_prompt"])}]


def validate_item(item: dict[str, Any]) -> None:
    required = {
        "example_id",
        "image_ref",
        "image_sha256",
        "rendered_prompt",
        "rendered_prompt_sha256",
        "work_item_hash",
    }
    missing = sorted(required - set(item))
    if missing:
        raise ValueError(f"work item missing fields: {missing}")
    if sha256_text(str(item["rendered_prompt"])) != item["rendered_prompt_sha256"]:
        raise ValueError(f"prompt hash mismatch for {item['example_id']}")
    image = Path(item["image_ref"])
    if not image.is_file():
        raise FileNotFoundError(image)
    if sha256_file(image) != item["image_sha256"]:
        raise ValueError(f"image hash mismatch for {item['example_id']}")


def read_items(path: Path) -> list[dict[str, Any]]:
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    hashes = [str(item.get("work_item_hash")) for item in items]
    if len(hashes) != len(set(hashes)):
        raise ValueError("duplicate work_item_hash")
    for item in items:
        validate_item(item)
    return items


def read_ledger(path: Path, arm: str, settings_sha256: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row["work_item_hash"])
        if key in rows:
            raise ValueError(f"duplicate ledger work item {key}")
        if row.get("arm") != arm or row.get("settings_sha256") != settings_sha256:
            raise ValueError("existing ledger arm/settings mismatch")
        rows[key] = row
    return rows


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical(row) + "\n")
        handle.flush()


def evaluate_items(
    items: list[dict[str, Any]],
    ledger: Path,
    arm: str,
    settings_sha256: str,
    generator: Callable[[dict[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    existing = read_ledger(ledger, arm, settings_sha256)
    expected = {item["work_item_hash"] for item in items}
    extra = set(existing) - expected
    if extra:
        raise ValueError(f"ledger contains {len(extra)} unexpected work items")
    for index, item in enumerate(items, 1):
        key = str(item["work_item_hash"])
        if key in existing:
            continue
        generated = generator(item)
        final_text, thought_sha256 = split_thinking_output(str(generated["text"]))
        grade = parse_grade(final_text)
        row = {
            "schema_version": SCHEMA_VERSION,
            "arm": arm,
            "settings_sha256": settings_sha256,
            "work_item_hash": key,
            "example_id": item["example_id"],
            "grade": grade,
            "valid_grade": grade is not None,
            "final_text": final_text,
            "response_sha256": sha256_text(str(generated["text"])),
            "thinking_sha256": thought_sha256,
            "elapsed_seconds": float(generated["elapsed_seconds"]),
            "prompt_tokens": int(generated["prompt_tokens"]),
            "generation_tokens": int(generated["generation_tokens"]),
            "peak_memory_gb": float(generated["peak_memory_gb"]),
            "finish_reason": generated.get("finish_reason"),
        }
        numeric = (row["elapsed_seconds"], row["peak_memory_gb"])
        if not all(math.isfinite(value) and value >= 0 for value in numeric):
            raise ValueError(f"non-finite runtime metric for {item['example_id']}")
        append_row(ledger, row)
        existing[key] = row
        print(
            json.dumps(
                {
                    "arm": arm,
                    "case": index,
                    "total": len(items),
                    "grade": grade,
                    "valid": grade is not None,
                    "seconds": round(row["elapsed_seconds"], 3),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return [existing[item["work_item_hash"]] for item in items]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def runtime_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def run_mlx(args: argparse.Namespace) -> dict[str, Any]:
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from PIL import Image

    model_path = args.model.resolve()
    items = read_items(args.work_items.resolve())
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    readme = (model_path / "README.md").read_text(encoding="utf-8")
    if args.source_revision not in readme:
        raise ValueError("model card does not bind the expected source revision")
    quantization = config.get("quantization")
    if args.arm == "bf16" and quantization is not None:
        raise ValueError("BF16 arm unexpectedly declares quantization")
    if args.arm == "6bit" and quantization != {"group_size": 64, "bits": 6, "mode": "affine"}:
        raise ValueError(f"unexpected 6-bit quantization: {quantization}")

    settings = {
        "schema_version": SCHEMA_VERSION,
        "arm": args.arm,
        "model_revision": args.model_revision,
        "source_revision": args.source_revision,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "enable_thinking": args.enable_thinking,
        "thinking_budget": args.thinking_budget,
        "images_per_item": 1,
        "system_role": False,
    }
    settings_sha = sha256_text(canonical(settings))

    load_started = time.perf_counter()
    model, processor = load(str(model_path))
    load_seconds = time.perf_counter() - load_started

    def generator(item: dict[str, Any]) -> dict[str, Any]:
        messages = build_messages(item)
        if [message["role"] for message in messages] != ["user"]:
            raise ValueError("judge ABI requires exactly one user turn")
        prompt = apply_chat_template(
            processor,
            model.config,
            messages,
            num_images=1,
            enable_thinking=args.enable_thinking,
        )
        with Image.open(item["image_ref"]) as source:
            image = source.convert("RGB")
        started = time.perf_counter()
        result = generate(
            model=model,
            processor=processor,
            prompt=prompt,
            image=[image],
            max_tokens=args.max_tokens,
            temperature=0.0,
            enable_thinking=args.enable_thinking,
            thinking_budget=args.thinking_budget if args.enable_thinking else None,
            verbose=False,
        )
        return {
            "text": result.text,
            "elapsed_seconds": time.perf_counter() - started,
            "prompt_tokens": result.prompt_tokens,
            "generation_tokens": result.generation_tokens,
            "peak_memory_gb": result.peak_memory,
            "finish_reason": result.finish_reason,
        }

    rows = evaluate_items(items, args.ledger.resolve(), args.arm, settings_sha, generator)
    times = [float(row["elapsed_seconds"]) for row in rows]
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "settings": settings,
        "settings_sha256": settings_sha,
        "model": {
            "path": str(model_path),
            "revision": args.model_revision,
            "source_revision": args.source_revision,
            "config_sha256": sha256_file(config_path),
            "weights_sha256": sha256_weights(model_path),
            "quantization": quantization,
        },
        "runtime": {
            "mlx_vlm": runtime_version("mlx-vlm"),
            "mlx": runtime_version("mlx"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "load_seconds": load_seconds,
        },
        "input": {
            "work_items_sha256": sha256_file(args.work_items.resolve()),
            "rows": len(items),
        },
        "output": {
            "ledger_sha256": sha256_file(args.ledger.resolve()),
            "rows": len(rows),
            "valid_rows": sum(bool(row["valid_grade"]) for row in rows),
            "grades": dict(sorted(collections.Counter(str(row["grade"]) for row in rows if row["grade"]).items())),
            "finish_reasons": dict(sorted(collections.Counter(str(row["finish_reason"]) for row in rows).items())),
            "latency_seconds": {
                "mean": statistics.fmean(times),
                "p50": percentile(times, 0.5),
                "p95": percentile(times, 0.95),
                "max": max(times),
            },
            "peak_memory_gb": max(float(row["peak_memory_gb"]) for row in rows),
        },
    }
    receipt_path = args.ledger.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(receipt_path), **receipt["output"]}, sort_keys=True))
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("bf16", "6bit"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--work-items", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--thinking-budget", type=int)
    return parser.parse_args()


def main() -> None:
    receipt = run_mlx(parse_args())
    if receipt["output"]["valid_rows"] != receipt["output"]["rows"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
