#!/usr/bin/env python3
"""Generate matched telepathic-context candidates and blind judge work items.

Generation ledgers contain private screenshots, prompts, and model outputs. They
must remain under the ignored private staging directory. Checked-in receipts are
created separately and contain hashes and aggregates only.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

SCHEMA_VERSION = "tt_telepathic_eval_v1"
EVAL_PROMPT_SHA256 = "afb58a57173811636a44aa37b18a445c7790d0b338a291a887ae9887a6416cbb"
SOURCE_REVISION = "fee6332c1abaafb77f6f9624236c63aa2f1d0187"
VIEW_POLICY = "global_plus_four_nonoverlapping_quadrants"
VIEW_ORDER = ("global", "top_left", "top_right", "bottom_left", "bottom_right")
PROHIBITED_WORK_ITEM_FIELDS = frozenset(
    {"arm", "model", "checkpoint", "split", "target", "target_json", "competing_output"}
)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_weights(model_path: Path) -> tuple[str, int]:
    weights = sorted(model_path.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"no safetensors in {model_path}")
    digest = hashlib.sha256()
    total_bytes = 0
    for path in weights:
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
        total_bytes += path.stat().st_size
    return digest.hexdigest(), total_bytes


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


def build_image_views(image: Any) -> list[Any]:
    """Return global, TL, TR, BL, BR RGB views with exact pixel coverage."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width < 2 or height < 2:
        raise ValueError(f"image dimensions must both be >=2, got {width}x{height}")
    x_mid, y_mid = width // 2, height // 2
    return [
        rgb,
        rgb.crop((0, 0, x_mid, y_mid)),
        rgb.crop((x_mid, 0, width, y_mid)),
        rgb.crop((0, y_mid, x_mid, height)),
        rgb.crop((x_mid, y_mid, width, height)),
    ]


def read_frozen_rows(
    *, arm: str, staging: Path, manifest_path: Path, count: int
) -> list[dict[str, str]]:
    if arm not in {"compact", "conditioned", "full"}:
        raise ValueError(f"unknown arm: {arm}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen_ids = [str(value) for value in manifest["validation_first_20_ids"]]
    if count < 1 or count > len(frozen_ids):
        raise ValueError(f"count must be in [1, {len(frozen_ids)}]")
    expected_ids = frozen_ids[:count]
    csv_path = staging / arm / "validation.csv"
    expected_sha = manifest["file_sha256"][f"{arm}/validation.csv"]
    if sha256_file(csv_path) != expected_sha:
        raise ValueError(f"{arm} validation CSV hash mismatch")
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows_by_id = {str(row["id"]): row for row in csv.DictReader(handle)}
    rows = [rows_by_id[example_id] for example_id in expected_ids]
    if [row["id"] for row in rows] != expected_ids:
        raise AssertionError("validation order mismatch")
    for row in rows:
        if row.get("image_view_policy") != VIEW_POLICY:
            raise ValueError(f"wrong view policy for {row['id']}")
        if not Path(row["image_path"]).is_file():
            raise FileNotFoundError(row["image_path"])
        if arm == "compact" and "system_prompt" in row:
            raise ValueError("compact validation ABI must omit the system_prompt column")
        if arm in {"conditioned", "full"} and not row.get("system_prompt", "").strip():
            raise ValueError(f"system-prompt row {row['id']} has no system prompt")
    return rows


def read_plan29_rows(
    *, csv_path: Path, ids_path: Path, count: int | None
) -> list[dict[str, str]]:
    ids = [str(value) for value in json.loads(ids_path.read_text(encoding="utf-8"))]
    if count is not None:
        if count < 1 or count > len(ids):
            raise ValueError(f"count must be in [1, {len(ids)}]")
        ids = ids[:count]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate Plan 29 IDs")
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows_by_id = {str(row["id"]): row for row in csv.DictReader(handle)}
    missing = [example_id for example_id in ids if example_id not in rows_by_id]
    if missing:
        raise ValueError(f"missing {len(missing)} Plan 29 IDs")
    rows = [rows_by_id[example_id] for example_id in ids]
    for row in rows:
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            row["image_path"] = str((csv_path.parent / image_path).resolve())
        if row.get("image_view_policy") != VIEW_POLICY:
            raise ValueError(f"wrong view policy for {row['id']}")
        if not Path(row["image_path"]).is_file():
            raise FileNotFoundError(row["image_path"])
        if not row.get("system_prompt", "").strip():
            raise ValueError(f"Plan 29 row {row['id']} has no full system prompt")
    return rows


def build_messages(row: dict[str, str], arm: str) -> list[dict[str, str]]:
    user = {"role": "user", "content": row["prompt"]}
    if arm == "compact":
        return [user]
    if arm in {"conditioned", "full"}:
        return [{"role": "system", "content": row["system_prompt"]}, user]
    raise ValueError(f"unknown arm: {arm}")


def parse_candidate_json(text: str) -> tuple[bool, str | None]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return False, None
    if not isinstance(value, dict) or set(value) != {"context_analysis"}:
        return False, None
    return True, sha256_text(canonical(value))


def read_generation_ledger(
    path: Path, *, arm: str, settings_sha256: str
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row["example_id"])
        if key in rows:
            raise ValueError(f"duplicate generation row: {key}")
        if row.get("arm") != arm or row.get("settings_sha256") != settings_sha256:
            raise ValueError("existing generation ledger arm/settings mismatch")
        rows[key] = row
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical(row) + "\n")
        handle.flush()


def seed_generation_rows(
    *, source_path: Path, source_receipt_path: Path, rows: list[dict[str, str]],
    ledger: Path, arm: str, settings_sha256: str, max_tokens: int,
) -> int:
    receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    expected = {
        "model": "mlx-community/gemma-4-e4b-it-bf16",
        "revision": "eec12d0899edea9b738ab1009af9159cdfd70d71",
        "thinking": False,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    for key, value in expected.items():
        if receipt.get("settings", {}).get(key) != value:
            raise ValueError(f"seed receipt setting mismatch: {key}")
    selected = {row["id"]: row for row in rows}
    existing = read_generation_ledger(ledger, arm=arm, settings_sha256=settings_sha256)
    seeded = 0
    for line in source_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        source = json.loads(line)
        row = selected.get(str(source["example_id"]))
        if row is None or row["id"] in existing:
            continue
        if (
            source.get("image_path") != row["image_path"]
            or source.get("image_sha256") != sha256_file(Path(row["image_path"]))
            or source.get("system_prompt") != row["system_prompt"]
            or source.get("user_prompt") != row["prompt"]
        ):
            raise ValueError(f"seed input mismatch: {row['id']}")
        candidate = str(source["candidate_output"]).strip()
        valid_json, normalized_sha = parse_candidate_json(candidate)
        copied = {
            "schema_version": SCHEMA_VERSION,
            "arm": arm,
            "settings_sha256": settings_sha256,
            "example_id": row["id"],
            "image_ref": row["image_path"],
            "image_sha256": source["image_sha256"],
            "system_instruction": row["system_prompt"],
            "user_message": row["prompt"],
            "candidate_output": candidate,
            "candidate_sha256": sha256_text(candidate),
            "normalized_json_sha256": normalized_sha,
            "raw_valid_json": valid_json,
            "elapsed_seconds": float(source["elapsed_seconds"]),
            "prompt_tokens": int(source["prompt_tokens"]),
            "generation_tokens": int(source["generation_tokens"]),
            "peak_memory_gb": float(source["peak_memory_gb"]),
            "finish_reason": source.get("finish_reason"),
            "reused_from": {
                "ledger_sha256": sha256_file(source_path),
                "receipt_sha256": sha256_file(source_receipt_path),
                "settings_sha256": source.get("settings_sha256"),
            },
        }
        append_jsonl(ledger, copied)
        existing[row["id"]] = copied
        seeded += 1
    return seeded


def generate_rows(
    rows: list[dict[str, str]],
    *,
    arm: str,
    ledger: Path,
    settings_sha256: str,
    generator: Callable[[dict[str, str], list[dict[str, str]]], dict[str, Any]],
) -> list[dict[str, Any]]:
    existing = read_generation_ledger(ledger, arm=arm, settings_sha256=settings_sha256)
    expected = {row["id"] for row in rows}
    extras = set(existing) - expected
    if extras:
        raise ValueError(f"generation ledger contains {len(extras)} unexpected rows")
    for index, source in enumerate(rows, 1):
        example_id = source["id"]
        if example_id in existing:
            continue
        generated = generator(source, build_messages(source, arm))
        candidate = str(generated["text"]).strip()
        valid_json, normalized_sha = parse_candidate_json(candidate)
        result = {
            "schema_version": SCHEMA_VERSION,
            "arm": arm,
            "settings_sha256": settings_sha256,
            "example_id": example_id,
            "image_ref": source["image_path"],
            "image_sha256": sha256_file(Path(source["image_path"])),
            "system_instruction": source.get("system_prompt", ""),
            "user_message": source["prompt"],
            "candidate_output": candidate,
            "candidate_sha256": sha256_text(candidate),
            "normalized_json_sha256": normalized_sha,
            "raw_valid_json": valid_json,
            "elapsed_seconds": float(generated["elapsed_seconds"]),
            "prompt_tokens": int(generated["prompt_tokens"]),
            "generation_tokens": int(generated["generation_tokens"]),
            "peak_memory_gb": float(generated["peak_memory_gb"]),
            "finish_reason": generated.get("finish_reason"),
        }
        if not math.isfinite(result["elapsed_seconds"]) or result["elapsed_seconds"] < 0:
            raise ValueError(f"invalid elapsed time for {example_id}")
        if not math.isfinite(result["peak_memory_gb"]) or result["peak_memory_gb"] < 0:
            raise ValueError(f"invalid peak memory for {example_id}")
        append_jsonl(ledger, result)
        existing[example_id] = result
        print(canonical({"arm": arm, "case": index, "total": len(rows), "valid_json": valid_json,
                         "seconds": round(result["elapsed_seconds"], 3)}), flush=True)
    return [existing[row["id"]] for row in rows]


def load_evaluation_prompt(path: Path) -> str:
    if sha256_file(path) != EVAL_PROMPT_SHA256:
        raise ValueError("evaluation prompt hash mismatch")
    return path.read_text(encoding="utf-8")


def render_evaluation_prompt(
    template: str, *, system_instruction: str, user_message: str, output: str
) -> str:
    values = {
        "${system_instruction}": system_instruction,
        "${user_message}": user_message,
        "${output}": output,
    }
    remainder = template
    for placeholder in values:
        if template.count(placeholder) != 1:
            raise ValueError(f"expected exactly one {placeholder}")
        remainder = remainder.replace(placeholder, "")
    if "${" in remainder:
        raise ValueError("unresolved evaluation prompt placeholder")
    rendered = template
    for placeholder, value in values.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


def build_blind_work_items(
    generated_rows: Iterable[dict[str, Any]], template: str
) -> list[dict[str, str]]:
    items = []
    for row in generated_rows:
        rendered = render_evaluation_prompt(
            template,
            system_instruction=str(row["system_instruction"]),
            user_message=str(row["user_message"]),
            output=str(row["candidate_output"]),
        )
        identity = canonical({
            "example_id": row["example_id"],
            "image_sha256": row["image_sha256"],
            "prompt_sha256": EVAL_PROMPT_SHA256,
            "rendered_prompt_sha256": sha256_text(rendered),
        })
        item = {
            "work_item_hash": sha256_text(identity),
            "example_id": str(row["example_id"]),
            "image_ref": str(row["image_ref"]),
            "image_sha256": str(row["image_sha256"]),
            "rendered_prompt": rendered,
            "rendered_prompt_sha256": sha256_text(rendered),
        }
        if PROHIBITED_WORK_ITEM_FIELDS.intersection(item):
            raise AssertionError("work item violates blind-evaluation contract")
        items.append(item)
    if len({item["work_item_hash"] for item in items}) != len(items):
        raise ValueError("duplicate blind work item hash")
    return items


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(canonical(row) + "\n" for row in materialized), encoding="utf-8")
    temporary.replace(path)


def run_mlx(args: argparse.Namespace) -> dict[str, Any]:
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from PIL import Image

    model_path = args.model.resolve()
    staging = args.staging.resolve() if args.staging is not None else None
    manifest = staging / "manifest.json" if staging is not None else None
    if args.csv is not None or args.ids_json is not None:
        if args.csv is None or args.ids_json is None or args.arm != "full":
            raise ValueError("Plan 29 requires --arm full plus both --csv and --ids-json")
        csv_path = args.csv.resolve()
        ids_path = args.ids_json.resolve()
        rows = read_plan29_rows(csv_path=csv_path, ids_path=ids_path, count=args.count)
    else:
        if staging is None or manifest is None or args.count is None:
            raise ValueError("legacy evaluation requires --staging and --count")
        csv_path = staging / args.arm / "validation.csv"
        ids_path = manifest
        rows = read_frozen_rows(
            arm=args.arm, staging=staging, manifest_path=manifest, count=args.count
        )
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("quantization") is not None:
        raise ValueError("Phase 2 stock control must use unquantized BF16 weights")
    conversion_identity = None
    if args.candidate_kind == "stock":
        if model_path.name != args.model_revision:
            raise ValueError("model snapshot path does not match the pinned MLX revision")
        readme = (model_path / "README.md").read_text(encoding="utf-8")
        if f"Source revision: `{SOURCE_REVISION}`" not in readme:
            raise ValueError("MLX model card does not bind the pinned source revision")
    else:
        conversion_path = model_path / "conversion-receipt.json"
        conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
        if (
            conversion.get("precision") != "bfloat16"
            or conversion.get("quantization") is not None
            or conversion.get("source_hf", {}).get("revision") != SOURCE_REVISION
        ):
            raise ValueError("candidate conversion receipt mismatch")
        for name, digest in conversion["output"]["files_sha256"].items():
            if sha256_file(model_path / name) != digest:
                raise ValueError(f"candidate conversion file mismatch: {name}")
        conversion_identity = conversion["conversion_identity_sha256"]
    weights_sha256, weight_bytes = sha256_weights(model_path)
    settings = {
        "schema_version": SCHEMA_VERSION,
        "arm": args.arm,
        "model_repo": args.model_repo,
        "model_revision": args.model_revision,
        "candidate_kind": args.candidate_kind,
        "conversion_identity_sha256": conversion_identity,
        "source_revision": SOURCE_REVISION,
        "precision": "bfloat16",
        "views": list(VIEW_ORDER),
        "thinking": False,
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "runtime": {
            "mlx_vlm": runtime_version("mlx-vlm"),
            "mlx": runtime_version("mlx"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "inputs": {
            "csv_sha256": sha256_file(csv_path),
            "ids_sha256": sha256_file(ids_path),
            "rows": len(rows),
        },
    }
    settings_sha = sha256_text(canonical(settings))
    if args.seed_ledger is not None or args.seed_receipt is not None:
        if args.candidate_kind != "stock" or args.seed_ledger is None or args.seed_receipt is None:
            raise ValueError("seed reuse requires stock plus both seed paths")
        seeded = seed_generation_rows(
            source_path=args.seed_ledger.resolve(),
            source_receipt_path=args.seed_receipt.resolve(), rows=rows,
            ledger=args.ledger.resolve(), arm=args.arm,
            settings_sha256=settings_sha, max_tokens=args.max_tokens,
        )
        print(canonical({"seeded": seeded, "total": len(rows)}), flush=True)
    started = time.perf_counter()
    model, processor = load(
        str(model_path), revision=args.model_revision if args.candidate_kind == "stock" else None
    )
    load_seconds = time.perf_counter() - started

    def generator(source: dict[str, str], messages: list[dict[str, str]]) -> dict[str, Any]:
        prompt = apply_chat_template(
            processor, model.config, messages, num_images=5, enable_thinking=False
        )
        with Image.open(source["image_path"]) as image:
            views = build_image_views(image)
        generation_started = time.perf_counter()
        result = generate(
            model=model,
            processor=processor,
            prompt=prompt,
            image=views,
            max_tokens=args.max_tokens,
            temperature=0.0,
            enable_thinking=False,
            verbose=False,
        )
        return {
            "text": result.text,
            "elapsed_seconds": time.perf_counter() - generation_started,
            "prompt_tokens": result.prompt_tokens,
            "generation_tokens": result.generation_tokens,
            "peak_memory_gb": result.peak_memory,
            "finish_reason": result.finish_reason,
        }

    generated = generate_rows(
        rows, arm=args.arm, ledger=args.ledger.resolve(), settings_sha256=settings_sha,
        generator=generator,
    )
    template = load_evaluation_prompt(args.evaluation_prompt.resolve())
    items = build_blind_work_items(generated, template)
    write_jsonl_atomic(args.work_items.resolve(), items)
    times = [float(row["elapsed_seconds"]) for row in generated]
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "settings": settings,
        "settings_sha256": settings_sha,
        "model": {
            "path": str(model_path),
            "config_sha256": sha256_file(config_path),
            "weights_sha256": weights_sha256,
            "weight_bytes": weight_bytes,
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
            "manifest_sha256": sha256_file(manifest) if manifest is not None else None,
            "csv_sha256": sha256_file(csv_path),
            "ids_sha256": sha256_file(ids_path),
            "rows": len(rows),
        },
        "output": {
            "ledger_sha256": sha256_file(args.ledger.resolve()),
            "work_items_sha256": sha256_file(args.work_items.resolve()),
            "rows": len(generated),
            "raw_valid_json": sum(bool(row["raw_valid_json"]) for row in generated),
            "finish_reasons": dict(sorted(collections.Counter(
                str(row["finish_reason"]) for row in generated
            ).items())),
            "latency_seconds": {
                "mean": statistics.fmean(times),
                "p50": percentile(times, 0.5),
                "p95": percentile(times, 0.95),
                "max": max(times),
            },
            "peak_memory_gb": max(float(row["peak_memory_gb"]) for row in generated),
        },
    }
    receipt_path = args.ledger.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(canonical({"receipt": str(receipt_path), **receipt["output"]}))
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("compact", "conditioned", "full"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-repo", default="mlx-community/gemma-4-e4b-it-bf16")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--candidate-kind", choices=("stock", "merged"), default="stock")
    parser.add_argument("--staging", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--ids-json", type=Path)
    parser.add_argument("--seed-ledger", type=Path)
    parser.add_argument("--seed-receipt", type=Path)
    parser.add_argument("--evaluation-prompt", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--work-items", type=Path, required=True)
    parser.add_argument("--count", type=int)
    parser.add_argument("--max-tokens", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    run_mlx(parse_args())


if __name__ == "__main__":
    main()
