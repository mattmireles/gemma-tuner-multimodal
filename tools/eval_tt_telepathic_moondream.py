#!/usr/bin/env python3
"""Run the frozen telepathic screenshot eval with Moondream 3 locally.

Private prompts and generations stay in an ignored append-only JSONL ledger.
Stdout contains aggregate-safe progress only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

MODEL_ID = "moondream/moondream3-preview"
MODEL_REVISION = "5112966d1a723413b1c9a1e8bea272b72e647b35"
ADVERTISED_CONTEXT = 32768
EVAL_CONTEXT = 12288


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_context_json(text: str) -> bool:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(value, dict) and list(value) == ["context_analysis"]


def read_rows(path: Path, count: int) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))[:count]
    if len(rows) != count:
        raise ValueError("validation CSV has fewer rows than requested")
    for row in rows:
        image = Path(row["image_path"])
        if not image.is_absolute():
            image = (path.parent / image).resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        row["image_path"] = str(image)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate selected IDs")
    return rows


def render_question(row: dict[str, str]) -> str:
    system = row.get("system_prompt", "")
    if not system:
        raise ValueError("row has no system prompt")
    return f"System instructions:\n{system}\n\nUser request:\n{row['prompt']}"


def read_prior(path: Path, settings_sha256: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("settings_sha256") != settings_sha256:
            raise ValueError("existing ledger settings mismatch")
        key = str(row["example_id"])
        if key in result:
            raise ValueError(f"duplicate ledger row: {key}")
        result[key] = row
    return result


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical(row) + "\n")
        handle.flush()


def load_model(device: str):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    # The release/model card advertises 32K, while this preview revision's
    # remote-code dataclass still defaults to 4096. Set the advertised cache
    # length before the lazy KV caches are created.
    object.__setattr__(model.model.config.text, "max_context", EVAL_CONTEXT)
    model.model._refresh_runtime_buffers()
    # Upstream enables FlexAttention decoding by default, but PyTorch rejects
    # FlexAttention tensors on MPS. The non-flex branch uses the same causal
    # attention mask through scaled_dot_product_attention.
    if device in {"mps", "cuda"}:
        model.model.use_flex_decoding = False
    return torch, model


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from PIL import Image

    csv_path = args.csv.resolve()
    rows = read_rows(csv_path, args.count)
    settings = {
        "schema_version": "tt_telepathic_moondream_eval_v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "csv_sha256": sha256_file(csv_path),
        "count": len(rows),
        "device": args.device,
        "reasoning": False,
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "image_policy": "native_moondream_multicrop_from_original",
        "prompt_policy": "verbatim_system_plus_user_with_role_delimiters",
        "mps_attention": "sdpa_causal_mask_no_flex",
        "cuda_compile": False,
        "advertised_context_positions": ADVERTISED_CONTEXT,
        "allocated_context_positions": EVAL_CONTEXT,
        "context_override_reason": "remote_code_defaults_4096;_12288_covers_frozen_eval_budget",
    }
    settings_sha = hashlib.sha256(canonical(settings).encode()).hexdigest()
    prior = read_prior(args.ledger.resolve(), settings_sha)
    torch, model = load_model(args.device)

    for index, row in enumerate(rows, start=1):
        if row["id"] in prior:
            continue
        if args.device == "mps":
            torch.mps.empty_cache()
        elif args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        question = render_question(row)
        encoded_image = model.encode_image(Image.open(row["image_path"]).convert("RGB"))
        prompt_tokens = len(model.model.tokenizer.encode(question).ids) + 4
        required_positions = int(encoded_image.pos) + prompt_tokens + args.max_tokens
        if required_positions > EVAL_CONTEXT:
            raise ValueError(
                f"row {row['id']} requires {required_positions} positions; "
                f"allocated {EVAL_CONTEXT}"
            )
        result = model.query(
            image=encoded_image,
            question=question,
            reasoning=False,
            settings={"temperature": 0.0, "max_tokens": args.max_tokens},
        )
        elapsed = time.monotonic() - started
        output = str(result["answer"])
        record = {
            "schema_version": "tt_telepathic_moondream_generation_v1",
            "settings_sha256": settings_sha,
            "example_id": row["id"],
            "candidate_output": output,
            "candidate_sha256": hashlib.sha256(output.encode()).hexdigest(),
            "reasoning_sha256": hashlib.sha256(
                canonical(result.get("reasoning", {})).encode()
            ).hexdigest(),
            "elapsed_seconds": elapsed,
            "image_positions": int(encoded_image.pos),
            "prompt_tokens": prompt_tokens,
            "generation_tokens": len(model.model.tokenizer.encode(output).ids),
            "raw_valid_json": strict_context_json(output),
            "peak_memory_bytes": int(torch.mps.driver_allocated_memory())
            if args.device == "mps"
            else int(torch.cuda.max_memory_allocated())
            if args.device == "cuda"
            else 0,
        }
        append_jsonl(args.ledger.resolve(), record)
        print(canonical({"completed": index, "total": len(rows), "elapsed_seconds": elapsed}))

    completed = read_prior(args.ledger.resolve(), settings_sha)
    ordered = [completed[row["id"]] for row in rows]
    return {
        "settings": settings,
        "settings_sha256": settings_sha,
        "rows": len(ordered),
        "raw_valid_json": sum(bool(row["raw_valid_json"]) for row in ordered),
        "mean_elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in ordered) / len(ordered),
        "max_peak_memory_bytes": max(int(row["peak_memory_bytes"]) for row in ordered),
        "ledger_sha256": sha256_file(args.ledger.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("count must be positive")
    if args.max_tokens < 1:
        raise ValueError("max tokens must be positive")
    print(canonical(evaluate(args)))


if __name__ == "__main__":
    main()
