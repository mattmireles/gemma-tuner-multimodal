#!/usr/bin/env python3
"""Evaluate a private telepathic-context LoRA without exposing row content.

The append-only ledger is private and contains model outputs. Stdout and the
checked receipt contain aggregate diagnostics and content hashes only.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

SOURCE_REVISION = "fee6332c1abaafb77f6f9624236c63aa2f1d0187"
MODEL_ID = "google/gemma-4-E4B-it"


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            row["image_path"] = str((path.parent / image_path).resolve())
    return rows


def select_rows(
    rows: list[dict[str, str]], *, ids: list[str] | None, count: int | None
) -> list[dict[str, str]]:
    if ids is not None:
        by_id = {str(row["id"]): row for row in rows}
        missing = [example_id for example_id in ids if example_id not in by_id]
        if missing:
            raise ValueError(f"missing {len(missing)} frozen row IDs")
        selected = [by_id[example_id] for example_id in ids]
    else:
        selected = rows
    if count is not None:
        if count < 1 or count > len(selected):
            raise ValueError(f"count must be in [1, {len(selected)}]")
        selected = selected[:count]
    if len({row["id"] for row in selected}) != len(selected):
        raise ValueError("duplicate selected IDs")
    return selected


def cyclic_permutation(rows: list[dict[str, str]]) -> dict[str, str]:
    if len(rows) < 2:
        raise ValueError("image-dependence check requires at least two rows")
    return {
        rows[index]["id"]: rows[(index + 1) % len(rows)]["image_path"]
        for index in range(len(rows))
    }


def one_sided_sign_test_p(*, positives: int, trials: int) -> float:
    """Exact P(X >= positives) for X ~ Binomial(trials, 0.5)."""
    if trials < 1 or positives < 0 or positives > trials:
        raise ValueError("invalid sign-test counts")
    return sum(math.comb(trials, value) for value in range(positives, trials + 1)) / (2**trials)


def summarize_dependence(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    evidence = [row["image_dependence"] for row in rows if row.get("image_dependence") is not None]
    if not evidence:
        return None
    margins = [float(row["permuted"] - row["correct"]) for row in evidence]
    base = sum(float(row["base_correct"]) for row in evidence) / len(evidence)
    adapted = sum(float(row["correct"]) for row in evidence) / len(evidence)
    positives = sum(value > 0 for value in margins)
    mean_margin = sum(margins) / len(margins)
    sign_test_p = one_sided_sign_test_p(positives=positives, trials=len(margins))
    return {
        "correct_lower_nll": positives,
        "trials": len(margins),
        "one_sided_sign_test_p": sign_test_p,
        "mean_permuted_minus_correct_nll": mean_margin,
        "passes_preregistered_gate": positives >= 22 and mean_margin > 0.0,
        "base_mean_correct_nll": base,
        "adapted_mean_correct_nll": adapted,
        "assistant_loss_reduction": 1.0 - adapted / base,
        "finite": all(math.isfinite(value) for value in [*margins, base, adapted]),
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical(row) + "\n")
        handle.flush()


def move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    """Move tensors while preserving scalar model kwargs such as logits_to_keep."""
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def existing_rows(path: Path, settings_sha256: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("settings_sha256") != settings_sha256:
            raise ValueError("existing ledger settings mismatch")
        example_id = str(row["example_id"])
        if example_id in result:
            raise ValueError(f"duplicate ledger row: {example_id}")
        result[example_id] = row
    return result


def messages_for_generation(row: dict[str, str], arm: str, views: list[Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if arm in {"conditioned", "full"}:
        prompt = row.get("system_prompt", "")
        if not prompt:
            raise ValueError("conditioned row has no system prompt")
        messages.append({"role": "system", "content": [{"type": "text", "text": prompt}]})
    elif arm != "compact":
        raise ValueError(f"unknown arm: {arm}")
    elif row.get("system_prompt"):
        raise ValueError("compact row contains a system prompt")
    messages.append({
        "role": "user",
        "content": [
            *({"type": "image", "image": view} for view in views),
            {"type": "text", "text": row["prompt"]},
        ],
    })
    return messages


def resolve_device(torch_runtime: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch_runtime.cuda.is_available():
        return "cuda"
    if hasattr(torch_runtime.backends, "mps") and torch_runtime.backends.mps.is_available():
        return "mps"
    return "cpu"


def reset_peak_memory(torch_runtime: Any, device: str) -> None:
    if device == "cuda":
        torch_runtime.cuda.reset_peak_memory_stats()
    elif device == "mps" and hasattr(torch_runtime.mps, "empty_cache"):
        torch_runtime.mps.empty_cache()


def release_device_cache(torch_runtime: Any, device: str) -> None:
    gc.collect()
    if device == "mps" and hasattr(torch_runtime.mps, "empty_cache"):
        torch_runtime.mps.empty_cache()


def peak_memory_bytes(torch_runtime: Any, device: str) -> int:
    if device == "cuda":
        return int(torch_runtime.cuda.max_memory_allocated())
    if device == "mps" and hasattr(torch_runtime.mps, "current_allocated_memory"):
        return int(torch_runtime.mps.current_allocated_memory())
    return 0


def adapter_integrity_sha256(adapter: Path | None) -> str | None:
    if adapter is None:
        return None
    for name in (".complete.json", ".integrity.json"):
        marker = adapter / name
        if marker.is_file():
            return sha256_file(marker)
    raise ValueError("adapter has no checkpoint completion or run integrity marker")


def load_runtime(adapter: Path | None, device: str, *, merge_adapter: bool = False):
    if merge_adapter and adapter is None:
        raise ValueError("cannot merge without an adapter")
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor

    from gemma_tuner.models.gemma.base_model_loader import load_base_model_for_gemma
    from gemma_tuner.models.gemma.family import detect_family

    family = detect_family(MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=SOURCE_REVISION)
    base = load_base_model_for_gemma(
        MODEL_ID,
        family=family,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        revision=SOURCE_REVISION,
    )
    model = (
        PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
        if adapter is not None
        else base
    )
    if merge_adapter:
        model = model.merge_and_unload()
    model = model.to(device)
    model.eval()
    return torch, processor, model, family


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from PIL import Image

    from gemma_tuner.models.common.collators import (
        DataCollatorGemmaImage,
        build_image_views,
    )
    from gemma_tuner.models.gemma.finetune import completion_only_causal_loss

    csv_path = args.csv.resolve()
    adapter = args.adapter.resolve() if args.adapter is not None else None
    ledger = args.ledger.resolve()
    source_rows = read_csv(csv_path)
    frozen_ids = None
    if args.ids_json:
        frozen_ids = [str(value) for value in json.loads(args.ids_json.read_text())]
    rows = select_rows(source_rows, ids=frozen_ids, count=args.count)
    settings = {
        "schema_version": "tt_telepathic_hf_eval_v1",
        "model_id": MODEL_ID,
        "model_revision": SOURCE_REVISION,
        "candidate": (
            "stock" if adapter is None else "merged_adapter" if args.merge_adapter else "adapter"
        ),
        "adapter_integrity_sha256": adapter_integrity_sha256(adapter),
        "csv_sha256": sha256_file(csv_path),
        "arm": args.arm,
        "count": len(rows),
        "max_new_tokens": args.max_new_tokens,
        "thinking": False,
        "temperature": 0.0,
        "image_dependence": args.image_dependence,
    }
    settings_sha = hashlib.sha256(canonical(settings).encode()).hexdigest()
    prior = existing_rows(ledger, settings_sha)
    unexpected = set(prior) - {row["id"] for row in rows}
    if unexpected:
        raise ValueError(f"ledger has {len(unexpected)} unexpected IDs")

    import torch

    device = resolve_device(torch, args.device)
    torch_runtime, processor, model, family = load_runtime(
        adapter, device, merge_adapter=args.merge_adapter
    )
    collator = DataCollatorGemmaImage(
        processor,
        text_column="response",
        family=family,
        image_path_column="image_path",
        prompt_column="prompt",
        image_token_budget=280,
        image_view_policy="global_plus_four_nonoverlapping_quadrants",
        system_prompt_column="system_prompt" if args.arm in {"conditioned", "full"} else None,
        require_telepathic_contract=args.arm != "full",
        completion_only_logits=True,
        max_length=16384,
        sub_mode="vqa",
    )
    shuffled = cyclic_permutation(rows) if args.image_dependence else {}

    for index, row in enumerate(rows, 1):
        if row["id"] in prior:
            continue
        with Image.open(row["image_path"]) as image:
            views = build_image_views(image, "global_plus_four_nonoverlapping_quadrants")
        messages = messages_for_generation(row, args.arm, views)
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = processor(text=[prompt], images=[views], return_tensors="pt", padding=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        input_length = int(encoded["input_ids"].shape[1])
        reset_peak_memory(torch_runtime, device)
        started = time.perf_counter()
        with torch_runtime.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
        elapsed = time.perf_counter() - started
        completion = generated[0, input_length:]
        text = processor.tokenizer.decode(completion, skip_special_tokens=True).strip()

        dependence: dict[str, float] | None = None
        if args.image_dependence:
            losses: dict[str, float] = {}
            for condition, image_path in (
                ("correct", row["image_path"]),
                ("permuted", shuffled[row["id"]]),
            ):
                ablated = dict(row)
                ablated["image_path"] = image_path
                batch = collator([ablated])
                labels = batch.pop("labels").to(device)
                prepared = move_batch_to_device(batch, device)
                with torch_runtime.inference_mode():
                    outputs = model(**prepared)
                    loss = completion_only_causal_loss(
                        outputs.logits,
                        labels,
                        prepared.get("attention_mask"),
                    )
                losses[condition] = float(loss.item())
                if condition == "correct":
                    if adapter is None:
                        base_outputs = outputs
                    else:
                        with model.disable_adapter(), torch_runtime.inference_mode():
                            base_outputs = model(**prepared)
                    base_loss = completion_only_causal_loss(
                        base_outputs.logits,
                        labels,
                        prepared.get("attention_mask"),
                    )
                    losses["base_correct"] = float(base_loss.item())
            dependence = {
                **losses,
                "correct_minus_permuted": losses["correct"] - losses["permuted"],
            }

        result = {
            "settings_sha256": settings_sha,
            "example_id": row["id"],
            "candidate_output": text,
            "candidate_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "raw_valid_json": strict_context_json(text),
            "elapsed_seconds": elapsed,
            "prompt_tokens": input_length,
            "generation_tokens": int(completion.numel()),
            "natural_stop": int(completion.numel()) < args.max_new_tokens,
            "peak_memory_bytes": peak_memory_bytes(torch_runtime, device),
            "image_dependence": dependence,
        }
        append_jsonl(ledger, result)
        prior[row["id"]] = result
        print(canonical({"case": index, "total": len(rows), "valid": result["raw_valid_json"]}), flush=True)
        # Do not retain the prior row's full generation and five-view batch
        # until the next assignment. On MPS that pins many gigabytes of Metal
        # allocations and can force the following row into swap.
        del encoded, generated, completion, views, messages, prompt
        release_device_cache(torch_runtime, device)

    ordered = [prior[row["id"]] for row in rows]
    receipt = {
        "schema_version": "tt_telepathic_hf_eval_receipt_v1",
        "settings": settings,
        "settings_sha256": settings_sha,
        "rows": len(ordered),
        "raw_valid_json": sum(bool(row["raw_valid_json"]) for row in ordered),
        "natural_stops": sum(bool(row["natural_stop"]) for row in ordered),
        "ledger_sha256": sha256_file(ledger),
        "image_dependence": summarize_dependence(ordered),
    }
    receipt_path = ledger.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(canonical(receipt))
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("compact", "conditioned", "full"), required=True)
    candidate = parser.add_mutually_exclusive_group(required=True)
    candidate.add_argument("--adapter", type=Path)
    candidate.add_argument("--stock", action="store_true")
    parser.add_argument("--merge-adapter", action="store_true")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--ids-json", type=Path)
    parser.add_argument("--count", type=int)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--image-dependence", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    if args.merge_adapter and args.adapter is None:
        parser.error("--merge-adapter requires --adapter")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
