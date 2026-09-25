"""Plan 33 unmerged-vs-merged HF parity canary for an E2B LoRA adapter.

Training computes the unmerged PEFT form, where the LoRA path bypasses the
vision clippable-linear clamps. Merged HF and MLX inference fold the delta
inside the clamps. On preregistered frozen real screenshots (first N
validation rows, full mode), compare projected image embeddings and a short
greedy continuation between the two forms. Emits a redacted receipt: numeric
deltas, token agreement, and first divergence only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/datasets/tt-screenshot-plan33-literal-v3-deploy/epoch-1"
MODEL_ID = "google/gemma-4-E2B-it"
REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"


def rows(count: int) -> list[dict]:
    csv.field_size_limit(1 << 30)
    with (DATASET / "validation.csv").open(encoding="utf-8", newline="") as handle:
        selected = [row for _, row in zip(range(count), csv.DictReader(handle))]
    for row in selected:
        row["image_path"] = str((DATASET / row["image_path"]).resolve(strict=True))
    return selected


def measure(model, processor, row: dict, new_tokens: int, device: torch.device) -> dict:
    from gemma_tuner.models.common.collators import (
        IMAGE_VIEW_GLOBAL_PLUS_QUADRANTS,
        _load_image_as_rgb,
        build_image_views,
    )
    from gemma_tuner.models.common.plan31_input_modes import render_plan31_input

    views = build_image_views(_load_image_as_rgb(row["image_path"]), IMAGE_VIEW_GLOBAL_PLUS_QUADRANTS)
    _, messages = render_plan31_input(mode="full", full_prompt=row["prompt"],
                                      system_prompt=row["system_prompt"], full_views=views)
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt", enable_thinking=False,
    ).to(device)
    captured: list[torch.Tensor] = []
    embed = next(module for name, module in model.named_modules() if name.endswith("model.embed_vision"))
    handle = embed.register_forward_hook(lambda _m, _i, out: captured.append(out.detach().float().cpu()))
    try:
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False)
    finally:
        handle.remove()
    prompt_length = inputs["input_ids"].shape[1]
    return {"image_embeddings": captured[0], "tokens": output[0, prompt_length:].tolist()}


def main() -> None:
    import transformers
    from peft import PeftModel

    from gemma_tuner.models.common.collators import apply_image_token_budget_to_processor
    from gemma_tuner.models.gemma.base_model_loader import load_base_model_for_gemma
    from gemma_tuner.models.gemma.family import GemmaFamily

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()
    if transformers.__version__ != "5.5.2":
        raise RuntimeError("merge parity must run under Transformers 5.5.2")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    processor = transformers.AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
    apply_image_token_budget_to_processor(processor, 280)
    base = load_base_model_for_gemma(MODEL_ID, family=GemmaFamily.GEMMA_4, torch_dtype=getattr(torch, args.dtype),
                                     attn_implementation="sdpa", revision=REVISION)
    model = PeftModel.from_pretrained(base, str(args.adapter)).to(device).eval()
    frozen = rows(args.rows)
    unmerged = [measure(model, processor, row, args.new_tokens, device) for row in frozen]
    started = time.time()
    model = model.merge_and_unload().eval()
    merged = [measure(model, processor, row, args.new_tokens, device) for row in frozen]

    results = []
    for index, (a, b) in enumerate(zip(unmerged, merged)):
        delta = (a["image_embeddings"] - b["image_embeddings"]).abs()
        scale = a["image_embeddings"].abs().mean()
        tokens_a, tokens_b = a["tokens"], b["tokens"]
        first = next((i for i, (x, y) in enumerate(zip(tokens_a, tokens_b)) if x != y),
                     None if len(tokens_a) == len(tokens_b) else min(len(tokens_a), len(tokens_b)))
        results.append({
            "preregistered_index": index,
            "image_embedding_max_abs_delta": round(float(delta.max()), 6),
            "image_embedding_mean_abs_delta": round(float(delta.mean()), 6),
            "image_embedding_mean_abs": round(float(scale), 6),
            "relative_mean_delta": round(float(delta.mean() / scale), 6),
            "greedy_tokens": [len(tokens_a), len(tokens_b)],
            "greedy_token_agreement": round(sum(x == y for x, y in zip(tokens_a, tokens_b)) / max(len(tokens_a), 1), 6),
            "first_divergent_token": first,
        })
        print(json.dumps(results[-1]), flush=True)
    receipt = {
        "schema_version": "plan33_merge_parity_v1",
        "adapter_complete_sha256": hashlib.sha256((args.adapter / ".complete.json").read_bytes()).hexdigest()
        if (args.adapter / ".complete.json").exists() else None,
        "transformers": transformers.__version__, "device": device.type, "dtype": args.dtype,
        "rows": results, "merge_seconds": round(time.time() - started, 2),
        "comparison": "PEFT unmerged (training form) vs merge_and_unload (MLX inference form), same HF runtime",
    }
    raw = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(raw, encoding="utf-8")
    print(raw, end="")


if __name__ == "__main__":
    main()
