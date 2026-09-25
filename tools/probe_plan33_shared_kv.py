"""Plan 33 stock-E2B shared-K/V cache-on/cache-off teacher-forcing probe.

Rows are preregistered: the first N training and first N validation rows of the
frozen epoch-1 projection, rendered in full mode through the real trainer
collator. Only ``use_cache`` differs between the two forwards. Emits a redacted
receipt (counts, losses, agreement) with no content or IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/datasets/tt-screenshot-plan33-literal-v3-deploy/epoch-1"
MODEL_ID = "google/gemma-4-E2B-it"
REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
# Preregistered gates, frozen before the first probe was read.
MIN_TOP1_AGREEMENT = 0.99
MAX_RELATIVE_NLL_DELTA = 0.01


def load_rows(split: str, count: int) -> list[dict]:
    csv.field_size_limit(1 << 30)
    with (DATASET / f"{split}.csv").open(encoding="utf-8", newline="") as handle:
        rows = [row for _, row in zip(range(count), csv.DictReader(handle))]
    for row in rows:
        row["image_path"] = str((DATASET / row["image_path"]).resolve(strict=True))
        row["input_mode"] = "full"
    return rows


def score(model, batch: dict, use_cache: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = {k: v for k, v in batch.items() if k not in ("labels", "logits_to_keep")}
    with torch.no_grad():
        logits = model(**inputs, use_cache=use_cache).logits.float()
    labels = batch["labels"]
    shifted = logits[:, :-1]
    targets = labels[:, 1:]
    mask = targets != -100
    nll = torch.nn.functional.cross_entropy(shifted[mask], targets[mask], reduction="none")
    return nll, shifted[mask].argmax(-1), shifted[mask]


def main() -> None:
    import transformers

    from gemma_tuner.models.common.collators import IMAGE_VIEW_GLOBAL_PLUS_QUADRANTS, DataCollatorGemmaImage
    from gemma_tuner.models.gemma.base_model_loader import load_base_model_for_gemma
    from gemma_tuner.models.gemma.family import GemmaFamily

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-per-split", type=int, default=2)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    processor = transformers.AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
    model = load_base_model_for_gemma(
        MODEL_ID, family=GemmaFamily.GEMMA_4, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", revision=REVISION,
    ).to(device).eval()
    collator = DataCollatorGemmaImage(
        processor, "response", family=GemmaFamily.GEMMA_4, prompt_column="prompt",
        system_prompt_column="system_prompt", input_mode_column="input_mode",
        image_view_policy=IMAGE_VIEW_GLOBAL_PLUS_QUADRANTS, image_token_budget=280, sub_mode="vqa",
    )

    results = []
    for split in ("train", "validation"):
        for index, row in enumerate(load_rows(split, args.rows_per_split)):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in collator([row]).items()}
            started = time.time()
            nll_on, top_on, logits_on = score(model, batch, use_cache=True)
            nll_off, top_off, logits_off = score(model, batch, use_cache=False)
            mean_on, mean_off = float(nll_on.mean()), float(nll_off.mean())
            agreement = float((top_on == top_off).float().mean())
            relative = abs(mean_on - mean_off) / max(mean_on, 1e-6)
            results.append({
                "split": split,
                "preregistered_index": index,
                "prompt_tokens": int(batch["input_ids"].shape[1]),
                "target_tokens": int(nll_on.numel()),
                "nll_cache_on": round(mean_on, 6),
                "nll_cache_off": round(mean_off, 6),
                "relative_nll_delta": round(relative, 6),
                "top1_agreement": round(agreement, 6),
                "max_abs_logit_delta": round(float((logits_on - logits_off).abs().max()), 4),
                "seconds": round(time.time() - started, 2),
                "passed": agreement >= MIN_TOP1_AGREEMENT and relative <= MAX_RELATIVE_NLL_DELTA,
            })
            print(json.dumps(results[-1]), flush=True)

    receipt = {
        "schema_version": "plan33_shared_kv_probe_v1",
        "model": MODEL_ID,
        "revision": REVISION,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "device": device.type,
        "dtype": "bfloat16",
        "mode": "full",
        "gates": {"min_top1_agreement": MIN_TOP1_AGREEMENT, "max_relative_nll_delta": MAX_RELATIVE_NLL_DELTA},
        "rows": results,
        "passed": all(row["passed"] for row in results),
    }
    raw = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(raw, encoding="utf-8")
    print(raw, end="")


if __name__ == "__main__":
    main()
