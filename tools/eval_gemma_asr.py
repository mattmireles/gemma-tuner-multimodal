#!/usr/bin/env python3
"""
Evaluate Gemma 3n ASR on a CSV dataset using jiwer (WER/CER).

- Loads a base Gemma 3n model and optional LoRA adapters
- Uses AutoProcessor to prepare multimodal inputs (audio + messages)
- Runs greedy generation and computes WER/CER against references

Usage:
  python tools/eval_gemma_asr.py \
      --csv data/datasets/myset/validation.csv \
      --model google/gemma-3n-E2B-it \
      --adapters ./output/my_adapter \
      --text-column text \
      --limit 200
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

try:
    from peft import PeftModel
except Exception:
    PeftModel = None  # Optional if no adapters are used

try:
    from jiwer import cer, wer
except Exception:
    wer = cer = None  # User must install jiwer

# Shared audio loader to support local files and GCS
try:
    from gemma_tuner.utils.dataset_prep import load_audio_local_or_gcs
except Exception:
    load_audio_local_or_gcs = None

# Canonical device selection (MPS > CUDA > CPU) from shared utils
from gemma_tuner.utils.device import get_device, probe_bfloat16


def build_messages(transcript_hint: Optional[str] = None) -> List[Dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": "<audio:attached>"},
                {"type": "text", "text": "Please transcribe this audio."},
            ],
        },
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Gemma ASR evaluation (WER/CER)")
    ap.add_argument("--csv", required=True, help="Validation CSV with audio_path and reference text")
    ap.add_argument("--model", default="google/gemma-4-E2B", help="Base Gemma model id")
    ap.add_argument("--adapters", help="Path to LoRA adapters (optional)")
    ap.add_argument("--text-column", default="text", help="Reference transcript column name")
    ap.add_argument("--limit", type=int, default=0, help="Max rows to evaluate (0 = all)")
    args = ap.parse_args()

    if wer is None or cer is None:
        print("[ERROR] jiwer not installed. Run: pip install jiwer")
        return 2

    device = get_device()

    # Prefer bf16 when available (MPS tensor probe, CUDA is_bf16_supported); else float32
    dtype = torch.bfloat16 if probe_bfloat16(device) else torch.float32

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    if args.adapters and PeftModel is not None:
        model = PeftModel.from_pretrained(model, args.adapters)
    model.eval()
    model.to(device)

    # Read CSV
    refs: List[str] = []
    hyps: List[str] = []
    rows = 0

    with Path(args.csv).open("r", newline="") as rf:
        reader = csv.DictReader(rf)
        if "audio_path" not in (reader.fieldnames or []):
            raise ValueError("CSV must contain 'audio_path' column")
        if args.text_column not in (reader.fieldnames or []):
            raise ValueError(f"CSV must contain '{args.text_column}' column for references")

        for row in reader:
            audio_path = (row.get("audio_path") or row.get("audio") or "").strip()
            ref = (row.get(args.text_column) or "").strip()
            if not audio_path:
                continue
            if args.limit and rows >= args.limit:
                break

            # Load audio
            if load_audio_local_or_gcs is None:
                raise RuntimeError("utils.dataset_prep.load_audio_local_or_gcs not available")
            # Honor processor sampling rate when available
            sr = None
            try:
                sr = getattr(processor, "sampling_rate", None)
                if sr is None and hasattr(processor, "feature_extractor"):
                    sr = getattr(processor.feature_extractor, "sampling_rate", None)
            except Exception:
                sr = None
            audio = load_audio_local_or_gcs(audio_path, sampling_rate=sr)

            messages = build_messages()
            prompts = processor.apply_chat_template(
                [messages],
                tokenize=False,
                add_generation_prompt=False,
            )
            enc = processor(
                text=prompts,
                audio=[audio],
                return_tensors="pt",
                padding=True,
                sampling_rate=sr,
            )
            enc = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc.items()}

            # Generate
            if device.type == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=128)
            if device.type == "mps":
                torch.mps.synchronize()
            _ = time.perf_counter() - t0

            # model.generate() returns [input_ids + new_tokens]; slice off the prompt
            # so we only decode the tokens the model actually generated. Without this
            # slice, batch_decode includes the prompt text in the hypothesis and every
            # WER/CER value produced would be inflated/wrong.
            input_len = enc["input_ids"].shape[1]
            new_tokens = out[:, input_len:]
            if hasattr(processor, "tokenizer"):
                hyp = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
            else:
                hyp = "<decoded text unavailable>"

            refs.append(ref)
            hyps.append(hyp)
            rows += 1
            if rows % 20 == 0:
                print(f"Processed {rows} samples...")

    # Compute metrics
    score_wer = wer(refs, hyps)
    score_cer = cer(refs, hyps)
    print(f"WER: {score_wer:.4f}  CER: {score_cer:.4f}  (n={rows})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
