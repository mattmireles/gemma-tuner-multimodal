# Gemma 4 on Apple Silicon: Multimodal LoRA Field Guide

> **Scope:** General Hugging Face + PyTorch MPS guidance for Gemma 4 multimodal fine-tuning. Training with **this repository**, use [`../../../README.md`](../../../README.md), [`../../../pyproject.toml`](../../../pyproject.toml), and [`../../../gemma_tuner/core/bootstrap.py`](../../../gemma_tuner/core/bootstrap.py) as the source of truth for pinned versions and MPS defaults.

**Related in this repo:** [`gemma3n.md`](gemma3n.md) (shared architecture context) · [`pytorch-mps.md`](pytorch-mps.md) · [`HF-transformers-MPS-guide.md`](HF-transformers-MPS-guide.md) · [`LoRA-Apple-Silicon-Guide.md`](LoRA-Apple-Silicon-Guide.md) · [`../../specifications/Gemma3n.md`](../../specifications/Gemma3n.md)

---

Google’s Gemma 4 and Gemma 3n lines support native multimodal stacks (text, image, audio) in a single model. On Apple Silicon, unified memory helps load larger effective parameter counts than typical discrete-GPU VRAM limits, but **PEFT on MPS** is a different problem than CUDA: undocumented edge cases, precision quirks, framework bugs, and modality-specific wiring (processors, collators, token-type tensors).

This guide focuses on **LoRA-style** adaptation for multimodal Gemma on macOS with the **MPS** backend: environment pins, memory limits, attention settings, PEFT pitfalls, and modality preprocessing. Where upstream issues matter, we link the GitHub threads; where behavior is **only defined in this repo** (CLI, bootstrap, pins), we point at source files.

## Stack versions: this repo vs “latest HF”

**Core dependency bounds** are in [`pyproject.toml`](../../../pyproject.toml) (and mirrored in [`requirements/requirements.txt`](../../../requirements/requirements.txt)). **`transformers` is pinned at `>=5.5.0` in the base tree** (Gemma 4’s minimum) so `pip install -e .` and `uv sync` do not resolve back to Transformers 4.x. There is **no upper cap** on `transformers` beyond that floor.

| Package | Base (this repo) | Optional [`requirements-gemma4.txt`](../../../requirements/requirements-gemma4.txt) (PEFT / explicit stack) |
| --- | --- | --- |
| `transformers` | `>=5.5.0` | same floor (repeated for manual / incremental installs) |
| `peft` | `>=0.9.0` | `>=0.18.1` (recommended for Gemma 4 LoRA) |
| `datasets` | `>=4.0.0` | (unchanged) |
| `accelerate` | `>=0.27.2` | (unchanged) |

**Gemma 3n** and **Gemma 4** both use the same base `transformers` floor here (5.5+). You can still `pip install -r requirements/requirements-gemma4.txt` to align **PEFT** with the Gemma 4 stack in one shot. Run tests and `gemma_preflight` after any major version bump. If you maintain a **separate** environment outside this project, consult [Hugging Face model cards](https://huggingface.co/google) and release notes for minimum versions.

Audio preprocessing in training stacks typically uses **librosa** + **soundfile**; vision stacks use **Pillow**. Match versions to your lockfile or `requirements/requirements.txt`.

## MPS memory and `PYTORCH_MPS_HIGH_WATERMARK_RATIO`

PyTorch’s MPS allocator can enforce a **high watermark**; under training spikes you may see errors suggesting `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` to lift the cap (see [PyTorch discussion](https://github.com/pytorch/pytorch/issues/152351)).

**In this codebase,** `gemma_tuner.core.bootstrap` runs on import and **clamps** `PYTORCH_MPS_HIGH_WATERMARK_RATIO` to strictly between **0 and 1** (exclusive). A value of **`0.0` is treated as invalid** and is replaced with the default fraction (~`0.80`). For day-to-day tuning here, follow [`README.md`](../../../README.md) (“Apple Silicon knobs”): try **`0.7`–`0.9`**, reduce batch size, and use gradient checkpointing before fighting the allocator.

Set any ratio **before** `import torch` if you need it to apply at allocator init; the README documents the intended workflow.

```python
import os

# Example: cap allocator appetite (must be strictly between 0 and 1 for this repo's bootstrap)
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.8"

import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
```

## Bfloat16, attention backends, and NaNs

Gemma is typically trained/evaluated in **bfloat16** where supported. On MPS, some **scaled dot-product attention** paths have had numerical rough edges; many teams force **`attn_implementation="eager"`** for stability when the stack allows it.

```python
model_id = "google/gemma-4-E2B-it"

model = AutoModelForMultimodalLM.from_pretrained(
    model_id,
    device_map="mps",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_implementation="eager",  # often recommended for MPS stability
)
```

Keep **`low_cpu_mem_usage=True`** when loading large models to avoid a full RAM resident copy before MPS migration.

If loss is **NaN** or **0.000** early on, treat it as a signal to try float32 for the run, smaller LR, or different attention settings—MPS behavior can differ from CUDA recipes.

## `torch.compile` on MPS

`torch.compile` can be immature on MPS for large, multimodal graphs. With **`Trainer`**, keep **`torch_compile=False`** in `TrainingArguments` unless you have a minimal repro and time to debug inductor/Metal failures.

## Hardware and memory budgeting

“E2B” / “E4B” denote **effective** parameter tiers; PLE and routing affect inference, but **training** still needs space for weights, activations, optimizer states, and (for full fine-tuning) much more than LoRA. Treat any RAM table as **heuristic**—measure with your sequence lengths and modalities.

| Unified RAM (illustrative) | E2B multimodal | E4B multimodal | Notes |
| --- | --- | --- | --- |
| 16 GB | Tight | Often OOM | Batch size 1; expect swap if anything else is resident |
| 32 GB | Usual sweet spot for E2B LoRA | Tighter | Reduce resolution / audio length aggressively |
| 64 GB+ | Comfortable for E2B | Better headroom | Still watch multimodal tensor sizes |

Use **gradient accumulation** to emulate larger batches: e.g. `per_device_train_batch_size=1` with `gradient_accumulation_steps=16` for effective batch 16.

Community **throughput** figures (samples/sec) vary widely by CPU, thermals, and modality—benchmark your own job rather than trusting a single number from a blog.

## PEFT: `Gemma4ClippableLinear` and LoRA targets

Hugging Face **PEFT** expects standard `nn.Linear` modules for many target selectors. Gemma 4 multimodal code paths include specialized linear wrappers (e.g. **`Gemma4ClippableLinear`**) that may **not** match PEFT’s linear checks, which can surface as errors when attaching adapters—see [PEFT #3129](https://github.com/huggingface/peft/issues/3129). Mitigations described in the wild include **monkey-patching** the class to inherit from `nn.Linear` before model construction; treat that as **advanced** and validate against your exact `transformers`/`peft` pair.

Avoid blindly using **`target_modules="all-linear"`** on multimodal stacks: it can hit heads, routers, or embeddings you did not intend. Prefer **explicit** module names from `model.named_modules()` for your checkpoint.

```python
from peft import LoraConfig, get_peft_model

# Example only — resolve real suffixes via named_modules() for your revision
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

Small datasets rarely benefit from huge rank; **r** in **8–32** with **alpha ≈ 2×r** is a common starting range.

## `mm_token_type_ids` and custom collators

Some Gemma 4 training paths expect **`mm_token_type_ids`** (and possibly **`token_type_ids`**) in the batch. If the processor does not emit them for your pipeline, you may need a **custom collator** that pads **`input_ids`** and adds missing tensors (e.g. zeros shaped like `input_ids`). See [transformers #45200](https://github.com/huggingface/transformers/issues/45200) for discussion.

```python
import torch

class Gemma4MultimodalCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        messages = [ex["messages"] for ex in examples]
        batch = self.processor.apply_chat_template(
            messages,
            padding=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        zeros = torch.zeros_like(batch["input_ids"])
        if "token_type_ids" not in batch:
            batch["token_type_ids"] = zeros
        if "mm_token_type_ids" not in batch:
            batch["mm_token_type_ids"] = zeros
        batch["labels"] = batch["input_ids"].clone()
        return batch
```

For huge corpora, **`datasets` streaming** (`streaming=True`) keeps resident memory flatter than materializing every asset.

## Text: instruction tuning and label masking

Starting from **`-it`** checkpoints usually reduces the work LoRA must do for chat formatting. For SFT, mask **prompt** tokens so loss applies only to the assistant span—often by setting ignored label positions (e.g. **`-100`**) for everything before the assistant turn. Exact control tokens depend on the tokenizer and template; use **`encode`** / **`apply_chat_template`** to locate boundaries rather than hard-coding fragile string splits.

Gemma 4 may emit **reasoning / “think”** segments; keep training examples **consistent** with the format you want at inference (omit thought traces if you want terse JSON-only outputs).

## Images: RGB and token budget

Force images to **RGB** before the processor when your dataset includes CMYK/RGBA assets—wrong layouts cause silent quality loss.

Gemma 4 vision uses a configurable **visual token budget** (tiers such as 70–1120 tokens depending on model and API). Higher budgets help OCR/dense documents; they cost memory and time.

## Audio: sample rate and scaling

Follow the **processor / model card** for input rate and layout. Many Gemma audio paths expect **mono**, **16 kHz**, float waveforms in a bounded range—**resample** with librosa (or equivalent) before batching.

```python
import librosa
import numpy as np

def prepare_audio_for_gemma(audio_path: str):
    audio_array, _ = librosa.load(audio_path, sr=16000, mono=True, dtype=np.float32)
    return np.clip(audio_array, -1.0, 1.0)
```

ASR-style eval often uses **WER/CER** (e.g. via `evaluate` + optional **`jiwer`** in this repo’s eval extras).

## Trainer on MPS: evaluation and metrics

Some users report **`Trainer.evaluate()`** stalling or odd memory metric behavior on MPS. **`skip_memory_metrics=True`** in `TrainingArguments` is a common mitigation ([transformers #27181](https://github.com/huggingface/transformers/issues/27181)). If evaluation still misbehaves, run eval in a **separate** script or disable interval eval and validate offline.

## Merging LoRA for export

Merge or export with the model on a path that avoids known MPS edge cases: many workflows **`to("cpu")`** before **`merge_and_unload()`**, then `save_pretrained(..., safe_serialization=True)`.

## When local training stops making sense

Consider cloud GPUs if:

- Dataset I/O or preprocessing dominates wall time for days.
- Epochs exceed your iteration budget (e.g. 12+ hours per epoch on your hardware for the target quality bar).
- You need **QLoRA / bitsandbytes**-style 4-bit training—**not** a first-class MPS training path today.

For **format-only** or **fact** tasks, strong prompting, few-shot examples, or **RAG** may beat small LoRA runs—match the tool to the metric.

## Conclusion

Apple Silicon can run serious Gemma multimodal **LoRA** work if you respect MPS allocator behavior, version pins, modality tensors, and PEFT/module quirks. For **this project**, keep **`pyproject.toml`**, **`bootstrap.py`**, and the main **[`README.md`](../../../README.md)** in view whenever advice from generic blogs conflicts with the trainer you actually run.

## References (selected)

- [Gemma 4 model card (Google AI)](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Gemma 4 on Hugging Face (example)](https://huggingface.co/google/gemma-4-E4B)
- [PyTorch MPS high watermark discussion](https://github.com/pytorch/pytorch/issues/152351)
- [PEFT Gemma4ClippableLinear / LoRA](https://github.com/huggingface/peft/issues/3129)
- [Transformers `mm_token_type_ids` discussion](https://github.com/huggingface/transformers/issues/45200)
- [Trainer memory metrics on MPS](https://github.com/huggingface/transformers/issues/27181)
- [Gemma audio capabilities (Google AI)](https://ai.google.dev/gemma/docs/capabilities/audio)
- [Gemma 4 prompt formatting (Google AI)](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4)
