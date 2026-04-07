# Image Fine-Tuning Plan

**Related:** `mm_token_type_ids` injection, family detection, and the `Gemma4ClippableLinear` patch are **shipped** by [gemma4-upgrade.md](gemma4-upgrade.md). Image mode must reuse `family_capabilities`, route loads through `_load_base_model_for_gemma`, and pass `family=` into the new collator — exactly as the audio collator does today.

**Date:** 2026-04-06  
**Status:** Complete  
**Completed:** 2026-04-07 (implementation landed on `main`; see git history for phase commits)

## Executive Summary

**Shipped:** An image fine-tuning path alongside existing **audio** and **text** paths. Targets **both Gemma 3n and Gemma 4** via existing family branching, clippable-linear patch, multimodal loader (`_load_base_model_for_gemma`), and `mm_token_type_ids` injection from [gemma4-upgrade.md](gemma4-upgrade.md). Delivers `DataCollatorGemmaImage` (caption + VQA), `modality=image` in config / `finetune.py` / `dataset_utils.py` / wizard, `image_token_budget` applied at processor load (`apply_image_token_budget_to_processor`), relative `image_path` resolution in `finetune.py`, export-time budget reapply from `metadata.json`, docs (README, Datasets, TROUBLESHOOTING, KNOWN_ISSUES), and tests including `tests/test_gemma_image_collator.py` and offline `tests/test_smoke_image.py`.

**Gated multimodal smoke (release gate):** `tests/test_smoke_image_multimodal.py` runs caption + VQA through `finetune.main()` for ≥10 steps on `google/gemma-3n-E4B-it`, asserting finite `train_loss`. It requires `HF_TOKEN` / `huggingface-cli login` and is marked `@pytest.mark.integration`; without credentials the tests **skip** so default CI stays green. The repo’s tiny HF stub still cannot substitute for this — it has no vision `AutoProcessor`.

## Problem Statement

- **Symptom:** Users who want to fine-tune Gemma on image-text pairs (VQA, captioning, OCR, document parsing, niche visual recognition) could not use this repo. Previously the trainer exposed only audio + text collators for multimodal CSV paths.
- **Root Cause (historical):** Collators and `finetune.py` had no `modality=image` branch or image-shaped chat + processor flow.
- **Impact:** Image-mode is one of the three native Gemma modalities and arguably the most commercially relevant for Mac developers (document OCR, screenshot understanding, niche visual classification). Leaving it unsupported caps the addressable use cases at ~1/3 of what the model can do.

## Mode Definitions

| Mode | Behavior | Why it matters |
| --- | --- | --- |
| `audio` (default) | Existing path: audio + transcript via `DataCollatorGemmaAudio`. | Preserves every existing profile with zero edits. |
| `text` | Text-only via `DataCollatorGemmaText` (instruction or completion). | Landed by [text-only-finetuning.md](text-only-finetuning.md). |
| `image` + `image_sub_mode=vqa` | Image + question + answer columns, chat-templated, prompt tokens masked. | Visual question answering and instruction-style image tuning. |
| `image` + `image_sub_mode=caption` | Image + caption columns, prompt is a fixed system instruction, response is the caption. | Captioning, OCR, document parsing. Single most common image-SFT use case. |

## Goals and Non-Goals

### Goals

- [x] A new `DataCollatorGemmaImage` supporting both VQA and caption sub-modes.
- [x] `modality = image` accepted by the profile loader, finetune.py, dataset_utils.py, wizard.
- [x] `image_token_budget` config key (∈ {70, 140, 280, 560, 1120}; default 280) plumbed to the processor.
- [x] Mandatory PIL `.convert("RGB")` enforced **inside the collator**, not the dataset loader, to prevent preprocessing skew silently corrupting CMYK/RGBA inputs.
- [x] Wizard "What kind of fine-tuning?" question gains image branches.
- [x] Image smoke coverage: `tests/test_smoke_image.py` (offline dataset + collator); gated full `finetune.main()` smoke on real vision via `tests/test_smoke_image_multimodal.py` (see Executive Summary).
- [x] README comparison table reflects image support honestly.
- [x] Image-mode memory estimator branch in `gemma_tuner/wizard/estimator.py`.

### Non-Goals

- **BLEU / CIDEr / VQA-accuracy eval callbacks.** The HF Trainer compute_metrics path is hostile to text generation; building a custom callback is deferred to v2. v1 reports eval loss and perplexity only.
- **Streaming image datasets.** v1 restricts image mode to the CSV adapter (same as text). **GCS URIs for image paths:** not implemented in v1; collator loads local paths (or absolute paths after `finetune.py` resolves relative to `data/datasets/<name>/`). Add `load_image_local_or_gcs` later if needed.
- **Multi-image-per-sample inputs.** Gemma 3n/4 supports it, but the schema and collator complexity isn't justified for v1. One image per sample.
- **Image augmentation pipeline** (random crop, color jitter, etc.). Out of scope.
- **Stripping the audio tower from `AutoModelForCausalLM` in image mode.** Same RAM-overhead trade-off documented in the text PR.
- **New Gemma 4 monkey-patches.** `Gemma4ClippableLinear` is already patched in [`gemma_tuner/models/gemma/gemma4_patches.py`](../../gemma_tuner/models/gemma/gemma4_patches.py) and applied by the loader; image mode inherits it for free. Do not add new vision-tower patches in v1.

## Scope and Constraints

- **Scope:** Image collator, finetune.py modality branch, dataset schema validation, profile schema, wizard branch, smoke test, README + TROUBLESHOOTING updates.
- **Constraints:** Must not regress existing audio or text profiles. Depends on [text-only-finetuning.md](text-only-finetuning.md) modality plumbing (**landed** before this work).
- **Guardrails:** Shared LoRA / Trainer / MPS / export / run-index infrastructure stays untouched.

## Ground Truth Contracts (Do Not Violate)

- **Default `modality = audio`.** Image mode is opt-in via profile or wizard. Existing INI files must work with zero edits.
- **RGB conversion happens at the collator boundary, not earlier.** Datasets can ship CMYK or RGBA; the collator must `.convert("RGB")` every image before handing it to the processor. Skipping this is the #1 silent failure mode (matches the gemma4-guide "preprocessing skew footgun").
- **Always use `processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)` followed by `processor(text=prompts, images=…, return_tensors="pt", padding=True)`.** This matches the audio collator's two-step flow today. Never call `processor(messages=…)` directly (broken on current transformers) and never hand-roll `<start_of_image>` strings. The control-token branching (`<start_of_turn>` for 3n vs `<|turn|>` for 4) is already handled by the shared `mask_gemma_prompt_tokens` helper — pass `family` and reuse it.
- **`image_token_budget` is a training/inference contract.** Whatever budget is used during training MUST be used at inference. The training profile (including `image_token_budget`) is stored in run **`metadata.json`** via the CLI; [`gemma_tuner/scripts/export.py`](../../gemma_tuner/scripts/export.py) reapplies the budget to the processor when `config.modality == image` and `config.image_token_budget` are present. A train/serve mismatch silently degrades quality.
- **Pad-token EOS masking via `attention_mask == 0`, not `labels == pad_id`.** Same footgun as text mode; the shared helper extracted by the text PR already does this correctly.

## Already Shipped (Do Not Re-Solve)

- **LoRA / PEFT injection** in [`gemma_tuner/models/gemma/finetune.py`](../../gemma_tuner/models/gemma/finetune.py) — modality-agnostic, reuse as-is.
- **MPS device + bfloat16 detection** in [`gemma_tuner/utils/device.py`](../../gemma_tuner/utils/device.py) and [`finetune.py:267`](../../gemma_tuner/models/gemma/finetune.py#L267) — reuse.
- **Run index, metrics persistence, export** in [`gemma_tuner/models/common/results.py`](../../gemma_tuner/models/common/results.py) and [`gemma_tuner/scripts/export.py`](../../gemma_tuner/scripts/export.py) — reuse.
- **Modality switch + shared collator helpers** delivered by [text-only-finetuning.md](text-only-finetuning.md) — `validate_bos_tokens_present`, attention-mask-based label masking.
- **Family detection + capability flags** in [`gemma_tuner/models/gemma/family.py`](../../gemma_tuner/models/gemma/family.py): `detect_family`, `family_capabilities`, `gate_gemma_model(model_id, entrypoint=...)`. Wizard calls `gate_gemma_model` during model pick; `finetune.main()` uses `detect_family` + `assert_family_supported` and passes `family` into `DataCollatorGemmaImage`.
- **Gemma 4 clippable-linear patch** in [`gemma_tuner/models/gemma/gemma4_patches.py`](../../gemma_tuner/models/gemma/gemma4_patches.py) — applied automatically by `_load_base_model_for_gemma`.
- **Family-aware loader** `_load_base_model_for_gemma(model_id, *, family, ...)` in [`gemma_tuner/models/gemma/finetune.py`](../../gemma_tuner/models/gemma/finetune.py) — picks `AutoModelForMultimodalLM` / `AutoModelForImageTextToText` for Gemma 4 and `AutoModelForCausalLM` for 3n. Image mode reuses this with no new branching.
- **`mm_token_type_ids` injection** via `inject_mm_token_type_ids` / `ensure_gemma_mm_token_type_ids` in [`gemma_tuner/models/common/collators.py`](../../gemma_tuner/models/common/collators.py), gated on `caps["needs_mm_token_type_ids_injection"]` (True for both families today).
- **Family-parameterised prompt masking** via `mask_gemma_prompt_tokens(..., control_token=caps["control_token"])` — reuse as-is in the image collator.
- **CSV loader, patch/blacklist/protection system** in [`gemma_tuner/utils/dataset_utils.py`](../../gemma_tuner/utils/dataset_utils.py) — reuse with one schema-validation tweak.
- **Profile schema scaffolding** (`modality` key, validation, defaults) — landed by the text PR.

## Fresh Baseline (Post-Implementation)

- **Architecture:** `finetune.py` branches `audio` / `text` / `image`; image uses `DataCollatorGemmaImage`, `AutoProcessor`, and `_load_base_model_for_gemma` like audio.
- **Profiles:** INI supports `modality = image` plus `image_sub_mode`, `image_path_column`, `image_token_budget` (see [`README/Datasets.md`](../Datasets.md), `config.ini.example`).
- **Wizard:** Five task kinds including image caption and image VQA; `configure_image_columns()`; estimator warns when image memory estimate exceeds ~80% RAM.
- **Tests:** `tests/test_gemma_image_collator.py`, `tests/data/image_caption_tiny/`, `tests/test_smoke_image.py` (offline pipeline), `tests/test_smoke_image_multimodal.py` (gated: real `google/gemma-3n-E4B-it`, ≥10 steps). Integration smokes for audio/text unchanged.

## Solution Overview

```
+-----------------+   modality=audio   +---------------------------+
|  finetune.py    | -----------------> | DataCollatorGemmaAudio    |
|  modality       |                    +---------------------------+
|  switch         |   modality=text    +---------------------------+
|                 | -----------------> | DataCollatorGemmaText     |
|                 |                    +---------------------------+
|                 |   modality=image   +---------------------------+
|                 | -----------------> | DataCollatorGemmaImage    |
+-----------------+                    | - vqa sub-mode            |
       |                               | - caption sub-mode        |
       v                               | - PIL.convert("RGB")      |
+--------------------+                 | - image_token_budget      |
| shared helpers     |                 +---------------------------+
| (from text PR)     |
+--------------------+
```

`AutoProcessor.from_pretrained(model_id)` plus `apply_image_token_budget_to_processor` (mutates `image_seq_length` / `full_image_sequence` on Gemma3-style processors). WER metrics are **audio-only**; image uses loss + perplexity in `train_results.json` like text.

## Implementation Phases

> Phases completed in order; each left audio + text profiles runnable. Prerequisite: [text-only-finetuning.md](text-only-finetuning.md) complete.

### Phase 0: Profile Schema Extension

**Goal:** Add image-mode config keys with audio-preserving defaults. No behavior change.

**Tasks:**

- [x] Extend the `modality` enum in `gemma_tuner/core/config.py` (or wherever the text PR registered it) to accept `image`.
- [x] Add `image_sub_mode` (default `caption`), `image_path_column` (default `image_path`), `image_token_budget` (default 280, validated against {70, 140, 280, 560, 1120}), `prompt_column` (already exists from text PR; reuse for VQA question column).
- [x] Update `config.ini.example` with commented-out image-mode keys.
- [x] Document new keys in [`README/Datasets.md`](../Datasets.md).

**Verification:** Existing audio + text profiles load and run unchanged.

---

### Phase 1: `DataCollatorGemmaImage`

**Goal:** Add the image collator with both sub-modes. Not yet wired into finetune.py.

**Tasks:**

- [x] Add `DataCollatorGemmaImage` class to `gemma_tuner/models/common/collators.py`. Signature: `__init__(self, processor, text_column, *, family: GemmaFamily, image_path_column="image_path", prompt_column=None, image_token_budget=280, sub_mode="caption")`. Store `self._caps = family_capabilities(family)` once at init (do not call per batch). Mirror `DataCollatorGemmaAudio` exactly for the family / caps / control-token plumbing.
- [x] Implement `_load_image_as_rgb(path)` helper that opens via PIL and **always** calls `.convert("RGB")`. Wrap in try/except with a clear error pointing at the offending row id.
- [x] **Caption sub-mode:** Build 2-turn message: user content = `[{"type": "image", "image": img}, {"type": "text", "text": "Describe this image."}]` (or a configurable system instruction), assistant content = caption. Render via `processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)`, then call `processor(text=prompts, images=images, return_tensors="pt", padding=True)`. Then run `inject_mm_token_type_ids` when `caps["needs_mm_token_type_ids_injection"]`.
- [x] **VQA sub-mode:** user content = `[{"type": "image", "image": img}, {"type": "text", "text": row[prompt_column]}]`, assistant content = row[text_column]. Same two-step apply_chat_template + processor flow.
- [x] Pass `image_token_budget` via `apply_image_token_budget_to_processor` (sets `image_seq_length` and rebuilds `full_image_sequence` on Gemma3-style processors); called from collator `__init__` and from `finetune.main()` after processor load.
- [x] Reuse the shared `validate_bos_tokens_present` and `mask_gemma_prompt_tokens(..., control_token=self._caps["control_token"])` helpers. Attention-mask-based label masking (NOT `labels == pad_id`).
- [x] Add `tests/test_gemma_image_collator.py` with both sub-modes. **Critical assertions:** (1) RGBA and CMYK fixture images load successfully and produce identical tensor shapes to RGB inputs; (2) VQA mask boundary; (3) padding masked via `attention_mask` (EOS/pad contract covered by that path in the fake processor test).

**Verification:** New collator tests pass. Audio + text collator tests still pass.

---

### Phase 2: `dataset_utils.py` Schema Branching

**Goal:** Loosen schema validation for image-mode CSVs.

**Tasks:**

- [x] Extend the `required_columns` branch added by the text PR in `gemma_tuner/utils/dataset_utils.py`. Image mode requires `id` + `image_path_column` + `text_column` + (`prompt_column` if VQA).
- [x] Extend `DatasetLoadContext` in `gemma_tuner/utils/dataset_sources.py` to carry `image_sub_mode` and `image_path_column` (defaults `caption` / `image_path`).
- [x] Reject `modality=image` in non-CSV adapters (BigQuery, Granary, GCS streaming) with a clear error pointing at v1 scope.
- [x] Add a `tests/data/image_caption_tiny/` fixture: 8 tiny images (RGB / RGBA / one CMYK as `.tif`) + `train.csv` with `id`, `image_path`, `caption` (split file name matches loader expectations).

**Verification:** Existing audio + text fixtures still load. The new image fixture loads successfully.

---

### Phase 3: `finetune.py` Image Branch

**Goal:** Wire the image collator into the trainer.

**Tasks:**

- [x] In the modality switch, add an `image` branch.
- [x] Image mode loads `AutoProcessor.from_pretrained(model_id)` (same as audio). **Model loading** goes through `_load_base_model_for_gemma(model_id, family=family, ...)`.
- [x] Branch collator instantiation: `DataCollatorGemmaImage(...)`. `family` from `detect_family` in `main()`.
- [x] Branch metrics: image uses `compute_metrics=None` and `preprocess_logits_for_metrics=None` (WER is **audio-only**). Perplexity for image in `results.py` like text.
- [x] `image_token_budget` is part of the profile and is written into run **`metadata.json`** with the rest of `config` by the CLI; export reapplies it to the saved processor when metadata is present beside the adapter.
- [x] `attn_implementation` from profile (group default `eager`).

**Verification:** Automated tests pass; **≥10-step image training** is manual with a real multimodal checkpoint (see Executive Summary).

---

### Phase 4: Wizard

**Goal:** Make image mode discoverable and foot-gun-free.

**Tasks:**

- [x] In `gemma_tuner/wizard/ui.py`, extend Step 1 choices: image caption/OCR and image VQA (`modality=image`, `image_sub_mode` set).
- [x] `configure_image_columns()` in `ui.py`: `image_path_column`, `text_column`, `prompt_column` (VQA), `image_token_budget` via guided select (70–1120).
- [x] In `gemma_tuner/wizard/estimator.py`, image branch scales memory/time vs `image_token_budget` and warns if estimate exceeds ~80% of total RAM.
- [x] `gemma_tuner/wizard/config.py` + `runner.py` merge image keys into generated profile.

**Verification:** `gemma-macos-tuner wizard` produces a working image-mode profile end-to-end.

---

### Phase 5: Validation and Cleanup

**Goal:** End-to-end smoke test, README + docs updates.

**Tasks:**

- [x] Add `tests/test_smoke_image.py`: loads image-modality CSV + runs `DataCollatorGemmaImage` one batch (offline; **not** full `finetune.main()` — see Executive Summary).
- [x] Update `README.md` comparison table and add **Image fine-tuning** section with INI snippets.
- [x] TROUBLESHOOTING: high `image_token_budget`, train/serve mismatch; [`README/KNOWN_ISSUES.md`](../KNOWN_ISSUES.md) image throughput notes.
- [x] Cross-reference from [`README/guides/README.md`](../guides/README.md).

**Verification:** Unit/integration tests (excluding gated full image training) green; README/docs aligned.

## Success Criteria

### Hard Requirements (Must Pass)

- [x] Existing audio + text profiles run with zero config edits (regression suite).
- [x] **Gated integration:** Image caption + VQA profiles train ≥10 steps with finite loss on `google/gemma-3n-E4B-it` (`tests/test_smoke_image_multimodal.py`; requires Hub auth — skips without `HF_TOKEN` / `huggingface-cli login`). Verifies real `apply_chat_template` + image blocks, `image_token_budget`, and `inject_mm_token_type_ids` on actual multimodal outputs.
- [x] CMYK / RGBA fixtures exercise `_load_image_as_rgb` + collator (`test_gemma_image_collator.py`).
- [x] VQA mask boundary unit test (`test_gemma_image_collator.py`).
- [x] Padding masked via `attention_mask == 0` in `DataCollatorGemmaImage` (unit test asserts non-padding labels not all `IGNORE`).
- [x] `image_token_budget` in profile → run `metadata.json` `config`; export reapplies budget to processor when metadata present.
- [x] Wizard can emit image profiles (`configure_image_columns` + `generate_profile_config`).

### Definition of Done

- [x] Automated tests passing for implemented scope; gated image multimodal integration (`tests/test_smoke_image_multimodal.py`) optional in CI (requires `HF_TOKEN` secret to execute, otherwise skips).
- [x] README comparison table + image section + Datasets + TROUBLESHOOTING + KNOWN_ISSUES updated.
- [x] Plan status **Complete** (this document).

## Open Questions

### Resolved

- **Q:** Need a separate model class for image mode?
- **A:** No. `AutoModelForCausalLM` + `AutoProcessor` handle both audio and image in Gemma 3n. The vision tower is already resident.

- **Q:** Where does RGB conversion belong — dataset loader or collator?
- **A:** Collator. Centralizing it at the boundary means dataset adapters (CSV, GCS, future streaming) don't each need to remember to do it. The one-line cost of `.convert("RGB")` per sample is negligible vs the silent corruption risk.

- **Q:** Support multi-image-per-sample inputs from day one?
- **A:** No. v1 = one image per sample. Multi-image is a real use case (document pages, image comparison) but the schema and collator complexity isn't justified yet.

- **Q:** BLEU / CIDEr / VQA-accuracy eval in v1?
- **A:** No. Generation-during-eval inside HF Trainer's compute_metrics is gnarly per the gemma4-guide. v1 ships eval loss + perplexity. Custom callback is v2.

### Resolved (implementation)

- **Per-profile `image_token_budget`:** **A** — single value in profile and processor; no per-sample dynamic budget in v1.
- **Default `image_token_budget`:** **280** in config defaults; wizard offers 70–1120.
- **GCS image paths:** **Deferred** — v1 uses local or dataset-relative paths after `finetune.py` path resolution; no `load_image_local_or_gcs` yet.

## References

### Internal

- [`README/plans/text-only-finetuning.md`](text-only-finetuning.md) — hard dependency
- [`README/guides/apple-silicon/gemma4-guide.md`](../guides/apple-silicon/gemma4-guide.md) — image preprocessing skew, token budgets, MPS bugs
- [`gemma_tuner/models/common/collators.py`](../../gemma_tuner/models/common/collators.py) — current audio collator
- [`gemma_tuner/models/gemma/finetune.py`](../../gemma_tuner/models/gemma/finetune.py) — training entrypoint
- [`gemma_tuner/utils/dataset_utils.py`](../../gemma_tuner/utils/dataset_utils.py) — dataset loader
- [`gemma_tuner/wizard/`](../../gemma_tuner/wizard/) — wizard flow
- [`README/Datasets.md`](../Datasets.md) — dataset config docs

### External

- [Gemma 4 model card — vision token budgets](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Hugging Face: `apply_chat_template` with images](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [PEFT LoRA docs](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)

## Files Likely to Change

| File | Change Type | Notes |
| --- | --- | --- |
| `gemma_tuner/models/common/collators.py` | Modify | Add `DataCollatorGemmaImage` (caption + vqa sub-modes); reuse text-PR helpers |
| `gemma_tuner/models/gemma/finetune.py` | Modify | Add `image` branch to modality switch; persist `image_token_budget` to run metadata |
| `gemma_tuner/utils/dataset_utils.py` | Modify | Schema validation accepts image-mode required columns |
| `gemma_tuner/utils/dataset_sources.py` | Modify | `DatasetLoadContext` gains `image_path_column`; non-CSV adapters reject image mode |
| `gemma_tuner/core/config.py` | Modify | Register `image` modality + `image_sub_mode`, `image_path_column`, `image_token_budget` |
| `gemma_tuner/wizard/config.py` | Modify | Modality question gains image options; new image-only questions |
| `gemma_tuner/wizard/estimator.py` | Modify | Image-mode memory heuristic |
| `gemma_tuner/wizard/runner.py` | Modify | Forward new keys to profile writer |
| `gemma_tuner/models/common/results.py` | Modify | Persist `image_token_budget` in metadata; perplexity for image runs |
| `gemma_tuner/scripts/export.py` | Modify | Validate `image_token_budget` matches run metadata at export time |
| `tests/test_gemma_image_collator.py` | Create | Both sub-modes; RGB-convert assertion; mask boundary; EOS survival |
| `tests/test_smoke_image.py` | Create | Offline dataset + collator smoke (not full `finetune.main()` image run) |
| `tests/data/image_caption_tiny/` | Create | 8 tiny images + `train.csv` |
| `config.ini.example` | Modify | Commented image-mode keys |
| `README.md` | Modify | Comparison table; new image-mode section |
| `README/Datasets.md` | Modify | Document image keys |
| `README/KNOWN_ISSUES.md` | Modify | image_token_budget=1120 throughput cost; train/serve contract |

## Risks and Mitigations

- **Preprocessing skew (CMYK / RGBA silent corruption).** → Mandatory `.convert("RGB")` inside the collator. Unit test exercises CMYK + RGBA fixtures. This is the #1 documented image-mode failure.
- **train/serve `image_token_budget` mismatch.** → Persist budget in run metadata; export validates it; README + TROUBLESHOOTING document the contract.
- **OOM at 1120-token budget on 32 GB Macs.** → Wizard estimator warns at >80% RAM. README documents batch_size=1, grad_accum=16 as the standard configuration for high-budget runs.
- **Vision tower kwarg drift between Gemma 3n and Gemma 4.** → Keep the processor call thin; introspect kwarg names at implementation time, do not hardcode based on the gemma4-guide (the guide is forward-looking).
- **Throughput cliff: vision encoder is slow on MPS.** → Documented in TROUBLESHOOTING. Smoke test uses `image_token_budget=70` for fast CI.
- **HF Trainer evaluation hang on MPS.** → Already handled by the audio path's `skip_memory_metrics=True` / `eval_strategy="no"` patterns. Inherit; do not re-solve.
- **`<bos>` validation false-positives on multimodal inputs.** → The shared `_validate_bos_tokens_present` helper already uses `torch.any()` (not position-0). Verified during text PR. No new work.
- **Gemma 4 `Gemma4ClippableLinear` PEFT injection failure.** → Already solved by [`gemma_tuner/models/gemma/gemma4_patches.py`](../../gemma_tuner/models/gemma/gemma4_patches.py) and the LoRA target validation in `finetune.py`. Image mode inherits this for free as long as it routes through `_load_base_model_for_gemma` and passes `family=` into the collator.
- **Multi-image-per-sample requests post-launch.** → Schema is forward-compatible: `image_path_column` could become `image_paths_column` (list). Defer until real demand exists.

## Critical Reminder

> SIMPLER IS BETTER. Image mode is one collator + one switch + one wizard branch + one estimator branch + one smoke test. Resist the urge to refactor the trainer, build a generation-eval callback, support multi-image inputs, or pre-emptively patch Gemma4ClippableLinear "while we're in there." The text PR did the architectural work; this plan rides on top of it.
