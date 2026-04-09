# Gemma Fine-Tuning Debug Notes

Institutional memory for Gemma (especially Gemma 4) LoRA fine-tuning on Apple Silicon: MPS memory, text collators, and long-input summarization. Multiple related issues live in this file; each issue is self-contained.

**Quick filter:** `grep -n "— Active" README/notes/debug-notes.md`

---

## Issue: Gemma 4 instruction tuning trains on prompt tokens (wrong `control_token`) — Resolved

**First spotted:** 2026-04-08
**Resolved:** 2026-04-08
**Status:** Resolved

### Summary

End-to-end Gemma 4 text instruction tuning on the bundled `sample-text` dataset "completed successfully" but the adapter learned nothing: `eval_loss ≈ 14.59`, `train_loss ≈ 14.62` (near random-init CE for a 256K vocab: `ln(256000) ≈ 12.4`). **Both** label-masking paths were silently broken for Gemma 4 and the collator fell through to an unsafe "leave labels unmasked" state — every row was training on **prompt + response** instead of response only.

Separate from (and stacked on top of) the earlier "Zero loss from degenerate `assistant_masks`" bug below. That fix made the collator fall through to `mask_gemma_prompt_tokens` when the primary path returned a degenerate mask. This bug is that the fallback itself was broken for Gemma 4 because of a wrong control-token string.

### Symptom

```log
DataCollatorGemmaText: assistant_masks has no supervised tokens for 1/1 row(s);
  using mask_gemma_prompt_tokens fallback (often: chat template missing the Jinja
  `generation` keyword so HF cannot build assistant spans).
mask_gemma_prompt_tokens: at least one sample has no control-token span '<|turn|>';
  skipping prompt masking for those rows.
{'eval_loss': '14.59', 'eval_runtime': '2.035', ...}
{'train_runtime': '30.17', 'train_loss': '14.62', 'epoch': '1'}
✅ Training completed successfully!
```

`loss` is not zero here (unlike the earlier degenerate-mask bug) — it is near the random-init baseline because the model is being penalized for failing to predict its own prompt.

### Root Cause

Two independent bugs that compounded:

1. **Wrong `control_token` string in `family.py:71`.** The file declared `"control_token": "<|turn|>"` (bars on both sides). **That string does not exist in the real Gemma 4 tokenizer at all.** `tokenizer.encode("<|turn|>", add_special_tokens=False)` returns a 4-subword sequence `[236820, 236909, 887, 111038]` that never appears in any rendered chat. The actual start-of-turn marker in Gemma 4 is `<|turn>` (single pipe, no trailing bar), a single token with id `105`. The closing marker is the asymmetric `<turn|>` (id 106). Both the declaration and the surrounding comments had the wrong string.

2. **Primary `assistant_masks` path is always degenerate for Gemma 4.** Gemma 4's shipped chat template (as of transformers 5.5.0) does not wrap assistant replies in the Jinja `{% generation %}` block that HF requires to build assistant spans. `apply_chat_template(return_assistant_tokens_mask=True)` returned a correctly-shaped but all-zeros `assistant_masks` tensor **and** printed a noisy per-batch HF warning. The previous fix correctly detected this and fell through — straight into bug 1.

Chain of failures:

- Primary: `return_assistant_tokens_mask=True` → all-zeros mask (bug 2) → per-row degenerate detection → fallback.
- Fallback: `mask_gemma_prompt_tokens` searches for the `control_token` as a token subsequence → the wrong string never matches → warns and **leaves labels untouched** for every row.
- Training: unmodified labels = full sequence, so CE loss is computed over prompt + response. For a 16-sample / 4-optimizer-step smoke run with `lr=1e-5`, loss stays stuck near the random-init level and nothing useful is learned.

### Verification that BOTH bugs were required for the failure

`tests/test_gemma_text_collator.py::test_instruction_gemma4_degenerate_all_zero_assistant_masks_uses_fallback` was **passing the whole time** because its fake tokenizer defined `encode("<|turn|>") = [50]` (matching the wrong string in family.py). The test was self-consistent but had nothing to do with real Gemma 4 behavior. The regression check below now catches this.

### Related Guides

- [Gemma 4 on Apple Silicon](../guides/apple-silicon/gemma4-guide.md) — Text SFT and label masking.

### Fix

**Files:**

- `gemma_tuner/models/gemma/family.py` — `family_capabilities(GEMMA_4)`: `control_token` changed from `"<|turn|>"` → `"<|turn>"`. Added new capability `supports_assistant_mask: bool` (False for Gemma 4, True for Gemma 3n) so collators can skip the primary HF path entirely when they know it will be degenerate. Extensive docstring explaining both traps for future maintainers.
- `gemma_tuner/models/common/collators.py` — `DataCollatorGemmaText._collate_instruction`: reads `self._caps["supports_assistant_mask"]` and only passes `return_assistant_tokens_mask=True` when supported. For Gemma 4 the call is skipped entirely, silencing the per-batch HF warning. The degenerate-row branch from the earlier fix is retained as belt-and-suspenders for any future tokenizer release that volunteers an all-zero mask without the flag. `_control_token_subsequence_ids` comments and fast-path updated to `<|turn>`.
- `tests/test_family.py` — `test_family_capabilities_keys` asserts `c4["control_token"] == "<|turn>"` and `c4["supports_assistant_mask"] is False`. Comment explains the `<|turn|>` vs `<|turn>` trap inline.
- `tests/test_gemma_text_collator.py` — fake tokenizers updated to mirror real Gemma 4 behavior (`<|turn>` → single token 50; `<|turn|>` → 4-subword sequence [900, 901, 902, 903] that never matches). Added three new tests:
  - `test_mask_gemma_prompt_tokens_gemma4_wrong_marker_warns_and_leaves_labels_unmasked` — regression guard: using the old wrong `<|turn|>` marker triggers the skip-masking warning and leaves labels unchanged.
  - `test_instruction_gemma4_skips_primary_assistant_mask_path` — spy tokenizer verifies the collator never requests `return_assistant_tokens_mask=True` for Gemma 4.
  - `test_instruction_gemma3n_still_uses_primary_assistant_mask_path` — matching spy test ensures Gemma 3n keeps using the primary path (no regression).
- `tests/test_gemma_image_collator.py` — `_TokenizerMaskingGemma4` fake updated to the same `<|turn>` / `<|turn|>` split; image-collator mask test now passes `control_token="<|turn>"`.

### Verification

End-to-end with the **real** Gemma 4 tokenizer (not a fake):

```bash
.venv/bin/python -c "
import warnings, torch
from transformers import AutoTokenizer
from gemma_tuner.models.common.collators import DataCollatorGemmaText
from gemma_tuner.models.gemma.family import GemmaFamily
from gemma_tuner.models.gemma.constants import GemmaTrainingConstants

tok = AutoTokenizer.from_pretrained('google/gemma-4-E2B-it')
collator = DataCollatorGemmaText(tok, text_column='response', family=GemmaFamily.GEMMA_4,
                                  prompt_column='prompt', max_length=128, sub_mode='instruction')
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    out = collator([
        {'prompt': 'Translate to French: Good morning.', 'response': 'Bonjour.'},
        {'prompt': 'What is the capital of Japan?', 'response': 'Tokyo.'},
    ])
assert len(caught) == 0, f'expected 0 warnings, got {[str(w.message) for w in caught]}'
ignore = GemmaTrainingConstants.IGNORE_TOKEN_ID
for i in range(out['input_ids'].size(0)):
    sup = [(p, int(t)) for p, t in enumerate(out['labels'][i].tolist()) if t != ignore]
    span = tok.decode([out['input_ids'][i, p].item() for p, _ in sup], skip_special_tokens=False)
    print(f'row {i}: {len(sup)} supervised tokens -> {span!r}')
"
```

Expected output (silence from warnings; response + end-of-turn supervised; prompt masked):

```
row 0: 4 supervised tokens -> 'Bonjour.<turn|>\n'
row 1: 4 supervised tokens -> 'Tokyo.<turn|>\n'
```

Plus the unit tests:

```bash
.venv/bin/python -m pytest tests/test_family.py tests/test_gemma_text_collator.py tests/test_gemma_image_collator.py -q
# Expect: 39 passed
```

### Follow-ups / things to watch

- The wizard default `learning_rate=1e-5` is very conservative — even with correct masking a 16-row / 4-step smoke run will not move loss noticeably. Consider bumping the wizard default to `5e-5` or `1e-4` for instruction tuning specifically, or offer a "smoke test" mode that picks aggressive defaults.
- If a future transformers release adds `{% generation %}` to Gemma 4's chat template, `supports_assistant_mask` can be flipped to True — but keep the regression guard test for the `<|turn>` marker either way.
- `<bos>` presence check: the Gemma 4 rendered chat starts with `<bos><|turn>user...`, so `validate_bos_tokens_present` still passes. Not related to this bug but worth noting alongside it.

---

## Issue: Zero loss from degenerate `assistant_masks` (all zeros) — Resolved

**First spotted:** 2026-04-07
**Resolved:** 2026-04-07
**Status:** Resolved

### Summary

`apply_chat_template(..., return_assistant_tokens_mask=True)` can return `assistant_masks` with **no supervised positions** (often when the chat template lacks the Jinja `generation` keyword; HF logs a warning). The old collator path did `labels[am == 0] = ignore`, which masks **every** token when `am` is all zeros → `loss=0`, `grad_norm=0`. **Fix:** per row, use `assistant_masks` only when `am[i].sum() > 0`; otherwise call `mask_gemma_prompt_tokens` for that row. One-time warning when fallback triggers.

### Symptom

```log
{'loss': '0', 'grad_norm': '0', 'learning_rate': '9.981e-06', 'epoch': '0.01363'}
return_assistant_tokens_mask==True but chat template does not contain `{% generation %}` keyword.
```

### Root Cause

`assistant_masks` present with correct shape but **all zeros**; combined with unconditional `labels[am == 0]` masking, every label became `IGNORE_TOKEN_ID`.

### Related Guides

- [Gemma 4 on Apple Silicon](../guides/apple-silicon/gemma4-guide.md) — Text SFT and label masking.

### Fix

**Files:**

- `gemma_tuner/models/common/collators.py` — `DataCollatorGemmaText._collate_instruction`: per-row degenerate `assistant_masks` handling + `_warned_degenerate_assistant_mask`.
- `tests/test_gemma_text_collator.py` — `test_instruction_gemma4_degenerate_all_zero_assistant_masks_uses_fallback`.

### Verification

```bash
.venv/bin/python -m pytest tests/test_gemma_text_collator.py -q
```

---

## Issue: Long-input summarization — truncation drops assistant span — Resolved

**First spotted:** 2026-04-07
**Resolved:** 2026-04-07
**Status:** Resolved

### Summary

For summarization, **right-side truncation** at `max_length` keeps the **beginning** of the chat; very long user prompts can fill the entire window so the `<|turn>model` / assistant content never appears. That is separate from the degenerate-mask bug. **Fix:** before `apply_chat_template`, measure token length per row and **left-truncate the user prompt** (drop characters from the start) until the chat fits `max_length`, then apply the usual HF truncation as a safety net. Optional `instruction_truncation="none"` restores tokenizer-only truncation. Very long assistant targets still need a smaller response or a larger `max_seq_length`; the collator may **prefix-truncate** the assistant string if the pair still does not fit. Chunking / hierarchical summarization for multi-window documents remains a data-pipeline concern.

### Root Cause

Tokenizer `truncation=True` + `max_length` applies to the full rendered chat; long prompts crowd out the assistant turn.

### Fix

**Files:**

- `gemma_tuner/models/common/collators.py` — `instruction_truncation` (default `"left_user"`), `_instruction_seq_len`, `_fit_instruction_pair_to_max_length`, and integration in `_collate_instruction`.
- `tests/test_gemma_text_collator.py` — `_FakeLongPromptTokenizer` plus tests for `left_user` vs `none`.

### Verification

```bash
.venv/bin/python -m pytest tests/test_gemma_text_collator.py -q
```

---

## Issue: MPS OOM during Gemma 4 multimodal text LoRA training — Resolved

**First spotted:** 2026-04-07
**Resolved:** 2026-04-07
**Status:** Resolved

### Summary

First training step crashed with `RuntimeError: MPS backend out of memory` (~41 GiB allocated) while fine-tuning `gemma-4-e2b-it` with LoRA on text CSV data. Mitigations: **MPS + text** forces `per_device_train_batch_size=1` and `per_device_eval_batch_size=1`, enables **gradient checkpointing** by default for MPS text (opt out via `GEMMA_MPS_ALLOW_NO_GRADIENT_CHECKPOINTING`), passes `gradient_checkpointing` into `TrainingArguments`, calls `enable_input_require_grads()` when checkpointing is on, and caps `max_seq_length` on MPS text using `GEMMA_MPS_MAX_SEQ_LENGTH` (default 2048). Added profile `gemma-csv-gemma4-mps-text` in `config/config.ini`.

### Symptom

```log
RuntimeError: MPS backend out of memory (MPS allocated: 40.96 GiB, ... max allowed: 41.47 GiB).
```

### Root Cause

Unified memory pressure from **Gemma 4 multimodal** weights + **LoRA** + **eager attention** + **micro-batch > 1** and **no gradient checkpointing** on long sequences. Confirmed by stack trace in attention softmax during the first step.

### Related Guides

- [Gemma 4 on Apple Silicon](../guides/apple-silicon/gemma4-guide.md) — MPS allocator, `PYTORCH_MPS_HIGH_WATERMARK_RATIO`, gradient checkpointing tradeoffs.

### Fix

**Files:**

- `gemma_tuner/models/gemma/finetune.py` — MPS text batch clamp, gradient checkpointing wiring, optional `GEMMA_MPS_ALLOW_NO_GRADIENT_CHECKPOINTING`, `GEMMA_MPS_MAX_SEQ_LENGTH` cap, `enable_input_require_grads`.
- `config/config.ini` — `[profile:gemma-csv-gemma4-mps-text]` with conservative batch/accumulation settings.

### Verification

```bash
gemma-macos-tuner finetune gemma-csv-gemma4-mps-text --json-logging
# Expect first training steps to progress without immediate MPS OOM (still subject to seq length / data).
```

---

## Appendix: Practical context limits (64 GB unified RAM, this stack)

**Not a bug ticket** — planning guidance from the same investigation.

- Causal LM **training** uses one sequence budget for **prompt + response** (plus chat template overhead). “8192 input” is not a separate bucket from the summary.
- For **`gemma-4-e2b-it` LoRA on MPS** in this repo, treat **~2k–3k total tokens** as a realistic training target unless proven otherwise; **~10k total** is not a safe default for local training on this hardware (attention memory scales roughly with sequence length squared).
- **Data:** median prompt length on sampled `gemma-csv` rows was on the order of **~8–9k tokens** (prompt-only encode); many rows exceed any single-window cap without **chunking**, **hierarchical summarization**, or **left-truncating** the source only.

<!--
USAGE NOTES (from template):

1. Prefer updating this file before creating new note files for the same domain.
2. New issue? Copy the "Active" template and paste at the top (newest first).
3. Fixed: rename "— Active" to "— Resolved" and add Resolved date.
4. Investigation Log: trim ruled-out hypotheses after resolution.
-->
