# Gemma 4 + 3n Dual-Support Upgrade Plan

**Date:** 2026-04-06
**Status:** In progress — Phases 0–5 core work is landed in code and CI; remaining items are documentation cross-links, polish, and the image-plan follow-up ticket below.

## Executive Summary

The codebase is in a broken intermediate state: configs and defaults reference `google/gemma-4-E2B` and friends ([config.ini.example:41-62](../../config.ini.example), [wizard/base.py:139-143](../../gemma_tuner/wizard/base.py#L139-L143), [gemma_dataset_prep.py:164](../../gemma_tuner/utils/gemma_dataset_prep.py#L164)) but the installed `transformers==4.57.6` (pinned `<5.0.0` in [pyproject.toml:28](../../pyproject.toml#L28)) cannot actually load Gemma 4 — the model type isn't registered until `transformers>=5.5.0` per the [gemma4-guide](../guides/apple-silicon/gemma4-guide.md). Gemma 3n works today; Gemma 4 IDs are tombstones. This plan keeps Gemma 3n as the default working path, adds a clean Gemma 4 training path behind a shared family/version gate, and makes every user-facing Gemma loader do one of two things consistently: either support Gemma 4 explicitly or fail fast with a clear message before any large download. Estimated effort: ~2 focused days, sequenced after the text PR but parallelizable with the image PR.

## Problem Statement

- **Symptom:** A fresh `pip install -e .` user who picks the default `gemma-4-e2b` profile from [config.ini.example:82](../../config.ini.example) gets a load-time crash like `KeyError: 'gemma4'` or `ValueError: Unknown model type gemma4`. The wizard's "estimator" rows for Gemma 4 are dead links.
- **Root Cause:** Two compounding mistakes:
  1. **Version pin lies.** `transformers>=4.46.0,<5.0.0` cannot load Gemma 4. The pin needs to either drop the Gemma 4 IDs or move to `>=5.5.0`.
  2. **No family abstraction.** The collator hardcodes Gemma 3n control tokens (`<start_of_turn>`), Gemma 3n processor kwargs (`processor(messages=…, audios=…)`), and the Gemma 3n model class (`AutoModelForCausalLM`). Gemma 4 needs `<|turn|>`-aware masking + model-header search, `mm_token_type_ids` injection when missing, the `Gemma4ClippableLinear` PEFT monkey-patch, and `AutoModelForMultimodalLM`. **(Resolved in this plan.)**
- **Impact:** The repo's marketing implies Gemma 4 support; the code does not deliver it. New users hit a crash on the first command. Existing 3n users are fine but cannot upgrade. The forthcoming [image](image-finetuning.md) plan also needs this resolved before its v2 deferred items can ship.

## Mode Definitions

| Family | Detection | Support rule | Collator path | Model class |
| --- | --- | --- | --- | --- |
| `gemma3n` | `model_id` matches `gemma-3n-` (case-insensitive) | Supported on the base pinned stack | `<start_of_turn>` masking; **inject missing** `token_type_ids` / `mm_token_type_ids` (zeros) when the capability flag is on — some processors omit them on certain transformers versions ([#45200](https://github.com/huggingface/transformers/issues/45200)) | `AutoModelForCausalLM` |
| `gemma4` | `model_id` matches `gemma-4-` (or class-style `gemma4`, see `family.py`) | Requires `transformers>=5.5.0`; unsupported entrypoints must fail loudly before load | `<|turn|>` boundary + model-header masking; same **injection** for missing modality keys when flag is on | `AutoModelForMultimodalLM` |

## Goals and Non-Goals

### Goals

- [x] `pip install -e .` + the documented quickstart never crashes due to a Gemma 4 / 3n mismatch.
- [x] A single `gemma_tuner/models/gemma/family.py` helper detects family from `model_id` and centralizes version/support checks.
- [x] The install story matches reality: base install stays 3n-first, Gemma 4 setup is a separate, documented compatibility install that does not fight the base pin.
- [x] Collator branches on family for control-token masking and `mm_token_type_ids` injection (both families when keys are missing — see #45200).
- [x] PEFT injection branches on family for the `Gemma4ClippableLinear` monkey-patch.
- [x] Wizard / preflight show capability mismatches (Gemma 4 label when `transformers` too old; `gate_gemma_model` before load).
- [x] Every user-facing Gemma loader either supports Gemma 4 explicitly or rejects it with the same clear, fast error before any `from_pretrained(...)` call.
- [x] An end-to-end smoke test exists for **both** families (skipped on the inactive family if its dependencies aren't installed).
- [x] [image-finetuning.md](image-finetuning.md)'s deferred Gemma 4 items (clippable-linear, mm_token_type_ids) are unblocked in shared collator/family code; image plan adoption remains a follow-up.

### Non-Goals

- **Auto-installing transformers 5.5 on demand.** Users opt in to the Gemma 4 stack through a documented compatibility install step; the repo does not mutate their environment at runtime.
- **Backporting Gemma 4 features to 3n** (e.g., the 256k context window, `<|think|>` reasoning blocks). Each family gets what its model card supports.
- **Dropping Gemma 3n support.** It works, it's the most-tested path, and 3n is the right default for 16 GB Macs per the gemma4-guide memory tier table.
- **Refactoring the trainer / LoRA / MPS plumbing.** Family detection lives at the edges (collator, PEFT injection, model load); the middle stays modality-agnostic and family-agnostic.
- **A new `AutoModelForMultimodalLM` shim for 3n.** 3n keeps `AutoModelForCausalLM`; only Gemma 4 uses the new class.

## Scope and Constraints

- **Scope:** Family detection helper, honest install story, collator branch, PEFT injection branch, finetune model-load branch, shared fast-fail gates for every Gemma-loading CLI/script, dual smoke tests, README updates.
- **Constraints:**
  - Must not regress existing Gemma 3n profiles. Default-everywhere stays 3n.
  - Must not require users to install Gemma 4 deps if they only want 3n.
  - Must coexist with the forthcoming [text](text-only-finetuning.md) and [image](image-finetuning.md) PRs — those PRs touch the same collator file. Coordinate sequencing.
- **Guardrails:** `detect_family(model_id)` is a pure `model_id -> GemmaFamily` function. `assert_family_supported(family, installed_transformers_version)` is the only version gate. No default-family fallback, no global state, no network calls.

## Ground Truth Contracts (Do Not Violate)

- **Detection is by `model_id`, not by config flag and not by installed version.** Users should not need to set `family=gemma4` in their profile. The model_id `google/gemma-4-E2B` is unambiguous; trust it.
- **Support is a separate check from detection.** `gemma-4-*` on `transformers==4.57.6` is still family `gemma4`; it just fails support validation immediately.
- **Capability mismatches fail loudly before any expensive load.** If user picks `gemma-4-e2b` on `transformers==4.57.6`, the shared gate says "Gemma 4 requires transformers>=5.5.0; your environment is still on the base Gemma 3n stack. Follow the Gemma 4 install steps in the README or pick a Gemma 3n model." No partial loads, no multi-GB download first.
- **`Gemma4ClippableLinear` monkey-patch must run before `from_pretrained`.** Per the gemma4-guide, patching after weights are instantiated does nothing. Family detection runs at the very top of `finetune.main()`, before any model load.
- **3n control tokens and 4 control tokens are NOT interchangeable.** Hand-rolling `<start_of_turn>` for a Gemma 4 model produces silent garbage. Always go through `tokenizer.apply_chat_template` or the family-aware mask helper.
- **Every user-facing Gemma loader shares the same gate.** `finetune`, `gemma_preflight`, `gemma_generate`, `gemma_profiler`, `gemma_dataset_prep`, `export`, `inference_common`, and `tools/eval_gemma_asr.py` must all call the same helper before any `from_pretrained(...)`.
- **Single source of truth for base-stack versions.** [pyproject.toml](../../pyproject.toml) is canonical for the base 3n path; the Gemma 4 compatibility install lives in one checked-in requirements file. Wizard / preflight / README read installed versions from `importlib.metadata.version("transformers")`, never hardcode a number in behavior logic.

## Already Shipped (Do Not Re-Solve)

- **Gemma 3n collator** in [collators.py](../../gemma_tuner/models/common/collators.py) — works correctly on transformers 4.57.6 + 3n. Reuse as the 3n branch.
- **LoRA / PEFT injection** in [finetune.py:294-359](../../gemma_tuner/models/gemma/finetune.py#L294-L359) — modality-agnostic and (mostly) family-agnostic. Wrap with a family branch for the clippable-linear patch.
- **MPS device + bfloat16 detection** in [finetune.py:267](../../gemma_tuner/models/gemma/finetune.py#L267) and [device.py](../../gemma_tuner/utils/device.py) — family-agnostic. Reuse.
- **Run index, metrics, export** — family-agnostic. Reuse.
- **Wizard model registry** in [wizard/base.py:139-146](../../gemma_tuner/wizard/base.py#L139-L146) — already lists both families. Just needs availability gating.
- **Forthcoming text-mode shared helpers** — `_validate_bos_tokens_present`, `_mask_prompt_tokens`, attention-mask-based label masking. Both family branches consume them.

## Fresh Baseline (Current State)

- **Installed:** `transformers==4.57.6`, `peft==0.18.1`, `torch==2.11.0`. Confirmed via `.venv/bin/python -c "..."`.
- **Pyproject pin:** `transformers>=4.46.0,<5.0.0` — explicitly excludes the version Gemma 4 needs.
- **Default model in profiles:** `gemma-4-e2b` (would crash if anyone actually used it).
- **Default model in code:** `google/gemma-4-E2B` ([finetune.py:256](../../gemma_tuner/models/gemma/finetune.py#L256)) — same crash risk.
- **Collator:** Hardcoded to Gemma 3n control tokens and processor API. Works on 3n, would silently or loudly fail on 4.
- **Preflight:** [gemma_preflight.py](../../gemma_tuner/scripts/gemma_preflight.py) validates Gemma 3n requirements per the docstring. No Gemma 4 awareness.
- **Wizard:** Lists both families with no availability check.
- **Tests:** `tests/test_gemma_collator.py`, `tests/test_smoke.py` — exercise the audio path on whatever model is configured. No explicit family coverage.
- **Known gaps:** No family abstraction, no version gate, no clippable-linear patch, no `mm_token_type_ids` injection, no Gemma 4 control-token handling, defaults point at unloadable models.

## Solution Overview

```
+-------------------------------+
| family.py                     |
| detect_family(model_id)       |
| assert_family_supported(...)  |
| assert_entrypoint_support(...)|
+-------------------------------+
          |                   |
          | shared fast-fail  | gemma4 allowed only where implemented
          v                   v
+--------------------+   +-----------------------------------+
| non-training CLIs  |   | finetune.py + collators           |
| reject unsupported |   | family-aware collator + model load|
| Gemma 4 early      |   | Gemma4 patch before load          |
+--------------------+   +-----------------------------------+
```

Family detection and support checks live in one helper. `finetune.py` gets the real dual-family implementation. Other user-facing loaders either reuse the same helper to support Gemma 4 where trivial or fail immediately with the same honest "not supported here yet" message. The middle of the trainer (LoRA config, Trainer setup, MPS handling, metrics plumbing) stays family-agnostic.

## Implementation Phases

> Sequencing: this plan can land **after the text PR** and **before or in parallel with the image PR**. The image PR's deferred Gemma 4 items get adopted by Phase 4 here.

### Phase 0: Honest Install Story

**Goal:** Stop lying about what's loadable. Base install stays a clean 3n path; Gemma 4 gets a real, non-conflicting install path.

**Tasks:**

- [x] Decide the default: `gemma-3n-e2b-it` (lean: yes — it works today on the pinned transformers).
- [x] Update [config.ini.example:82](../../config.ini.example) and the active default in [`GemmaTrainingConstants.DEFAULT_BASE_MODEL_ID`](../../gemma_tuner/models/gemma/constants.py) (used by [finetune.py](../../gemma_tuner/models/gemma/finetune.py)) to `google/gemma-3n-E2B-it`.
- [x] Update [gemma_dataset_prep.py](../../gemma_tuner/utils/gemma_dataset_prep.py) (via constants), [gemma_generate.py](../../gemma_tuner/scripts/gemma_generate.py), [gemma_profiler.py](../../gemma_tuner/scripts/gemma_profiler.py) defaults, wizard [estimator/runner](../../gemma_tuner/wizard/) hints, and [blacklist.py](../../gemma_tuner/scripts/blacklist.py) `model_name_or_path` to 3n IDs where they were still Gemma 4.
- [x] Keep the base `pyproject.toml` pin as the 3n-tested stack.
- [x] Add a checked-in `requirements-gemma4.txt` compatibility install file that upgrades the environment to the Gemma 4-capable stack (`transformers>=5.5.0`, `peft>=0.18.1`, and any other proven-needed overrides). Do **not** use a `gemma4` extra that conflicts with the base pin.
- [x] Update README install docs to use a 3n-first quickstart plus a separate Gemma 4 setup path: `pip install -e .` then `pip install -r requirements-gemma4.txt` when the user explicitly wants Gemma 4.

**Verification:** Fresh `pip install -e .` + `gemma-macos-tuner finetune --profile <default>` runs at least one training step without a model-load error. A separate fresh env can follow the Gemma 4 install steps without dependency-resolution conflicts.

---

### Phase 1: Family Detection Helper

**Goal:** Land the family abstraction with no callers yet. Pure code addition.

**Tasks:**

- [x] Create `gemma_tuner/models/gemma/family.py` exporting:
  - `class GemmaFamily(str, Enum): GEMMA_3N = "gemma3n"; GEMMA_4 = "gemma4"`.
  - `def detect_family(model_id: str) -> GemmaFamily`. Substring match: `"gemma-3n"` → 3N; `"gemma-4"` → 4; class-style ids with `gemma4` → 4; `tiny-random` Gemma stubs → 3N; raises `ValueError` on neither (the codebase only supports these two).
  - `def family_capabilities(family: GemmaFamily) -> dict` returning behavior data shared across callers: `{"control_token": "<start_of_turn>" | "<|turn|>", "needs_clippable_patch": bool, "needs_mm_token_type_ids_injection": bool, "min_transformers_version": "4.46.0" | "5.5.0"}`.
  - `def assert_family_supported(family: GemmaFamily) -> None`. Reads installed transformers via `importlib.metadata.version`. If user picked Gemma 4 on transformers < 5.5, raises with a single remediation message that points to the README Gemma 4 install steps.
  - `def assert_entrypoint_support(entrypoint: str, family: GemmaFamily) -> None`. For entrypoints that are still 3n-only, raises a clear "Gemma 4 is not implemented in `<entrypoint>` yet; use `finetune` or switch to Gemma 3n" error before any load.
- [x] Add `tests/test_family.py` covering: detection from each known ID format (`gemma-3n-E2B-it`, `google/gemma-4-E4B`, mixed case), `assert_family_supported` raising on the version mismatch, `assert_entrypoint_support` rejecting Gemma 4 on a declared 3n-only entrypoint, ValueError on a non-Gemma model_id.

**Verification:** New tests pass. No other code changed.

---

### Phase 2: Shared Fast-Fail Gate

**Goal:** Surface family/version mismatches and unsupported entrypoints at the friendliest possible point (before training, before model download).

**Tasks:**

- [x] In [gemma_preflight.py](../../gemma_tuner/scripts/gemma_preflight.py), call `detect_family(model_id)` + `assert_family_supported(family)` early. Print the resolved family in the preflight banner.
- [x] Wizard: suffix for Gemma 4 when `transformers` is below `MIN_TRANSFORMERS_GEMMA4` from family.py — implemented in [wizard/ui.py](../../gemma_tuner/wizard/ui.py) using `detect_family(hf_id)` + `ModelSpecs.MODELS[*].hf_id` (not string prefix on display names). `gate_gemma_model` runs after selection; `finetune.main()` repeats the same gate (no duplicate prompts).
- [x] In `finetune.main()`, run `detect_family + assert_family_supported` immediately after parsing `model_id` from the profile, before any tokenizer / processor / model load.
- [x] Call the same helper from every other user-facing Gemma loader before any `from_pretrained(...)`: `gemma_generate.py`, `gemma_profiler.py`, `gemma_dataset_prep.py`, `scripts/inference_common.py`, `scripts/export.py`, and `tools/eval_gemma_asr.py`.
- [x] For non-training entrypoints that are not being upgraded in this plan, call `assert_entrypoint_support(...)` so Gemma 4 is rejected honestly instead of crashing under `AutoModelForCausalLM`.

**Verification:** A user with `transformers==4.57.6` who picks `gemma-4-e2b-it` in the wizard or via any supported CLI path gets the remediation error within ~1 second, not after a 5 GB model download. A user who tries Gemma 4 through a still-3n-only entrypoint gets a clear "not implemented here yet" error, not a crash.

---

### Phase 3: Collator Family Branch

**Goal:** Make the audio collator (and the post-text-PR text/image collators) family-aware. This is the surgical part.

**Tasks:**

- [x] In [collators.py](../../gemma_tuner/models/common/collators.py), accept a required keyword-only `family: GemmaFamily` constructor argument on every collator (`DataCollatorGemmaAudio`, `DataCollatorGemmaText`; `DataCollatorGemmaImage` when image PR lands). No silent default.
- [x] Refactor prompt masking to take `control_token: str` from `family_capabilities` (`mask_gemma_prompt_tokens`). Gemma 4 uses `<|turn|>` boundary + model-header search (not a single `<|turn|>model` span).
- [x] `inject_mm_token_type_ids` creates zero tensors matching `input_ids` shape when keys are missing. Called when `needs_mm_token_type_ids_injection` is True **for both families** (3n and 4) so regressions against transformers#45200 are avoided. Cite [HF transformers issue #45200](https://github.com/huggingface/transformers/issues/45200) in the docstring.
- [x] Update `tests/test_gemma_collator.py` and `tests/test_gemma_text_collator.py` for `family` and injection behavior.

**Verification:** All collator tests pass on the 3n branch. The Gemma 4 branch is exercised by unit tests using a mock processor (no real model load required).

---

### Phase 4: Model Load + PEFT Family Branch

**Goal:** Wire family branching into model instantiation.

**Tasks:**

- [x] In [finetune.py](../../gemma_tuner/models/gemma/finetune.py), branch on `family`:
  - Gemma 3n: existing `AutoModelForCausalLM.from_pretrained(...)`.
  - Gemma 4: `AutoModelForMultimodalLM` (lazy import inside the branch). Apply the `Gemma4ClippableLinear` monkey-patch **before** `from_pretrained`. Patch lives in `gemma_tuner/models/gemma/gemma4_patches.py`.
- [x] Pass `family` into the collator constructor.
- [x] In `gemma_tuner/models/gemma/gemma4_patches.py`, define `apply_clippable_linear_patch()` (Patched linear inheriting from `nn.Linear`, idempotent). Mirrors upstream `Gemma4ClippableLinear` signature (transformers 5.5+).
- [x] Add `tests/test_gemma4_patches.py` (skip when `transformers<5.5`; subprocess test that `import gemma_tuner` + finetune does not eager-load `transformers.models.gemma4`).

**Verification:** On a machine with `transformers==4.57.6`, all 3n tests pass and Gemma 4 tests skip with a clear reason. On a machine with `transformers>=5.5.0` and `peft>=0.18.1` installed via `requirements-gemma4.txt`, both branches' unit tests pass.

---

### Phase 5: Validation and Cleanup

**Goal:** End-to-end dual-family smoke tests and honest documentation.

**Tasks:**

- [ ] Add `tests/test_smoke_gemma3n.py` (rename of existing smoke if needed): runs the existing 8-row audio fixture through `finetune.main()` with a 3n model, 2 steps, finite loss assertion.
- [x] Add `tests/test_smoke_gemma4.py`: same shape but with a Gemma 4 model_id. Decorator: `@pytest.mark.skipif(transformers_version < "5.5.0", reason="Gemma 4 requires transformers>=5.5.0")`.
- [x] CI: `unit` + `unit-gemma4` jobs in [.github/workflows/ci.yml](../../.github/workflows/ci.yml) (`base` + `requirements-gemma4.txt`).
- [ ] Update [README.md](../../README.md) to spell out: "Gemma 3n works out of the box. Gemma 4 training requires the separate Gemma 4 install step." Also document which non-training entrypoints are still 3n-only if any remain after this plan lands.
- [ ] Update [README/specifications/Gemma3n.md](../specifications/Gemma3n.md) to reference [README/guides/apple-silicon/gemma4-guide.md](../guides/apple-silicon/gemma4-guide.md) for the Gemma 4 path.
- [ ] Add a TROUBLESHOOTING entry: "Gemma 4 model load fails with `KeyError: 'gemma4'`" → follow the Gemma 4 install steps; "Gemma 4 is not implemented in `<entrypoint>` yet" → use `finetune` or switch to Gemma 3n.
- [ ] Cross-reference this plan from [README/guides/README.md](../guides/README.md) and from [text-only-finetuning.md](text-only-finetuning.md) + [image-finetuning.md](image-finetuning.md) (which both depend on this work for full Gemma 4 coverage).
- [ ] **Follow-up (tracked):** Open or file a ticket for [image-finetuning.md](image-finetuning.md): adopt family-aware `mm_token_type_ids` / collator paths for v2 image features (unblocked by this plan).

**Verification:** Both smoke tests pass on a Gemma 4-configured machine; only the 3n smoke test runs on a base install, and the 4 test skips with a clear reason. README claims match code reality.

## Success Criteria

### Hard Requirements (Must Pass)

- [x] `pip install -e .` + the README quickstart never crashes due to model-family / transformers-version mismatch.
- [x] A user picking `gemma-4-e2b-it` on `transformers<5.5` gets a clear remediation error within ~1 second (preflight / wizard), not after a multi-GB download or a confusing collator crash.
- [x] A clean env can follow the documented Gemma 4 install steps without dependency-resolution conflicts.
- [ ] A Gemma 4-configured env trains a Gemma 4 profile for ≥10 steps with finite, decreasing loss (manual / extended validation).
- [ ] `pip install -e .` (base) + a Gemma 3n profile trains for ≥10 steps with finite, decreasing loss (regression check).
- [x] `tests/test_family.py` covers detection + version-gate behavior.
- [x] `tests/test_gemma4_patches.py` proves the clippable-linear monkey-patch works (or skips cleanly).
- [x] Every user-facing Gemma loader either supports Gemma 4 or rejects it with the shared fast-fail error before any `from_pretrained(...)` call.
- [x] No Gemma 4 import statements at module-import time on a base install (lazy imports inside branches).

### Definition of Done

- [x] All tests passing (3n always; 4 always or skip-with-reason).
- [ ] README + TROUBLESHOOTING accurate about both families.
- [x] CI includes a real Gemma 4 job, not just skip-based coverage on base.
- [ ] Image plan's deferred Gemma 4 items updated to reference this plan as the unblocker.
- [ ] Plan status updated to Complete.


## Open Questions

### Resolved

- **Q:** Drop Gemma 4 IDs entirely vs add them as opt-in?
- **A:** Opt-in via a separate Gemma 4 compatibility install, not a package extra. The model family is real, the gemma4-guide is forward-looking-but-detailed, and the image plan needs the path to exist. A separate requirements file is simpler and avoids impossible resolver constraints.

- **Q:** Default to 3n or 4?
- **A:** 3n. Works on the base install, the gemma4-guide explicitly says 3n is the right tier for 16 GB Macs, and 3n is what the audio path was tested against. 4 is opt-in.

- **Q:** Detect family from a config flag (`family=gemma4`) or from the model_id?
- **A:** From the model_id. Model_id is unambiguous (`gemma-3n-` vs `gemma-4-`), users already provide it, and a separate flag would create another foot-loaded inconsistency to debug.

- **Q:** Apply the clippable-linear patch globally on import vs lazily on Gemma 4 model load?
- **A:** Lazily, inside the family branch in `finetune.py`. Global import-time patching pollutes the namespace for 3n users and breaks "no Gemma 4 imports on base install."

### Unresolved

- **Q:** Should the Gemma 4 compatibility install also pin a specific torch version? The gemma4-guide mentions PyTorch 2.x with MPS but doesn't pin.
- **Options:** (A) Don't pin; trust the user's existing torch unless a real incompatibility appears. (B) Pin `torch>=2.5` defensively in `requirements-gemma4.txt`. **Lean: A.** Extra pins fight upstream constraints; surface a runtime warning if torch is unexpectedly old.

- **Q:** Should we add a `gemma_tuner doctor` CLI subcommand that prints family + version + capability matrix?
- **Options:** (A) Yes — high-leverage debugging UX. (B) No — preflight already does this. **Lean: A** if it's <50 lines (just calls `detect_family` + `assert_family_supported` + prints `family_capabilities`). Defer to a follow-up if it grows.

- **Q:** Granary / BigQuery / GCS adapters — do they need family awareness?
- **Options:** (A) No; data adapters are pre-tokenization and family-blind. (B) Yes; certain dataset shapes only make sense for one family. **Lean: A.** No evidence the adapters care.

## References

### Internal

- [`README/guides/apple-silicon/gemma4-guide.md`](../guides/apple-silicon/gemma4-guide.md) — definitive source for Gemma 4 footguns
- [`README/specifications/Gemma3n.md`](../specifications/Gemma3n.md) — Gemma 3n notes
- [`README/plans/text-only-finetuning.md`](text-only-finetuning.md) — landed first; provides shared collator helpers
- [`README/plans/image-finetuning.md`](image-finetuning.md) — depends on this plan for full Gemma 4 coverage
- [`gemma_tuner/models/common/collators.py`](../../gemma_tuner/models/common/collators.py) — gets a family branch
- [`gemma_tuner/models/gemma/finetune.py`](../../gemma_tuner/models/gemma/finetune.py) — gets a family branch
- [`gemma_tuner/scripts/gemma_preflight.py`](../../gemma_tuner/scripts/gemma_preflight.py) — gets family-version assertion
- [`gemma_tuner/wizard/base.py`](../../gemma_tuner/wizard/base.py) — gets availability gating
- [`pyproject.toml`](../../pyproject.toml) — base 3n stack stays canonical
- `requirements-gemma4.txt` — Gemma 4 compatibility install

### External

- [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Gemma 3n model card](https://ai.google.dev/gemma/docs/core)
- [HF transformers issue #45200 — mm_token_type_ids](https://github.com/huggingface/transformers/issues/45200)
- [HF peft issue #3129 — Gemma4ClippableLinear](https://github.com/huggingface/peft/issues/3129)

## Files Likely to Change

| File | Change Type | Notes |
| --- | --- | --- |
| `pyproject.toml` | Modify | Keep base 3n stack honest; no conflicting Gemma 4 extra |
| `requirements-gemma4.txt` | Create | Gemma 4 compatibility install without resolver conflict |
| `config.ini.example` | Modify | Default profile points at a 3n model |
| `gemma_tuner/models/gemma/finetune.py` | Modify | Family detection + branched model load + clippable patch + collator family arg |
| `gemma_tuner/models/gemma/family.py` | Create | `GemmaFamily` enum, `detect_family`, `assert_family_supported`, `family_capabilities` |
| `gemma_tuner/models/gemma/gemma4_patches.py` | Create | `apply_clippable_linear_patch()` (idempotent, lazy import) |
| `gemma_tuner/models/common/collators.py` | Modify | Family arg on every collator; control-token parameterization; `_inject_mm_token_type_ids` helper |
| `gemma_tuner/scripts/gemma_preflight.py` | Modify | Call `assert_family_supported` early; print resolved family |
| `gemma_tuner/wizard/base.py` | Modify | Gate Gemma 4 entries with availability check |
| `gemma_tuner/scripts/gemma_profiler.py` | Modify | Shared fast-fail gate before model load |
| `gemma_tuner/scripts/gemma_generate.py` | Modify | Default model → 3n; shared fast-fail gate before load |
| `gemma_tuner/scripts/inference_common.py` | Modify | Shared fast-fail gate before model load |
| `gemma_tuner/scripts/export.py` | Modify | Shared fast-fail gate before model load |
| `tools/eval_gemma_asr.py` | Modify | Shared fast-fail gate before model load |
| `gemma_tuner/utils/gemma_dataset_prep.py` | Modify | `DEFAULT_MODEL_ID` → 3n |
| `gemma_tuner/scripts/blacklist.py` | Modify | Default model → 3n |
| `tests/test_family.py` | Create | Detection + version gate |
| `tests/test_gemma4_patches.py` | Create | Clippable-linear patch correctness; skip on base install |
| `tests/test_smoke_gemma3n.py` | Create or rename | 2-step smoke on 3n |
| `tests/test_smoke_gemma4.py` | Create | 2-step smoke on 4; skip on base install |
| `README.md` | Modify | Document Gemma 4 compatibility install; quickstart uses 3n |
| `README/KNOWN_ISSUES.md` | Modify | Family-version mismatch troubleshooting |
| `README/guides/README.md` | Modify | Cross-reference this plan |
| `.github/workflows/*` | Modify | Add explicit base + Gemma 4 CI coverage |

## Risks and Mitigations

- **Lazy import for Gemma 4 leaks at module-import time.** → Add a unit test that imports `gemma_tuner` on a base install and asserts `transformers.models.gemma4` is NOT in `sys.modules`. Catches accidental top-level imports.
- **Clippable-linear monkey-patch double-applied.** → Make `apply_clippable_linear_patch()` idempotent: check if `Gemma4ClippableLinear.__bases__` already includes `nn.Linear` before patching.
- **`mm_token_type_ids` shape mismatch on Gemma 4.** → The injection helper creates zeros matching `input_ids.shape` exactly. Unit test asserts the shapes match for batched inputs of varying lengths.
- **Version-pin drift over time.** → A single `min_transformers_version` constant per family in `family.py`. README and preflight read from it; never duplicate.
- **User follows the Gemma 4 install steps with an incompatible torch / peft.** → Preflight runs `assert_family_supported` which checks transformers version. Add a follow-up check for peft >= 0.18.1 and surface a warning. Out of scope for the version assertion proper.
- **Image PR ships before this plan and re-hardcodes `<start_of_turn>`.** → Coordinate sequencing: image PR consumes the family helper if it lands first, otherwise this plan retrofits it. Either path is ~10 lines.
- **Collator constructor signature change breaks downstream callers.** → `family` is required and keyword-only, so missing wiring fails immediately in tests instead of silently treating Gemma 4 as 3n.
- **Gemma 4 tests run on CI with no compatibility install and silently always-skip.** → CI matrix has two jobs: `base` (3n only, 4 tests skip with reason) and `gemma4` (`pip install -e .` + `pip install -r requirements-gemma4.txt`, both run). The `unit-gemma4` job sets `GEMMA_MACOS_EXPECT_GEMMA4_STACK=1`; `tests/test_gemma4_ci_guard.py` asserts `transformers>=5.5` so a bad resolver fails the job instead of skipping smoke.
- **Docs overpromise repo-wide Gemma 4 support when only `finetune` is fully upgraded.** → README must explicitly separate "Gemma 4 training path supported" from "non-training entrypoints still 3n-only" until those callers are upgraded.

## Critical Reminder

> SIMPLER IS BETTER. The whole point of this plan is one helper file (`family.py`), one patch file (`gemma4_patches.py`), one checked-in Gemma 4 compatibility install file, and surgical branches at three points (shared gate, collator, finetune model load). Everything else either stays family-blind or fails fast honestly. Resist the urge to "abstract the model family across the entire codebase while we're in there." Family knowledge belongs at the edges; the middle stays clean.
