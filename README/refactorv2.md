# Whisper Fine-Tuner — Refactor v2 (To‑Do)

Goal: A-grade reliability with simpler architecture, no feature loss.

## Phase 0 — Must-fix correctness (now)
- [x] scripts/blacklist.py: remove duplicate metrics call (keep one)
  - Location: see file around the end; search for two consecutive `compute_metrics(...)` calls.
- [x] scripts/export.py: align behavior with docs (Option A chosen)
  - Option A (rename): `export_ggml` → `export_model_dir` (pure HF/SafeTensors export), update README.
  - Option B (implement): add real GGML/CT2 export path behind `--format ggml|ct2` flag and keep name.
- [x] utils/dataset_utils.py (and similar): replace `print()` diagnostics with unified logger
  - Use `core.logging.init_logging()`; keep verbosity behind INFO/DEBUG.

## Phase 1 — Unify inference/evaluation (remove duplication)
- [x] Create `core/inference.py` (single source of truth)
  - `prepare_features(audio_or_path, feature_extractor) -> input_features`
  - `generate(model, input_features, language_mode, forced_language) -> token_ids`
  - `decode_and_score(tokenizer, preds, refs, normalizer) -> {wer, cer, strings...}`
- [x] Refactor `scripts/evaluate.py` to call `core/inference.py`
- [x] Refactor `scripts/blacklist.py` to call `core/inference.py`

## Phase 1b — Shared dataset preprocessing
- [ ] Add `utils/dataset_prep.py`
  - `load_audio_local_or_gcs(path, sr, timeout=10, retries=2)`
  - `encode_labels(tokenizer, text, max_len)`
  - `resolve_language(language_mode, sample_lang, forced_lang)`
- [ ] Use in:
  - `models/whisper/finetune.py`
  - `models/whisper_lora/finetune.py`
  - `models/distil_whisper/finetune.py`
  - `scripts/evaluate.py`
  - `scripts/blacklist.py`

## Phase 2 — Config + logging hardening
- [x] `core/config._validate_profile_config`:
  - Default `language_mode` to `'strict'` if missing
  - Coerce ints/floats/bools centrally (no scattered `int(...)`/`float(...)`)
  - Validate LoRA params types if present
- [x] Replace remaining ad-hoc prints in:
  - `utils/dataset_utils.py`, `scripts/gather.py`, `manage.py` (debug paths) → logger

## Phase 3 — Cleanup redundancy
- [ ] `models/whisper/finetune.py`: remove `CustomDataCollator`; use `models/common/collators.DataCollatorWhisperStrict`
- [ ] Ensure all trainers use `models/common/args.build_common_training_kwargs(...)`

## Phase 4 — Tests (fast, targeted)
- [ ] `tests/test_config.py`: required keys, type coercion, invalid splits
- [ ] `tests/test_runs.py`: `get_next_run_id`, `create_run_directory`, metadata/metrics merge, completion marker
- [ ] `tests/test_dataset_utils.py`: patch precedence (override → do_not_blacklist → blacklist), streaming filtering
- [ ] `tests/test_language_mode.py`: strict/mixed/override prompt behavior
- [ ] `tests/test_inference.py`: tiny local wav and fake-GCS (monkeypatch) round-trip → non-empty decode + metric call

## Phase 5 — Docs / CLI polish
- [ ] README export section: reflect chosen export path (rename vs real conversion)
- [ ] Document Typer CLI usage as the recommended interface (keep `main.py`)
- [ ] Note new `core/inference.py` and `utils/dataset_prep.py` for contributors

## Acceptance criteria
- One inference path used by evaluation and blacklist; no duplicate dataset mapping.
- All scripts log via unified logger (human or JSON).
- Export task name/behavior matches README; CI still green.
- New tests pass locally on macOS; smoke tests unchanged in runtime.

## Rollout plan (PR slices)
1) Phase 0 fixes + log swaps (small)  
2) Phase 1 core inference + evaluate refactor  
3) Phase 1b dataset_prep + trainer/eval/blacklist adoption  
4) Phase 3 cleanup + args/collators convergence  
5) Phase 4 tests  
6) Phase 5 docs

Notes:
- Keep CI fast; tests are unit-level and avoid heavy model pulls.
- GCS reads: add short retries + WARN; keep silence fallback for CI.