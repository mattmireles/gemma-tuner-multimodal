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
- [x] `models/whisper/finetune.py`: remove `CustomDataCollator`; use `models/common/collators.DataCollatorWhisperStrict`
- [x] Ensure all trainers use `models/common/args.build_common_training_kwargs(...)`

## Phase 4 — Tests (fast, targeted)
- [x] `tests/test_config.py`: required keys, type coercion, invalid splits
- [x] `tests/test_runs.py`: `get_next_run_id`, `create_run_directory`, metadata/metrics merge, completion marker
- [x] `tests/test_dataset_utils.py`: patch precedence (override → do_not_blacklist → blacklist), streaming filtering
- [x] `tests/test_language_mode.py`: strict/mixed/override prompt behavior
- [x] `tests/test_inference.py`: tiny local wav and fake-GCS (monkeypatch) round-trip → non-empty decode + metric call

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

## Phase 6 — CI + E2E Reliability (A+ bar)
- [ ] macOS CI workflow (GitHub Actions):
  - Import smoke, tiny streaming prepare, evaluation run (whisper-tiny), artifact upload (logs/metrics).
  - Matrix: Python 3.10/3.11; cache HF hub; enforce `requirements.lock` install via uv/pip-tools.
  - Optional mini-train guarded by `RUN_MINI_TRAIN=1` (1–2 batches, LoRA tiny), with timeout and artifact on failure.
  - Concurrency: cancel in-progress on new pushes; fail CI on non-zero exit codes and coverage below threshold (see Phase 12).
- [ ] CI failure diagnostics:
  - Upload `output/**/metadata.json`, `metrics.json`, and `run.log` as artifacts on failure.
  - Emit summarized error and remediation hints in job summary (e.g., MPS fallback, batch-size reduction).

## Phase 7 — Determinism & Resilience
- [ ] Deterministic mode:
  - Global seeding in one place (torch/numpy/random), `torch.use_deterministic_algorithms(True)` where supported; document MPS caveats.
  - Dataloader worker seeding; pinned shuffle seeds for HF datasets.
- [ ] Robust GCS streaming:
  - Bounded timeouts + exponential backoff + limited retries; structured WARN on fallback (CI-safe silence preserved only in CI mode).
  - Sample-level try/except guards with skipped-sample counters written to `metrics.json`.
- [ ] Run recovery:
  - Detect stale “running” runs without `completed` marker; mark as `failed` with reason on next startup; add `manage.py cleanup --stale`.

## Phase 8 — Observability & Performance
- [ ] Extend `core/runs.write_metrics(...)`:
  - Persist per-phase timings (prepare/train/eval), throughput (samples/sec), and peak memory (MPS/CUDA) per run.
  - Merge step logs into rolling aggregates (mean/min/max loss per N steps) to keep files small.
- [ ] Lightweight benchmarking:
  - `manage.py overview --perf` to print averages of throughput/memory across completed runs.

## Phase 9 — Packaging & CLI Distribution
- [ ] Add `pyproject.toml` with console scripts:
  - Entry point `whisper-tuner` → Typer CLI; extras: `[viz]`, `[gcs]`, `[dev]`.
  - Ensure importable modules (`core/*`, `models/common/*`, `utils/*`) are packaged.
- [ ] Installation docs:
  - pip/pipx instructions; minimal env bootstrap that respects torch-per-platform guidance.

## Phase 10 — Documentation & Examples
- [ ] End-to-end tiny example:
  - Add a licensed 3–5 file local dataset (`data/datasets/test_local/`) to validate full flow offline.
  - Quickstart scripts: prepare → finetune (LoRA tiny, 1–2 batches) → evaluate → gather; copy/paste friendly.
- [ ] Export section refresh:
  - Reflect Option A (HF/SafeTensors export). If GGML/CT2 added later, document flags and dependencies.
- [ ] Architecture notes:
  - Add a diagram of `core/inference.py` consumption paths (evaluate + blacklist) to reduce contributor errors.

## Phase 11 — Test Suite Expansion (high-signal, fast)
- [ ] LoRA smoke:
  - 1–2 steps training; assert `adapter_model/` exists; verify `train_results.json` merged into `metrics.json`.
- [ ] Distillation dry-run:
  - Instantiate teacher/student; single forward pass on 1–2 samples; assert no OOM and proper loss composition; a `--fast` flag to skip full train.
- [ ] Visualizer callback test:
  - Ensure callback attaches; throttled `on_log` does not raise; no server dependency required (mock `get_visualizer()`).
- [ ] Gather/Manage:
  - Tests for `gather.py` (columns present, language join) and `manage.py overview/list` happy-paths.

## Phase 12 — Linting, Types, Coverage Gates
- [ ] Pre-commit + CI gates:
  - Ruff (errors + select rules), isort/black style (if desired consistency), mypy (focus on `core/*`, `utils/*`, `scripts/evaluate.py`, `scripts/blacklist.py`).
  - Coverage: 80%+ for `core/*` and `utils/*`, 60% global minimum (exclude heavy trainers).
- [ ] Type hints:
  - Add typings to new modules (`core/inference.py`, `utils/dataset_prep.py`, `core/config.py` validators).

## Phase 13 — Error Taxonomy & UX
- [ ] `core/errors.py` with typed exceptions:
  - `ConfigError`, `DatasetError`, `DeviceError`, `StreamingError`, `ExportError`; map to exit codes in `main.py`.
  - Write `error_code` alongside `error_message` in run metadata for quick triage.
- [ ] Helpful CLI remediation:
  - Common hint mapping (e.g., “Try PYTORCH_ENABLE_MPS_FALLBACK=1”, “Reduce batch size”, “Install torch with correct index-url”).