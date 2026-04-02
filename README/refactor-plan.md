## Whisper Fine-Tuner Refactor Plan

### Where we started
- **Monolithic orchestration in `main.py`**: mixed CLI, device init, run ID/metadata, config merging, and operation dispatch all in one large file. Hard to change safely.
- **Duplication across trainers**: `models/whisper/finetune.py`, `models/whisper_lora/finetune.py`, and `models/distil_whisper/finetune.py` share similar scaffolding (args, dataset prep, metrics) with drift risk.
- **Visualizer not wired**: server can start, but no training callbacks push metrics, so the UI won’t show real data.
- **Inconsistent logging**: mix of prints and logger usage; no single logger init; uneven error surfaces.
- **MPS footguns in config**: defaults favor `float16`/`sdpa`, which can be problematic on Apple Silicon; need normalization to `float32`/`eager` on MPS.
- **Reproducibility gaps**: semi-pinned requirements but no lockfile or CI to validate across machines.
- **Legacy test (`tests/test_dataset.py`)**: assumes a `[dataset]` section no longer used by the modern loader; fails in clean environments.

### What we’ve done (this refactor)
- **Introduced `core/` package** to separate orchestration concerns:
  - `core/config.py`: `load_profile_config(...)`, `load_model_dataset_config(...)` extracted from `main.py`.
  - `core/runs.py`: `get_next_run_id(...)`, `create_run_directory(...)`, `update_run_metadata(...)`, `mark_run_as_completed(...)`, `find_latest_finetuning_run(...)` extracted/centralized.
  - `core/logging.py`: `init_logging(...)` to standardize log setup.
  - `core/ops.py`: thin adapters to existing `scripts/*` operations (prepare/finetune/evaluate/export/blacklist).
- **Slimmed `main.py` to a thin CLI**: keeps early MPS env guard and device setup, delegates config/run/ops/logging to `core/`.
- **Kept behavior stable**: retained a `find_finetuning_run_dir` alias for blacklist path semantics; left model trainers unchanged; no functional changes to training/eval/export.
- **Sanity checks**: lints clean; targeted tests (MPS) pass. Legacy dataset test is gated behind `RUN_LEGACY_DATASET_TEST=1`.

### Status and roadmap

#### Completed

- Phase 1: Reliability and safety
  - **Central logging rollout**: replaced prints with module loggers in `scripts/prepare_data.py`, `scripts/export.py`, `scripts/evaluate.py`, `scripts/blacklist.py`, and trainer orchestrator `scripts/finetune.py`. `init_logging()` called once in `main.py`.
    - Impact: consistent, filterable logs; easier debugging in long runs.
  - **Signal-safe run finalization**: added SIGINT/SIGTERM handlers around `finetune` and `blacklist` to set `status="cancelled"` and `end_time` in `metadata.json`.
    - Impact: fewer “incomplete” runs; clearer lifecycle state.
  - **Device-safe config normalization**: on MPS, enforce `dtype=float32` and `attn_implementation="eager"` at dispatch-time for `finetune`, `evaluate`, and `blacklist`.
    - Impact: avoids MPS operator issues and silent degradations.
  - **Visualizer server hardening**: bind `127.0.0.1`, auto-pick a free port, default `open_browser=False`, and only allow unsafe werkzeug when `VIZ_ALLOW_UNSAFE_WERKZEUG=1`.
    - Impact: safer defaults, fewer surprises on shared systems.
  - **Fix/guard legacy dataset test**: `tests/test_dataset.py` uses modern API when available; otherwise gated behind `RUN_LEGACY_DATASET_TEST=1`.
    - Impact: green CI by default; keeps legacy utility available on demand.
- Phase 2: Simplicity and integration
  - **Wire visualizer to training**: added minimal callback in each trainer that emits `loss`, `lr`, `grad_norm`, `memory_gb` every N steps when `visualize=True`.
    - Implemented `models/common/visualizer.py` and integrated in `models/whisper/finetune.py`, `models/whisper_lora/finetune.py`, and `models/distil_whisper/finetune.py`.
    - Impact: delivers visualizer with negligible overhead when off (default).
  - **Config validation**: basic validators in `core/config.py` enforce required keys and sane ranges.
    - Impact: earlier, clearer failures; fewer runtime surprises.
  - **Evaluation/export resolution**: added `core/runs.find_latest_completed_finetuning_run(...)` and use it in `main.py` (evaluate) and `core/runs.create_run_directory(...)` (linking runs). Export remains a manual path/id.
    - Impact: correctness under concurrent runs and filesystem anomalies.
  - **Extract shared collators**: unified collators now used by all trainers:
    - `DataCollatorWhisperStrict` (standard), `DataCollatorWhisperLoRA`, `DataCollatorWhisperDistill`.
    - Impact: removes duplicated collator logic across trainers; simpler maintenance.
  - **Extract training arg builders/normalization**: shared helpers centralize TrainingArguments setup and platform-safe normalization (dtype/attn already enforced at dispatch; worker caps unified).
    - Implemented in `models/common/args.py`: `get_effective_preprocessing_workers(...)`, `get_effective_dataloader_workers(...)`, `build_common_training_kwargs(...)`.
    - Integrated into all trainers.
    - Impact: reduces drift in Trainer argument creation and platform tweaks.

- Phase 3: Reproducibility and CI
- **Lockfile support**: Documented lockfile generation via `uv` or `pip-tools` (`requirements.lock`). Torch remains user-installed per platform.
  - Impact: deterministic setup guidance across machines.
- **CI on macOS**: Added `.github/workflows/ci-macos.yml` (Python 3.10/3.11) to run smoke imports, prepare tiny streaming dataset, and run a short evaluation with `openai/whisper-tiny`.
  - Impact: prevents regressions; validates basics on macOS runners.
- **Dataset pipeline hardening**: Added CSV schema validation in `utils/dataset_utils.load_dataset_split()` and split fallback to `{dataset}_prepared.csv`. Auto-detect `gs://` URIs in `scripts/prepare_data.py` to force streaming mode. Capped preprocessing/dataloader workers on MPS by default with config overrides.
  - Impact: earlier failures, predictable performance, fewer memory spikes on MPS.

#### Completed
- Phase 4: Nice-to-haves
  - **Structured logs + metrics**
    - Optional JSON logging via `--json_logging` or `LOG_JSON=1`. File logging attached per run (`run.log`) with `--log_file` override.
    - Added `core/runs.write_metrics(...)` and now write a `metrics.json` per run. Evaluation metrics are persisted; training writes `train_results.json` from the Trainer and merges into `metrics.json` with duration.
    - Acceptance: JSON logs enabled on demand; `metrics.json` includes key stats (e.g., loss/wer/cer) and timing.
  - **CLI ergonomics**
    - Clearer help text and defaults using `ArgumentDefaultsHelpFormatter`.
    - New flags: `--json_logging`, `--log_file` (non-breaking). Existing commands unchanged.
    - Acceptance: `whisper-tuner -h` is cleaner; behavior preserved.
  - **Visualizer polish**
    - Emission throttle wired: `viz_update_steps` (defaults to `logging_steps`).
    - Frontend auto-reconnect/backoff added; server logs hardened.
    - Lightweight feature flags via URL params to toggle heavy panels (attention/token cloud/spectrogram/3D).
    - Acceptance: UI stays responsive; reconnect works; heavy features toggleable.
  - **Docs & examples**
    - Added tiny end-to-end examples (prepare → eval; optional micro-train) and logging notes in README.
    - Acceptance: quickstart path documented; new flags described.
  - **Optional CI smoke (mini-train)**
    - Guarded mini-train step added in `.github/workflows/ci-macos.yml` toggled by `RUN_MINI_TRAIN=1`.
    - Acceptance: off by default to keep CI fast; passes locally.

#### To Do (follow-ups)
- Visualizer: expose feature toggles in UI (not only via URL params) and minor UI polish.
- CI: ensure macOS GitHub Actions workflow is present and green; include lint + smoke (prepare tiny streaming dataset, short eval). Keep optional guarded mini-train.
- Typer CLI: document optional Typer interface (`cli_typer.py`) in README and add a small smoke test.
- Tests: add focused unit tests for `core/config` validation paths, `core/ops` dispatch error paths, and `core/runs` metadata/metrics merges.
- Docs: fix duplicate "Completed" header in this file; note Typer CLI and LLM-first headers added to smaller helper modules (`core/__init__.py`, `utils/__init__.py`, `models/common/__init__.py`, root `__init__.py`).

### Risk management
- Changes are staged to keep functional behavior stable at each step.
- We extracted code, not logic; rollbacks are easy (imports point back to old helpers).
- Visualizer integration is fully optional and off by default.

### Acceptance criteria
- Core flows work end-to-end: `prepare`, `finetune`, `evaluate`, `export`.
- Run metadata is always consistent (`running` → `completed`/`failed`/`cancelled` with `end_time`).
- Visualizer shows live data when enabled; adds no overhead when disabled.
- CI green on macOS runners; lockfile in repo; docs updated.

### File map (post-refactor so far)
- `core/config.py`: config/profile merging
- `core/runs.py`: run IDs, directories, metadata, completion
- `core/logging.py`: logging init
- `core/ops.py`: operation dispatch to `scripts/*`
- `main.py`: thin CLI; early MPS env, device setup, delegates to `core/*`
- Trainers unchanged (next phases): `models/whisper/*`, `models/whisper_lora/*`, `models/distil_whisper/*`
 - `visualizer.py`: hardened server; localhost bind, free port selection, optional browser open
- `models/common/visualizer.py`: minimal HF Trainer callback to stream metrics when visualization is enabled
 - `models/common/collators.py`: shared collators (`WhisperStrict`, `WhisperLoRA`, `WhisperDistill`)
 - `models/common/args.py`: shared training arg/normalization helpers
 - `models/common/metrics.py`: WER/CER builder (scaffold for future integration)

