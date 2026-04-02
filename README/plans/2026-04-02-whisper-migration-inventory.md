# Whisper Migration Inventory

Date: 2026-04-02
Status: Phase 0 completed

This inventory freezes the repository boundaries before any code or history moves. It is the execution companion to `2026-04-02-whisper-focused-reorganization-plan.md`.

## Boundary Decisions

- New standalone Mamba repository: `/Users/mm/Documents/GitHub/mamba-asr-mps`
- Local publish strategy: create the standalone repository locally first; leave `origin` unset during the split so the extraction is not blocked on a remote-creation step.
- Cloud streaming architecture: keep in the Whisper repo only. No Mamba code currently depends on the GCS/BigQuery streaming path, so duplication is unnecessary in this migration.
- Gemma product surface: keep in the Whisper repo. Current Gemma code and tests have no `mamba`, `Mamba-ASR`, `exogym`, or `gym.exogym` references.

## Migration Checklist

| Path or surface | Move to Mamba repo | Delete with ExoGym | Keep in Whisper repo | Needs new home in Whisper repo | Notes |
| --- | --- | --- | --- | --- | --- |
| `Mamba-ASR-MPS/` | x |  |  |  | Full standalone product tree; current size about 509 MB |
| `Mamba-ASR-NVIDIA/` |  |  |  |  | Empty placeholder; remove from Whisper repo during Mamba extraction |
| `MambaASR.mlmodelc/` | x |  |  |  | Mamba deployment artifact bundle |
| `MambaASR.mlpackage/` | x |  |  |  | Mamba Core ML package |
| `CMHello.swift` | x |  |  |  | Core ML validation utility for `MambaASR` |
| `cmhello` | x |  |  |  | Built binary for `CMHello.swift`; belongs with Mamba tooling until later cleanup |
| `README/guides/apple-silicon/Mamba-Apple-Silicon-guide.md` | x |  |  |  | Mamba-specific Apple Silicon guide |
| `README/guides/research/mamba-asr-landscape.md` | x |  |  |  | Mamba research landscape note points users to `Mamba-ASR-MPS/` |
| `README.md` Mamba sections |  |  |  | x | Replace with pointer to new Mamba repo |
| `README/guides/README.md` Mamba references |  |  |  | x | Replace/remove entries that point into Mamba material |
| `gym/` |  | x |  |  | Git submodule and ExoGym code |
| `.gitmodules` `gym` entry |  | x |  |  | Remove submodule metadata |
| `distributed/` |  | x |  |  | Entire ExoGym-backed distributed surface |
| `train_distributed.py` |  | x |  |  | ExoGym launcher |
| `test_distributed.py` |  | x |  |  | Distributed validation script |
| `distributed_hosts.json.example` |  | x |  |  | No surviving distributed workflow after this migration |
| `cli_typer.py` distributed commands |  | x |  |  | Remove distributed command surface; keep canonical Whisper CLI |
| `main.py` distributed references |  | x |  |  | Legacy CLI should stop advertising removed flows |
| `wizard/runner.py` distributed hooks |  | x |  |  | Remove distributed setup path from wizard execution |
| `README/specifications/distributed-training-gym.md` |  | x |  |  | No longer true after ExoGym removal |
| `README/guides/integrations/exolabs-gym.md` |  | x |  |  | No longer true after ExoGym removal |
| `DISTRIBUTED_TRAINING.md` |  | x |  |  | Obsolete top-level distributed guide |
| `.github/workflows/nightly-distributed-check.yml` |  | x |  |  | Workflow exists only for distributed checks |
| `.github/workflows/ci.yml` distributed test invocations |  | x |  |  | Remove distributed references from surviving CI |
| `core/` |  |  | x |  | Core Whisper package surface survives |
| `models/whisper/` |  |  | x |  | Whisper trainer survives |
| `models/whisper_lora/` |  |  | x |  | Whisper LoRA trainer survives |
| `models/distil_whisper/` |  |  | x |  | Whisper distillation survives |
| `models/gemma/` |  |  | x |  | Gemma stays in Whisper repo |
| `utils/dataset_utils.py` |  |  | x |  | Streaming dataset loading and patch application stay in Whisper |
| `utils/dataset_prep.py` |  |  | x |  | Local and GCS audio loading stays in Whisper |
| `core/bigquery.py` |  |  | x |  | BigQuery import/export path stays in Whisper |
| `scripts/prepare_data.py` |  |  | x |  | Surviving data-prep flow |
| `wizard/config.py` |  |  | x |  | Surviving wizard configuration path |
| `wizard/` remaining guided setup |  |  | x |  | Keep wizard, remove only distributed branch |
| `scripts/` remaining Whisper/Gemma workflows |  |  | x |  | Prepare, finetune, evaluate, export, blacklist, pseudo-label, Gemma |
| `visualizer.py` |  |  |  | x | Keep product feature, but fold under future package namespace |
| `VISUALIZER_README.md` |  |  |  | x | Keep content, relocate into product docs |
| `explainer.md` |  |  |  | x | Distillation explainer belongs under product docs |
| `verified_generations_with_audio_VIEW_fixed.sql` |  |  |  | x | Keep as cloud-data helper; relocate into docs or SQL tooling home |

## Whisper Workflows That Must Survive

- Canonical CLI: `cli_typer.py` and the `whisper-tuner` entrypoint
- Training and evaluation core: `core/`, `models/`, `utils/`, `scripts/`
- Guided setup: `wizard/`
- Data quality and export tooling: blacklist, evaluation, GGUF/Core ML export, visualizer
- Cloud-resident data with local compute: BigQuery import, GCS audio loading, dataset streaming

## Cloud Streaming Architecture Inventory

These files are the current cloud-local contract and must survive the migration:

- `core/bigquery.py`: BigQuery discovery, query execution, and local materialization helpers
- `utils/dataset_prep.py`: local-or-GCS audio loading helpers
- `utils/dataset_utils.py`: streaming-aware dataset loading and patch application
- `scripts/prepare_data.py`: data preparation flow that consumes streamed or prepared manifests
- `wizard/config.py`: wizard-side BigQuery and dataset setup
- `core/inference.py`: direct support for `gs://` audio inputs
- `models/whisper/finetune.py`: streamed dataset loading through shared dataset utilities
- `models/gemma/finetune.py`: streamed dataset loading and cloud-aware audio loading

Decision: keep this architecture in the Whisper repo. The current Mamba tree does not import `core.bigquery`, `utils.dataset_prep`, `utils.dataset_utils`, or any Google Cloud client library.

## Gemma Audit

Gemma stays in the Whisper repo. The following surfaces were checked and have no Mamba or ExoGym dependency:

- `models/gemma/finetune.py`
- `scripts/gemma_preflight.py`
- `scripts/gemma_profiler.py`
- `scripts/gemma_tiny_overfit.py`
- `scripts/gemma_generate.py`
- `utils/gemma_dataset_prep.py`
- `tests/test_gemma_collator.py`
- `tests/test_gemma_dataset_prep.py`
- `README/specifications/Gemma3n.md`

## Current Size and Repo Hygiene Notes

- `Mamba-ASR-MPS/`: about 509 MB
- `MambaASR.mlmodelc/`: about 1.7 MB
- `MambaASR.mlpackage/`: about 1.5 MB
- `gym/`: about 684 KB as a git submodule checkout

The current worktree is already dirty. Subsequent phases must avoid reverting unrelated user changes while removing the product surfaces above.
