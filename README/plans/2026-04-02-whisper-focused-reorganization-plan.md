# Whisper-Focused Repository Reorganization Plan

**Date:** 2026-04-02
**Status:** In Progress

## Execution Status

- [x] Phase 0 completed
- [ ] Phase 1 completed
- [ ] Phase 2 completed
- [ ] Phase 3 completed
- [ ] Phase 4 completed

## Executive Summary

This plan narrows the current repository to a single product boundary: the Whisper fine-tuner. The work splits the Mamba-ASR product surface into a new standalone repository, removes the `gym`/ExoGym dependency from this repo, and then reorganizes the remaining Whisper code into a clearer package and docs structure without a high-risk import-path big bang.

## Problem Statement

- **Symptom:** The repository currently mixes the Whisper trainer, a separate Mamba-ASR product, and an ExoGym submodule-backed distributed training layer in one top-level tree.
- **Root Cause:** Independent experiments and supporting projects were co-located without enforcing package, repo, or release boundaries.
- **Impact:** Ownership is unclear, packaging is porous, onboarding is slow, CI is hard to trust, and unrelated generated artifacts make the repo larger and noisier than it should be.

## Goals and Non-Goals

### Goals

- [ ] Keep this repository focused on Whisper fine-tuning, evaluation, export, and the guided CLI workflow.
- [ ] Preserve the cloud streaming architecture, where metadata and audio can remain in cloud systems while training and evaluation run locally.
- [ ] Create a new standalone repository for `Mamba-ASR-MPS/`, preserving its code, docs, and release history as cleanly as possible.
- [ ] Resolve the rest of the Mamba-owned surface in the root tree, including `Mamba-ASR-NVIDIA/`, `MambaASR.mlmodelc/`, and `MambaASR.mlpackage/`.
- [ ] Remove ExoGym from this repository completely, including the submodule, imports, CLI surface, docs, and tests that depend on it.
- [ ] Reorganize the remaining Whisper code into a clearer package boundary with fewer top-level namespaces and fewer legacy entrypoints, without requiring an immediate `src/` migration.
- [ ] Remove tracked generated artifacts and build outputs from the Whisper repo so git reflects source, docs, fixtures, and intentional assets only.

### Non-Goals

- Replacing ExoGym with a new distributed training framework in the same change set. This migration removes the dependency first.
- Rewriting core Whisper training algorithms or retraining models as part of the repo split.
- Moving Gemma-related support in this pass unless it blocks the Whisper-focused boundary.

## Scope and Constraints

- **Scope:** Repository boundaries, packaging, docs, CLI surface, git hygiene, CI, and migration sequencing.
- **Constraints:** The current worktree is already dirty; the plan must avoid assuming a clean baseline or reverting unrelated user changes.
- **Guardrails:** Standard Whisper workflows must remain intact: prepare, finetune, evaluate, export, blacklist, and wizard-driven setup.

## Ground Truth Contracts (Do Not Violate)

- **Whisper installability:** The root package defined in `pyproject.toml` must remain installable and runnable as the Whisper product after the migration.
- **Primary CLI continuity:** `whisper-tuner` remains the canonical entrypoint, even if legacy shims are removed later.
- **Run/output compatibility:** Existing Whisper run directories and metadata contracts remain readable by the surviving management and export tools.
- **Cloud-local architecture continuity:** The ability to keep data in cloud storage or BigQuery while compute runs locally must either stay in the Whisper repo or be intentionally duplicated into the new Mamba repo if Mamba still depends on that workflow.
- **No hidden nested projects:** After the migration, the Whisper repo must not contain another product repo or a git submodule masquerading as first-party code.

## Already Shipped (Do Not Re-Solve)

- **Typer CLI:** `cli_typer.py` already centralizes the primary CLI flow.
- **Whisper core pipeline:** `core/`, `models/`, `utils/`, `scripts/`, and `wizard/` already implement the main Whisper workflows.
- **Cloud streaming path:** GCS/BigQuery-backed local-compute flows already exist through `core/bigquery.py`, `utils/dataset_prep.py`, `utils/dataset_utils.py`, `scripts/prepare_data.py`, `wizard/config.py`, and model training code that consumes streamed audio.
- **Gemma support:** Gemma-related training and export flows remain in scope for the Whisper repo unless Phase 0 finds a hidden dependency on Mamba or ExoGym.
- **Deprecation direction:** `main.py` already marks itself as legacy and points users toward the Typer CLI.
- **Packaging baseline:** The root `pyproject.toml` already defines the installable Whisper package and script entrypoint.

## Fresh Baseline (Current State)

What exists today, with concrete data:

- **Architecture:** The root repo packages Whisper code from multiple top-level directories while also containing a `gym` git submodule, a full `Mamba-ASR-MPS/` product tree, an empty `Mamba-ASR-NVIDIA/` directory, root-level Mamba model artifacts, and several loose files with no clear product home.
- **Metrics:** `Mamba-ASR-MPS/` is roughly 509 MB in the current working tree; `MambaASR.mlmodelc/` is about 1.7 MB; `MambaASR.mlpackage/` is about 1.5 MB; `gym/` is configured as a git submodule; the main package exports multiple top-level packages instead of one namespace package.
- **Known gaps:** Distributed code imports ExoGym directly and uses `sys.path` hacks; wizard and tests still expose distributed flows; generated artifacts are tracked despite ignore rules; CI workflows exist but are disabled; the cross-repo fate of cloud streaming code is not yet explicitly captured.

## Solution Overview

First separate product boundaries, then simplify the Whisper repo around what remains.

```text
+------------------------------+        +---------------------------+
| Current repo                 |        | Target state              |
|                              |        |                           |
| whisper code                 |        | whisper-fine-tuner repo   |
| + Mamba-ASR-MPS              |  -->   | - whisper-only code       |
| + gym/exogym submodule       |        | - no submodules           |
| + tracked generated outputs  |        | - clean package boundary  |
+------------------------------+        +---------------------------+
                 \
                  \-------> new standalone Mamba-ASR repo
```

The migration order is intentional:

1. Extract Mamba-ASR first so it stops shaping the Whisper repo.
2. Remove ExoGym next so the remaining code has a clear dependency graph.
3. Reorganize the Whisper codebase after the scope is reduced.

Two implementation choices are fixed up front:

1. Use `git subtree split` as the default Mamba extraction method.
2. Consolidate under a top-level `whisper_tuner/` package first; defer any `src/` layout migration to a later, separate decision.

## Implementation Phases

> Do one phase at a time. Verify before proceeding.

### Phase 0: Prerequisites and Inventory

**Goal:** Freeze the target boundaries and capture everything that must move, stay, or be deleted.

**Tasks:**

- [x] Inventory all Mamba-owned files under `Mamba-ASR-MPS/` plus the adjacent Mamba surface: `Mamba-ASR-NVIDIA/`, `MambaASR.mlmodelc/`, `MambaASR.mlpackage/`, and Mamba-specific docs and scripts.
- [x] Inventory all ExoGym-dependent files, including `.gitmodules`, `gym/`, `distributed/`, `train_distributed.py`, `test_distributed.py`, `wizard/runner.py`, tests, docs, and CLI commands.
- [x] Record current Whisper workflows that must survive unchanged: `cli_typer.py`, `core/`, `models/`, `utils/`, `scripts/`, `wizard/`, and relevant tests.
- [x] Inventory the cloud streaming architecture explicitly, including GCS audio loading, streaming dataset handling, BigQuery import/export flow, and any model-specific streaming hooks.
- [x] Inventory loose root-level files and assign each one to `keep`, `move`, `delete`, or `relocate`: `CMHello.swift`, `cmhello/`, `verified_generations_with_audio_VIEW_fixed.sql`, `visualizer.py`, `VISUALIZER_README.md`, and `explainer.md`.
- [x] Inventory docs beyond the root README, including `README/specifications/distributed-training-gym.md`, `README/guides/integrations/exolabs-gym.md`, and `README/guides/apple-silicon/Mamba-Apple-Silicon-guide.md`.
- [x] Confirm Gemma code paths and tests do not depend on Mamba or ExoGym, and explicitly mark Gemma as `keep in Whisper repo` if clean.
- [x] Decide whether the new Mamba repo needs a duplicated copy of the cloud streaming architecture or only documentation pointing to the Whisper implementation.
- [x] Decide the target name and remote for the new Mamba repository before moving code.

**Verification:** A migration checklist exists with four columns: `move to Mamba repo`, `delete with ExoGym`, `keep in Whisper repo`, and `needs new home in Whisper repo`, and the cloud streaming architecture is listed as either `keep` or `duplicate`.

Phase 0 execution notes are captured in [2026-04-02-whisper-migration-inventory.md](./2026-04-02-whisper-migration-inventory.md).

---

### Phase 1: Extract `Mamba-ASR-MPS/` Into a New Repository

**Goal:** Move the Mamba-ASR product into its own repo without leaving Whisper dependent on it.

**Tasks:**

- [ ] Create the new Mamba repository from `Mamba-ASR-MPS/` using `git subtree split --prefix=Mamba-ASR-MPS` as the default history-preserving extraction method.
- [ ] Move Mamba-specific docs, scripts, benchmarks, Swift runner code, exports guidance, and model assets ownership into the new repository.
- [ ] If Mamba depends on cloud-resident data with local compute, duplicate the required cloud streaming components or design-equivalent interfaces into the new Mamba repo instead of silently dropping that capability.
- [ ] Decide the fate of `Mamba-ASR-NVIDIA/`, `MambaASR.mlmodelc/`, and `MambaASR.mlpackage/`: move them to the new Mamba repo if they are still product assets, otherwise delete them from the Whisper repo.
- [ ] Add or tighten standalone repo metadata for Mamba, including its own `.gitignore`, README, and packaging/bootstrap story if needed.
- [ ] Decide whether any intentional large binaries in the new Mamba repo should use Git LFS; do not carry generated exports into git by default just because they existed here.
- [ ] Make the new repo obviously independent on day one by updating its README title, setup instructions, and default branch metadata as needed.
- [ ] Remove `Mamba-ASR-MPS/` from this repo after the new repo is validated.
- [ ] Replace Mamba sections in `README.md` and related guides, including Mamba-specific Apple Silicon and Core ML guidance, with a short pointer to the new repository.

**Verification:** The Whisper repo no longer contains `Mamba-ASR-MPS/`, `Mamba-ASR-NVIDIA/`, or root-level Mamba artifact bundles unless explicitly reclassified; the new Mamba repo can explain how to train and evaluate its pipeline without depending on this repo.

---

### Phase 2: Remove ExoGym and All Dependent Surface Area

**Goal:** Delete ExoGym cleanly instead of leaving broken imports, commands, or docs behind.

**Tasks:**

- [ ] Remove the `gym` submodule cleanly: `git submodule deinit -f gym`, remove the `gym` entry from `.gitmodules`, remove any `.git/config` submodule entry if present, delete `.git/modules/gym` if needed, and delete the `gym/` tree from the Whisper repo.
- [ ] Delete or retire ExoGym-bound distributed code paths in `distributed/trainer.py`, `distributed/network_trainer.py`, `distributed/worker_entry.py`, `distributed/__init__.py`, `distributed/launcher.py`, `distributed/whisper_wrapper.py`, `train_distributed.py`, and `wizard/runner.py`.
- [ ] Remove Typer and legacy CLI commands that expose ExoGym-backed distributed flows from `cli_typer.py` and `main.py`.
- [ ] Delete or explicitly replace distributed test coverage, including `test_distributed.py`, `tests/test_distributed_check.py`, `tests/test_distributed_dry_run.py`, and `tests/test_distributed_launcher.py`.
- [ ] Remove distributed training docs that are no longer true, including README sections and supporting guides such as `README/specifications/distributed-training-gym.md` and `README/guides/integrations/exolabs-gym.md`.

**Verification:** A clean install of the Whisper repo succeeds without `gym`, no code imports `exogym` or `gym.exogym`, and the exposed CLI only references supported Whisper functionality.

---

### Phase 3: Reorganize the Remaining Whisper Repository

**Goal:** Turn the surviving Whisper code into a cleaner, single-product repository.

**Tasks:**

- [ ] Move toward a single package namespace under `whisper_tuner/` with a clearly bounded package layout; do not require an immediate `src/` migration in this phase.
- [ ] Consolidate top-level modules and packages from `core/`, `models/`, `utils/`, `wizard/`, and `scripts/` under the chosen namespace.
- [ ] Keep `cli_typer.py` as the only first-class CLI entrypoint and plan the retirement of legacy shims such as `main.py` and `manage.py`.
- [ ] Reorganize docs so product docs live in one place instead of splitting responsibility across `README.md`, `README/`, and scattered top-level markdown files.
- [ ] Update imports, packaging, and tests to match the new directory layout.

**Verification:** `pip install -e .` works, `whisper-tuner --help` works, and the surviving Whisper test suite passes against the reorganized layout.

---

### Phase 4: Validation and Cleanup

**Goal:** Verify end-to-end Whisper behavior and remove temporary scaffolding.

**Tasks:**

- [ ] Remove tracked generated artifacts and build outputs that do not belong in source control, then tighten `.gitignore` so they stay out.
- [ ] Re-enable CI with a reduced, trustworthy Whisper-only test matrix in `.github/workflows/`, and remove distributed test references from workflow definitions.
- [ ] Align `pyproject.toml` metadata, including `[project.urls]`, with the real Whisper repository after the new boundaries are in place.
- [ ] Decide whether to document PyTorch as a manual install prerequisite or add an explicit dependency/extra so install expectations match reality.
- [ ] Run and document the core verification flows: install, prepare, finetune smoke path, evaluate, export, and wizard smoke tests.
- [ ] Add short migration notes for contributors so they understand where Mamba went and why ExoGym disappeared.

**Verification:** CI is active, repo size/noise is reduced, and contributor docs describe the Whisper-only boundary clearly.

## Success Criteria

### Hard Requirements (Must Pass)

- [ ] The Whisper repo contains no `Mamba-ASR-MPS/` directory, no stray Mamba product siblings or root Mamba bundles, and no `gym` submodule.
- [ ] The Whisper repo has no runtime imports of `exogym` or `gym.exogym`.
- [ ] `whisper-tuner` still supports the core Whisper product flows after the migration.
- [ ] The new Mamba repo is independently usable and linked from Whisper docs.
- [ ] The cloud streaming architecture remains available where needed: retained in Whisper and duplicated in Mamba only if Mamba still requires it.
- [ ] Generated artifacts and build outputs are no longer tracked in the Whisper repo.

### Definition of Done

- [ ] All Whisper-focused tests passing
- [ ] Documentation updated for new repo boundaries
- [ ] CI re-enabled for the surviving Whisper surface
- [ ] New Mamba repository created and referenced from this repo

## Open Questions

### Resolved

- **Q:** Should this repository stay Whisper-focused?
- **A:** Yes. Whisper is the product boundary for this repo.

- **Q:** Should `Mamba-ASR-MPS/` stay here?
- **A:** No. It becomes a standalone repository.

- **Q:** Should ExoGym remain in this repo?
- **A:** No. Delete the submodule and the features that depend on it.

- **Q:** Should ExoGym-backed distributed commands be removed immediately or replaced with a smaller native Whisper-only implementation?
- **A:** remove now and reintroduce later.

- **Q:** Which extraction method should be the default for the new Mamba repo?
- **A:** Use `git subtree split` first. It is simpler and sufficient for this boundary split.

- **Q:** Should this plan require an immediate `src/` layout migration?
- **A:** No. Consolidate under `whisper_tuner/` first and defer `src/` to a later decision if still useful.

- **Q:** Does Gemma support still belong in this repo after the Whisper-only boundary is enforced?
- **A:** Yes.

- **Q:** Should the cloud streaming architecture survive the split?
- **A:** Yes. Keep it in Whisper, and duplicate it into the Mamba repo only if Mamba still needs cloud-resident data with local compute.

### Unresolved

- **Q:** Should the package consolidation under `whisper_tuner/` happen in the same PR series as the repo split, or only after Mamba extraction and ExoGym removal land?
- **A:** after Mamba extraction and ExoGym removal

## References

### Internal

- [Root Packaging](../../pyproject.toml)
- [Main README](../../README.md)
- [BigQuery Utilities](../../core/bigquery.py)
- [ExoGym Submodule](../../.gitmodules)
- [Primary CLI](../../cli_typer.py)
- [Legacy CLI](../../main.py)
- [Audio Loading](../../utils/dataset_prep.py)
- [Streaming Dataset Utils](../../utils/dataset_utils.py)
- [Data Preparation](../../scripts/prepare_data.py)
- [Disabled CI Workflow](../../.github/workflows/ci.yml)
- [Distributed Launcher](../../distributed/launcher.py)
- [Wizard Runner](../../wizard/runner.py)
- [Distributed Spec](../../README/specifications/distributed-training-gym.md)

### External

- [git subtree documentation](https://git-scm.com/docs/git-subtree)
- [git submodule documentation](https://git-scm.com/docs/git-submodule)
- [Git LFS](https://git-lfs.com/)
