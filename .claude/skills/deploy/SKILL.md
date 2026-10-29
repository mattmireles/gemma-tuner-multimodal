---
name: deploy
description: Prepare and ship a **gemma-macos-tuner** release (PyPI) or verify release readiness. Before any publish, run the same blocking gates as CI (Ruff, fast tests, pre-commit when applicable); fix failures before tagging. Use when the user asks to publish to PyPI, cut a release, verify a version tag against pyproject.toml, or confirm release automation. Do not use for ad-hoc training runs or local-only work unless release hygiene is explicitly the goal.
---

# Deploy / release (gemma-tuner-multimodal)

This skill targets **this repository**: **Python** package **`gemma-macos-tuner`** ([`pyproject.toml`](../../../pyproject.toml)), **macOS-first** CI ([`.github/workflows/ci.yml`](../../../.github/workflows/ci.yml)), and **PyPI publish** on version tags ([`.github/workflows/publish-pypi.yml`](../../../.github/workflows/publish-pypi.yml)). There is **no Cloudflare Worker** or `wrangler` surface in this repo.

## What gets shipped

| Surface | How it ships |
| --- | --- |
| **PyPI package** | Tag `v<version>` where `<version>` matches `project.version` in `pyproject.toml`; workflow builds sdist/wheel and runs `twine upload` (needs `PYPI_TOKEN` in repo secrets). |

Canonical version truth: **`pyproject.toml`** `project.version`.

## Pre-publish gate (blocking)

Do **not** tag or advise a release until every applicable step succeeds (exit 0):

1. **Install dev deps:** `pip install -e '.[dev]'` (and CPU `torch` for tests if needed, matching CI—see [`.github/workflows/ci.yml`](../../../.github/workflows/ci.yml)).
2. **`ruff check .`**
3. **`ruff format --check .`**
4. **`pre-commit run --all-files`** (when verifying full CI parity locally; optional if Ruff alone was requested).
5. **Fast tests:** `pytest -q -m "not slow"` (or the subset CI runs—see workflow for explicit test list).

**On failure:** stop, fix root cause, re-run from step 2.

## Procedure (agent)

1. Read [`README.md`](../../../README.md) and [`.github/workflows/publish-pypi.yml`](../../../.github/workflows/publish-pypi.yml) for tag ↔ version rules and secrets.
2. Run [Pre-publish gate (blocking)](#pre-publish-gate-blocking); do not continue until green.
3. **Tag discipline:** `publish-pypi` requires **`v$VERSION`** where `$VERSION` equals `project.version` in `pyproject.toml` (e.g. tag `v1.1.0` for version `1.1.0`).
4. **Credentials:** `PYPI_TOKEN` must be configured on GitHub; do not commit tokens. User may run `python -m build` and `twine check dist/*` locally without uploading.
5. After a successful publish (or dry-run build), note the artifact paths under `dist/` if relevant.

## CI parity

`.github/workflows/ci.yml` runs **Ruff**, **pre-commit**, and **fast pytest** on **macOS arm64**. Manual release prep should honor the same checks.

## Anti-patterns

- Tagging when Ruff or tests are red.
- Bumping `pyproject.toml` version without updating the tag format expected by CI.
- Lowering coverage or lint thresholds to “make release green.”

## Related docs

- [`README.md`](../../../README.md)
- [`docs/CONTRIBUTING.md`](../../../docs/CONTRIBUTING.md)
- [`README/notes/debug-notes.md`](../../../README/notes/debug-notes.md)
