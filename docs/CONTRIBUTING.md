# Contributing

Thank you for contributing! This repo uses:

- Typer CLI as the primary interface
- Ruff for lint/format
- Pytest for tests (fast tests only in CI)

## Setup

```
pip install -e '.[dev]'
pre-commit install
```

## Development

- Run fast tests:
```
pytest -q tests -k 'not slow'
```
- Lint/format:
```
ruff check . && ruff format --check .
```

## Plan documents

`README/plans/` is **gitignored** by default (local implementation plans). To commit an updated plan, use `git add -f README/plans/<file>.md`.

## Image fine-tuning release checks

Caption/VQA training on a real gated multimodal checkpoint is covered by `tests/test_smoke_image_multimodal.py` (requires Hub auth). Locally:

```bash
HF_TOKEN=... pytest tests/test_smoke_image_multimodal.py -m integration -v
```

Repository maintainers can add a **`HF_TOKEN`** (read) [secret](https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions) and run the **Gated multimodal image smoke** workflow manually (`workflow_dispatch`). If the secret is missing, those tests **skip** and the workflow still passes.

## PRs

- Include a short description and screenshots/logs for UX changes
- Keep edits focused; avoid unrelated formatting churn
- Green CI required
