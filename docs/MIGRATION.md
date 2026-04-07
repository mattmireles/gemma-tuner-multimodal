# Migration Guide: Legacy Scripts → Typer CLI

This guide maps common legacy commands to the modern `gemma-macos-tuner` CLI.

- Train (profile)
  - Legacy: `python entrypoints/main.py finetune <profile> --config config/config.ini` (or `python -m gemma_tuner.main …`)
  - New: `gemma-macos-tuner finetune <profile> --config config/config.ini`

- Evaluate (profile or model+dataset)
  - Legacy: `python entrypoints/main.py evaluate <profile|model+dataset> --config config/config.ini`
  - New: `gemma-macos-tuner evaluate <profile|model+dataset> --config config/config.ini`

- Blacklist generation
  - Legacy: `python entrypoints/main.py blacklist <profile> --config config/config.ini`
  - New: `gemma-macos-tuner blacklist <profile> --config config/config.ini`

- Runs management (was manage.py)
  - Legacy: `python manage.py list|overview|details|cleanup`
  - New: `gemma-macos-tuner runs list|overview|details|cleanup`

Notes:
- `entrypoints/main.py` and `gemma_tuner/main.py` are compatibility shims. Temporary wrappers are available:
  - `gemma-macos-tuner legacy main`
  - `gemma-macos-tuner legacy manage`
- Configuration lives at **`config/config.ini`** by default (a `config.ini` at the repo root is still supported). Toggles like `gradient_checkpointing` and
  `attn_implementation` can be set per-profile. On Apple Silicon, MPS defaults will
  enforce `dtype=float32` and `attn_implementation=eager` for stability.
