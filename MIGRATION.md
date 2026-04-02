# Migration Guide: Legacy Scripts → Typer CLI

This guide maps common legacy commands to the modern `whisper-tuner` CLI.

- Train (profile)
  - Legacy: `python main.py finetune <profile> --config config.ini`
  - New: `whisper-tuner finetune <profile> --config config.ini`

- Evaluate (profile or model+dataset)
  - Legacy: `python main.py evaluate <profile|model+dataset> --config config.ini`
  - New: `whisper-tuner evaluate <profile|model+dataset> --config config.ini`

- Blacklist generation
  - Legacy: `python main.py blacklist <profile> --config config.ini`
  - New: `whisper-tuner blacklist <profile> --config config.ini`

- Runs management (was manage.py)
  - Legacy: `python manage.py list|overview|details|cleanup`
  - New: `whisper-tuner runs list|overview|details|cleanup`

Notes:
- `main.py` and `manage.py` are compatibility shims. Temporary wrappers are available:
  - `whisper-tuner legacy main`
  - `whisper-tuner legacy manage`
- Configuration remains in `config.ini`. Toggles like `gradient_checkpointing` and
  `attn_implementation` can be set per-profile. On Apple Silicon, MPS defaults will
  enforce `dtype=float32` and `attn_implementation=eager` for stability.
