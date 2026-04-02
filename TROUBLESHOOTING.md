# Troubleshooting Guide

This guide covers common issues for Apple Silicon (MPS).

## Apple Silicon (MPS) Pitfalls

- Symptom: `MPS not available` or `PyTorch not compiled with MPS`
  - Check Python architecture: `python -c "import platform; print(platform.platform())"` must show `arm64`.
  - Reinstall PyTorch for Apple Silicon. Avoid Rosetta/x86_64 environments.

- Symptom: Training is extremely slow or hangs
  - Disable debugging fallback in production: unset `PYTORCH_ENABLE_MPS_FALLBACK`.
  - Set memory watermark to avoid swapping: `export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7`.

- Symptom: Random `NotImplementedError` during ops
  - Temporarily set `PYTORCH_ENABLE_MPS_FALLBACK=1` while identifying the op.
  - Prefer `dtype=float32` and `attn_implementation='eager'` on MPS.

- Symptom: Frequent out-of-memory or system lag
  - Reduce batch size; enable gradient accumulation.
  - Use attention slicing or gradient checkpointing (note: checkpointing can be unstable on MPS for some models).

## Migrating to the Typer CLI

- Legacy scripts `main.py` and `manage.py` are deprecated.
- Use modern commands:
  ```bash
  whisper-tuner finetune <profile>
  whisper-tuner evaluate <profile | model+dataset>
  whisper-tuner runs list
  ```
