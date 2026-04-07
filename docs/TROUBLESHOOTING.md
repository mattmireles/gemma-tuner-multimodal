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

## Image fine-tuning (vision token budget)

- **High `image_token_budget` (e.g. 1120)** on a 32 GB Mac: prefer `per_device_train_batch_size = 1` and scale with `gradient_accumulation_steps`. Throughput is often **much** lower than text-only training because of the vision encoder on MPS.
- **Train/serve mismatch:** if inference uses a different `image_token_budget` than training, quality can drop silently. Match the profile value; export reapplies the budget from run `metadata.json` when present.

## Migrating to the Typer CLI

- Legacy scripts `main.py` and `manage.py` are deprecated.
- Use modern commands:
  ```bash
  gemma-macos-tuner finetune <profile>
  gemma-macos-tuner evaluate <profile | model+dataset>
  gemma-macos-tuner runs list
  ```
