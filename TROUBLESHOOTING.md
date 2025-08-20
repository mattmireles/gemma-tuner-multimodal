# Troubleshooting Guide

This guide covers common issues for Apple Silicon (MPS) and distributed training.

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

## Passwordless localhost SSH (single-machine distributed)

1. Generate a key if you don't have one:
   ```bash
   ssh-keygen -t ed25519 -C "local" -f ~/.ssh/id_ed25519 -N ''
   ```
2. Authorize it for localhost:
   ```bash
   mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys
   cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
   chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys
   ssh-keyscan -H 127.0.0.1 >> ~/.ssh/known_hosts
   ssh-keyscan -H localhost >> ~/.ssh/known_hosts
   ```
3. Minimal `distributed_hosts.json`:
   ```json
   {
     "master": "127.0.0.1",
     "workers": ["127.0.0.1"],
     "ssh_user": "$(whoami)",
     "python_env": "$(which python)",
     "project_path": "$PWD"
   }
   ```
4. Validate:
   ```bash
   whisper-tuner distributed-check --hosts-config distributed_hosts.json --verbose
   ```

## Migrating to the Typer CLI

- Legacy scripts `main.py` and `manage.py` are deprecated.
- Use modern commands:
  ```bash
  whisper-tuner finetune <profile>
  whisper-tuner evaluate <profile | model+dataset>
  whisper-tuner runs list
  ```
