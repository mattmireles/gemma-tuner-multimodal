---
name: debug
description: Debug Gemma Multimodal Fine-Tuner failures across Python environments, Hugging Face models, PEFT/LoRA, datasets, audio/image processors, MPS, wizard subprocesses, visualizer state, evaluation, or export.
---

# Debug

Reproduce the narrowest failing layer, record the exact profile/model/dataset
and environment, distinguish configuration from model/data/device failure, and
add a regression before changing behavior when possible. Use `uv run pytest`
and `uv run ruff` with the smallest relevant target before the full suite.
