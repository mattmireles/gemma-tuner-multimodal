---
name: audit
description: Audit Gemma Multimodal Fine-Tuner for training correctness, dataset leakage, MPS/device behavior, checkpoint/export integrity, evaluation validity, privacy, and CLI/wizard/visualizer regressions.
---

# Audit Gemma Tuner

Trace configuration through dataset preparation, collation, training,
evaluation, export, and recorded metrics. Prioritize leakage, unpinned model
revisions, incorrect modality routing, silent CPU fallback, invalid loss or
metrics, unsafe artifact writes, and tests that do not exercise the real path.
Use focused evidence and recommend the smallest proper fix.
