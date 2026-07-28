---
name: execute-plan
description: Execute an approved Gemma Tuner implementation or experiment plan phase by phase with frozen gates and recorded evidence.
---

# Execute plan

Read the full plan, preserve unrelated changes, implement one phase, run Ruff
and focused Pytest plus the relevant wizard/visualizer/package smoke, then
record evidence. Stop on failed data, privacy, correctness, or evaluation gates.
Do not commit or push without explicit authorization.
