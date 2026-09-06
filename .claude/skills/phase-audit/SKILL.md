---
name: phase-audit
description: Audit a Gemma Tuner plan phase against its acceptance criteria, tests, artifacts, leakage controls, hardware claims, and stop conditions.
---

# Phase audit

Require direct evidence for every checked item. Inspect representative,
adversarial, and failure cases; verify split and revision hashes; distinguish
MPS execution from silent fallback; and reject conclusions that exceed the
frozen metrics.

Return the audit to the caller or chat. Do not create a routine evidence
artifact, append an execution summary to the plan, or write audit output
under `README/`.
