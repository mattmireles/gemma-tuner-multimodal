---
name: execute-plan
description: Execute an approved Gemma Tuner implementation or experiment plan phase by phase with frozen gates and recorded evidence.
---

# Execute plan

Read the full plan, preserve unrelated changes, implement one phase, run Ruff
and focused Pytest plus the relevant wizard/visualizer/package smoke, then
record evidence. Stop on failed data, privacy, correctness, or evaluation gates.
Do not commit or push without explicit authorization.

## Progress and Evidence Contract

- The task checkboxes under each phase are the only progress tracker.
- Each checkbox owns one independently completable fact. Do not add roll-up
  boxes derived from child boxes or mirror work owned by another phase.
- A checked box means its stated verification passed. Git history and CI are
  the normal evidence; do not save routine test, typecheck, full-check, or
  review output under `README/`.
- Create a separate evidence artifact only for an important fact that cannot be
  reproduced from the commit and CI. Store it under
  `README/Notes/receipts/plan-NNN/`.
- Keep an exceptional evidence artifact compact. It must contain no secrets,
  credentials, customer or job input/output content, or personal data.
- Put durable root causes, platform traps, and reusable engineering lessons in
  the appropriate notes file. Notes never carry phase status.
- Rewrite stale plan text instead of preserving an execution history.

