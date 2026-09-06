---
name: write-notes
description: Record Gemma Tuner investigations, experiment evidence, failures, and operational learnings in the repo's existing notes structure.
---

# Write notes

Record date, question, environment, model/data revisions, commands, artifacts,
observations, confirmed facts, inferences, decision, and next gate. Preserve
negative results and exclude private examples or secrets.

Never write a plan execution log here. Phase progress belongs only in the
plan's task checkboxes; the plan header states only its overall lifecycle
(`Planned`, `In-Progress`, or `Complete`). Routine test and review output is
transient; Git and CI are the evidence. Create a separate artifact only for an
important external fact that cannot be reproduced from the commit. Store it
under `README/Notes/receipts/plan-NNN/`.
