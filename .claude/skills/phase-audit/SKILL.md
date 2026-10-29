---
name: phase-audit
description: Audit a completed plan phase in this repo. Use when a phase has just been implemented and needs a findings-first review against the active plan, changed files, linked guides, and the canonical audit rubric before commit or before moving to the next phase. Do not use for implementing fixes, broad product critique, or replacing tests and CI execution.
---

# Phase Audit

## Purpose

Use this skill to audit a completed implementation phase like a skeptical
senior engineer doing code review. The job is to decide whether the phase is actually complete and ready to commit or push, not to help implement it.

## Use When

- A plan phase was just implemented.
- The current plan checkboxes need validation against reality.
- The repo needs a findings-first review before commit or before moving to the
  next phase.

## Do Not Use When

- The request is to implement or patch code.
- The request is broad product or UX feedback.
- The request is to replace tests, CI, or runtime verification with a code
  review.

## Procedure

1. Read [references/index.md](references/index.md) first.
2. If the phase changes architecture, reliability boundaries, or complexity budgets, run `elon-musk` first and keep only the surviving scope; note deletions or rejected requirements in the audit output.
3. Read the active plan and isolate the exact phase being audited:
   - phase goal
   - checked tasks
   - phase verification text
   - linked canonical references in [`README.md`](../../../README.md), [`README/`](../../../README/), [`README/guides/`](../../../README/guides/), [`README/skills/`](../../../README/skills/), and [`README/notes/`](../../../README/notes/) when those paths exist
3. Inspect the implementation evidence:
   - changed files and diffs
   - tests or checks that were run
   - any notes that explain scope decisions
4. Compare the implementation against the canonical rubric and linked guides.
5. Produce a findings-first audit:
   - severity ordered
   - concrete file references when applicable
   - explicit call on whether the phase is complete
6. Explicitly check:
   - missing scope
   - missing edge cases
   - missing tests or weak verification
   - checkbox drift between the plan and the code
   - commit readiness
   - push and CI readiness
7. If there are no findings, say so directly and note residual risks or
   verification gaps.

## Output Contract

- Findings first.
- Severity ordered.
- Concrete file references when applicable.
- Brief summary only after the findings.

## Boundaries

- Do not edit files.
- Do not silently fix the problem instead of reporting it.
- Do not replace tests, CI, or runtime verification with review prose.

## Canonical Docs

Read [references/index.md](references/index.md) first. It maps the workflow
guide, the audit rubric, and the repo artifacts that define whether a phase is
actually done.

## Handoff Rules

- Hand findings back to `execute-plan` or the user for remediation.
- Re-run this skill after fixes if the phase changed materially.
