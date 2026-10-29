---
name: execute-plan
description: Execute a checked-in implementation plan for this repo phase by phase. Commits land once per completed phase locally; push and CI monitoring run once after every phase is done (not after each phase). Use when the user provides a concrete plan path and wants end-to-end implementation, phase audits, plan updates, commits, push, and CI monitoring. Do not use for creating a plan, one-off fixes without a checked-in plan, or read-only review work.
---

# Execute Plan

## Purpose

Use this skill to execute an existing repo plan without collapsing the work into
one giant unreviewed change. The unit of progress is one completed phase at a
time.

## Use When

- A concrete checked-in plan path exists (commonly `docs/plans/...` or a team-agreed path; `README/plans/` is gitignored here for local-only drafts).
- The plan is implementation-ready.
- The user wants the full implementation loop, not just discussion.

## Do Not Use When

- No checked-in plan exists yet.
- The request is a one-off fix without a real plan.
- The request is planning, brainstorming, or read-only review.

## Authority Model

- Explicit invocation of `$execute-plan` counts as authorization for this
  workflow's git side effects. Direct naming also counts, for example
  "use execute-plan":
  - **During phases (step 4):** narrow **phase commits** on the current branch
    only. You may **`git fetch`** and merge/rebase **locally** to integrate
    `origin/main` when needed—**do not** **`git push`** to `origin` and **do
    not** run the full **`git-push`** skill until **all** phases are complete.
  - **After all phases (step 5):** one **`git-push`** pass: integrate upstream,
    push the branch, monitor GitHub Actions, and fix failures until green.
- If `execute-plan` was routed implicitly or inferred rather than invoked
  directly, it may prepare local implementation work but must stop before the
  first commit or push and explain why.
- At the start of the workflow, state that `execute-plan` is running: **phase
  commits first, then a single push/CI tail**—unless the user opts out of
  remote writes.

## Procedure

1. Read [references/index.md](references/index.md) first.
2. Read the target plan and linked docs ([`README.md`](../../../README.md), [`README/guides/`](../../../README/guides/), [`README/skills/`](../../../README/skills/), and [`README/notes/`](../../../README/notes/) if present) before changing code.
3. Refuse to start if the plan is missing, too vague to execute honestly, or
   still requires planning work.
4. Execute one phase at a time:
   - 4a. Execute the current phase. Implement only the active phase scope.
     Make it perfect. No god modules. No bugs. Just elegant, modular code.
   - 4b. Audit the execution with `phase-audit`. Prefer delegated review when
     the runtime supports it cleanly; otherwise run the same checklist locally
     against the canonical rubric. Double-check the work and fix findings
     before moving on. Make it perfect.
   - 4c. Update the plan. Make the phase checkboxes match reality before
     calling the phase complete.
   - 4d. Commit only your changes. Stage **only** the files for the completed
     phase (narrower than the default `git-commit` whole-tree staging) and
     follow the `git-commit` skill for the commit **message** (what and why).
     **Stop after the commit—do not push.** Per-phase push is an anti-pattern
     here.
5. After all phases are complete:
   - follow `git-push` for syncing with `origin`, pushing the branch, watching
     GitHub Actions, and fixing failures until green (phase commits stay
     narrower than default `git-commit`; `git-push` still applies merge/push/CI
     loop).

## Worktree Rules

- Ignore unrelated dirty files unless they directly conflict with the active
  phase.
- Never revert unrelated user work.
- Stop only when an existing change creates a real safety or correctness
  conflict with the current phase.

## Audit Contract

Before each phase commit, explicitly verify:

- the phase goal is actually complete
- the plan checkboxes match the implementation
- obvious edge cases are handled
- tests or checks are appropriate for the change
- the current phase is commit-ready

## Boundaries

- Do not create a new plan inside this workflow.
- Do not silently skip the audit step.
- **Do not push to `origin` between phase commits.** Reserve **`git push`**
  and the **`git-push`** skill for step 5 (after every phase is implemented and
  committed), unless the user explicitly asks for a different cadence.
- Do not push with known failing CI unless the user explicitly overrides that
  rule.

## Canonical Docs

Read [references/index.md](references/index.md) first. It maps the shared
workflow guide, the phase audit rubric, and the artifacts needed to execute a
plan cleanly.

## Handoff Rules

- Hand off to `phase-audit` when clean delegated review is available.
- Hand off to `create-plan` only if the user actually needs a new plan instead
  of execution.
