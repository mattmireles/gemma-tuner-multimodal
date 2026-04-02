# Plan Workflow Skills Guide

Canonical workflow guide for the repo's plan-oriented skills:
`create-plan`, `execute-plan`, and `phase-audit`.

## Purpose

These skills exist to turn a repeated manual pattern into a stable, narrow
workflow:

- create a real plan from repo knowledge
- execute an approved plan one phase at a time
- audit each completed phase before moving on

This guide is the source of truth for that workflow. The skills should wrap
this guide, not invent a parallel process.

## Shared Rules

- All three workflow skills are injected by default so they stay visible in
  Codex and available for normal routing.
- Keep each skill narrow:
  - `create-plan` writes plans
  - `execute-plan` executes plans
  - `phase-audit` reviews completed phases
- Canonical knowledge stays in `README/...`.
- Stable workflow guidance lives here and in the rubric, not only in prompts.
- If a runtime cannot support delegated review cleanly, the workflow must still
  work with a local audit fallback.
- `create-plan` should prefer one fresh Codex review and one fresh Claude Code
  review on the saved draft when both local CLIs are available, but that
  cross-agent pass is optional rather than a hard dependency.
- `execute-plan` should prefer one fresh Codex phase audit and one fresh
  Claude Code phase audit on the current uncommitted phase before the phase is
  marked complete or committed, but that cross-agent pass is optional rather
  than a hard dependency.

## Side-Effect Classes

- `create-plan`: repo-write, no git side effects beyond normal file edits
- `phase-audit`: read-only review
- `execute-plan`: git-write workflow that may commit, sync, push, and monitor
  CI

## Authority Rule

- Explicit invocation of a workflow skill authorizes the side effects
  documented for that skill. Direct naming counts, for example `$execute-plan`
  or "use execute-plan".
- Implicit routing does not authorize git writes. If `execute-plan` was not
  invoked explicitly, it may prepare local changes but must stop before commit
  or push and say why.

## Skill: `create-plan`

### `create-plan` Job

Turn a concrete request into a repo-native implementation plan using
[Plans Template](../../Templates/Plans-template.md).

### `create-plan` Use When

- The work is large enough to need a real implementation plan.
- The repo already contains enough context to scope the work.
- The output should be a checked-in plan under `README/plans/...`.

### `create-plan` Do Not Use When

- The user wants immediate implementation instead of a plan.
- The request is still brainstorming and the scope is not concrete yet.
- The output should be a note, not a plan.

### Research Order

Always gather local context before external research:

1. Read the directly relevant plan, guide, and note files in `README/...`.
2. Read adjacent `README/guides` and `README/notes` material that constrains the
   implementation.
3. Use Context7 only when the plan depends on current library, framework, or
   API behavior that may have changed since model training.
4. If Context7 is insufficient, use official vendor docs.

### Output Contract

- Use [Plans Template](../../Templates/Plans-template.md).
- Make the plan specific enough to implement without inventing missing policy.
- Include clear phases, verification, hard requirements, and rollback.
- Name concrete files whenever the implementation path is already knowable.
- Save the draft to the target `README/plans/...` file before asking external
  reviewers to inspect it.

### Cross-Agent Review Loop

After the first draft exists on disk:

1. Invoke one fresh Codex CLI review thread on the draft.
2. Invoke one fresh Claude Code CLI review thread on the same draft.
3. Integrate concrete findings that improve implementation readiness.
4. Prefer canonical repo docs when a reviewer suggestion conflicts with local
   ground truth.
5. Move unresolved disagreements into `Open Questions` instead of inventing
   certainty.
6. Re-run the local audit before stopping.

Preferred artifact helper:

- `scripts/run-cross-agent-plan-review.sh`
- The helper shells out to `codex exec` and `claude -p` so the reviews happen
  in separate fresh CLI threads, not inside the original planning thread.

Fallback rule:

- If one or both reviewer CLIs are unavailable or fail, `create-plan` must
  still perform the same planning audit locally and note the missing reviewer
  path in the final handoff.

## Skill: `execute-plan`

### `execute-plan` Job

Execute an existing checked-in plan end-to-end without collapsing the work into
one giant unreviewed change.

### `execute-plan` Use When

- The user gives a concrete plan path.
- The plan is implementation-ready.
- The user wants the full execution loop, not just discussion.

### `execute-plan` Do Not Use When

- No checked-in plan exists yet.
- The work is a one-off bugfix without a real plan.
- The request is read-only review, analysis, or planning.

### Required Loop

For each phase:

1. Read the phase and the linked canonical guides.
2. Implement only that phase's scope.
3. Audit the phase:
   - invoke one fresh Codex CLI audit thread
   - invoke one fresh Claude Code CLI audit thread
   - integrate concrete findings before phase completion
   - otherwise run the same rubric locally
4. Fix audit findings before proceeding.
5. Update the plan checkboxes to reflect reality.
6. Commit the phase with a comprehensive commit message.

After all phases:

1. Sync with `origin/main`.
2. Push the branch.
3. Monitor GitHub Actions until the run is complete.
4. If CI fails, fix only the regression and repeat until green.

### Worktree Rule

- Ignore unrelated dirty files unless they directly conflict with the active
  phase.
- Never revert unrelated user work.

### Cross-Agent Phase Audit Loop

After the implementation for the current phase exists in the working tree and
before the phase is marked complete or committed:

1. Invoke one fresh Codex CLI audit thread on the current phase.
2. Invoke one fresh Claude Code CLI audit thread on the same phase.
3. Integrate concrete findings into the implementation.
4. Re-run the local rubric before marking the phase complete.
5. Update the plan to reflect the post-audit reality.

Preferred artifact helper:

- `scripts/run-cross-agent-phase-audit.sh <plan-path> <phase-selector>`
- The helper shells out to `codex exec` and `claude -p` so the audits happen
  in separate fresh CLI threads, not inside the original implementation
  thread.

Fallback rule:

- If one or both reviewer CLIs are unavailable, fail, or time out,
  `execute-plan` must still run the same rubric locally, fix findings, and
  note the missing reviewer path in the phase handoff before commit.

## Skill: `phase-audit`

### `phase-audit` Job

Review a completed phase like a skeptical senior reviewer before the next phase
or before push.

### `phase-audit` Use When

- A phase was just implemented.
- The plan checkboxes need validation.
- The repo needs a findings-first audit before commit or push.

### `phase-audit` Do Not Use When

- The request is to implement code.
- The request is broad product critique.
- The request is general CI monitoring rather than a phase review.

### Review Style

- Findings first
- Severity ordered
- Concrete file references
- Focus on:
  - missing scope
  - missing edge cases
  - missing tests
  - checkbox drift
  - push and CI readiness

Read [phase-audit-rubric.md](./phase-audit-rubric.md) before auditing.

## Delegated Audit Fallback

If the runtime supports delegated or forked review cleanly, `execute-plan` can
invoke `phase-audit` as a separate reviewer. If not, `execute-plan` must still
perform the same rubric locally and continue.

This is a portability rule, not an optimization.

## Invocation Policy

- Default injection is enabled for all three workflow skills.
- Prefer explicit `$create-plan`, `$execute-plan`, or `$phase-audit` when the
  user is asking for that exact workflow or when you want to force a precise
  handoff.
- Keep the descriptions and boundaries tight so default routing does not drift
  into generic implementation or generic review work.
