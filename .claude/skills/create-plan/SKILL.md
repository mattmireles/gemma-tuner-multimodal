---
name: create-plan
description: Create an implementation plan for this repo. Use when the user wants a checked-in plan (commonly under `docs/plans/`), wants the work scoped into phases, and expects the plan to be built from the README tree and repo README first, with Context7 only when current external library or framework behavior materially affects the plan. Do not use for implementing the work, informal brainstorming, or lightweight notes.
---

# Create Plan

## Purpose

Use this skill to turn a concrete request into a repo-native implementation
plan. The output is a **real plan file checked into git** (typically under
`docs/plans/...`—follow existing layout if present). **`README/plans/`** is
gitignored in this repo for local-only drafts; use a **tracked** path unless the
user explicitly wants a private scratch plan.

## Use When

- The work is large enough to need a real implementation plan.
- The user wants a plan written into the repo.
- The implementation needs phases, verification steps, and concrete files.

## Do Not Use When

- The user wants direct implementation instead of planning.
- The request is still too vague to scope honestly.
- The output should be a note, scratchpad, or brainstorm instead of a plan.

## Procedure

1. Read [references/index.md](references/index.md) first.
2. Gather repo context in this order:
   - [`README.md`](../../../README.md), [`README/guides/README.md`](../../../README/guides/README.md), and [`docs/CONTRIBUTING.md`](../../../docs/CONTRIBUTING.md)
   - [`README/notes/debug-notes.md`](../../../README/notes/debug-notes.md) when the work ties to past incidents
   - optional [`README/skills/`](../../../README/skills/) when the work interacts with agent/plan conventions
   - neighboring plans in the target plan subtree for local style
3. Use Context7 only when the plan depends on current library, framework, or API behavior that may have changed.
4. If Context7 is insufficient, fall back to official vendor docs.
5. Choose the most specific existing subdirectory under the plan root that fits the work. If none exists yet, create a minimal sensible path (avoid noisy folder sprawl).
6. Write the first draft to the real target plan file with a clear structure and make it implementation ready:
   - concrete phases
   - specific files where knowable
   - verification per phase
   - hard requirements
   - rollback or kill switch when relevant
7. **Optional external review:** If `scripts/run-cross-agent-plan-review.sh` (or a similar checked-in helper) exists, run it on the saved draft, integrate findings, and move unresolved disagreements into `Open Questions`. If no such script exists, skip this step and do a careful self-audit instead.
8. Audit the final plan before stopping:
   - no missing policy that an implementer would have to invent
   - no fake certainty where the repo context is incomplete
   - no implementation work performed while planning
   - no unresolved reviewer finding silently ignored without explanation

## Canonical Docs

Read [references/index.md](references/index.md) first.

## Handoff Rules

- Hand off to `execute-plan` only after the plan is checked in and the user wants implementation.
- Hand off to a domain skill only if the planning request narrows into a domain-specific technical question that must be answered before the plan can be completed.
