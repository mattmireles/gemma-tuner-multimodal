---
name: research-debugging
description: Investigate external bugs, vendor limits, SDK mismatches, and workarounds, then save a durable debug guide under README/. Use when a failure depends on web research outside the repo. Do not use for purely local bugs, generic implementation work, or broad planning research not tied to a concrete failure.
---

# Research Debugging

## Purpose

**Adapted from Gist.** Use when debugging requires **external** confirmation (vendor issues, official documentation, trackers) and the result should live under **`README/`** for reuse.

## Use When

- The failure depends on vendor behavior, quotas, or API shape.
- The research should be saved for the next incident.

## Do Not Use When

- The bug is fully explained by local code and tests.
- The task is implementation without an external mystery.

## Procedure

1. Read [references/index.md](references/index.md) first.
2. Capture: symptom, minimal repro, what was ruled out locally.
3. Use **Context7** and official docs; add targeted web research when needed.
4. Write or update `README/<topic>-debug.md` (or append to [`README/notes/debug-notes.md`](../../../README/notes/debug-notes.md) for short items).
5. Hand off to code changes or **`research-planning`** when the root cause is clear and work should continue outside note-taking.

## Handoff Rules

- **`research-planning`** for broad technology surveys without a single failure.
- **`debug`** (repo debug skill) for systematic local investigation first.
