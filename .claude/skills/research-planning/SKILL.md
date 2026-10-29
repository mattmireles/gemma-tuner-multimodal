---
name: research-planning
description: Use Gemini (or similar) deep research to create or refresh external field guides under README/ before planning or implementation. Use when the user wants exhaustive web research on a technology, API, framework, platform, or domain with practical examples and failure modes. Do not use for repo-only architecture work, local bug reproduction, or direct implementation without external research.
---

# Research Planning

## Purpose

**Adapted from Gist.** Use when **gemma-macos-tuner** needs external research before planning or implementation. The durable artifact is usually a markdown file under **`README/`** (often `README/guides/` for long guides), not a chat-only summary.

## Use When

- The user wants a new or refreshed guide doc.
- The work depends on current vendor, SDK, API, or platform behavior.
- The guide must answer what the thing is, how it works in practice, what breaks, and how to choose a path.

## Do Not Use When

- The task is concrete implementation with enough local context.
- The task is a purely local bug with a live repro.
- The output should be temporary scratch instead of durable research.

## Procedure

1. Read [references/index.md](references/index.md) first.
2. Lock scope: topic, versions, target path under `README/`, and what decision the guide unblocks.
3. Prefer updating an existing doc over creating a near-duplicate.
4. Run research using whichever tooling exists:
   - If the repo adds a helper script (compare gist: `scripts/gemini-deep-research.js`), use it and the paths it expects.
   - Otherwise: structured web research + **Context7** for library facts + synthesis into `README/guides/<path>/<slug>.md` or another agreed location.
5. Review and edit the draft: strip filler, verify versions and links, separate vendor facts from repo-specific integration notes.
6. Add cross-links from [`README.md`](../../../README.md) or related docs when appropriate.
7. Hand off to **`create-plan`** when the guide is clean enough to scope work.

## Prompt contract

Use the section order and constraints from the **gist** copy of this skill if you need a long-form template (audience: senior engineers; exhaustive; code examples; Works Cited). Replace “Gist repo” with **gemma-tuner-multimodal** and target paths with `README/...`.

## Handoff Rules

- **`research-debugging`** when the work narrows to a concrete external failure hunt.
- **`create-plan`** when the guide unblocks implementation scoping.
