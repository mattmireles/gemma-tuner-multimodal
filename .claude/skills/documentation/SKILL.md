---
name: documentation
description: Write or review inline code documentation that captures domain knowledge, non-obvious constraints, non-greppable cross-file contracts, and state lifecycle. Use when the task is adding or reviewing docstrings, module headers, constant rationale, or training/data invariants. Do not use for README files, markdown guides, plan documents, or code changes where comments are incidental.
---

# Documentation

## Purpose

Enforce the repo's inline code documentation standards. Document what a future
editor cannot safely derive from the code itself.

## Use When

- Adding or reviewing **module/class/function docstrings** or short file headers.
- Documenting **training state**, **checkpoint**, or **config** invariants.
- A review flags missing or low-quality documentation.

## Do Not Use When

- Writing README, plan, or notes documents.
- General code changes where documentation isn't the focus.
- Markdown formatting issues (use **`markdown`**).

## Procedure

1. Read [references/index.md](references/index.md) first.
2. Inspect the target file and the smallest set of related files needed to
   confirm what context is truly missing.
3. Add or tighten docs only where they capture:
   - domain knowledge
   - non-obvious constraints
   - non-greppable cross-file contracts
   - state lifecycle or constant rationale
4. Prefer short, durable comments over boilerplate:
   - short file headers when a whole module has one responsibility story
   - docstrings that explain **why** or constraints—not a rephrasing of the signature
   - notes on MPS vs CUDA assumptions when non-obvious
5. Do not add manual call graphs, line-by-line prose, or comments that are more
   likely to drift than to help.
6. If the missing context belongs in a canonical guide, update the
   guide under **`README/guides/`** instead of burying the whole explanation in code comments.

## References

Read [references/index.md](references/index.md) first.

## Handoff Rules

- Hand off to **`debug`** if the problem is a runtime defect, not missing docs.
- Hand off to **`markdown`** for prose-heavy documentation files.
