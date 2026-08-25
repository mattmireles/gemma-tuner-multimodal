---
name: markdown
description: Edit Gemma Tuner Markdown while preserving links, runnable commands, tables, heading structure, and honest model/data/hardware claims.
---

# Markdown

Keep one H1, ordered headings, fenced languages, valid relative links, blank
lines around lists/tables, and a final newline. Verify commands and artifact
paths rather than copying stale examples.

## Prose Wrapping

- Do not hard-wrap ordinary Markdown prose at a fixed column. Keep each paragraph on one source line unless Markdown semantics or a document format requires breaks (for example, lists, tables, code, blockquotes, or fixed-width email/plain-text output).
- Treat editor and browser word wrap as presentation, not a reason to insert newlines.
- Do not reflow prose merely to satisfy Markdownlint MD013. Disable or configure that rule for prose-heavy documentation when appropriate.
- Preserve existing paragraph line structure when editing unrelated text; avoid drive-by reflow.
