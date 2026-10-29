---
name: markdown
description: Write or repair repo markdown files. Use when the task is editing README files, guides, plans, notes, or markdown lint failures. Apply the repo's markdown rules, preserve the right template or document structure, and keep prose tight. Do not use for inline code comments or code changes that only happen to touch markdown strings.
---

# Markdown

## Purpose

Use this skill to keep repo markdown clean, consistent, and easy to maintain.

## Use When

- Writing or revising markdown docs.
- Fixing markdown lint failures.
- Editing plans, notes, guides, or README files.

## Do Not Use When

- The real task is inline code documentation.
- The work is code implementation, not doc authoring.
- The file is not markdown.

## Procedure

1. Read [references/index.md](references/index.md) first.
2. Identify the document family:
   - guide
   - note
   - plan
   - general README or content doc
3. Use the canonical template when the document family has one.
4. Preserve the existing structure of the doc family unless the task is
   explicitly a reorganization.
5. Apply the repo markdown rules:
   - real markdown links, not bare URLs
   - no unnecessary inline HTML
   - blank lines around headings and lists
   - language-tagged fenced code blocks when known
   - single trailing newline
6. Keep prose lean. Prefer one canonical explanation over duplicated text across
   several files.
7. If the task is markdown-focused or touches several markdown files, run a
   markdown lint pass before stopping when repo tooling is available.

## References

Read [references/index.md](references/index.md) first.

## Handoff Rules

- Hand off to `documentation` if the real work is inline code docs rather than
  markdown documents.
- Hand off to `write-notes` if the main question is where a note belongs and how
  to consolidate it without note sprawl.
