---
name: guide-ingest
description: >-
  End-to-end field-guide workflow: draft long-form technical guides (research export, vendor docs, or manual draft), then ingest—normalize markdown, verify APIs and behavior with Context7 MCP, add cross-links. Use when adding or refreshing a guide from research output; when the user says guide-ingest, research-to-guide, technical field guide workflow, or wants corpus cross-linking.
---

# Guide ingest

## Purpose

Turn raw research into **durable documentation** that matches this repo’s markdown
conventions, reflects **current** library and platform facts where Context7 can
verify them, and sits correctly alongside **`README/`**, the repo **`README.md`**, and
guides under **`README/guides/`**.

## End-to-end workflow (mirrors how we work here)

| Phase | What happens |
| --- | --- |
| **1. Draft** | Produce an **in-depth** guide from research export, vendor documentation dump, or paste. **Coverage beats brevity**—there is no target word count. |
| **2. Ingest** | Run **this skill** on the draft: proper markdown, **Context7** verification of documented facts, outbound + inbound links. |

Do **not** treat “short” as success. Treat **correct, exhaustive, and well-linked** as success.

## Use When

- Importing research output into `README/guides/` (or `README/`).
- The user invokes **guide-ingest**, **research-to-guide**, **technical field guide workflow**, or the same intent in natural language.
- A new doc needs **outbound** links to related notes/guides and **inbound** links from existing docs.

## Do Not Use When

- The task is only a small markdown typo fix (use **`markdown`**).
- The task is only where to put a note (use **`write-notes`**).
- The work is inline code documentation (use **`documentation`**).

## Prerequisites

- **Context7 MCP** for library docs when verifying API facts. Read the tool schema in the MCP descriptors **before** calling tools.
- **Raw source**: file path or pasted content.

## Procedure (ingest phase)

### 1. Normalize to proper markdown

1. Read the **`markdown`** skill and match tone/structure of sibling files under `README/guides/`.
2. Produce guide-shaped markdown:
   - real markdown links, not bare URLs
   - blank lines around headings and lists
   - language-tagged fences where applicable
   - single trailing newline; no unnecessary HTML
3. Add a short top-of-file purpose blurb if sibling docs use one.
4. **Length and tone:** Do **not** trim for length unless the user asks. Prefer keeping tables, gotchas, and code. If the draft is padded with generic filler, you may tighten phrasing **without** dropping technical detail.

### 2. Verify claims with Context7 (fix stale research)

1. Extract **verifiable** claims: library APIs, CLI flags, framework behavior, config keys, deprecation notices, default versions, and similar **documented** facts.
2. For each claim, use Context7: resolve the library/tool ID, then pull the relevant docs. Prefer **current** official behavior over the draft.
3. **Update the doc** when Context7 contradicts the export.
4. For claims **outside** Context7’s scope, do **not** fake verification. Either leave as narrative, mark as opinion/heuristic, or point to an authoritative repo note.

### 3. Add outbound links (guides and notes)

1. While reading sections, identify concepts that already have a home in **`README/`**, **`README.md`**, **`README/notes/`**, or **`docs/`**.
2. Add **inline** markdown links on the first strong mention in each section (or a compact “See also” list). Use **repo-relative** paths from the new file.
3. Prefer linking to **one canonical** doc per topic rather than duplicating long explanations.

### 4. Add inbound links (corpus updates)

1. **Discover candidates**: search `README/` and `README.md` for overlapping keywords. Include files that already link to adjacent topics.
2. **Edit sparingly**: add a link in “Related”, “See also”, or the most relevant paragraph—**minimal** diff, no drive-by rewrites.
3. If **`README/notes/debug-notes.md`** (or another note) covers the same subsystem, add a short bullet with a link where appropriate (see **`write-notes`**).

### 5. Close the loop

1. Run **`ruff`** / **`pre-commit`** only if you touched Python—otherwise rely on **`markdown`** conventions and CI for docs.
2. Give the user a **short summary**: what was corrected via Context7, which files gained inbound links, and any claims left unverified.

## Handoff Rules

| Situation | Hand off to |
| --- | --- |
| Repo markdown rules, lint, structure | **`markdown`** |
| Where a note should live, note sprawl | **`write-notes`** |
| Docstrings / code comments | **`documentation`** |
| Deep modeling or training tradeoff | **`ilya-sutskever`** |

## References

- [`README.md`](../../../README.md)
- [`README/templates/guide-template.md`](../../../README/templates/guide-template.md)
- [`README/guides/README.md`](../../../README/guides/README.md)
