---
name: write-notes
description: Write or update repo notes (prefer README/notes/debug-notes.md in this repo, or README/Notes if present). Use when the user wants debugging notes, investigation notes, audit notes, or institutional memory captured in the repo. Prefer updating the right high-level notes document over creating a fresh file for every session. Do not use for plans, long-form guides, or inline code comments.
---

# Write Notes

## Purpose

Capture useful institutional memory without creating note sprawl. In **gemma-tuner-multimodal**, default to **`README/notes/debug-notes.md`** unless the repo already uses another agreed layout.

## Use When

- The user wants a bug, investigation, audit, or debugging trail captured in
  repo notes.
- A change should leave behind durable troubleshooting context.
- An existing notes bundle should be updated with a new issue section.

## Do Not Use When

- The user wants a plan.
- The output belongs in a guide or `README.md` instead of session notes.
- The content is really inline code documentation.
- The note would be throwaway session chatter with no lasting value.

## Procedure

1. Read [references/index.md](references/index.md) first.
2. Scan `README/notes/debug-notes.md` and, if present, `README/Notes/` for the best existing home before creating any new file.
3. Default to consolidation:
   - update the right high-level domain note file
   - add a new issue section inside that file
   - keep the issue self-contained with the notes-template structure
4. Create a new notes file only when the topic is durable enough to deserve its
   own entry point:
   - recurring subsystem problem
   - cross-cutting audit
   - report likely to be searched directly later
5. Use topic-based names for new files. Do not create a new file just because
   there is a new date or session.
6. Keep the note high signal:
   - summary
   - symptom
   - root cause or `TBD`
   - related guides
   - fix or current status
   - verification or next step
7. Keep active issues near the top. Convert fixed issues to resolved status and
   prune dead investigation branches after resolution.

## References

Read [references/index.md](references/index.md) first.

## Handoff Rules

- Hand off to `markdown` when the real task is markdown cleanup rather than note
  placement or note structure.
- Hand off to `create-plan` when the user wants a real implementation plan
  instead of notes.
