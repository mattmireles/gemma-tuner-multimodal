---
name: debug
description: >-
  Systematic debugging for gemma-macos-tuner: consult `README/` and `README.md` first,
  pull current library docs via Context7 MCP, parallelize investigation with
  multiple subagents when stuck, prove fixes before calling success, then capture
  one consolidated note (e.g. `README/notes/debug-notes.md` via **write-notes**) as the **final**
  step before ending the session. For extreme cases, delegate a multi-agent
  audit via the Claude Code CLI. Use when the user invokes **debug**, **use
  debug**, asks to debug or fix a tricky bug, or work is blocked by unclear
  failure modes after quick local checks.
---

# Debug

## Purpose

Reduce guesswork and wheel-spinning: **`README/` and existing notes first**,
**Context7 when library behavior is uncertain**, **parallel hypotheses when
stuck**, **proof before “fixed”**, **one write-notes pass at session end**,
**CLI multi-agent audit** only for exceptional escalation.

## Use When

- The user says **debug**, **use the debug skill**, **debug this**, or
  equivalent, or the task is clearly **bug investigation** / **root-cause**
  analysis that is stalling.
- Failures involve **frameworks, SDKs, or cloud APIs** where training data may be
  stale—Context7 should be used before assuming APIs.

## Do Not Use When

- The user only wants a **trivial fix** with an obvious stack trace and no
  doc ambiguity (still skim `README/` if the area is known to be finicky).
- The task is **greenfield feature build** with no defect—use normal dev flow.

## Workflow (in order)

### 1. Read repo knowledge first (mandatory)

Before writing code or deep-diving the stack:

1. **Skim `README/`** — especially [`README/notes/debug-notes.md`](../../../README/notes/debug-notes.md) and relevant guides under [`README/guides/`](../../../README/guides/) — for subsystem match (MPS, collators, wizard, BigQuery/GCS paths). Look for **known issues**, **workarounds**, and **checklists**.
2. **Skim `README.md`** for structure, scripts, and links to other docs. Use [`README/skills/`](../../../README/skills/) when debugging agent/plan routing.

Many answers live here; **do not skip** this step to “save time.”

### 2. Context7 MCP (library and platform docs)

When the bug touches **specific libraries, CLIs, or cloud products** (PyTorch,
Transformers, PEFT, librosa, GCP clients, Typer, etc.):

1. Use the **Context7** MCP server: resolve the library ID, then fetch focused
   docs for the API or behavior in question (errors, limits, migration notes).
2. Prefer Context7 over memory or generic web search **for API shape and
   version-sensitive behavior**.

If Context7 is unavailable, say so briefly and fall back to official docs in
the browser or MCP fetch—**do not** invent APIs.

### 3. Reproduce and narrow

- Confirm **minimal repro** or the exact **request / job / log line** that
  fails.
- State a **one-sentence current hypothesis** and what evidence would falsify
  it.

### 4. Parallel investigation when stuck

If the problem is **hard**, **cross-cutting**, or you have been **going in
circles** after the reproduce-and-narrow pass (step 3):

- In **one turn**, spin up **multiple** `Task` subagents with **`readonly: true`**
  and **narrow charters** (e.g. “collator only”, “config/INI only”, “MPS device path only”).
- Ask each for **plausible hypotheses** ranked by likelihood and **concrete
  next checks** (files, logs, experiments)—not generic advice.
- **Merge** overlapping hypotheses; avoid duplicate deep reads of the same file.

**When in doubt, parallelize** (same posture as **`audit`**, but
hypothesis-driven and allowed to propose experiments).

### 5. Exceptionally frustrating or elusive bugs — CLI + audit

When you are **still stuck** after parallel investigation, or the blast radius
is **high** (money, data loss, production):

1. Use the **Claude Code CLI** from the repo root (or the user’s preferred
   invocation) to run a **multi-agent `audit`** over the **scoped slice** that
   matters (paths, feature, or git range).
2. In the CLI handoff prompt, **explicitly attach or cite**:
   - The relevant files under **`README/`** (and optional **`README/Guides/`** / **`README/Notes/`** if present)
     (paths or paste key excerpts),
   - A short **symptom + what was already ruled out**,
   - Instruction to use the **`audit` skill** with **multiple readonly
     subagents** and findings-first output.

This is for **escalation**, not first resort—**after** `README/`, notes, Context7,
and a fair parallel pass.

### 6. Prove the fix (before claiming success)

Do **not** say the bug is **fixed**, **resolved**, **done**, or imply victory
until there is **objective proof** the original failure mode is gone: e.g.
failing test passes, manual repro clean, bad log line absent, pipeline green—
whatever matches the bug—not because a change *ought* to work.

If you cannot verify yet, state **unverified** or **needs proof** and list the
exact check still missing.

### 7. Write notes once (mandatory last step before ending the session)

**After** steps 2–6, perform **a single** notes update using
[**`write-notes`**](../write-notes/SKILL.md) (typically appending or restructuring
`README/notes/debug-notes.md`; consolidate the whole session; do not dribble edits earlier).

- **Do not** repeatedly open or patch note files while investigating; keep
  working detail in memory or chat until this step.
- One pass should capture: symptom, repro fingerprint, what was ruled out, root
  cause (`TBD` if unknown), fix (if any), **verification** (proven / not proven
  / not run), pointers to relevant docs.
- **Skip** only when the user explicitly wants no note or the outcome is a
  trivial obvious one-liner; when in doubt, **write once** here.

## Output expectations

- **What was read** from `README/` / notes (document names or paths).
- **Verification:** what proved the fix (or honest **unverified** / **blocked**).
- **Which notes file(s)** were updated in step 7 (or why skipped).
- **Whether Context7** was used and for which library/topic.
- **Leading hypothesis(es)** if still open, else **root cause** summary.
- If CLI audit was invoked: **scope given to Claude Code** and **what you need
  back** (graded findings, P0 list).

## Anti-patterns

- Jumping straight to code changes without **`README/`** / **`README.md`**.
- Assuming API behavior without **Context7 or official docs** when versions
  matter.
- **Serial** thrashing (same file, same guess) instead of **parallel**
  charters.
- Calling **audit** for every small typo—keep escalation proportionate.
- **Declaring victory** without **verification** (step 6).
- **Patching notes throughout** the session instead of **one** end-of-session
  pass (step 7).

## Relation to other skills

- **`audit`:** structured repo review; use via CLI escalation when debug skill
  maxes out.
- **`write-notes`:** follow **[`write-notes`](../write-notes/SKILL.md)** **once at
  the end** of the debug session (step 7), after honest verification status;
  consolidates the full investigation for the next engineer’s step 1 (prefer **`README/notes/debug-notes.md`** in this repo unless a different notes layout exists).
- **`phase-audit`:** plan-phase checks—not the same as production debugging.
