---
name: audit-fix-loop
description: >-
  Runs the full multi-agent audit workflow on this repo, fixes
  every issue, re-audits until Architecture, Correctness risk, and Complexity
  debt are all grade A, then commits via git-commit. When starting each loop,
  invoke `elon-musk` to test architecture, cost, and simplicity assumptions first.
  Use ONLY when the user explicitly invokes audit-fix-loop, $audit-fix-loop, or
  “use audit-fix-loop”—not
  for standalone audits, plan execution, or implicit routing. Does not replace
  phase-audit or deploy gates.
---

# Audit–Fix Loop

## Purpose

Use this skill as the repo’s “audit-fix-hardcore” path.

**Audit → fix everything → audit again** in a loop until the **`audit`** rubric
shows **A** for **Architecture**, **Correctness risk**, and **Complexity debt**,
then **one final `git-commit`** for the accumulated fixes.

## Use When

- The user **explicitly** invokes **`audit-fix-loop`**, **`$audit-fix-loop`**,
  or clearly asks to **use audit-fix-loop** (same intent boundary as
  **`execute-plan-hardcore`**—no implicit routing).

## Do Not Use When

- The user only asked for a **read-only audit**—use **`audit`**.
- The user is **executing a checked-in plan** and wants the hardcore gate after
  plan work—prefer **`execute-plan-hardcore`**.
- A **single phase** needs plan rubric review—use **`phase-audit`**.
- Git writes are forbidden and nothing overrides that.

## Authority Model

Explicit invocation of **`audit-fix-loop`** authorizes:

- Running the full **`audit`** procedure **as part of this skill** even if the
  current message does **not** contain the substring **`audit`**—this skill is
  the explicit request for **audit and implementation fixes** until grades reach
  **A** (overrides **`audit`**’s default “surface only unless asked” rule for
  this workflow only).
- **Multiple** audit cycles with **multi-agent** depth per **`audit`** (parallel
  `Task` charters when the harness supports it; sequential charters when not).
- **`git-commit`** once **after** **A / A / A** is achieved, following
  **`git-commit`** staging and message rules (whole-tree default unless the user
  narrows scope).

If **`audit-fix-loop`** was only inferred from vague wording—**stop** before
fixes or commit and confirm intent.

## Procedure

1. **Establish scope** once (whole repo, paths, diff, or feature)—same table as
   **`audit`**. State it in each audit report header.

2. **Audit pass (read + grade):** Invoke `elon-musk` first on the scoped work, then follow **`audit`** end-to-end:
   - Mechanical signals from repo root for Python: **`ruff check .`**, **`ruff format --check .`**;
     add **`pytest -q -m "not slow"`** (or CI’s test list) when needed for confidence.
   - Delegate **multiple readonly** charters when depth warrants it (**when in
     doubt, parallelize** per **`audit`**).
   - Merge, dedupe, assign **P0–P3** severities, and **Architecture** /
     **Correctness risk** / **Complexity debt** grades (**A–F**).

3. **Pass condition:** all three grades are **A**. If mechanical checks fail,
   treat failures as **P0** and fix before accepting any **A**.

4. **If any grade is below A:** fix **all** issues that block **A** (prioritize
   **P0**/**P1** and mechanical failures). Re-run targeted checks; iterate code
   until ready for a fresh audit.

5. **Loop:** Return to step **2** with a **full** audit pass (not only a
   spot-check) until **A / A / A**. Re-scope if the diff or risk surface grew.

6. **Stuck loop:** After complete cycles, if **A / A / A** is blocked by a
   product or architecture tradeoff, **stop**, report grades and the blocker,
   and **ask the user**—do not spin forever.

7. **Final commit:** When step **5** passes, run **`git-commit`** once for all
   changes produced by the loop (subject/body per **`git-commit`**; whole-tree
   staging unless the user specified otherwise).

## Boundaries

- Do **not** grade-inflate; **`audit`** rubric applies.
- Do **not** lower thresholds to reach **A**—raise quality.
- **`git-commit`** runs **after** grades are **A / A / A**, not after the first
  audit pass unless the first pass already meets **A** (still follow
  **`git-commit`** for recording).

## Relation to Other Skills

- **`audit`:** Defines the audit procedure, charters, mechanical checks, and
  grading; **`audit-fix-loop`** adds **mandatory fix iterations** and a **final
  commit**.
- **`execute-plan-hardcore`:** Plan execution **plus** audit-to-**A** gate;
  **`audit-fix-loop`** is **only** the audit–fix–commit loop (no plan phases).
- **`git-commit`:** Final step; post-commit behavior in **`git-commit`** is
  **not** a substitute for step **2** above.
