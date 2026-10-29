---
name: execute-plan-hardcore
description: >-
  Execute a checked-in implementation plan like execute-plan (phase-by-phase,
  commits, push, CI), then run the full audit skill on the execution scope and
  require Architecture, Correctness risk, and Complexity debt grades all A—fix
  findings and repeat until all three are A. Use when the user explicitly
  invokes this skill or wants the hardcore post-audit gate after plan execution.
  Do not use for planning-only, read-only review, or when the user wants standard
  execute-plan without the audit-to-A loop.
---

# Execute Plan (Hardcore)

## Purpose

Same end-to-end plan execution as **`execute-plan`**, plus a **mandatory
post-execution loop**: run the **`audit`** skill on the work, require **A**
grades on **all three** rubric dimensions, **fix everything** that stands in the
way, and **repeat** until those grades are **A** across the board.

## Use When

- A concrete checked-in plan path exists (commonly under `docs/plans/` or as agreed by the team; `README/plans/` is gitignored here for local drafts).
- The user explicitly invokes **`$execute-plan-hardcore`**, **use
  execute-plan-hardcore**, or otherwise clearly asks for this workflow (not
  implicit routing).
- The user wants **execute-plan** behavior **and** a full **audit** gate with
  **A / A / A** before the work is considered done.

## Do Not Use When

- No checked-in plan exists, or the request is planning-only / read-only.
- The user wants standard **`execute-plan`** without the audit-to-**A** loop—use
  **`execute-plan`**.
- The user forbids additional fix iterations after the plan is “done” by
  execute-plan alone.

## Authority Model

- Explicit invocation of **`execute-plan-hardcore`** authorizes:
  - everything **`execute-plan`** authorizes: phase commits **without** pushing
    until all phases are done, then **one** sync/push/CI tail per
    **`execute-plan`** step 5, plus follow-up fixes;
  - **after** the plan is fully executed and the **`execute-plan`** tail
    (`git-push` loop) has completed as that skill describes, **additional**
    commits and changes needed to satisfy the **post-audit** gate below;
  - running the **`audit`** workflow **as part of this skill** even though the
    user message may not contain the substring **`audit`**—this skill **is** the
    explicit request for audit **and** for **implementation fixes** until grades
    reach **A** (overrides **`audit`**’s default “no fixes unless asked” rule for
    this phase only).

- If **`execute-plan-hardcore`** was only inferred or routed implicitly, stop
  before the first commit or push and explain why—same boundary as
  **`execute-plan`**.

- At workflow start, state that **`execute-plan-hardcore`** is running: phase
  commits through Part A, **then** one push/CI tail, **then** **audit → fix →
  repeat until A/A/A** (additional commits/pushes only as needed for that
  loop—not per-phase push during Part A).

## Procedure

### Part A — Execute the plan (same as `execute-plan`)

1. Read the same primers as **`execute-plan`**
   ([README/guides/README.md](../../../README/guides/README.md),
   [README/skills/plan-workflow-skills-guide.md](../../../README/skills/plan-workflow-skills-guide.md),
   [README/skills/phase-audit-rubric.md](../../../README/skills/phase-audit-rubric.md));
   see also
   [execute-plan/references/index.md](../execute-plan/references/index.md).
2. Follow the **`execute-plan`** skill end-to-end: read the plan and linked
   guides/notes, one phase at a time, **`phase-audit`** per phase, plan
   checkboxes, narrow phase commits, then after all phases **`git-push`** (merge,
   push, CI green) per that skill.

Use the **`execute-plan`**
[SKILL.md](../execute-plan/SKILL.md) as the canonical
procedure for Part A.

### Part B — Hardcore audit gate (after Part A is complete)

Run only when **all phases** are implemented, the plan reflects reality, and the
**`execute-plan`** push/CI expectations are satisfied (no known failing CI from
this work unless the user explicitly overrides).

Before Part B audit passes, invoke `elon-musk` because the work already assumes plan-level architecture decisions and likely includes system-shape tradeoffs.

1. **Scope** the audit to the **plan execution**: prefer a **`git`**-defined
   slice (e.g. merge base with `origin/main` or the branch point where this plan
   work started → `HEAD`), or the directories/files the plan touched. State the
   scope in the audit report header (same expectation as **`audit`**).

2. **Run `audit`** in full:
   - mechanical signals from repo root (**`ruff check .`**, **`ruff format --check .`**
     for Python; add **`pytest -q -m "not slow"`** when
     needed for confidence);
   - depth: solo vs parallel charters per **`audit`** (when in doubt,
     parallelize);
   - produce the **`audit`** output template including **Architecture**,
     **Correctness risk**, and **Complexity debt** grades **and** findings.

3. **Pass condition:** **Architecture**, **Correctness risk**, and **Complexity
   debt** are all **A**. (If all three are **A**, overall is **A** per the
   worst-of-three rule in **`audit`**.)

4. **If any of the three grades is below A:**
   - Fix **all** issues that block raising grades to **A** (treat **P0**/**P1**
     first; do not ignore mechanical check failures).
   - Re-run relevant checks; **`git-commit`** (and push if needed) per repo
     conventions so fixes are recorded.
   - **Repeat** from step 2 (re-scope if the diff grew) until **A / A / A**.

5. **Stuck loop:** If multiple complete **audit** cycles still cannot reach
   **A / A / A** due to a tradeoff that needs product or architecture buy-in,
   **stop**, report grades and the blocker, and **ask the user**—do not spin
   forever.

## Worktree Rules

Same as **`execute-plan`**: ignore unrelated dirty files unless they conflict;
do not revert unrelated user work.

## Boundaries

- Do not skip Part B after Part A.
- Do not **grade inflate**; **`audit`** rubric applies.
- Do not lower **`audit`** thresholds—**raise** code/test quality until grades
  merit **A**.

## Relation to Other Skills

- **`execute-plan`:** Part A is identical in intent; **`execute-plan-hardcore`**
  adds Part B only.
- **`audit`:** Part B is a full **`audit`** pass with fix iterations authorized
  by this skill.
- **`phase-audit`:** Still used **per phase** inside Part A; Part B is **broader**
  **`audit`** after the full execution lands.

## Canonical Docs

Start with
[README/guides/README.md](../../../README/guides/README.md),
[README/skills/plan-workflow-skills-guide.md](../../../README/skills/plan-workflow-skills-guide.md), and
[README/skills/phase-audit-rubric.md](../../../README/skills/phase-audit-rubric.md),
then the active plan and linked **`README/guides`**, **`README/skills`**, and **`README/notes`**.
