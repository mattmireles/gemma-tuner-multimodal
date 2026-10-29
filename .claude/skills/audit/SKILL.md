---
name: audit
description: Triggered when the user’s message includes the word **audit** (primary routing hook). Findings-first review of this repo or a scoped slice (paths, diff, commits)—runs Python mechanical signals (Ruff, tests) when in scope, optionally delegates readonly subagents when depth warrants (**when in doubt, parallelize**), merges and dedupes findings, and assigns A–F grades for architecture, correctness risk, and complexity debt. Do not use when the user wants implementation fixes unless they explicitly ask to fix issues after the audit—for plan-phase checklists against an active plan, prefer phase-audit.
---

# Audit

## Purpose

Run a **structured, paranoid-friendly** audit focused on **bugs** and **needless complexity**. The orchestrator **does not** silently rewrite code; it **surfaces**
issues with severity, paths, and letter grades.

**Posture:** use **judgment**. Narrow scope (few files, small diff, localized
change) can be a **single-agent** pass. **Whole-repo**, **large diff**, **high
blast radius**, or **you are unsure** → prefer **multiple readonly passes** with
**narrow charters** (parallel when the harness supports it—see
[Harness compatibility](#harness-compatibility)), then **merge and dedupe** into
one report. **When in doubt, use multiple passes.**

## Use When

- The user’s message includes **`audit`** as a substring (primary trigger)—e.g.
  **audit**, **audit this**, **security audit**, **use the audit skill** (case
  insensitive in normal routing).

Before running `audit`, invoke `elon-musk` if the audit is broad (whole-repo or large diff), touches training architecture (data paths, model router, wizard, export), or includes major complexity/cost decisions. Use it to challenge requirements and simplify first, then map findings.

## Do Not Use When

- The user wants **implementation** only—unless they ask for fixes **after**
  the audit lands.
- The work is **only** validating an `execute-plan` phase against the plan and
  rubric—use **`phase-audit`** instead (this skill is broader).

## Scope (ask or infer once)

Establish **audit scope** before delegating:

| Kind | How to bound |
| --- | --- |
| **Whole codebase** | Repo root; subagents apportion by area (`gemma_tuner/`, `tests/`, `config/`, `README/`) or by charter only. |
| **Paths** | User-provided globs or directories; all charters focus there. |
| **Git delta** | User provides base (`main`, `origin/main`, tag) or “last N commits”; use `git diff`, `git log -n`, `git diff-tree --name-only`. |
| **Single feature** | User names flows (e.g. “wizard finetune”, “image collator”); map to [`README.md`](../../../README.md), [`gemma_tuner/`](../../../gemma_tuner/), and [`config/`](../../../config/). |

State the **chosen scope** in the final report header.

## Mechanical signals (orchestrator, once)

From **repository root**, when the audited surface includes **Python** (almost always here):

1. **`ruff check .`**
2. **`ruff format --check .`**

Report failures as **P0 / Critical** findings with command output (summarize if
huge). Do not “fix” unless the user later asks.

**Optional:** `pytest -q -m "not slow"` (or CI’s explicit test list) when the user wants maximum confidence or the
change touches core training/config logic; failures are **P0**.

When subagents run, they should assume mechanical checks ran **once**; they
focus on **review**, not on re-running gates unless a charter requires
spot-checking a file.

## Subagent delegation (use judgment)

**When one agent is enough:** tiny or localized scope (e.g. a handful of files,
one module); low coupling; user asked for a quick
pass. The orchestrator performs the full charter coverage **solo** (still hit
architecture, correctness, security/ops, and complexity—just in one pass).

**When to parallelize:** whole-repo or large `git` range;
security- or data-path-sensitive areas; multimodal collator / dataloader / export concerns; or
**any uncertainty** about depth. **When in doubt, use multiple chartered passes.**

### Harness compatibility

Not every runtime exposes Cursor’s **`Task`** tool. Same skill, two shapes:

| Harness | How to run multiple charters |
| --- | --- |
| **Cursor (has `Task`)** | In **one** turn, launch **up to four** `Task` subagents with `readonly: true`, **`subagent_type`:** `generalPurpose` (or `explore` for layout mapping). Use **`model: fast`** unless the user asks for maximum depth. |
| **No `Task` (e.g. Codex CLI, plain Claude Code, terminals)** | Run the **same charters sequentially**: one charter per pass, same scope each time, then **merge and dedupe**. Do **not** skip depth because you cannot parallelize—sequential is fine. |

You may use **fewer than four** charters if scope is medium (e.g. two passes:
correctness+security vs architecture+complexity).

**If parallelizing (Cursor `Task` only):** in **one assistant turn**, launch **up to four**
`Task` subagents with `readonly: true`, **`subagent_type`:** `generalPurpose`
(or `explore` when the request is mostly mapping file layout). Use **`model: fast`**
unless the user asks for maximum depth.

Give each subagent (or each sequential charter pass):

- The **exact scope** (paths, diff summary, or “whole repo”).
- One **charter** from below (copy the charter text into the task prompt).
- Instruction: **concrete file paths**, **severity** (P0–P3), **one paragraph
  max per finding**, **no generic advice**.

### Charter 1 — Architecture & modules

God modules; separation of concerns; public API surface; clear split between **CLI/wizard**, **config**, **training router**, **model implementations**, and **utilities**; fan-in
choke points; misplaced orchestration vs domain logic. Flag files approaching
**~1000 LOC** or cramming unrelated responsibilities (**repo norm:** well
under 1k LOC per file).

### Charter 2 — Correctness & reliability

Logic bugs; error paths; swallowed exceptions; off-by-one; impossible states;
**device/MPS/CUDA** assumptions; dataloader and **multimodal** edge cases; checkpoint and export consistency; training numerical stability hooks.

### Charter 3 — Security, privacy, operational

Secrets in logs or repo; unsafe paths or shell-outs; credential files that should stay gitignored (**`config/config.ini`**, tokens); operational foot-guns (partial config, wrong `base_model`) **when relevant to scope**.

### Charter 4 — Complexity, duplication, maintainability

Needless abstraction; configuration or branching explosion; **duplication with
drift**; dead code; **comment/doc lies**; weak or lying tests; naming that hides
behavior; dependency weight for trivial wins.

## Consolidation (orchestrator)

After **subagents return** or after a **solo** review pass:

1. **Dedupe:** merge findings that cite the same root cause or file+theme.
2. **Severity:** **P0** critical (wrongness, security, data loss, build breaks),
   **P1** high, **P2** medium, **P3** low / hygiene.
3. **Grades:** assign three letter grades **A–F** using [Grading rubric](#grading-rubric).
   **Overall grade = worst of the three** (if any dimension is **D**, overall
   cannot be **B** or higher—cap at **D** unless you justify an exception in one
   sentence).
4. **New teammate test:** one line—would a newcomer likely break this area in
   week one? (y/n + why)

## Grading rubric

Assign **Architecture**, **Correctness risk**, and **Complexity debt** separately.

| Grade | Meaning |
| --- | --- |
| **A** | Clear boundaries, hard to misuse, simple where it matters, issues are cosmetic. |
| **B** | Solid with minor debt; a few focused fixes would raise to A. |
| **C** | Meaningful issues; would block merges in a strict shop without a plan. |
| **D** | Serious structural or reliability risk; bounded scope but unsafe. |
| **F** | Unsafe, incomprehensible, or broken; needs pause and redesign or revert. |

**Correctness risk** includes likelihood of **latent bugs** and **failure-mode**
holes—not only existing failing tests.

## Output template

```text
## Audit scope
- ...

## Mechanical checks
- ruff check: pass | fail (summary)
- ruff format --check: pass | fail (summary)
- tests (if run): ...

## Grades
- Architecture: ?
- Correctness risk: ?
- Complexity debt: ?
- Overall (worst-of-three): ?

## Findings (severity order)
### P0 — Critical
- ...

### P1 — High
- ...

### P2 — Medium
- ...

### P3 — Low
- ...

## Delegation / overlap notes
- Solo vs multi-charter (parallel `Task` or sequential); deduped: ...

## Residual risks / what we did not run
- ...

## New teammate test
- ...
```

## Anti-patterns

- **Whole-repo or high-risk audit in a single shallow pass** when depth was
  needed—**when in doubt, parallelize.**
- **Four subagents for a three-line diff**—wasted latency; judge proportionally.
- **Findings without paths**—every substantive issue should anchor to a file or
  symbol when possible.
- **Auto-implementing** during audit when the user only asked for review.
- **Grade inflation**—if Correctness is **D**, overall is not **B**.
- **Ignoring mechanical failures**—Ruff red is at least **P0** for
  code health.

## Relation to other skills

- **`phase-audit`:** plan-phase completion vs rubric and plan checkboxes.
- **`git-commit`:** post-commit **non-blocking** **`ruff check .`** + **`HEAD`**
  diff scan only—**not** a substitute for **`audit`** (no full test gate by default, no
  grades, no multi-agent).
- **`deploy`:** **blocking** checks before release; audit does not replace release
  gates when the goal is to ship.
