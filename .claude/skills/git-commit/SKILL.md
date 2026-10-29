---
name: git-commit
description: Produces comprehensive Git commit messages (what and why) and, when authorized, stages and commits. By default stages the entire working tree (all pending changes), including work from other agents or earlier sessions—not only files the current agent touched. After a successful commit (non-blocking), runs `ruff check .` and a **lightweight** read of **HEAD** only (bug risks, god-module smells, needless complexity, duplication)—not a full **audit** (no grades, no multi-agent); for that, the user invokes the **`audit`** skill. Surfacing findings as optional heads-up, never as a reason to block or unwind the commit unless the user asks. No automatic fixes. Message-only requests use the same template without running git until commit is explicitly requested. Also use inside authorized workflows (for example execute-plan phase commits). Do not use for read-only review, hypothetical history, or when git writes are not authorized.
---

# Git Commit

## Purpose

Produce **reviewable commits**: a clear subject line and a body that explains
**what** changed and **why**. **Staging is whole-tree by default** so one commit
reflects everything currently pending on the branch, not a cherry-pick of what
this session edited.

## Authority

- **`git commit` allowed when** the user explicitly asks you to commit, or a
  checked-in workflow skill (for example `execute-plan`) authorizes commits as
  part of its procedure.
- **Message text only** when the user wants a commit message but did **not**
  ask you to run Git: use the subject and body rules below, optionally after
  `git status` / `git diff` for context. Do **not** `git add` or `git commit`.
- If committing would be a surprise—no explicit commit request and no
  workflow granting commit authority—**stop** before `git commit` and confirm
  intent.

## Use When

- A commit with a strong message is requested or workflow-authorized.
- The user wants help composing commit message text (with or without you
  running `git commit`—see Authority).

## Do Not Use When

- Git writes are forbidden by context and nothing overrides that.

## Staging scope

- **Default (this skill alone):** Stage **all** changes in the working tree that
  Git will track—modified, deleted, and **untracked** files—typically
  `git add -A` at the repo root. That includes edits made by **other agents**,
  the user, or tooling. `.gitignore` still applies (do not fight it to force
  secrets into the commit).
- **Parent workflow override:** If the active procedure **explicitly** limits
  what may be staged (for example `execute-plan` phase commits: only files for
  the completed phase), **follow that staging scope** and still use the message
  rules below.
- If `git status` shows something surprising (unexpected paths, credentials,
  huge artifacts), **surface it** before committing; do not silently drop
  co-workers’ changes to keep the commit “tidy.”

## Procedure

1. **Inspect** full `git status` and `git diff` (and `git diff --staged` if
   anything is already staged). Understand **everything** that will be included.
2. **Stage** per [Staging scope](#staging-scope) above.
3. **Subject line** (about 50 characters target, 72 hard cap):
   - Imperative mood: "Add", "Fix", "Refactor" — not past tense ("Added",
     "Fixed") or third-person singular ("Adds", "Fixes") as the **subject**.
   - The **body** may include issue closers such as `Fixes #123` when that is the
     project convention.
   - Be specific about area or behavior, not "Update code" or "WIP".
4. **Body** (one blank line after the subject; wrap near 72 columns):
   - **What**: bullets or short paragraphs covering **all** substantive areas in
     this commit (not only the files you touched this turn).
   - **Why**: motivation, tradeoffs, what was broken or awkward, or why this
     approach over alternatives.
   - **Context** (optional): issue links, plan paths, or follow-ups when they
     help the next reader.
5. **Commit** (commit path only), for example in bash or zsh:
   - `git commit -m "subject" -m $'paragraph...\n\n- bullet'`
   - or `git commit` with an editor when the body is long.
6. **Verify** (commit path only): `git show --stat HEAD` matches intent (entire
   staged set landed).
7. **Post-commit audit** (commit path only): after the commit succeeds, follow
   [Post-commit audit](#post-commit-audit). Run **`ruff check .`** first; then
   scan **only** the `HEAD` diff for bug risks, god modules, needless complexity,
   and duplication (**lightweight**—see [What this is not](#what-this-is-not)).
   **Tell the user** about findings in the same turn (paths,
   severity, next step if obvious)—as **heads-up**, not scolding. The commit is
   already done; do **not** delay the commit, imply it was a mistake, or push
   amend/revert unless the user asks. Do **not** silently “fix” findings unless
   the user asked; surfacing is the goal. If everything passes and nothing
   worrisome stands out in the diff, you may omit commentary.

On the **message-only** path, perform step 1 as needed for context, skip
staging and steps 5–7, and output the subject and body (use the template
below).

## Message template

```text
Add/improve/fix <behavior> in <area>

What:
- ...

Why:
- ...
```

## Post-commit audit

Run this **only after** `git commit` completes successfully (same repo, same
branch).

### Intent (non-blocking)

- **Commit always wins.** Nothing here vetoes, reorders, or shames a commit that
  already landed. People should keep committing; this step is **extra awareness**
  for the author, not policy for the team.
- **Alerts, not gates.** Ruff or review findings are **heads-up** so you can
  fix forward (`ruff check .` failed after commit—here is the output) or
  choose to ignore. Do not treat red Ruff as “the commit was wrong.”
- **Optional follow-up.** Suggest a fix commit or local cleanup only when helpful;
  never pressure amend/revert unless the user explicitly wants that.
- **Deeper review:** For **format check + tests**, **A–F grades**, optional **multi-agent**
  passes, and broader scope, the user should invoke the **`audit`** skill
  (message contains **`audit`**)—do not inflate post-commit into a full audit.

### What this is not

Stay **narrow** so commits stay fast:

- **Do** run **`ruff check .`** and read **`git show -p HEAD`** (this commit only).
- **Do not** run **`ruff format --check`**, full **`pytest`**, or integration tests here by
  default—those belong to **`audit`**, **`deploy`**, or CI unless the user’s task
  already called for them.

### Ruff (primary static pass)

From the **repository root**, with dev deps installed (`pip install -e '.[dev]'`):

- **`ruff check .`**

If the command is missing, dependencies are not installed, or the run fails,
**report the full relevant output to the user** with neutral framing (for example
“Heads-up: `ruff check .` failed after your commit”). The commit stays;
the user can fix in a follow-up when ready. Do not hide lint errors.

### Diff scope

Use the committed change as the source of truth—e.g. `git show HEAD` or
`git show -p HEAD`, and the paths touched:
`git diff-tree --no-commit-id --name-only -r HEAD`.

### Bug scan

Re-read the change as a quick self-review **on changed lines only**. Flag **likely**
problems for the user (this is not a substitute for CI, **`audit`**, or the full
test suite unless already part of the task):

- Logic errors, wrong conditions, off-by-one, incorrect defaults.
- Missing or broken error handling / early returns where failures are plausible.
- Obvious regressions: removed guards, weakened validation, race or stale-state
  risks introduced by the new code.
- API or type mismatches, impossible states, or changes that contradict the
  commit message.
- Training/device edge cases if the diff touches `gemma_tuner/models/` or `device.py`.
- **Security / privacy (glance):** obvious secrets or PII logged in the diff; if
  it needs auth review, nudge toward **`audit`** instead of digging here.
- Anything that would make you say “wait, that can’t be right” on a PR.

### God module scan

Align with repo philosophy: **clear separation of concerns**, no dumping
unrelated responsibilities into one place.

Flag when this commit **creates or substantially grows** a module that looks
like a grab-bag “does everything” file—for example:

- **Size**: one file approaches or exceeds the project’s per-file size guidance
  (this repo aims for well under ~1000 LOC per file—use judgment near that
  band).
- **Scope creep**: unrelated domains, layers, or features fused into a single
  module (many disparate exports, orchestration + persistence + UI helpers in
  one file, etc.).
- **Fan-in smell**: a change that makes one file the obvious choke point for
  unrelated call sites when a split would be natural.

When in doubt, **flag lightly** with reasoning; avoid crying wolf, but do not
skip obvious smells to avoid bothering the user.

### Needless complexity and duplication

- **Needless complexity**: extra layers, over-abstraction, clever patterns where
  a straight line would do, new dependencies or infrastructure for trivial
  wins, configuration explosions, or branching that obscures the actual
  behavior—especially when it violates **simpler is better** for this repo.
- **Duplication**: copy-pasted blocks, parallel implementations of the same rule
  or transform, near-identical helpers that should be one function, or “another
  copy” of logic that already exists elsewhere—when unifying would obviously
  reduce drift risk.

Flag when this commit **introduces or worsens** these; cite paths and why it
matters.

### What to output

- **Ruff**: pass/fail; if fail, enough output to act on, framed as post-commit
  awareness—not a blocked workflow.
- **If other issues**: short summary, bullet list with **file paths**, what you
  saw, and whether it looks like a definite bug vs. a risk vs. a maintainability
  smell.
- **If clean**: say nothing or one line—no boilerplate required.

## Anti-patterns

- One-word subjects ("fix", "wip", "updates").
- A body that only repeats the subject.
- **Cherry-staging** only the files you personally edited when the user asked
  for a commit and no narrower workflow override applies—other agents’ work
  must ship too.
- Claiming a “single logical change” while omitting co-present dirty files
  without explicit user direction.
- **Skipping the post-commit audit** after a successful commit when this skill’s
  commit path ran—especially omitting **`ruff check .`** or staying silent when
  it fails or the diff clearly introduces risk.
- **Turning post-commit into a full audit:** running **`pytest`**, **`ruff format --check`**,
  multi-agent **`Task`** delegation, or **A–F grades** here—use the
  **`audit`** skill when the user says **`audit`**.
- **Blocking mindset:** implying the user should not have committed, or urging
  amend/revert, because of post-commit findings—unless they asked for that.

## Related skills

- **`audit`:** user message includes **`audit`**; optional tests, optional
  multi-agent, **A–F** grades, broader scope.
- **`deploy`:** **blocking** Ruff + tests before release.
