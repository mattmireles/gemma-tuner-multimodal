---
name: git-push
description: Commits any pending work via git-commit, syncs with origin (fetch, merge-oriented pull), pushes the current branch, watches GitHub Actions, and fixes root causes until CI is green—including coverage gates without lowering thresholds. Use when the user asks to push, sync and push, or ship the branch and chase CI. Requires explicit invocation or a parent workflow that authorizes push and CI remediation. Do not use without permission to write to git remote or when the user forbids automated CI fix loops.
---

# Git Push

## Purpose

Close the loop from **dirty working tree → clean push → green GitHub Actions**:
commit everything pending when needed, integrate upstream with **merge** (not
silent rebase unless the user prefers otherwise), push, watch workflows, and
iterate on real failures until passes.

## Authority

- **`git push` allowed when** the user invokes `$git-push`, names `git-push`,
  or clearly asks you to sync, push, and fix CI until green—or a checked-in
  workflow authorizes the same (for example **`execute-plan`** step 5 **after
  all phases** are committed). **`execute-plan`** does **not** authorize
  running **`git-push`** after **each** phase—only the final tail.
- This authorizes **`git commit`**, **`git pull`/`git merge`** (integrating
  `origin`), **`git push`**, reading CI logs, and **code/test fixes** needed to
  get workflows green.
- If push or CI remediation would be a surprise, **stop** and confirm.

## Use When

- The user wants the branch **on the remote** and **CI passing**.
- You need the full “commit if needed → sync → push → babysit CI” routine.

## Do Not Use When

- The user only wants a **dry run** or local-only commands.
- You are not allowed to touch **`origin`** or fix CI (no override).

## Preconditions (agent)

1. Know the **current branch** and whether it has an **upstream**
   (`git status -sb`, `git rev-parse --abbrev-ref @{u}` when set).
2. `git fetch origin` before deciding you are in sync.

## Procedure

### 1. Uncommitted changes

- If **`git status --porcelain`** is non-empty (including untracked you intend
  to keep), run **`git-commit`** first so the working tree is committed with a
  full **what / why** message.
- **Do not** bypass `git-commit` with a lazy one-liner unless the user
  explicitly overrides.

### 2. Integrate `origin` (merge-oriented)

- **Preferred** for this skill: after `git fetch origin`, integrate with
  **merge**, not rebase—unless the user explicitly asks for rebase.
- If an upstream exists: e.g. `git pull --no-rebase` (or `git merge`
  `origin/<upstream-branch>` after fetch) so local commits combine with remote
  updates.
- If there is **no upstream** yet, a reasonable first push is
  `git push -u origin HEAD` after the branch is ready—set upstream for future
  pulls.
- **Merge conflicts**: resolve carefully; if intent is ambiguous, **stop** and
  ask rather than guessing.

### 3. Push

- `git push` to the appropriate remote (usually `origin`) and branch.
- If push is rejected because the remote advanced, **fetch**, **merge** (per
  above), resolve conflicts, then **push again**—do not force-push unless the
  user explicitly requests it and it is safe for the branch.

### 4. Monitor GitHub Actions

- Identify the run(s) for the pushed commit / branch (for example
  `gh run list --branch <branch> --limit 5` then
  `gh run watch <run-id> --exit-status`, or the Actions UI).
- Wait until the relevant workflow(s) **finish**.

### 5. On failure: fix root cause, repeat

- Read logs; reproduce locally when possible (`ruff check .`, `pytest ...`, targeted
  tests, matching what CI failed).
- Fix the **underlying issue**—not symptoms only—then **commit** (again via
  **`git-commit`** when there are new changes) and **push**, then **re-watch**
  CI.
- **Loop** until green or until you hit a blocker (permissions, flaky external,
  ambiguous product intent).

### 6. Coverage and quality gates

- If CI fails because **test coverage** (or a similar **required threshold**)
  is not met:<br />
  **Never satisfy the gate by lowering the threshold or disabling checks** as
  a shortcut.
- Correct response: **add or strengthen tests**, **exercise untested paths**,
  or **refactor for testability** so coverage **earns** the bar.

## Anti-patterns

- Pushing with a dirty tree (except when every pending change is intentionally
  left out—and then say so explicitly; default is commit first).
- Force-push to shared branches without explicit approval.
- “Fixing” CI by weakening lint, coverage, or type gates.
- Re-running CI repeatedly without changing what failed.

## Related skills

- **`git-commit`**: message shape and default **whole-tree** staging (unless a
  narrower parent workflow applies).
- **`execute-plan`**: may narrow what gets committed **per phase**; this skill
  still applies to the **push / merge / CI loop** once commits are ready.
