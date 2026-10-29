# Execute Plan References

**Git cadence:** phase **commits** repeat per completed phase; **`git push`**
and the **`git-push`** skill run **once after all phases** (see parent
`SKILL.md`). Do not push after each phase unless the user explicitly overrides.

Canonical docs for `execute-plan` live under **`README/skills/`** (lowercase):

- [`README/skills/plan-workflow-skills-guide.md`](../../../../README/skills/plan-workflow-skills-guide.md)  
  Read first for the shared workflow contract and the explicit execution loop.
- [`README/skills/phase-audit-rubric.md`](../../../../README/skills/phase-audit-rubric.md)  
  Read for the canonical local-audit fallback when delegated review is not
  available.

Use **`README/skills/`** when the active plan interacts with other repo skills and
routing constraints (start with the guides above; extend to other files in that
folder as they are added).

Always inspect:

- the concrete checked-in plan file before any implementation
- every guide or note linked from the active phase
- the changed files and verification outputs before updating plan checkboxes or
  committing
