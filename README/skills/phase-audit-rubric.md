# Phase Audit Rubric

Canonical review checklist for `phase-audit` and the local audit fallback used
by `execute-plan`.

## Review Goal

Decide whether the current phase is actually complete, safely committed, and
ready for the next phase or push.

## Required Checks

### 1. Scope Completion

Double-check and audit all the work. Review the code. 

- Did the implementation complete the exact scope promised by the current phase?
- Did we miss anything?

### 2. Checkbox Accuracy

Check the boxes and update the plan. 

- Do the checked boxes in the plan match reality?
- Is anything marked complete that was only partially done?

### 3. Canonical-Guide Alignment

- Does the implementation still follow the linked `README/guides` contracts?
- Did it create a parallel workflow or duplicate a canonical doc by accident?

### 4. Edge Cases

- Are the obvious failure modes handled?
- Did the change preserve existing guardrails and invariants?

### 5. Tests and Verification

- Were the relevant tests or validations run?
- Is there a meaningful verification statement for the phase?
- If tests were not run, is that gap stated clearly?

### 6. Commit Readiness

- Is the change coherent enough to commit as one phase?
- Are there stray edits, debug leftovers, or half-finished scaffolding?

### 7. Push and CI Readiness

- Is the repo ready for the next push?
- Is there any obvious reason the change will fail CI?

## Findings Format

Report findings in this order:

1. High severity findings
2. Medium severity findings
3. Low severity findings
4. Residual risks or test gaps

If there are no findings, say so explicitly and still mention any remaining
verification gaps.

## Decision Rule

- If scope, checkbox accuracy, or safety is wrong, the phase is not complete.
- If findings require fixes, do not mark the phase complete yet.
- If no blocking findings remain, the phase can be marked complete and
  committed.
