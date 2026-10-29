---
name: audit-fix-hardcore
description: Compatibility alias for `audit-fix-loop`; keep system-shape and architecture decisions under review while executing the audit-fix loop.
---

# Audit Fix Hardcore (Compatibility Alias)

This is a compatibility entry for workflows that invoke `audit-fix-hardcore`.
Use the same procedure as [`audit-fix-loop`](../audit-fix-loop/SKILL.md) and run `elon-musk` as the first pre-pass when the scope involves architecture, complexity, and tradeoff risk before starting each audit-fix cycle.

## Procedure

1. Invoke `audit-fix-loop` behavior exactly (scope, audit/fix iterations, final commit).
2. Ensure each cycle is preceded by `elon-musk` before running the `audit` steps.
