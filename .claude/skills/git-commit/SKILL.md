---
name: git-commit
description: Create a scoped Gemma Tuner commit after tests pass. Use only when the user explicitly asks to commit.
---

# Git commit

Review the dirty worktree, stage only requested files, exclude models, private
datasets, secrets, outputs, and unrelated changes, run focused gates, inspect
the staged diff, and write a specific commit message. Never lower tests to make
a commit pass.
