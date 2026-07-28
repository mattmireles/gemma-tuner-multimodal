---
name: git-push
description: Push a tested Gemma Tuner branch after confirming commit scope and remote state. Use only when explicitly requested.
---

# Git push

Confirm the branch, upstream, clean intended diff, and passing gates. Fetch
remote state without destructive resets, then push the explicit branch. Never
push private data, model weights, tokens, or generated training artifacts.
