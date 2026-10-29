---
name: ilya-sutskever
description: Applies an AI research and engineering lens to **gemma-macos-tuner**: base-model and LoRA choices, multimodal data and collator design, training objective, evaluation, and inference/export tradeoffs. Use when the work is about extracting smarter behavior from the training loop, not about generic app plumbing. Do not use for routine Python packaging, unrelated CI hygiene, or small refactors with no modeling impact.
---

# Ilya Sutskever

## Purpose

Use this skill for **AI engineering and research** in **gemma-macos-tuner**:
dataset design, collators, model capacity vs budget, training stability,
evaluation, and what should be **learned** vs hand-specified.

The stance is simple: learning, scaling, and empirical evidence beat ornate
hand-tuned rules.

## Use When

- Choosing or comparing **Gemma** checkpoints, LoRA rank/settings, or training recipes.
- Designing **text / image / audio** supervision, budgets, or sampling.
- The task touches **evaluation metrics**, held-out quality, or regression risks for fine-tunes.
- A pipeline is becoming too clever and needs the **simplest empirical loop** that can prove quality.

## Do Not Use When

- The task is **only** Ruff, pytest layout, or release tagging—other skills cover that.
- The failure is **purely** environment or packaging (missing torch, wrong Python)—fix infra first.
- The user needs **only** copy or docs with no modeling decision.

## Procedure

1. Name the intelligence required: classification, generation, captioning, ASR-style mapping, VQA, etc.
2. Read [ilya-sutskever.md](ilya-sutskever.md), then [references/index.md](references/index.md) and open only entries that match the task.
3. Ask: does this change belong in **training**, **data**, **config**, or **export**—and what is the minimal experiment that validates it?
4. Start with the simplest setup that could work; add complexity only when evidence demands it.
5. Match model capacity to **memory, latency, and data**; verify with concrete evals or smoke tests rather than intuition.
6. Treat **configs and prompts** (if any) as versioned artifacts tied to runs.
7. Test degenerate and representative inputs before trusting a quality claim.
8. Before stopping, ask: does this respect the **Bitter Lesson** mindset—is anything hand-coded that should be learned? Is there a simpler architecture that works?

## Output Expectations

- Explain choices in terms of scaling, data, empirical evidence, and simplicity.
- Name which guides or prior notes informed the decision when that matters.
- Prefer one well-tested training recipe over many speculative branches.
- Flag when an approach fights empirical validation or smuggles unbounded complexity into the data path.

## Handoff Rules

- **`audit`** / **`phase-audit`**: after large training-path changes when structured code review is needed.
- **`write-notes`**: capture durable training or eval findings in `README/notes/`.
- **`research-planning`** / **`research-debugging`**: when behavior depends on external vendor docs or upstream bugs.
