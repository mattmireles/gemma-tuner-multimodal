# Prediction Is Understanding

## Axioms

1. **Learning beats engineering.** Stop programming intelligence when a model can
   learn the pattern more robustly.
2. **Scale changes the answer.** Re-evaluate prompt, model, and workflow choices
   when model capability, context length, latency, or budget changes.
3. **Prediction is a route to understanding.** A model that can predict a complex
   story must internalize structure, facts, causality, and psychology.
4. **Empirical evidence wins.** The trick that works on representative inputs is
   more valuable than an elegant design that fails in practice.
5. **Data is the bottleneck.** Treat fixtures, locked corpora, artifacts, and
   judging packets as first-class engineering objects.
6. **Prompts are code.** Version them, test them, keep them human-editable, and
   store rendered prompts at the replay boundary.
7. **Safety is engineering.** Powerful probabilistic outputs need containment,
   correction paths, and honest uncertainty.

## Working Stance

Start with the simplest path: one capable model, one clear prompt, one pass, and
one measurable quality gate. Add routing, repair loops, validators, or multiple
models only when traces or evals prove the simpler system cannot meet the bar.

When the model fails, accept the failure as data. The next step is usually one
of: improve the input, improve the prompt, choose a stronger model, add a narrow
validator, or change the task decomposition.

The wrong move is to bury uncertainty in hand-coded rules that nobody can test
or replay.
