---
name: build-vs-buy
description: Decide whether to adopt an open-source library or build in-house before committing to a design in gemma-macos-tuner. Use when scoping a feature, writing a plan, evaluating architecture, or when the user suggests implementing infrastructure, parsing, SDK wrappers, eval tooling, or other reusable capabilities from scratch. Runs a short OSS scan and records a dependency cost note. Prefer PyPI packages that are maintained, license-compatible, and fit Python 3.10+, PyTorch, and Apple Silicon (optional CUDA) without unnecessary native build pain.
---

# Build vs Buy

## Purpose

Use this skill as a quick "does this already exist?" pass before inventing a
custom implementation in **gemma-macos-tuner**.

Default bias:

- Prefer a well-supported library when it clearly fits.
- Prefer building in-house when library fit is weak, heavy, abandoned, unsafe,
  or would add more complexity than it removes.
- Prefer dependencies that align with [`pyproject.toml`](../../../pyproject.toml): **transformers**, **PEFT**, **datasets**, optional **GCP** extras, etc.

## Use When

- Scoping a new feature or subsystem.
- Writing a checked-in plan.
- Proposing reusable IO, metrics, dataset utilities, or trainer hooks.
- The work smells like something PyPI may already solve.
- The user asks whether a package or SDK already exists.

## Do Not Use When

- The task is repo-specific training logic that should stay custom.
- The repo already has an established library for the job.
- The user explicitly wants a scratch implementation.
- A five-minute search would cost more than the code itself.

## Procedure

1. Define the capability in one sentence.
2. List the non-negotiables:
   - Python version and **torch** / **MPS** / **CUDA** fit
   - license and **optional extras** (`dev`, `gcp`, `viz`, etc.)
   - lockfile / install ergonomics (`uv`, `pip`)
   - required features
3. Run a short **WebSearch** pass for established open-source options.
4. Shortlist at most **3** candidates.
5. Reject weak fits quickly.
6. Recommend one of:
   - **Use library**
   - **Build thin wrapper around library**
   - **Build from scratch**
7. Include a one-paragraph **dependency cost note** with the decision.

Keep this pass lightweight. Usually **5 to 15 minutes**, not a research rabbit
hole.

## Evaluation Rubric

Check each candidate for:

- **Fit**: solves the actual problem, not a nearby one.
- **Weight**: dependency size, API sprawl, operational overhead.
- **Maintenance**: recent activity, issue health, release cadence.
- **Adoption**: real usage; stars are only a weak signal.
- **Compatibility**: Python 3.10+, PyTorch, Apple Silicon arm64 (and CUDA when relevant).
- **License**: acceptable for this repo.
- **Escape hatch**: can we wrap or replace it later without surgery?

Reject candidates fast if they are abandoned, overbuilt, fight the **transformers/PEFT** stack, are poorly documented, or impose painful native build chains for marginal value.

## Recommendation Format

Return a short decision note:

```md
Capability: <one sentence>

Candidates:
- <library 1> - why it is promising or rejected
- <library 2> - why it is promising or rejected
- <library 3> - why it is promising or rejected

Decision:
- Use library / Wrap library / Build from scratch

Why:
- <2-4 concrete reasons>

Dependency cost note:
- <install impact, transitive deps, operational cost, lock-in risk>

Next step:
- <what to plan or implement next>
```

## Decision Rules

- If one candidate clearly fits and lowers complexity, use it.
- If a candidate is close but exposes too much surface area, wrap it behind a
  local adapter.
- If every candidate adds more risk or complexity than value, build the smallest
  custom solution that can work.
- When uncertain, prefer the option that is simpler to maintain by ordinary
  teammates six months from now.

## Handoff

- During planning: include the shortlist and decision in the plan so future
  implementers know the build-vs-buy check already happened.
- During implementation: either adopt the chosen library or note explicitly why
  this repo is intentionally building a custom solution.
- If a library is the likely winner and current API details matter, fetch
  current docs with Context7 before writing implementation steps or code.
