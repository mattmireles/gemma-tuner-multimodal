# Training visualizer debug notes

Institutional memory for the Gemma training dashboard (Socket.IO + Chart.js + Three.js): wiring bugs, empty charts, and multimodal panel gaps. Multiple related issues live here; each issue is self-contained.

**Quick filter:** `grep -n "— Active" README/notes/visualizer-debug-notes.md`

---

## Issue: Meta and hero loss update but charts / 3D / panels stay empty — Resolved

**First spotted:** 2026-04-08  
**Resolved:** 2026-04-08  
**Status:** Resolved

### Summary

Charts stayed empty while the hero loss moved because **`idx` was undefined** in the galaxy `forEach`, which threw during **3D init before `initCharts()`**, and a **second `io()`** in the template masked the failure for the hero only. **Fix:** define `idx` in the callback, run **`initCharts()` first**, wrap 3D init in **try/catch**, move hero loss into the viz bundle, remove the duplicate inline Socket.IO bridge, fix **`.panel`** selector for the galaxy title, hydrate **memory** from history, and guard chart updaters if charts failed to construct. **Do not** pass **`broadcast=`** to **`SocketIO.emit`** (Flask-SocketIO 5 / python-socketio): omit `to=` to broadcast to all; `broadcast=True` raises and breaks the emit worker so **live updates never arrive** (history still works via the request handler).

### Symptom

```log
(charts blank; hero loss still updated; possible Console: ReferenceError: idx is not defined)
```

### Root cause

Confirmed in code review: **`mesh.userData = { ring: p.ring, idx }`** referenced **`idx`** without a binding in `positions.forEach((p) => …)`. Init order ran **3D before charts**, so the exception prevented Chart.js setup. The template’s inline script used a **separate** `io()` connection and still updated `#loss-value`.

### Related Guides

- [Gemma 4 on Apple Silicon](../guides/apple-silicon/gemma4-guide.md) — multimodal stacks, MPS trainer behavior (context for what the server can reasonably stream).
- [README/notes/debug-notes.md](debug-notes.md) — prior Gemma 4 collator / masking issues (orthogonal to dashboard JS, but same training runs).

### Fix

**Files:**

- `static/visualizer.js` — `forEach((p, idx) => …)`; `initCharts()` before 3D; try/catch on 3D; `updateHeroLoss` + history/memory hydration; chart `if (!lossChart) return` guards; `.panel` for title.
- `templates/index.html` — removed duplicate inline `io()` bridge; cache-bust `visualizer.js?v=5`.
- `gemma_tuner/visualizer.py` — viz worker uses `socketio.emit(..., namespace="/")` only (no `broadcast=` kwarg).
- **2026-04-08 follow-up:** `GemmaVizTrainer` + `VisualizerTrainerCallback.bind_trainer()` plumb `batch` / `outputs` from the last forward into `build_training_event` (attention, token top‑k, mel or image preview). With **gradient checkpointing**, some stacks still omit attentions — see training log warning.

### Verification

```bash
ruff check gemma_tuner/
```

Browser: reload dashboard with `--visualize`; expect Chart.js lines to draw and 3D galaxy to render; Console free of `idx` errors.

### Investigation log

**2026-04-08 — Fix applied**

- **Root cause:** Undefined `idx` + wrong init order + duplicate Socket.IO client.
- **Outcome:** Resolved as above. Multimodal panels (attention / mel / tokens) remain data-limited on the HF `on_log` path unless the trainer passes `batch`/`outputs` — unchanged by this fix.

### If this recurs

- [ ] DevTools Console on first paint — any error before `initCharts` / Chart constructors.
- [ ] Single `io()` in Network → WS (no duplicate bridges in `index.html`).
- [ ] `training_update` payloads include scalars when you expect chart motion.

```bash
grep -n "forEach((p, idx)" static/viz/gemma-viz-scene.js
```

---

## Audit grades (2026-04-08, pre-fix snapshot — superseded by fix above)

| Dimension | Grade | Note |
|-----------|-------|------|
| **Architecture** | **B** (post-fix) | Single `io()` in viz bundle; charts init before 3D; `SocketIO.emit` without invalid `broadcast=` kwarg. |
| **Correctness risk** | **B** (post-fix) | `idx` fixed; try/catch + chart guards limit blast radius. |
| **Complexity debt** | **C** | Partial `training_update` shapes + multimodal nulls on HF path unchanged. |

---

<!--
USAGE NOTES (from README/templates/Notes-template.md):

1. Prefer updating this file before adding another visualizer note file.
2. When an issue is fixed, rename "— Active" to "— Resolved", add **Resolved:** date, and trim Investigation Log to confirmed facts.
-->
