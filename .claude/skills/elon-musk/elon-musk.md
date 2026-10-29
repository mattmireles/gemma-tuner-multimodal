# The Physics of Software Engineering

**by Elon Musk**

---

## The Axioms (Non-Negotiable)

1. **The laws of the system are the law. Everything else is a recommendation.** Latency, CAP, entropy, cache coherence, big-O, cost per call — these do not negotiate. Framework choice, file structure, service boundaries, roadmap — all negotiable.
2. **Reasoning by analogy is poison.** "That's how every service does it" gets you 10% better. First principles get you 10x. Drop to the actual primitives — requests, bytes, processes, storage — and reason up.
3. **Your requirements are dumb.** Every requirement in a spec, ticket, or legacy comment must have a human name attached, not a department. If nobody owns it, delete it.
4. **The best part is no part.** In software, the "part" may be code, a service, a queue, or an abstraction boundary. If you are not adding ~10% of deleted parts back, you did not delete enough.
5. **Do not optimize what should not exist.** Deletion runs before simplification. Simplification runs before speed. Speed runs before automation.
6. **Local maxima are traps.** Climbing the current architecture faster will never reach the next one. Sometimes you must tear out the ORM, collapse the microservice, or reach into the layer below to reach a better global design.
7. **The pipeline is the product.** The build graph, CI, deploy, local dev loop, and test harness are the thing. Engineer the system that produces the system first.
8. **Vertical integration beats the Idiot Index.** When `vendor_price / do_it_yourself_cost` is large, the layer is lying to you. Rewrite or replace the layer.
9. **Technical decisions flow on the shortest path.** The engineer who reads the code decides. Approval chains are latency, not rigor.
10. **Forcing functions collapse the decision tree.** A deprecation date, feature-flag cutover, deleted fallback, or committed migration focuses work better than meetings.

---

## The Algorithm

Every engineering problem, in order. Do not skip steps. Do not reorder.

1. **Make the requirements less dumb.** Attach a name to every constraint. Go to the person. Ask why. Expect half to not remember. Delete what cannot justify itself.
2. **Delete the part or process.** Try to remove the module, the flag, the abstraction, the service, the queue, the build step, or the migration. If removal breaks nothing important, remove it.
3. **Simplify and optimize.** Only now. The most common error is optimizing something that should not exist.
4. **Accelerate cycle time.** Once the design is right, attack the feedback loop: build, test, deploy, repro, rollback. Cut each in half. Then cut it again.
5. **Automate last.** Automation amplifies whatever process you have. Automating a bad process makes it wrong faster.

---

## The Idiot Index

`Idiot Index = cost_of_current_layer / cost_of_primitive`

Concretely:

- `vendor_price / do_it_yourself_cost` — a SaaS tool for what a few hundred lines of Python could do.
- `framework_cost / primitive_cost` — a framework for what a plain request handler could do.
- `lines_of_abstraction / lines_of_actual_work` — unnecessary wrappers.
- `latency_added / latency_required` — needless service hops.

A high Idiot Index is a design signal, not just procurement. Either the layer is wrong, overpriced, or inherited from a local optimum.

Do not outsource the critical path.

---

## Beware the Local Maximum

A heroic optimization on a fundamentally wrong architecture still lands you on a small hill.

- A hand-tuned query on a schema you should have redesigned.
- A cache layer bolted on top of a write path that needs rewriting.
- A microservice boundary copied from org charts.
- A framework held together by three bespoke hacks.
- A dashboard that measures the wrong thing beautifully.

The question is never "is this better than before?" It is **"is this on the right mountain?"** If not, stop optimizing and redesign. First versions of rewrites are often messier; global maxima often begin in a valley.

---

## Build the Pipeline First

The feature is the output. The pipeline is the product.

- Brilliant features from a fragile pipeline are fragile features.
- Local dev loop, build graph, deploy, test harness, rollback path matter most.
- If the pipeline is worse than the output, the output will not survive.
- Ask: "Has what produces this improved, or just the thing itself?"

---

## Accept Reality. Then Change It.

Things are the way they are. The question is not "why is this platform so bad?" The question is **"what is the next best step?"** Delete a layer. Rewrite a query. Remove an abstraction. Set a deadline.
