---
name: steer-inject
description: Future Work #5 — inject a validated role vector into the residual stream during generation and test whether the role PERSISTS across a multi-turn conversation (vs prompted-only personas, which drift to the model default in 3-9 turns). Use when the user wants to run/evaluate steering, test role persistence, set up the injection hook, or sweep the steering coefficient. Defaults to a transparent simulation (dummy); real injection is an nnsight intervention.
---

# steer-inject — steering injection & multi-turn persistence

## When to run
"run the steering eval", "does the role persist across turns?", "wire up injection", "sweep the coefficient".

## Command
```bash
PYTHONPATH=src python -c "from rolevec.steer import evaluate_persistence as e; \
print(e(role_id='mna_lawyer', turns=9, coeff=8.0).summary())"
# or as part of the full run:
PYTHONPATH=src python -m rolevec.run_all --quick
```

## What it tests
- **Hypothesis (deck):** prompted-only personas drift to the default in 3-9 turns. A validated role
  vector injected at layer l (`h' = h + coeff·v_r`) should hold the role across the conversation.
- **Metric:** role score (0-3 rubric) per turn for steered vs prompted-only.
- **PASS:** steered final-turn score ≥ floor (default 2.0) AND exceeds the prompted baseline.

## Going live (real model)
1. Extract + validate vectors first (`role-extract` → `vector-metrics`).
2. Implement `NNSightBackend.generate_steered()` — add `coeff·v_r` to the chosen layer's output
   during generation (nnsight intervention; runnable remotely on **NDIF**).
3. Call `evaluate_persistence(..., simulate=False, backend=..., role_vector=v_r)` with a real judge.

## Guardrails
- Steer with the **validated** vector (passed Q3/Q4) — don't inject an unvalidated direction.
- Watch coherence, not just role score: an over-large coeff causes caricature/saturation (rubric 0).
- Pick the injection layer from the stability band (10-20); sweep coeff to find role-without-breakage.
