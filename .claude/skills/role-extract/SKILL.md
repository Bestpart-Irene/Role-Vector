---
name: role-extract
description: Extract prompted role personas into activation-space vectors — role-minus-default activations, score-2-plus weighting, balanced in-domain + out-of-domain construction (v = v_in + v_ood). Use when the user wants to build/re-extract role vectors, run the extraction pipeline, compare construction variants (raw vs rmd vs balanced), or set up a model backend. Model choice is deferred: defaults to the dummy backend; supports TransformerLens / nnsight (incl. NDIF remote).
---

# role-extract — vector construction

## When to run
"extract role vectors", "run the pipeline", "build the balanced construction", "switch to model X",
"compare raw vs role-minus-default".

## Command
```bash
PYTHONPATH=src python -m rolevec.pipeline --backend dummy --runs 30
# real model later (nothing else changes):
PYTHONPATH=src python -m rolevec.pipeline --backend nnsight --model meta-llama/Llama-3.1-8B --runs 30
```

## What it does (Prez slides 24-25)
1. Every role answers all 90 questions (15 in-domain + 75 OOD).
2. `role-judge` scores each answer 0-3 -> score-2-plus weight (3→3, 2→2, 0/1→0).
3. Per layer: `d_i = h_role(answer_i) - h_default(answer_i)` (role-minus-default).
4. `v_in = Σw·d over in-domain`, `v_ood = Σw·d over OOD`; **final = v_in + v_ood**.
5. Saves `runs/<ts>/vectors.npz` + `meta.json`; updates `runs/latest`.

## Backends (model deferred)
- `dummy` — random but stable activations; runs with no model. Default; use for wiring/tests.
- `transformer_lens` — local white-box HF model (Llama/Qwen/Gemma), `run_with_cache`.
- `nnsight` — extraction + intervention; **can run remotely on NDIF** (free [redacted] access to
  large open-weight models incl. Llama-3.1-405B) — no local GPU needed.
  Fill the integration point in `src/rolevec/backends.py`.

## Guardrails
- Never hard-code a model id; require `--model` for non-dummy backends.
- Analysis band is layers 10-20. Keep all three constructions saved for Q2 comparison.
- Questions are realistic scenarios, not "act as a ..." requests — keep it that way.
