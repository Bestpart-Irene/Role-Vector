---
name: run-all
description: One-command automated end-to-end run of the whole Role Vector roadmap with a validation gate — occupational track, cross-model replication (#1), Big Five persona track (#2/#3/#4), and steering persistence (#5) — then validates every track against its success criteria and writes runs/report.md. Use when the user wants to run everything, reproduce all results, check the validation gate, or get a single PASS/FAIL.
---

# run-all — full automated pipeline + validation gate

## When to run
"run everything", "reproduce all results", "does it all pass?", "run the full roadmap", "give me the report".

## Command
```bash
PYTHONPATH=src python -m rolevec.run_all --quick          # fast smoke (8 runs)
PYTHONPATH=src python -m rolevec.run_all --runs 30        # full
# real model later:
PYTHONPATH=src python -m rolevec.run_all --backend nnsight --model <hf-id> --runs 30
```

## What it does
1. **Track A** occupational roles (base method) → Q1/Q3/Q4 checks.
2. **Track A'** cross-model replication (#1) → second model; separability-agreement check.
3. **Track B** Big Five personas (#2/#3/#4) → same pipeline on trait roles/questions.
4. **Steering** persistence (#5) → per-role multi-turn PASS/FAIL.
5. **Validation gate** → aggregates all required checks into one OVERALL PASS/FAIL,
   writes `runs/report.md`, and exits non-zero on failure (CI-runnable).

## Notes
- With the `dummy` backend the whole thing runs with no GPU. Q1 (domain-sensitivity) needs a real
  judge, so under dummy it is *reported but not required* — flagged in the gate output.
- Required checks: Q3 stability (margins>0, perfect clustering), Q4 separability (all pairs >1),
  steering persistence, cross-model agreement.

## Guardrails
- A FAIL is information, not an error to hide — read which check failed in `runs/report.md`.
- Don't loosen a success criterion to force PASS; fix the data/prompt/coefficient instead.
