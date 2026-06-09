---
name: vector-metrics
description: Compute the stability & separability metrics on extracted role vectors — same-role cosine, same-role Pearson, stability margin (C_same − C_cross), role-cluster assignment, and symmetric normalized separation across all role pairs — then evaluate pass/fail against the Q1-Q4 success criteria. Use when the user wants to analyze a run, check if roles are stable/separable, compare constructions, or generate the results numbers for the paper.
---

# vector-metrics — stability & separability (Prez slides 26, 31-32)

## When to run
"compute the metrics", "are the roles separable?", "is extraction stable?", "evaluate the run",
"which construction wins?", "give me the results numbers".

## Command
```bash
PYTHONPATH=src python -m rolevec.metrics --runs-dir runs/latest --construction balanced
# compare constructions for Q2:
for c in raw rmd balanced; do PYTHONPATH=src python -m rolevec.metrics --runs-dir runs/latest --construction $c; done
```

## Metrics (implemented in `src/rolevec/metrics.py`)
| Metric | Tests | Good |
|---|---|---|
| same-role cosine | directional reproducibility across runs | ~1 |
| same-role Pearson | coordinate-level pattern reproducibility | ~1 |
| stability margin `M_r = C_same − C_cross` | same-role tighter than cross-role | > 0 |
| cluster assignment | each run-vector nearest its own role centroid | 1.0 (chance 1/6 = 0.17) |
| symmetric normalized separation | how far apart two role centroids are | >1 = well-separated |

## Success criteria evaluated
- **Q1** in-domain score-3 rate > OOD, and OOD non-zero (from `meta.json`).
- **Q3** all stability margins > 0; cluster accuracy = 1.0.
- **Q4** all 15 pairwise separations > 1 (incl. the two close-role contrasts).
Report prints `[PASS]/[FAIL]` per check.

## Guardrails
- Separation is the *primary* separability metric (adds magnitude); cosine is a secondary directional check.
- Average over layers 10-20 before computing (the pipeline already band-averages on save).
- Watch the close-role pairs (mna_lawyer|corp_law_prof, general_prof|corp_law_prof) — they are the hard cases.
