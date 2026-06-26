# Future Work — roadmap (from Prez slide 40)

The five future-work directions from the deck, each landed into the repo as a concrete artifact and
wired into the one-command automated runner (`python -m rolevec.run_all`).

| # | Limitation (deck) | Future direction (deck) | Landed as |
|---|---|---|---|
| 1 | Single small model | Replicate on larger / different-architecture models | **Cross-model sweep** in `run_all` (loops backends/models; with dummy it simulates two "models" via distinct seeds) → compare metrics. Real: pass several `--model` ids. |
| 2 | Professional roles only | Extend to non-occupational personas (social, cultural, demographic, personality) | **Big Five persona track** — `data/roles_bigfive.yaml` (personality personas, not occupations). Same pipeline, `--roles`. |
| 3 | Professional domain-oriented extraction | Trait-based question sets (Big Five / value frameworks); test generalization to psychological constructs | **`data/questions_bigfive.yaml`** — trait-targeted situational items, 5 trait families. Same pipeline, `--questions`. |
| 4 | Minimal prompting only | Richer, behavioral-science-grounded role prompts; check sharper/more coherent personas while preserving stability & separability | **Behaviorally-grounded prompts** in `roles_bigfive.yaml`; A/B against the minimal occupational prompts via the cross-track metrics in `report.md`. |
| 5 | Steering injection after validation untested | Inject the validated role vector; test persistent target-role behavior, coherent across multi-turn | **`src/rolevec/steer.py`** + `steer-inject` skill — apply v_r during generation, measure role-score persistence over T turns vs prompted-only baseline (which drifts in 3-9 turns per the deck). |

## How they generalize the original pipeline
The core insight: **#2, #3, #4 need no new pipeline code** — the extractor/judge/metrics are
role-set- and question-set-agnostic. They are exercised purely by swapping `--roles` / `--questions`
data. Only **#1** (sweep/aggregate) and **#5** (injection + persistence) add new code.

## Validation gate
Every track is validated against its success criteria (Q1 domain-sensitivity, Q3 stability,
Q4 separability; plus a persistence check for #5). `run_all` aggregates these into an overall
PASS/FAIL and exits non-zero if any required check fails, so the whole thing is CI-runnable.

## Status
- Today (dummy backend): the full roadmap runs end-to-end and self-validates with no GPU.
- Real models: choose a backend (recommended: `nnsight` on **NDIF**, free academic access),
  implement the two backend integration points, and the same `run_all` produces real results.
