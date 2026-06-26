# Role Vector Validation — Project Guide

**One-line:** A reusable *construct-validation* pipeline for LLM agent personas. We extract prompted
role personas into activation-space vectors and test whether they (1) carry domain-sensitive role
signal, (2) are stably reproducible across repeated extraction, and (3) are separable from other roles.

> From "Prompted Agents" to "Validated Personas" — Jose L. Sampedro Mazon.
> PI: Dr. Nadim Saad · Advisor: Dr. Ilmi Yoon. Source of record: `Role_Vector_Interpretability_Prez_vFinal.pdf`.

## Research question
> Can prompted role personas be extracted into activation-space vectors that faithfully capture
> role-specific behaviour across domains, remain stable across repeated extraction, and are
> separable from other roles?

| Sub-Q | Question | Primary metric | Success criterion |
|---|---|---|---|
| Q1 | Is the signal domain-sensitive (strongest in-domain, measurable OOD)? | Score-3 rate in- vs out-of-domain; raw separation | In-domain ≫ OOD; OOD separation non-zero & consistent |
| Q2 | Which construction best isolates role-specific signal? | Pairwise symmetric normalized separation across variants | One global construction clearly beats raw vectors |
| Q3 | Are repeated extractions stable in direction + coordinate pattern? | Same-role cosine, Pearson, stability margin, cluster assignment | Same-role > cross-role baseline; margin > 0; correct cluster |
| Q4 | Are role vectors separable, incl. close-role contrasts? | Symmetric normalized separation over all 15 pairs | All 15 pairs separable; close-role pairs distinguishable |

## Pipeline (4 analytic dimensions → 5 stages)
1. **Mixed-domain questions** — 90 Q, 6 domain families × 15. Every role answers every question
   (15 in-domain + 75 OOD per role). → `data/questions.yaml`
2. **Role-adherence judging** — domain-aware 0–3 rubric; filters generic helpfulness, wrong-role
   drift, over-forced role-play. → `role-judge` skill / `rolevec.judge`
3. **Vector construction** — role-minus-default activations, score-2-plus weighting, balanced
   `v = v_in_domain + v_out_of_domain`. → `role-extract` skill / `rolevec.extract`
4. **Stability & separability** — 30 repeated runs; 5 metrics; success-criteria checks.
   → `vector-metrics` skill / `rolevec.metrics`

## Key formulas
- **Score-2-plus weighted vector:** `v_{r,l,s} = Σ_i w_i·h_i / Σ_i w_i`, `w=0` (score 0/1), `w=2` (score 2), `w=3` (score 3).
- **Final candidate:** `v̂_{r,l,s} = v_in_domain + v_out_of_domain` (both score-2-plus, role-minus-default).
- **Symmetric normalized separation:** `S_{a,b} = ½(‖v_a−v_b‖/‖v_a‖ + ‖v_a−v_b‖/‖v_b‖)`, averaged over layers 10–20 and 30 runs. `~0` identical · `~1` distance≈length · `>1` well-separated.
- **Stability margin:** `M_r = C_same − C_cross` (avg same-role cosine minus avg cosine to other roles).

## Roles (6 reported + 1 baseline)
`default` is the subtraction baseline (removed from every vector, never reported). Reported roles span
heterogeneous personas plus two deliberate close contrasts. See `data/roles.yaml`.

## Repo layout
- `src/rolevec/` — core library. `backends.py` is **model-agnostic**: a `DummyBackend` runs the whole
  pipeline today (synthetic activations); plug in `TransformerLens`/`nnsight`/HF later by setting a config flag.
  `steer.py` = Future Work #5 injection+persistence; `run_all.py` = one-command multi-track run + validation gate.
- `.claude/skills/` — `lit-scan`, `role-extract`, `role-judge`, `vector-metrics`, `paper-draft`,
  `steer-inject` (#5), `run-all` (full pipeline + gate).
- `data/` — `roles.yaml`/`questions.yaml` (occupational); `roles_bigfive.yaml`/`questions_bigfive.yaml`
  (Future Work #2/#3/#4 persona track). `docs/auto-research-survey.md` (survey), `docs/future-work.md` (roadmap).

## Run everything (with validation gate)
```bash
PYTHONPATH=src python -m rolevec.run_all --quick      # all 5 future-work tracks, self-validating
```
Tracks: occupational · cross-model replication (#1) · Big Five personas (#2/#3/#4) · steering
persistence (#5). Writes `runs/report.md`; exits non-zero if any required check fails.
**Dummy numbers are synthetic** (the dummy backend de-emphasizes the baseline center so roles come
out separable, matching the deck) — they verify the *plumbing + validation*, not real model behavior.

## Conventions
- **Model choice is deferred.** Never hard-code a model id; read `config.MODEL` / `--model`. Default backend is `dummy`.
- Activations are captured per layer; the canonical analysis band is **layers 10–20**.
- Keep metrics pure-numpy and model-free so they stay unit-testable without a GPU.
