---
name: paper-draft
description: Turn a completed metrics run into a paper-ready write-up — fills the method, results, and limitations sections of a LaTeX/Markdown draft from runs/<ts>/meta.json and the metric report, with the success criteria and related-work positioning from the survey. Use when the user wants to draft/update the paper, write the results section, generate tables from a run, or assemble the methodological argument.
---

# paper-draft — report writing (Agent-Laboratory "Report Writing" stage)

## When to run
"draft the paper", "write the results section", "make the results table", "update the write-up from the latest run".

## Inputs
- `runs/latest/meta.json` + the `vector-metrics` report (run it first if stale).
- `docs/auto-research-survey.md` for related-work positioning.
- The fixed thesis (do not re-derive): *output plausibility is necessary but not sufficient;
  activation-level validation is an independent check on whether the intended role is represented
  consistently and distinctly inside the model.*

## Sections to produce
1. **Method** — roles (6 + default baseline), 90 mixed-domain questions, 0-3 judging, role-minus-default
   + score-2-plus weighting, balanced `v = v_in + v_ood`, layers 10-20, 30 runs.
2. **Results** — one table per sub-question:
   - Q1 score-3 rate in vs OOD per role.
   - Q2 separation by construction (raw/rmd/balanced) — show balanced wins.
   - Q3 same-role cosine/Pearson, stability margin, cluster accuracy.
   - Q4 the 15 pairwise separations, close-role pairs highlighted.
   Each table ends with the PASS/FAIL against its success criterion.
3. **Limitations** — cross-check against "Hidden Pitfalls" (arXiv 2509.08713): leakage, judge bias,
   single-model scope, question coverage (families at <15 questions).

## Guardrails
- Pull every number from the saved run; never invent results. State the model + run count used.
- Cite nearest prior work ("Designing Role Vectors", 2502.12055) and differentiate (we *validate* vectors as research instruments; they *use* them to steer).
- If any check is FAIL, report it plainly in Results — do not bury it.
