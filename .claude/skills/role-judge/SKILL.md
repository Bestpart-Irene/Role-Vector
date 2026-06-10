---
name: role-judge
description: Apply the domain-aware 0-3 role-adherence rubric to model answers — scores how well an answer expresses the TARGET ROLE (not how helpful it is), filtering generic helpfulness, wrong-role drift, and over-forced role-play. Scores map to score-2-plus vector weights. Use when the user wants to judge/score answers, calibrate the rubric, wire up the judge model, or inspect score distributions as quality control.
---

# role-judge — role-adherence scoring (Prez slides 22-23)

## When to run
"score these answers", "set up the judge", "why is this a 2 not a 3", "check the score distribution".

## Rubric (0-3) → weight
| Score | Meaning | Weight |
|---|---|---|
| 3 Strong | natural role expression; in-domain uses domain reasoning, OOD uses role's characteristic habits | 3 |
| 2 Partial | visible role expression, possibly incomplete/generic | 2 |
| 1 Weak | generic / overly cautious / marginally wrong framing | 0 (excluded) |
| 0 Unusable | wrong role, caricature, or no role signal | 0 (excluded) |

Generic traits (clarity, safety, escalation) only count when expressed through the role's
characteristic habits — never as generic-assistant behavior.

## How (`src/rolevec/judge.py`)
Three pluggable judges, selected by `--judge-backend` / `ROLEVEC_JUDGE_BACKEND`:
- **`local` (default, FREE):** `LocalJudge` — open-weight instruct model via HF transformers, no API
  key, runs on cluster GPU. `ROLEVEC_JUDGE_MODEL=Qwen/Qwen2.5-7B-Instruct` (any HF instruct model).
- **`anthropic` (paid):** `LLMJudge` — Claude API; set `ANTHROPIC_API_KEY`, model `claude-haiku-4-5`.
- **`heuristic`:** offline deterministic, wiring/tests only (auto-used by the dummy backend).

`LocalJudge`/`LLMJudge` share `PromptJudge.score` (builds the rubric prompt, parses the 0-3 integer).
Score distributions per (role × family) are quality control: if in-domain isn't skewing high, the
questions/rubric for that family are misaligned — fix before trusting the vectors.

## Guardrails
- Score ROLE expression, not answer quality. A correct-but-generic answer is a 1, not a 3.
- Keep the judge model separate from the extraction model to avoid self-grading bias.
