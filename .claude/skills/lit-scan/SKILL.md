---
name: lit-scan
description: Scan arXiv / GitHub / OpenReview for role-vector, persona-vector, activation-steering, and representation-engineering work, then map each finding to this project's four sub-questions (Q1 domain-sensitivity, Q2 construction, Q3 stability, Q4 separability). Use when the user asks to find related work, build a related-work table, update the survey, or check whether a method/paper is already covered. Produces a cited table, not raw link dumps.
---

# lit-scan — related-work scanning for Role Vector Validation

## When to run
The user wants related work found, compared, or positioned against this project — e.g. "find papers
on persona vectors", "is X already covered?", "update the survey", "who's nearest to our method?".

## Inputs
- Optional query/topic. Default seed terms: `role vector`, `persona vector`, `activation steering`,
  `representation engineering`, `difference-in-means steering`, `LLM ABM personas`, `construct validation LLM`.

## Procedure
1. Search arXiv, GitHub, OpenReview, lab blogs (WebSearch). Prefer 2024-2026.
2. For each hit capture: title, venue/source, URL, one-line method, and **which sub-question it touches**.
3. Flag the *nearest prior work* (currently "Designing Role Vectors", arXiv 2502.12055) and any method
   that already does our construction/metrics — we must cite and differentiate.
4. Cross-check against `docs/auto-research-survey.md`; append new rows, don't duplicate.

## Output (always this shape)
A markdown table — `| Work | Source/URL | Method (1 line) | Sub-Q | Note vs ours |` — followed by a
2-3 sentence "positioning" paragraph. Offer to write it into `docs/auto-research-survey.md`.

## Guardrails
- Output plausibility ≠ validation: tag whether a paper validates at output level or activation level.
- No raw link dumps; every link gets a method + sub-question mapping.
- End with a `Sources:` list of URLs used.
