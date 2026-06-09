# Role Vector Validation

A reusable construct-validation pipeline for LLM agent personas: extract prompted role personas into
activation-space vectors and validate that they are **domain-sensitive**, **stable**, and **separable**.

See [CLAUDE.md](CLAUDE.md) for the full method and [docs/auto-research-survey.md](docs/auto-research-survey.md)
for the external-ecosystem survey behind the design.

## Quickstart (no model required)
The default backend is `dummy` (random activations), so the whole extract→metrics pipeline runs today:

```bash
pip install -r requirements.txt
python -m rolevec.pipeline --backend dummy --runs 30      # extract + score all roles
python -m rolevec.metrics  --runs-dir runs/latest          # compute the 5 metrics + success checks
```

Swap in a real model later — nothing else changes:

```bash
python -m rolevec.pipeline --backend transformer_lens --model <hf-model-id> --runs 30
```

## Claude Code skills
Invoke inside Claude Code with `/`:
| Skill | Does |
|---|---|
| `lit-scan` | scan arXiv/GitHub for role/persona-vector work, map to sub-questions |
| `role-extract` | extract role-minus-default, score-2-plus weighted vectors |
| `role-judge` | apply the 0–3 role-adherence rubric (LLM judge) |
| `vector-metrics` | compute cosine / Pearson / stability margin / cluster / separation + success criteria |
| `paper-draft` | fill metric results into a LaTeX method/results/limitations draft |

## Layout
```
src/rolevec/      core library (model-agnostic backends)
data/             roles.yaml, questions.yaml
docs/             survey + notes
.claude/skills/   the 5 skills
tests/            metric unit tests
runs/             extraction outputs (gitignored)
```
