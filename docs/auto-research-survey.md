# Auto-Research Pipelines & Activation-Tooling — Survey

Survey of open-source auto-research pipelines and representation-engineering tooling, mapped to the
**Role Vector Validation** project. Compiled 2026-06-09.

## TL;DR
Full-pipeline "AI Scientist" systems are *too heavy* for this project — we do not need automated idea
generation; our method is already specified. What we actually integrate is:
- an **Agent-Laboratory–style 3-stage skeleton** (lit-review → experiment → report), realised as the
  5 skills under `.claude/skills/`, and
- an **activation-extraction layer** (TransformerLens / nnsight / vllm-lens) behind a model-agnostic
  backend interface.

## A. Full-pipeline autonomous-research systems

| System | Pipeline / setup | Borrowable for us |
|---|---|---|
| **The AI Scientist v1/v2** (Sakana) | idea → code experiment → run → LaTeX write-up → automated review; v2 adds agentic tree-search over hypotheses; ~$15/paper; v2 paper passed a workshop peer review | The **automated-reviewer** stage maps onto our 0–3 role-adherence judging |
| **Agent Laboratory** (Schmidgall, OSS) | 3 phases: Literature Review → Experimentation → Report Writing; autonomous + co-pilot modes; tools: arXiv/HF/Python/LaTeX | The **3-phase skeleton** is our skill split; co-pilot checkpoints = human-in-the-loop |
| **AgentRxiv** | preprint server for agents; results accumulate & compound (improves MATH-500 over iterations) | An **archive/reuse** store for our 30-run × multi-role outputs |
| **AI-Researcher / AutoAgent (HKUDS)** | zero-code, CLI-driven fully-automated agent framework | CLI orchestration patterns |
| **From AI for Science to Agentic Science** (survey, arXiv 2508.14111) | taxonomy of autonomous scientific discovery | Positioning / related-work framing |

⚠️ Caveat paper — **"The More You Automate, the Less You See"** (arXiv 2509.08713): hidden pitfalls of
AI-scientist systems (leakage, metric gaming). This is external backing for our core thesis: *output
plausibility is necessary but not sufficient — validate at the activation level.*

- The AI Scientist — https://sakana.ai/ai-scientist/
- Agent Laboratory — https://github.com/SamuelSchmidgall/AgentLaboratory · https://arxiv.org/pdf/2501.04227
- AgentRxiv — https://agentrxiv.github.io/ · https://arxiv.org/pdf/2503.18102
- AI-Researcher — https://arxiv.org/pdf/2505.18705
- AutoAgent (MetaChain) — https://github.com/HKUDS/AutoAgent
- Agentic Science survey — https://arxiv.org/pdf/2508.14111
- Hidden Pitfalls — https://arxiv.org/pdf/2509.08713

## B. Activation / representation-engineering tooling (our implementation layer)

| Tool | What it gives us | Fit |
|---|---|---|
| **Designing Role Vectors** (arXiv 2502.12055) | difference-in-means role vectors steering domain expertise while preserving general ability | Nearest-neighbour prior work — **must cite & compare** |
| **TransformerLens** | PyTorch hooks, `run_with_cache` → per-layer residual-stream activations | Primary extraction backend for white-box HF models |
| **nnsight + NDIF** ⭐ | nnsight extends PyTorch with deferred remote execution; **NDIF** (National Deep Inference Fabric) executes it on shared GPUs | **[redacted] home-team infra** — see callout below |
| **nnsight** | extraction + injection (intervention) phases; baukit successor | Role-minus-default intervention + steering validation |
| **vllm-lens** (UK AISI) | extract residual activations & apply steering vectors on vLLM | High-throughput for 30-run repeated extraction |
| **llm_steer / repeng** | lightweight steering-vector experimentation | Quick prototyping |
| **Persona Vectors / BILLY / Steering at the Source** | persona-trait directions; merging multiple persona vectors | Related work for Q4 separability |

- Designing Role Vectors — https://arxiv.org/pdf/2502.12055
- TransformerLens — https://github.com/TransformerLensOrg/TransformerLens
- nnsight — https://nnsight.net/
- vllm-lens — https://github.com/UKGovernmentBEIS/vllm-lens
- llm_steer — https://github.com/Mihaiii/llm_steer
- Persona Vectors (OpenReview) — https://openreview.net/forum?id=HpUDi5Pe8S
- BILLY — https://bai1026.github.io/LLM_Persona/

### ⭐ NDIF — National Deep Inference Fabric ([redacted] home-team infra)
NSF-funded (~$9M) inference fabric **at [redacted]** ([redacted]) giving U.S. researchers **free remote
access to open-weight model internals**, up to **Llama-3.1-405B**, via `nnsight`'s deferred remote
execution — no local GPU required. This is the single most relevant external resource for this project:
it lets us extract role-minus-default activations from large white-box models for free, and run all 30
repeated extractions at scale. Recommended path once a model is chosen: `nnsight` backend with
`remote=True` (set NDIF API key). Apply for pilot access as a [redacted] student.
- NDIF — https://ndif.us/
- nnsight + NDIF paper — https://ndif.us/ · https://openreview.net/forum?id=MxbEiFRf39
- GitHub — https://github.com/ndif-team
- nnsight — https://nnsight.net/

## C. Mapping external pieces → our sub-questions
- **Q1 (domain-sensitive signal):** TransformerLens per-layer cache + AI-Scientist-style automated judge.
- **Q2 (construction):** repeng / Designing-Role-Vectors difference-in-means vs our role-minus-default + score-2-plus.
- **Q3 (stability):** repeated extraction via vllm-lens throughput; metrics in `rolevec.metrics`.
- **Q4 (separability):** Persona-Vectors / BILLY separability framing; symmetric normalized separation as primary metric.

## D. Decision
- **Adopt:** Agent-Laboratory 3-stage skeleton (→ skills); TransformerLens **or** nnsight extraction behind one backend interface; AgentRxiv-style local results archive (`runs/`).
- **Defer:** full AI-Scientist idea-generation loop (not needed; method is fixed).
- **Watch:** "Hidden Pitfalls" failure modes as a validation checklist for our own reported metrics.
