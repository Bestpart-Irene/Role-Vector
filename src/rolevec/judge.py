"""Role-adherence judging — domain-aware 0-3 rubric (Prez slides 22-23).

The judge filters generic helpfulness, wrong-role drift, and over-forced role-play. Scores map to
vector weights via the score-2-plus policy (config.SCORE_WEIGHTS): 3->3, 2->2, 0/1->0 (excluded).

`Judge` is model-agnostic. `HeuristicJudge` runs offline for wiring/tests. `LLMJudge` is the real
one: implement `_call_model` against your chosen judge model (deferred).
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

from .config import SCORE_WEIGHTS

RUBRIC = {
    3: "Strong, natural, context-appropriate role expression.",
    2: "Partial: visible role expression, possibly incomplete/generic. Usable.",
    1: "Weak/generic role signal or marginally wrong framing. Excluded.",
    0: "Unusable: wrong role, caricature, or no role signal. Excluded.",
}

JUDGE_PROMPT = """You score how well an answer expresses a target ROLE, not how helpful it is.
Role: {role_name} — {role_prompt}
This question is {domain} for this role. Behavioral dimensions to look for: {dimensions}.

Question: {question}
Answer: {answer}

Score 0-3 using this rubric (penalize generic-assistant helpfulness, wrong-role drift, and
over-forced caricature):
3 = strong, natural role expression ({domain_clause}).
2 = partial but visible role expression.
1 = weak/generic or marginally wrong framing.
0 = wrong role, caricature, or no role signal.
Reply with ONLY the integer score."""


def weight_for(score: int) -> int:
    return SCORE_WEIGHTS.get(int(score), 0)


class Judge(ABC):
    @abstractmethod
    def score(self, *, role_name: str, role_prompt: str, question: str, answer: str,
              in_domain: bool, dimensions: list[str]) -> int: ...


class HeuristicJudge(Judge):
    """Offline placeholder: deterministic pseudo-scores for pipeline wiring. NOT for real results."""

    def score(self, *, role_name, role_prompt, question, answer, in_domain, dimensions) -> int:
        h = (abs(hash((role_name, question, answer))) % 100) / 100.0
        base = h + (0.15 if in_domain else 0.0)        # in-domain skews higher, like real data
        return 3 if base > 0.75 else 2 if base > 0.45 else 1 if base > 0.2 else 0


class LLMJudge(Judge):
    """Real judge. Implement `_call_model` against your chosen judge model (deferred)."""

    def __init__(self, model: str | None = None):
        self.model = model

    def _call_model(self, prompt: str) -> str:  # pragma: no cover - integration point
        raise NotImplementedError(
            "Implement _call_model() to query your judge model and return its text reply."
        )

    def score(self, *, role_name, role_prompt, question, answer, in_domain, dimensions) -> int:
        prompt = JUDGE_PROMPT.format(
            role_name=role_name, role_prompt=role_prompt, question=question, answer=answer,
            domain="IN-DOMAIN" if in_domain else "OUT-OF-DOMAIN",
            dimensions=", ".join(dimensions) or "the role's characteristic habits",
            domain_clause="uses domain reasoning" if in_domain
            else "uses the role's characteristic habits in an unfamiliar context",
        )
        reply = self._call_model(prompt)
        m = re.search(r"[0-3]", reply)
        return int(m.group()) if m else 0
