"""Role-adherence judging — domain-aware 0-3 rubric (Prez slides 22-23).

The judge filters generic helpfulness, wrong-role drift, and over-forced role-play. Scores map to
vector weights via the score-2-plus policy (config.SCORE_WEIGHTS): 3->3, 2->2, 0/1->0 (excluded).

Three judges, all model-agnostic at the interface:
  HeuristicJudge  offline, deterministic — wiring/tests only (used by the dummy backend).
  LocalJudge      FREE open-weight judge via HF transformers (runs on your GPU / SLURM cluster).
  LLMJudge        Claude via the Anthropic API (needs ANTHROPIC_API_KEY; paid).
Keep the judge model SEPARATE from the extraction model to avoid self-grading bias.
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

    def score_batch(self, items: list[dict]) -> list[int]:
        """Score many (role_name, role_prompt, question, answer, in_domain, dimensions) dicts.
        Default loops `score`; PromptJudge overrides with a batched model call."""
        return [self.score(**it) for it in items]


class HeuristicJudge(Judge):
    """Offline placeholder: deterministic pseudo-scores for pipeline wiring. NOT for real results."""

    def score(self, *, role_name, role_prompt, question, answer, in_domain, dimensions) -> int:
        h = (abs(hash((role_name, question, answer))) % 100) / 100.0
        base = h + (0.15 if in_domain else 0.0)        # in-domain skews higher, like real data
        return 3 if base > 0.75 else 2 if base > 0.45 else 1 if base > 0.2 else 0


class PromptJudge(Judge):
    """Shared 0-3 scoring: build the rubric prompt, call a model, parse the integer.
    Subclasses implement `_call_model(prompt) -> text` (and may override `_call_model_batch`)."""

    @abstractmethod
    def _call_model(self, prompt: str) -> str: ...

    def _call_model_batch(self, prompts: list[str]) -> list[str]:
        return [self._call_model(p) for p in prompts]

    @staticmethod
    def _build_prompt(role_name, role_prompt, question, answer, in_domain, dimensions) -> str:
        return JUDGE_PROMPT.format(
            role_name=role_name, role_prompt=role_prompt, question=question, answer=answer,
            domain="IN-DOMAIN" if in_domain else "OUT-OF-DOMAIN",
            dimensions=", ".join(dimensions) or "the role's characteristic habits",
            domain_clause="uses domain reasoning" if in_domain
            else "uses the role's characteristic habits in an unfamiliar context",
        )

    @staticmethod
    def _parse(reply: str) -> int:
        m = re.search(r"[0-3]", reply)
        return int(m.group()) if m else 0

    def score(self, *, role_name, role_prompt, question, answer, in_domain, dimensions) -> int:
        return self._parse(self._call_model(
            self._build_prompt(role_name, role_prompt, question, answer, in_domain, dimensions)))

    def score_batch(self, items: list[dict]) -> list[int]:
        prompts = [self._build_prompt(**it) for it in items]
        return [self._parse(r) for r in self._call_model_batch(prompts)]


class LocalJudge(PromptJudge):
    """FREE open-weight judge via HF transformers — no API key, runs on your GPU / SLURM cluster.

    Default model is a small instruct model; override with `model=` or ROLEVEC_JUDGE_MODEL. Greedy
    decode, a handful of tokens (we only need the score)."""

    def __init__(self, model: str | None = None, max_new_tokens: int = 4):
        self.model_id = model or "Qwen/Qwen2.5-7B-Instruct"
        self.max_new_tokens = max_new_tokens
        self._pipe = None

    def _pipe_lazy(self):
        if self._pipe is None:
            from transformers import pipeline  # lazy: only needed for real judging
            self._pipe = pipeline(
                "text-generation", model=self.model_id, torch_dtype="auto", device_map="auto",
            )
        return self._pipe

    def _call_model(self, prompt: str) -> str:
        return self._call_model_batch([prompt])[0]

    def _call_model_batch(self, prompts: list[str]) -> list[str]:
        msgs = [[{"role": "user", "content": p}] for p in prompts]
        outs = self._pipe_lazy()(
            msgs, max_new_tokens=self.max_new_tokens, do_sample=False,
            return_full_text=False, batch_size=min(len(msgs), 16),
        )
        # transformers pipeline returns a list per input
        return [(o[0] if isinstance(o, list) else o)["generated_text"] for o in outs]


class LLMJudge(PromptJudge):
    """Claude via the Anthropic API. Needs `pip install anthropic` + ANTHROPIC_API_KEY. Paid.
    Override the model with `model=` or ROLEVEC_JUDGE_MODEL (Haiku is the cheap high-volume default)."""

    def __init__(self, model: str | None = None):
        self.model = model or "claude-haiku-4-5"
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            from anthropic import Anthropic  # lazy
            self._client = Anthropic()
        return self._client

    def _call_model(self, prompt: str) -> str:
        msg = self._client_lazy().messages.create(
            model=self.model, max_tokens=5,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
