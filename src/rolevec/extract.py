"""Vector construction (Prez slides 24-25).

For one role and one run (seed s), per layer l:
  - generate the role's answer to each of the 90 questions
  - judge each answer 0-3 -> score-2-plus weight w_i
  - role-minus-default activation per answer: d_i = h_role(answer_i) - h_default(answer_i)
  - weighted mean, split by domain:
        v_in_domain  = Σ_{i in-domain}  w_i d_i / Σ w_i
        v_out_domain = Σ_{i out-domain} w_i d_i / Σ w_i
  - final candidate (balanced): v = v_in_domain + v_out_domain

Construction variants kept for Q2 comparison:
  raw  : weighted mean of h_role only (no default subtraction)
  rmd  : weighted mean of (h_role - h_default)             [single pooled]
  balanced : v_in_domain + v_out_domain (role-minus-default)  [reported]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backends import ActivationBackend
from .data import Question, Role
from .judge import Judge, weight_for


@dataclass
class RoleVector:
    role_id: str
    run: int
    # per construction -> per layer -> vector
    vectors: dict[str, dict[int, np.ndarray]]
    score3_rate_in: float
    score3_rate_ood: float


def _wmean(vecs: list[np.ndarray], weights: list[float], dim: int) -> np.ndarray:
    wsum = sum(weights)
    if wsum == 0:
        return np.zeros(dim)
    return np.sum([w * v for w, v in zip(weights, vecs)], axis=0) / wsum


def extract_role_vector(
    *, backend: ActivationBackend, judge: Judge, role: Role, baseline: Role,
    questions: list[Question], dimensions_by_family: dict[str, list[str]], run: int,
) -> RoleVector:
    dim = backend.hidden_dim
    layers = list(backend.cfg.layers)

    # 1) generate all answers for this role in ONE batched call (the bottleneck — batch it).
    qtexts = [q.text for q in questions]
    answers = backend.generate_batch(role.prompt, qtexts)

    # 2) judge all answers in one batch.
    in_flags = [q.in_domain_role == role.id for q in questions]
    items = [dict(role_name=role.name, role_prompt=role.prompt, question=q.text, answer=a,
                  in_domain=ind, dimensions=dimensions_by_family.get(q.family, []))
             for q, a, ind in zip(questions, answers, in_flags)]
    scores = judge.score_batch(items)

    # 3) extract activations for (role, answer) and (default, answer), batched.
    qa = list(zip(qtexts, answers))
    h_role_all = backend.hidden_states_batch(role.prompt, qa)
    h_def_all = backend.hidden_states_batch(baseline.prompt, qa)

    rows = []
    s3_in = s3_n_in = s3_ood = s3_n_ood = 0
    for in_domain, score, h_role, h_def in zip(in_flags, scores, h_role_all, h_def_all):
        rows.append((in_domain, weight_for(score), h_role, h_def))
        if in_domain:
            s3_n_in += 1; s3_in += int(score == 3)
        else:
            s3_n_ood += 1; s3_ood += int(score == 3)

    vectors: dict[str, dict[int, np.ndarray]] = {"raw": {}, "rmd": {}, "balanced": {}}
    for l in layers:
        raw_v = [r[2][l] for r in rows]
        rmd_v = [r[2][l] - r[3][l] for r in rows]
        w_all = [r[1] for r in rows]
        in_mask = [r[0] for r in rows]

        vectors["raw"][l] = _wmean(raw_v, w_all, dim)
        vectors["rmd"][l] = _wmean(rmd_v, w_all, dim)
        v_in = _wmean([v for v, m in zip(rmd_v, in_mask) if m],
                      [w for w, m in zip(w_all, in_mask) if m], dim)
        v_ood = _wmean([v for v, m in zip(rmd_v, in_mask) if not m],
                       [w for w, m in zip(w_all, in_mask) if not m], dim)
        vectors["balanced"][l] = v_in + v_ood

    return RoleVector(
        role_id=role.id, run=run, vectors=vectors,
        score3_rate_in=s3_in / max(s3_n_in, 1),
        score3_rate_ood=s3_ood / max(s3_n_ood, 1),
    )
