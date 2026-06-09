"""Future Work #5: steering injection + multi-turn persistence.

The deck shows PROMPTED-only personas drift toward the model default in 3-9 turns. The open question:
if we inject a *validated* role vector into the residual stream during generation, does the role
PERSIST across a multi-turn conversation while staying coherent?

This module evaluates that. The real intervention is a backend hook (nnsight: add `coeff * v_r` to
layer-l output during generation — see `NNSightBackend`). Until a model is wired, `simulate=True`
runs a transparent simulation that encodes the deck's empirical pattern:
  - prompted-only role score decays geometrically over turns (drift),
  - steered role score stays roughly flat (the hypothesis under test),
so the persistence metric + validation gate are exercised end-to-end with no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PersistenceResult:
    turns: int
    coeff: float
    prompted_scores: list[float]   # role score per turn, prompted-only baseline
    steered_scores: list[float]    # role score per turn, with vector injected
    prompted_final: float
    steered_final: float
    persistence_gain: float        # steered_final - prompted_final
    passed: bool                   # steered stays high AND beats prompted drift

    def summary(self) -> str:
        return (f"coeff={self.coeff:g} turns={self.turns} | "
                f"prompted {self.prompted_scores[0]:.2f}->{self.prompted_final:.2f} | "
                f"steered {self.steered_scores[0]:.2f}->{self.steered_final:.2f} | "
                f"gain={self.persistence_gain:+.2f} | {'PASS' if self.passed else 'FAIL'}")


def evaluate_persistence(
    *, role_id: str, turns: int = 9, coeff: float = 8.0,
    simulate: bool = True, backend=None, role_vector=None,
    floor: float = 2.0, seed: int = 0,
) -> PersistenceResult:
    """Measure whether injected role behavior persists across `turns` (rubric scale 0-3).

    PASS criterion: steered role score stays >= `floor` at the final turn AND exceeds the
    prompted-only baseline (i.e., the vector counteracts the documented 3-9 turn drift).
    """
    if not simulate:
        if backend is None or role_vector is None:
            raise ValueError("non-simulated mode needs a backend with generate_steered() and a role_vector")
        return _evaluate_real(role_id, turns, coeff, backend, role_vector, floor)

    rng = np.random.default_rng(abs(hash((role_id, seed))) % (2**32))
    s0 = 3.0
    decay = 0.78                       # prompted drift per turn (within the deck's 3-9 turn window)
    prompted, steered = [], []
    for t in range(turns):
        prompted.append(float(np.clip(s0 * (decay ** t) + rng.normal(0, 0.08), 0, 3)))
        # injected vector holds the role near the top; effect scales with coeff, mild noise
        hold = 3.0 * min(1.0, coeff / 6.0)
        steered.append(float(np.clip(hold + rng.normal(0, 0.06), 0, 3)))

    p_final, s_final = prompted[-1], steered[-1]
    passed = (s_final >= floor) and (s_final > p_final)
    return PersistenceResult(
        turns=turns, coeff=coeff, prompted_scores=prompted, steered_scores=steered,
        prompted_final=p_final, steered_final=s_final,
        persistence_gain=s_final - p_final, passed=passed,
    )


def _evaluate_real(role_id, turns, coeff, backend, role_vector, floor):  # pragma: no cover - integration
    """Real eval: drive a multi-turn chat with `backend.generate_steered(..., role_vector, coeff)`
    and a judge, collecting per-turn role scores. Implement once a model + judge are wired."""
    raise NotImplementedError(
        "Wire backend.generate_steered() (nnsight intervention) + a judge, then collect per-turn "
        "role scores here. Until then call with simulate=True."
    )
