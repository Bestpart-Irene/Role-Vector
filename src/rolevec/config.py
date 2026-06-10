"""Central, model-agnostic configuration.

Nothing here hard-codes a model. `MODEL` is None by default; the dummy backend ignores it and
real backends require it to be set (via --model or env ROLEVEC_MODEL).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"

# Canonical analysis band (Prez: separation averaged over layers 10-20).
ANALYSIS_LAYERS = tuple(range(10, 21))

# Score -> vector weight (score-2-plus policy). 0/1 excluded.
SCORE_WEIGHTS = {0: 0, 1: 0, 2: 2, 3: 3}


@dataclass
class Config:
    backend: str = "dummy"                 # dummy | transformer_lens | nnsight | hf
    model: str | None = field(default_factory=lambda: os.environ.get("ROLEVEC_MODEL"))
    # Judge is SEPARATE from the extraction model (avoid self-grading bias).
    #   judge_backend: local (FREE, open-weight via transformers) | anthropic (paid Claude) | heuristic
    #   judge_model:   resolved per backend if left None (local->Qwen instruct, anthropic->Haiku)
    judge_backend: str = field(
        default_factory=lambda: os.environ.get("ROLEVEC_JUDGE_BACKEND", "local"))
    judge_model: str | None = field(default_factory=lambda: os.environ.get("ROLEVEC_JUDGE_MODEL"))
    baseline_prompt: str | None = None     # set by the pipeline; dummy backend de-emphasizes it
    layers: tuple[int, ...] = ANALYSIS_LAYERS
    runs: int = 30                         # repeated extractions for stability (Q3)
    hidden_dim: int = 256                  # dummy backend only; real backends infer from model
    seed: int = 0
    roles_path: Path = DATA_DIR / "roles.yaml"
    questions_path: Path = DATA_DIR / "questions.yaml"
    runs_dir: Path = RUNS_DIR

    def require_model(self) -> str:
        if self.backend != "dummy" and not self.model:
            raise ValueError(
                f"backend={self.backend!r} needs a model id. "
                "Pass --model <id> or set ROLEVEC_MODEL. (Model choice is deferred by default.)"
            )
        return self.model or "dummy"
