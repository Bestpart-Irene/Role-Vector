"""End-to-end driver: extract role vectors across roles x runs, save to runs/, print a summary.

    python -m rolevec.pipeline --backend dummy --runs 30
    python -m rolevec.pipeline --backend transformer_lens --model <hf-id> --runs 30 --construction balanced

Outputs runs/<timestamp>/ (and a `runs/latest` symlink): vectors.npz + meta.json.
Then run `python -m rolevec.metrics --runs-dir runs/latest` for the metric report.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from .backends import get_backend
from .config import Config
from .data import load_questions, load_roles
from .extract import extract_role_vector


def band_average(per_layer: dict[int, np.ndarray], layers) -> np.ndarray:
    return np.mean([per_layer[l] for l in layers], axis=0)


def run_pipeline(cfg: Config, tag: str = "") -> "Path":  # type: ignore[name-defined]
    baseline, roles = load_roles(cfg.roles_path)
    cfg.baseline_prompt = baseline.prompt          # dummy backend de-emphasizes this center
    questions = load_questions(cfg.questions_path)
    if cfg.max_questions_per_family:               # fast-smoke subsample
        from collections import defaultdict
        by_fam: dict[str, list] = defaultdict(list)
        for q in questions:
            by_fam[q.family].append(q)
        questions = [q for fam in by_fam.values() for q in fam[:cfg.max_questions_per_family]]
    dims_by_family = {
        f["id"]: f.get("judge_dimensions", [])
        for f in yaml.safe_load(cfg.questions_path.read_text())["families"]
    }
    from .judge import HeuristicJudge, LLMJudge, LocalJudge
    if cfg.backend == "dummy" or cfg.judge_backend == "heuristic":
        judge = HeuristicJudge()
    elif cfg.judge_backend == "anthropic":
        judge = LLMJudge(cfg.judge_model)           # paid Claude
    else:
        judge = LocalJudge(cfg.judge_model)         # FREE open-weight (default)

    print(f"backend={cfg.backend} model={cfg.require_model()} "
          f"roles={len(roles)} questions={len(questions)} runs={cfg.runs}")

    # Load the backend (and its model) ONCE and reuse across runs. Reloading per run reloads the
    # whole model from disk every time — fine for the dummy backend, crippling for a real GPU model.
    # Run-to-run variation comes from sampled generation, not from reseeding the backend.
    backend = get_backend(cfg)

    arrays: dict[str, np.ndarray] = {}
    s3_in: dict[str, list[float]] = {r.id: [] for r in roles}
    s3_ood: dict[str, list[float]] = {r.id: [] for r in roles}
    for run in range(cfg.runs):
        for role in roles:
            rv = extract_role_vector(
                backend=backend, judge=judge, role=role, baseline=baseline,
                questions=questions, dimensions_by_family=dims_by_family, run=run,
            )
            for constr, per_layer in rv.vectors.items():
                key = f"{constr}|{role.id}|run{run}"
                arrays[key] = band_average(per_layer, cfg.layers)
            s3_in[role.id].append(rv.score3_rate_in)
            s3_ood[role.id].append(rv.score3_rate_ood)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out = cfg.runs_dir / (f"{tag}-{stamp}" if tag else stamp)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "vectors.npz", **arrays)
    meta = {
        "backend": cfg.backend, "model": cfg.require_model(), "runs": cfg.runs,
        "layers": list(cfg.layers), "constructions": ["raw", "rmd", "balanced"],
        "roles": [r.id for r in roles],
        "score3_rate_in": {r: float(np.mean(v)) for r, v in s3_in.items()},
        "score3_rate_ood": {r: float(np.mean(v)) for r, v in s3_ood.items()},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    latest = cfg.runs_dir / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    try:
        latest.symlink_to(out.name)
    except OSError:
        shutil.copytree(out, latest, dirs_exist_ok=True)
    print(f"saved -> {out}")
    return out


def build_config(argv=None) -> Config:
    p = argparse.ArgumentParser(description="Extract role vectors.")
    p.add_argument("--backend", default="dummy",
                   choices=["dummy", "transformer_lens", "nnsight"])
    p.add_argument("--model", default=None, help="HF model id (required for non-dummy backends)")
    p.add_argument("--judge-backend", default=None, choices=["local", "anthropic", "heuristic"],
                   help="local=FREE open-weight (default) | anthropic=paid Claude | heuristic")
    p.add_argument("--judge-model", default=None, help="judge model id (ROLEVEC_JUDGE_MODEL)")
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--roles", default=None, help="path to a roles.yaml (default: data/roles.yaml)")
    p.add_argument("--questions", default=None, help="path to a questions.yaml (default: data/questions.yaml)")
    a = p.parse_args(argv)
    cfg = Config(backend=a.backend, model=a.model, runs=a.runs, seed=a.seed)
    if a.judge_backend:
        cfg.judge_backend = a.judge_backend
    if a.judge_model:
        cfg.judge_model = a.judge_model
    if a.roles:
        cfg.roles_path = Path(a.roles)
    if a.questions:
        cfg.questions_path = Path(a.questions)
    return cfg


if __name__ == "__main__":
    run_pipeline(build_config())
