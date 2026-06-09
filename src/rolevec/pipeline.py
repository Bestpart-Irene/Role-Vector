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

import numpy as np
import yaml

from .backends import get_backend
from .config import Config
from .data import load_questions, load_roles
from .extract import extract_role_vector


def band_average(per_layer: dict[int, np.ndarray], layers) -> np.ndarray:
    return np.mean([per_layer[l] for l in layers], axis=0)


def run_pipeline(cfg: Config) -> "Path":  # type: ignore[name-defined]
    baseline, roles = load_roles(cfg.roles_path)
    questions = load_questions(cfg.questions_path)
    dims_by_family = {
        f["id"]: f.get("judge_dimensions", [])
        for f in yaml.safe_load(cfg.questions_path.read_text())["families"]
    }
    backend = get_backend(cfg)
    from .judge import HeuristicJudge, LLMJudge
    judge = HeuristicJudge() if cfg.backend == "dummy" else LLMJudge(cfg.model)

    print(f"backend={cfg.backend} model={cfg.require_model()} "
          f"roles={len(roles)} questions={len(questions)} runs={cfg.runs}")

    arrays: dict[str, np.ndarray] = {}
    s3_in: dict[str, list[float]] = {r.id: [] for r in roles}
    s3_ood: dict[str, list[float]] = {r.id: [] for r in roles}
    for run in range(cfg.runs):
        rcfg = Config(**{**cfg.__dict__, "seed": cfg.seed + run})
        backend = get_backend(rcfg)
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

    out = cfg.runs_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
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
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    return Config(backend=a.backend, model=a.model, runs=a.runs, seed=a.seed)


if __name__ == "__main__":
    run_pipeline(build_config())
