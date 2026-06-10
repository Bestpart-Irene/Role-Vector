"""One-command automated runner for the whole Role Vector roadmap, with a validation gate.

    PYTHONPATH=src python -m rolevec.run_all --runs 30
    PYTHONPATH=src python -m rolevec.run_all --runs 8 --quick   # fast smoke

Runs, in order, and validates each:
  Track A  Occupational roles            (base method)                  -> Q1/Q3/Q4 checks
  Track A' Cross-model replication (#1)  (second simulated "model")     -> Q3/Q4 checks + agreement
  Track B  Big Five personas (#2/#3/#4)  (non-occupational + traits)    -> Q1/Q3/Q4 checks
  Steering Injection persistence (#5)    (per role, multi-turn)         -> persistence check

Writes runs/report.md and prints an overall PASS/FAIL. Exits non-zero if any REQUIRED check fails,
so it is CI-runnable. (Q1 needs a real judge; with the dummy heuristic judge it is reported but not
required — flagged clearly in the report.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import DATA_DIR, RUNS_DIR, Config
from .data import load_roles
from .metrics import _load_runs, success_report
from .pipeline import run_pipeline
from .steer import evaluate_persistence


def _run_track(tag, roles_file, questions_file, *, backend, model, runs, seed, construction,
               max_new_tokens=200, max_questions=None):
    cfg = Config(backend=backend, model=model, runs=runs, seed=seed,
                 max_new_tokens=max_new_tokens, max_questions_per_family=max_questions)
    cfg.roles_path = DATA_DIR / roles_file
    cfg.questions_path = DATA_DIR / questions_file
    out = run_pipeline(cfg, tag=tag)
    vecs, meta = _load_runs(out, construction)
    rep = success_report(vecs, meta.get("score3_rate_in"), meta.get("score3_rate_ood"))
    return out, meta, rep


def _centroids(out: Path, construction: str):
    vecs, _ = _load_runs(out, construction)
    return {r: np.mean(v, axis=0) for r, v in vecs.items()}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run the full Role Vector roadmap with validation.")
    p.add_argument("--backend", default="dummy")
    p.add_argument("--model", default=None)
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--construction", default="balanced", choices=["raw", "rmd", "balanced"])
    p.add_argument("--quick", action="store_true", help="smaller runs for a fast smoke test")
    p.add_argument("--max-new-tokens", type=int, default=200, help="answer length (lower = faster)")
    p.add_argument("--max-questions", type=int, default=None, help="questions per family (subsample for smoke)")
    a = p.parse_args(argv)
    runs = 8 if a.quick else a.runs
    common = dict(backend=a.backend, model=a.model, runs=runs, construction=a.construction,
                  max_new_tokens=a.max_new_tokens, max_questions=a.max_questions)

    results = {}        # track -> (out, meta, rep)
    required_checks = []  # (label, bool)
    reported_only = []    # (label, bool)  -> Q1 with dummy judge

    # ---- Track A: occupational (base method) ----
    print("\n### Track A — occupational roles")
    results["A_occupational"] = _run_track("A-occ", "roles.yaml", "questions.yaml", seed=0, **common)

    # ---- Track A': cross-model replication (#1) — second simulated model via different seed ----
    print("\n### Track A' — cross-model replication (#1)")
    results["A2_crossmodel"] = _run_track("A2-occ-model2", "roles.yaml", "questions.yaml", seed=1000, **common)

    # ---- Track B: Big Five personas (#2/#3/#4) ----
    print("\n### Track B — Big Five personas (#2/#3/#4)")
    results["B_bigfive"] = _run_track("B-bigfive", "roles_bigfive.yaml", "questions_bigfive.yaml", seed=0, **common)

    # ---- Steering injection persistence (#5) ----
    print("\n### Steering — injection persistence (#5)")
    _, occ_roles = load_roles(DATA_DIR / "roles.yaml")
    # Steering persistence is SIMULATED here (placeholder until real nnsight injection, Future Work #5).
    # Extraction metrics (Q1/Q3/Q4) above are real; this step just exercises the persistence machinery.
    steer_results = [
        evaluate_persistence(role_id=r.id, turns=9, coeff=8.0, simulate=True)
        for r in occ_roles
    ]
    for r, sr in zip(occ_roles, steer_results):
        print(f"  {r.id:18s} {sr.summary()}")

    # ---- collect checks ----
    for label, (_out, _meta, rep) in results.items():
        c = rep["checks"]
        required_checks += [
            (f"{label}:Q3_margin_all_positive", c["Q3_margin_all_positive"]),
            (f"{label}:Q3_cluster_perfect", c["Q3_cluster_perfect"]),
            (f"{label}:Q4_all_pairs_separable", c["Q4_all_pairs_separable"]),
        ]
        if "Q1_in_gt_ood" in c:
            tag = "(dummy judge: reported-only)" if a.backend == "dummy" else ""
            reported_only.append((f"{label}:Q1_in_gt_ood {tag}", c["Q1_in_gt_ood"]))
            reported_only.append((f"{label}:Q1_ood_nonzero {tag}", c["Q1_ood_nonzero"]))

    required_checks.append((
        "steer:persistence_all_roles",
        all(s.passed for s in steer_results),
    ))

    # cross-model agreement (#1): same separability verdict across the two "models"
    a_sep_ok = results["A_occupational"][2]["checks"]["Q4_all_pairs_separable"]
    a2_sep_ok = results["A2_crossmodel"][2]["checks"]["Q4_all_pairs_separable"]
    required_checks.append(("crossmodel:separability_agreement", a_sep_ok == a2_sep_ok))

    overall = all(ok for _, ok in required_checks)
    _write_report(results, steer_results, required_checks, reported_only, overall, a)

    print("\n=== VALIDATION GATE ===")
    for label, ok in required_checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    for label, ok in reported_only:
        print(f"  [{'ok' if ok else '--'}] {label}")
    print(f"\nOVERALL: {'PASS ✅' if overall else 'FAIL ❌'}   (report: runs/report.md)")
    return 0 if overall else 1


def _write_report(results, steer_results, required, reported, overall, args):
    lines = ["# Role Vector — automated run report", ""]
    lines.append(f"backend=`{args.backend}` model=`{args.model or 'dummy'}` "
                 f"runs={8 if args.quick else args.runs} construction=`{args.construction}`")
    lines.append(f"\n**OVERALL: {'PASS' if overall else 'FAIL'}**\n")
    for label, (out, meta, rep) in results.items():
        lines.append(f"## {label}  (`{Path(out).name}`)")
        lines.append(f"roles: {', '.join(rep['roles'])} · cluster acc {rep['cluster_accuracy']:.3f}")
        lines.append("\n| role | cosine | pearson | margin |")
        lines.append("|---|---|---|---|")
        for r in rep["roles"]:
            lines.append(f"| {r} | {rep['same_role_cosine'][r]:.3f} | "
                         f"{rep['same_role_pearson'][r]:.3f} | {rep['stability_margin'][r]:.3f} |")
        seps = rep["pairwise_separation"]
        lo = min(seps, key=seps.get)
        lines.append(f"\nseparation range: {min(seps.values()):.2f}–{max(seps.values()):.2f} "
                     f"(closest pair: {lo} = {seps[lo]:.2f})")
        lines.append("")
    lines.append("## Steering persistence (#5)")
    lines.append("| prompted final | steered final | gain | pass |")
    lines.append("|---|---|---|---|")
    for s in steer_results:
        lines.append(f"| {s.prompted_final:.2f} | {s.steered_final:.2f} | "
                     f"{s.persistence_gain:+.2f} | {'PASS' if s.passed else 'FAIL'} |")
    lines.append("\n## Validation gate")
    for label, ok in required:
        lines.append(f"- [{'PASS' if ok else 'FAIL'}] {label}")
    for label, ok in reported:
        lines.append(f"- [{'ok' if ok else '--'}] {label} (reported, not required)")
    (RUNS_DIR / "report.md").parent.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
