"""Stability & separability metrics (Prez slides 26 / 31-32). Pure numpy, model-free.

Inputs are role vectors at a fixed construction, organised as:
    vecs[role_id] -> list of per-run vectors (each a 1-D array; analysis-band-averaged).

Metrics:
  same_role_cosine(role)        directional reproducibility across runs        [-1, 1], ~1 good
  same_role_pearson(role)       coordinate-level pattern reproducibility       [-1, 1], ~1 good
  stability_margin(role)        C_same - C_cross                               > 0 good
  cluster_assignment()          each run-vector nearest to its own centroid    accuracy, chance=1/R
  separation(a, b)              symmetric normalized separation                ~0 same .. >1 separated

Plus `success_report()` -> pass/fail against the Prez success criteria (Q1-Q4).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    da, db = np.linalg.norm(a), np.linalg.norm(b)
    if da == 0 or db == 0:
        return 0.0
    return float(np.dot(a, b) / (da * db))


def same_role_cosine(runs: list[np.ndarray]) -> float:
    """Mean pairwise cosine across the role's repeated extractions."""
    pairs = [_cos(x, y) for x, y in combinations(runs, 2)]
    return float(np.mean(pairs)) if pairs else 0.0


def same_role_pearson(runs: list[np.ndarray]) -> float:
    pairs = [_pearson(x, y) for x, y in combinations(runs, 2)]
    return float(np.mean(pairs)) if pairs else 0.0


def stability_margin(role: str, vecs: dict[str, list[np.ndarray]]) -> float:
    """M_r = C_same - C_cross. Positive rules out a generic shared direction."""
    c_same = same_role_cosine(vecs[role])
    cross = [_cos(x, y) for other, lst in vecs.items() if other != role
             for x in vecs[role] for y in lst]
    c_cross = float(np.mean(cross)) if cross else 0.0
    return c_same - c_cross


def cluster_assignment(vecs: dict[str, list[np.ndarray]]) -> float:
    """Fraction of run-vectors whose nearest role-centroid is their own role. Chance = 1/n_roles."""
    centroids = {r: np.mean(lst, axis=0) for r, lst in vecs.items()}
    correct = total = 0
    for role, lst in vecs.items():
        for v in lst:
            nearest = min(centroids, key=lambda r: np.linalg.norm(v - centroids[r]))
            correct += int(nearest == role); total += 1
    return correct / max(total, 1)


def separation(v_a: np.ndarray, v_b: np.ndarray) -> float:
    """Symmetric normalized separation: ½(‖a-b‖/‖a‖ + ‖a-b‖/‖b‖). Primary separability metric."""
    d = np.linalg.norm(v_a - v_b)
    na, nb = np.linalg.norm(v_a), np.linalg.norm(v_b)
    if na == 0 or nb == 0:
        return 0.0
    return 0.5 * (d / na + d / nb)


def pairwise_separation(vecs: dict[str, list[np.ndarray]]) -> dict[tuple[str, str], float]:
    """Separation between role centroids for all unique role pairs."""
    centroids = {r: np.mean(lst, axis=0) for r, lst in vecs.items()}
    return {(a, b): separation(centroids[a], centroids[b])
            for a, b in combinations(sorted(centroids), 2)}


def success_report(
    vecs: dict[str, list[np.ndarray]],
    score3_in: dict[str, float] | None = None,
    score3_ood: dict[str, float] | None = None,
) -> dict:
    """Evaluate the Prez success criteria. Returns per-role/per-pair stats + pass flags."""
    roles = sorted(vecs)
    margins = {r: stability_margin(r, vecs) for r in roles}
    cosines = {r: same_role_cosine(vecs[r]) for r in roles}
    pearsons = {r: same_role_pearson(vecs[r]) for r in roles}
    cluster_acc = cluster_assignment(vecs)
    sep = pairwise_separation(vecs)

    rep = {
        "roles": roles,
        "same_role_cosine": cosines,
        "same_role_pearson": pearsons,
        "stability_margin": margins,
        "cluster_accuracy": cluster_acc,
        "pairwise_separation": {f"{a}|{b}": v for (a, b), v in sep.items()},
        "checks": {
            # Q3: stable in direction + coordinate pattern; margin > 0; perfect clustering
            "Q3_margin_all_positive": all(m > 0 for m in margins.values()),
            "Q3_cluster_perfect": cluster_acc == 1.0,
            # Q4: all pairs separable (separation > 1 = well-separated)
            "Q4_all_pairs_separable": all(v > 1.0 for v in sep.values()),
        },
    }
    if score3_in and score3_ood:
        # Q1: in-domain score-3 rate meaningfully higher than OOD; OOD non-zero
        rep["score3_rate_in"] = score3_in
        rep["score3_rate_ood"] = score3_ood
        rep["checks"]["Q1_in_gt_ood"] = all(
            score3_in[r] > score3_ood[r] for r in roles if r in score3_in
        )
        rep["checks"]["Q1_ood_nonzero"] = all(
            score3_ood[r] > 0 for r in roles if r in score3_ood
        )
    return rep


def _load_runs(runs_dir, construction: str):
    """Load saved vectors.npz -> {role_id: [run vectors]} for one construction."""
    import json
    from pathlib import Path

    runs_dir = Path(runs_dir)
    data = np.load(runs_dir / "vectors.npz")
    meta = json.loads((runs_dir / "meta.json").read_text())
    vecs: dict[str, list[np.ndarray]] = {}
    for key in data.files:
        constr, role, _run = key.split("|")
        if constr != construction:
            continue
        vecs.setdefault(role, []).append(data[key])
    return vecs, meta


def _main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description="Compute role-vector metrics from a saved run.")
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--construction", default="balanced", choices=["raw", "rmd", "balanced"])
    a = p.parse_args(argv)

    vecs, meta = _load_runs(a.runs_dir, a.construction)
    rep = success_report(vecs, meta.get("score3_rate_in"), meta.get("score3_rate_ood"))

    print(f"\n== Role-Vector report ({a.construction}, {meta['runs']} runs, "
          f"backend={meta['backend']}) ==")
    for r in rep["roles"]:
        print(f"  {r:18s} cos={rep['same_role_cosine'][r]:+.3f} "
              f"pearson={rep['same_role_pearson'][r]:+.3f} "
              f"margin={rep['stability_margin'][r]:+.3f}")
    print(f"  cluster accuracy: {rep['cluster_accuracy']:.3f} "
          f"(chance {1/len(rep['roles']):.3f})")
    print("  separation (centroids):")
    for pair, v in rep["pairwise_separation"].items():
        print(f"    {pair:38s} {v:.3f}")
    print("  CHECKS:")
    for name, ok in rep["checks"].items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")


if __name__ == "__main__":
    _main()
