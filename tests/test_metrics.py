"""Sanity tests for the metrics — pure numpy, no model needed."""
import numpy as np

from rolevec import metrics as m


def _clustered(n_roles=3, runs=10, dim=32, spread=0.1, seed=0):
    rng = np.random.default_rng(seed)
    vecs = {}
    for r in range(n_roles):
        center = rng.standard_normal(dim) * 3
        vecs[f"role{r}"] = [center + rng.standard_normal(dim) * spread for _ in range(runs)]
    return vecs


def test_same_role_cosine_high_for_tight_cluster():
    vecs = _clustered(spread=0.05)
    for runs in vecs.values():
        assert m.same_role_cosine(runs) > 0.9


def test_stability_margin_positive_when_separated():
    vecs = _clustered(spread=0.1)
    for r in vecs:
        assert m.stability_margin(r, vecs) > 0


def test_cluster_assignment_perfect_when_separated():
    assert m.cluster_assignment(_clustered(spread=0.1)) == 1.0


def test_separation_zero_for_identical_and_large_for_far():
    a = np.ones(8)
    assert m.separation(a, a.copy()) == 0.0
    assert m.separation(a, -a) > 1.0


def test_success_report_shape():
    rep = m.success_report(_clustered())
    assert set(rep["checks"]) >= {"Q3_margin_all_positive", "Q3_cluster_perfect", "Q4_all_pairs_separable"}
