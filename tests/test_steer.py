"""Sanity test for steering persistence simulation (#5)."""
from rolevec.steer import evaluate_persistence


def test_steered_persists_above_prompted_drift():
    r = evaluate_persistence(role_id="mna_lawyer", turns=9, coeff=8.0, simulate=True)
    assert r.steered_final >= 2.0          # holds the role at the last turn
    assert r.steered_final > r.prompted_final  # beats prompted-only drift
    assert r.passed


def test_low_coeff_does_not_pass_floor():
    # coeff ~0 -> no injection -> should not hold the role
    r = evaluate_persistence(role_id="mna_lawyer", turns=9, coeff=0.0, simulate=True)
    assert not r.passed
