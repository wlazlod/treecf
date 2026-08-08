"""Derivation soundness: a derived bound never excludes a point the Linear accepts."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from treecf import Linear
from treecf.constraints.compile import CompiledConstraints, compile_constraints

NAMES = ("a", "b")
FINITE = st.floats(allow_nan=False, allow_infinity=False, min_value=-1e12, max_value=1e12)


@settings(max_examples=200, deadline=None)
@given(
    coef=FINITE.filter(lambda c: c != 0.0),
    rhs=FINITE,
    op=st.sampled_from(["<=", ">=", "=="]),
    other=FINITE,
    candidate=FINITE,
)
def test_derived_bound_never_excludes_feasible_point(
    coef: float, rhs: float, op: str, other: float, candidate: float
) -> None:
    lin = Linear({"a": coef}, op, rhs)
    with_bound = compile_constraints([lin], NAMES)
    without = CompiledConstraints(
        feature_names=with_bound.feature_names,
        constraints=with_bound.constraints,
        linears=with_bound.linears,
        implications=with_bound.implications,
        onehot_groups=with_bound.onehot_groups,
        allow_missing=with_bound.allow_missing,
        binary_features=with_bound.binary_features,
        derived_ranges=(),
    )
    row = np.array([[candidate, other]])
    x = np.zeros(2)
    # restrict to points satisfying the linear EXACTLY (tolerance-free): bounds
    # carry no slack, so a point feasible only via check_matrix's 1e-9 linear
    # slack may legitimately be excluded by the derived bound (repair clips
    # candidates onto the bound itself, so end-to-end feasibility is unaffected)
    total = coef * candidate
    exact = total <= rhs if op == "<=" else total >= rhs if op == ">=" else total == rhs
    if not exact:
        return
    assert without.check_matrix(row, x)[0]
    assert with_bound.check_matrix(row, x)[0]
