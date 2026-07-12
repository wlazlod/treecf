"""Property suite (spec §12.4): every returned counterfactual satisfies every
constraint and the target in float space, independently re-checked here
(not via the library's own verifier)."""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from treecf import (
    AllowMissing,
    Counterfactual,
    Explainer,
    Freeze,
    Monotone,
    Range,
    Target,
    constraint,
)
from treecf.constraints.objects import Constraint, Linear
from treecf.ir.evaluate import raw_score

from ..conftest import make_random_ir

P = 4


def _build_case(seed: int) -> tuple[object, np.ndarray, list[Constraint], float]:
    rng = np.random.default_rng(seed)
    ir = make_random_ir(rng, n_features=P, n_trees=int(rng.integers(2, 5)), depth=3)
    x = rng.normal(scale=2.0, size=P)
    if rng.random() < 0.3:
        x[int(rng.integers(0, P))] = np.nan

    constraints: list[Constraint] = []
    names = ir.feature_names
    for j, name in enumerate(names):
        roll = rng.random()
        if roll < 0.12 and not math.isnan(x[j]):
            constraints.append(Freeze(name))
        elif roll < 0.25 and not math.isnan(x[j]):
            constraints.append(Monotone(name, "increase" if rng.random() < 0.5 else "decrease"))
        elif roll < 0.38:
            lo = float(rng.uniform(-6, 0))
            constraints.append(Range(name, lo, lo + float(rng.uniform(2, 8))))
        if roll > 0.6:
            constraints.append(AllowMissing(name, delta_miss=float(rng.uniform(0.2, 2.0))))
    if rng.random() < 0.5:
        a, b = rng.choice(P, size=2, replace=False)
        constraints.append(constraint(f"{names[a]} <= {names[b]}"))

    scores = [raw_score(ir, rng.normal(scale=3.0, size=P)) for _ in range(30)]
    lo_t = float(np.percentile(scores, rng.uniform(30, 90)))
    return ir, x, constraints, lo_t


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_returned_counterfactuals_satisfy_everything(seed: int) -> None:
    ir, x, constraints, lo_t = _build_case(seed)
    try:
        exp = Explainer(ir, normalizers=np.ones(P), constraints=constraints)
    except Exception:
        return  # contradictory random constraint sets may be rejected at compile: fine
    res = exp.explain(x, target=Target.raw(op=">=", value=lo_t), sparsity_weight=0.05, seed=0)
    if not isinstance(res, Counterfactual):
        return

    cf = res.x_cf
    assert raw_score(ir, cf) >= lo_t  # target, float space

    allow = {c.feature for c in constraints if isinstance(c, AllowMissing)}
    names = ir.feature_names
    for c in constraints:
        if isinstance(c, Freeze):
            j = names.index(c.feature)
            assert cf[j] == x[j] or (math.isnan(cf[j]) and math.isnan(x[j]))
        elif isinstance(c, Monotone):
            j = names.index(c.feature)
            if not math.isnan(cf[j]) and not math.isnan(x[j]):
                assert cf[j] >= x[j] if c.direction == "increase" else cf[j] <= x[j]
        elif isinstance(c, Range):
            j = names.index(c.feature)
            if not math.isnan(cf[j]):
                assert c.lo <= cf[j] <= c.hi
        elif isinstance(c, Linear):
            values = [cf[names.index(n)] for n in c.coefficients]
            if not any(math.isnan(v) for v in values):
                coefs = c.coefficients.values()
                total = sum(coef * v for coef, v in zip(coefs, values, strict=True))
                assert total <= c.rhs + 1e-9

    for j, name in enumerate(names):
        if math.isnan(cf[j]) and not math.isnan(x[j]):
            assert name in allow  # NaN only where AllowMissing granted it
