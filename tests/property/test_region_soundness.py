"""Property: recourse regions are sound.

Every point sampled from a region -- interior draws, box corners, and each
feature's own endpoints in isolation -- must independently re-verify: still
in-target, still constraint-feasible, still plausible when plausibility is
configured. The region must also contain its own counterfactual, never widen
a degenerate coordinate, and never grow when the target interval it was built
against shrinks.

Cases are built from a fixed-seed ``numpy`` draw (the house hypothesis
convention in this package: see ``tests/property/test_constraints_hypothesis.py``),
covering the same constraint shapes ``test_exact_vs_genetic.py`` uses --
single-feature ``Linear``, the canonical order-pair ``Linear`` shape,
``OneHot``, ``Implies``, ``Freeze``, ``Monotone``, ``Range``,
``AllowMissing`` -- plus an occasional isolation-forest ``Plausibility``.
Every case solves through the public ``explain(..., region=True)`` API, not
the region builder directly.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from treecf import (
    AllowMissing,
    Counterfactual,
    Equals,
    Explainer,
    Freeze,
    Implies,
    Linear,
    Monotone,
    OneHot,
    Plausibility,
    Range,
    RecourseRegion,
    Target,
    TreecfWarning,
)
from treecf.constraints.compile import compile_constraints
from treecf.constraints.objects import Constraint
from treecf.ir.evaluate import raw_score
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree

from ..conftest import make_random_ir

N_FEATURES = 3
_CAP = 1e6  # finite stand-in used only when sampling an unbounded region edge


def _order_pair(a: str, b: str) -> Linear:
    return Linear({a: 1.0, b: -1.0}, op="<=", rhs=0.0)


def _draw_single_feature_constraints(
    rng: np.random.Generator, names: tuple[str, ...], x: np.ndarray, skip: set[int]
) -> list[Constraint]:
    constraints: list[Constraint] = []
    for j, name in enumerate(names):
        if j in skip:
            continue
        anchor = 0.0 if math.isnan(x[j]) else float(x[j])
        kind = str(rng.choice(["none", "none", "freeze", "range", "monotone", "linear"]))
        if kind == "freeze":
            constraints.append(Freeze(name))
        elif kind == "range":
            width = float(rng.uniform(0.5, 3.0))
            shift = float(rng.uniform(-1.5, 1.5))
            constraints.append(Range(name, anchor + shift - width, anchor + shift + width))
        elif kind == "monotone":
            constraints.append(Monotone(name, "increase" if rng.random() < 0.5 else "decrease"))
        elif kind == "linear":
            coef = float(rng.choice([-2.0, -1.0, 1.0, 2.0]))
            op = str(rng.choice(["<=", ">="]))
            rhs = coef * (anchor + float(rng.uniform(-2.0, 2.0)))
            constraints.append(Linear({name: coef}, op=op, rhs=rhs))
        if kind != "freeze" and rng.random() < 0.25:
            constraints.append(AllowMissing(name, delta_miss=float(rng.uniform(0.5, 3.0))))
    return constraints


def _draw_relational(
    rng: np.random.Generator, names: tuple[str, ...], x: np.ndarray
) -> tuple[list[Constraint], set[int]]:
    kind = str(rng.choice(["", "", "", "order", "onehot", "implies"]))
    if not kind:
        return [], set()
    a, b = (int(i) for i in rng.choice(N_FEATURES, size=2, replace=False))
    if kind == "order":
        return [_order_pair(names[a], names[b])], {a, b}
    if kind == "onehot":
        x[a], x[b] = 1.0, 0.0
        return [OneHot((names[a], names[b]))], {a, b}
    x[a] = float(rng.integers(0, 2))
    x[b] = float(rng.integers(0, 2))
    return [Implies(Equals(names[a], 1.0), Equals(names[b], 1.0))], {a, b}


def _maybe_plausibility(rng: np.random.Generator, names: tuple[str, ...]) -> Plausibility | None:
    if rng.random() > 0.4:
        return None
    feature = int(rng.integers(0, N_FEATURES))
    stump = Tree(
        nodes=(
            Node(0, feature, float(rng.normal()), SplitOp.LT, True, 1, 2, None),
            Node(1, None, None, None, None, None, None, 0.0),
            Node(2, None, None, None, None, None, None, float(rng.uniform(1.0, 4.0))),
        )
    )
    if_ir = EnsembleIR(
        trees=(stump,), base_score=0.0, link=Link.IDENTITY, n_features=N_FEATURES,
        feature_names=names, meta={"max_samples": 16.0},
    )
    theta = float(rng.uniform(0.5, 0.95))  # loose-ish: keeps some cases feasible
    return Plausibility(if_ir=if_ir, max_anomaly_score=theta)


def _build_case(seed: int) -> tuple[Explainer, np.ndarray, Target] | None:
    """One randomized ``(Explainer, x, Target)`` problem, or ``None`` when the
    draw compiled to something contradictory -- fine, every draw is
    independent and hypothesis tries plenty of others."""
    rng = np.random.default_rng(seed)
    ir = make_random_ir(rng, n_features=N_FEATURES, n_trees=int(rng.integers(3, 6)), depth=3)
    names = ir.feature_names
    x = np.round(rng.normal(scale=2.0, size=N_FEATURES), 2)
    for j in range(N_FEATURES):
        if rng.random() < 0.15:
            x[j] = math.nan

    relational, claimed = _draw_relational(rng, names, x)
    constraints = _draw_single_feature_constraints(rng, names, x, claimed) + relational
    try:
        compile_constraints(constraints, names)
    except Exception:
        return None

    plausibility = None if np.isnan(x).any() else _maybe_plausibility(rng, names)
    try:
        exp = Explainer(
            ir, normalizers=np.ones(N_FEATURES), constraints=constraints,
            plausibility=plausibility,
        )
    except Exception:
        return None

    scores = [raw_score(ir, rng.normal(scale=3.0, size=N_FEATURES)) for _ in range(60)]
    lo = float(np.percentile(scores, float(rng.choice([30.0, 50.0, 70.0]))))
    return exp, x, Target.raw(op=">=", value=lo)


def _feasible_case(seed: int) -> tuple[Explainer, np.ndarray, Target, Counterfactual] | None:
    case = _build_case(seed)
    if case is None:
        return None
    exp, x, target = case
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TreecfWarning)
        result = exp.explain(x, target, backend="exact", seed=0, region=True, time_budget_s=5.0)
    if not isinstance(result, Counterfactual):
        return None
    assert isinstance(result.region, RecourseRegion)
    return exp, x, target, result


def _finite(v: float) -> float:
    return math.copysign(_CAP, v) if math.isinf(v) else v


def _non_degenerate(exp: Explainer, region: RecourseRegion) -> list[int]:
    index = {name: j for j, name in enumerate(exp.ir.feature_names)}
    return sorted(index[name] for name in region.feature_intervals)


def _sample_points(
    rng: np.random.Generator, x_cf: np.ndarray, region: RecourseRegion, non_degenerate: list[int]
) -> list[np.ndarray]:
    samples: list[np.ndarray] = []

    for _ in range(5):  # interior draws
        z = x_cf.copy()
        for j in non_degenerate:
            lo_j, hi_j = _finite(region.lo[j]), _finite(region.hi[j])
            z[j] = lo_j + float(rng.uniform(0.0, 1.0)) * (hi_j - lo_j)
        samples.append(z)

    for mask in range(2 ** len(non_degenerate)):  # every lo/hi corner
        z = x_cf.copy()
        for bit, j in enumerate(non_degenerate):
            edge = region.hi[j] if (mask >> bit) & 1 else region.lo[j]
            z[j] = _finite(edge)
        samples.append(z)

    for j in non_degenerate:  # one feature moved to each of its own endpoints
        for edge in (region.lo[j], region.hi[j]):
            z = x_cf.copy()
            z[j] = _finite(edge)
            samples.append(z)

    return samples


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(min_value=0, max_value=1_000_000))
def test_region_samples_stay_feasible(seed: int) -> None:
    case = _feasible_case(seed)
    if case is None:
        return
    exp, x, target, result = case
    region = result.region
    assert region is not None
    interval = target.raw_interval(exp.ir.link)
    rng = np.random.default_rng(seed)
    non_degenerate = _non_degenerate(exp, region)
    for z in _sample_points(rng, result.x_cf, region, non_degenerate):
        reason = exp._verify(x, z, interval)
        assert reason is None, f"seed {seed}: sample {z} failed verification: {reason}"


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(min_value=0, max_value=1_000_000))
def test_region_contains_its_own_counterfactual(seed: int) -> None:
    case = _feasible_case(seed)
    if case is None:
        return
    _, _, _, result = case
    assert result.region is not None
    assert result.region.contains(result.x_cf)


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(min_value=0, max_value=1_000_000))
def test_degenerate_coordinates_are_never_widened(seed: int) -> None:
    case = _feasible_case(seed)
    if case is None:
        return
    exp, _, _, result = case
    region = result.region
    assert region is not None
    non_degenerate = set(_non_degenerate(exp, region))
    for j in range(len(result.x_cf)):
        if j in non_degenerate:
            continue
        xj = result.x_cf[j]
        if math.isnan(xj):
            assert math.isnan(region.lo[j]) and math.isnan(region.hi[j])
        else:
            assert region.lo[j] == region.hi[j] == xj


@settings(max_examples=15, deadline=None)
@given(seed=st.integers(min_value=0, max_value=1_000_000))
def test_shrinking_the_target_never_enlarges_the_region(seed: int) -> None:
    case = _feasible_case(seed)
    if case is None:
        return
    exp, x, target, result = case
    region = result.region
    assert region is not None
    interval = target.raw_interval(exp.ir.link)
    score = result.score_raw
    narrow_lo = max(interval[0], score - 0.01)
    narrow_hi = min(interval[1], score + 0.01) if math.isfinite(interval[1]) else score + 0.01
    if not narrow_lo < narrow_hi:
        return
    narrow_region = exp.recourse_region(x, result.x_cf, Target.raw(range=(narrow_lo, narrow_hi)))
    for j in range(len(result.x_cf)):
        if math.isnan(region.lo[j]):
            continue
        assert narrow_region.lo[j] >= region.lo[j] - 1e-9
        assert narrow_region.hi[j] <= region.hi[j] + 1e-9
