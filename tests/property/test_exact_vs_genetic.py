"""Property: whenever the genetic backend lands a counterfactual, the exact
backend finds one at least as cheap.

The exact search proves optimality (or gives an honest, documented reason it
cannot: a warm start it took on trust, or an order-pair repair that withdrew
its own certificate), so it can never do worse than a heuristic search over
the same problem — ``exact.distance <= genetic.distance + 1e-9`` — and it
never comes back ``Infeasible`` when the genetic search already found a row.
Both backends run through the public ``explain()`` API, not the solvers
directly, so this exercises verification, snapping and pruning exactly as a
caller would see them.

Cases are built from a fixed-seed ``numpy`` draw (the house hypothesis
convention in this package: see ``tests/property/test_constraints_hypothesis.py``),
using only constraint shapes the exact backend supports today — single-feature
``Linear``, the canonical order-pair ``Linear`` shape, ``OneHot``, ``Implies``,
``Freeze``, ``Monotone``, ``Range``, ``Equals``, ``AllowMissing`` — and
occasional missing-value factuals. A random factual may violate its own
random constraints; the ``TreecfWarning`` that raises is not the property
under test here, so it is filtered rather than left to trip
``filterwarnings = error``.
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
    Range,
    Target,
    TreecfWarning,
)
from treecf.constraints.compile import compile_constraints
from treecf.constraints.objects import Constraint
from treecf.ir.evaluate import raw_score

from ..conftest import make_random_ir

N_FEATURES = 3


def _order_pair(a: str, b: str, missing_policy: str) -> Linear:
    return Linear({a: 1.0, b: -1.0}, op="<=", rhs=0.0, missing_policy=missing_policy)


def _draw_single_feature_constraints(
    rng: np.random.Generator, names: tuple[str, ...], x: np.ndarray, skip: set[int]
) -> list[Constraint]:
    """Freeze/Range/Monotone/single-feature Linear + AllowMissing, one draw per
    feature not already claimed by a relational constraint."""
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
            # "==" excluded on purpose: its derived bound is only a superset of
            # what check_matrix's own 1e-9 slack actually accepts (see
            # tests/property/test_derived_bounds.py), so a NaN-factual/AllowMissing
            # candidate landing on that bound's own edge can be built by
            # _build_domains and then rejected by the arbiter a float-ulp later —
            # a known boundary case, not a genetic-vs-exact dominance failure.
            op = str(rng.choice(["<=", ">="]))
            rhs = coef * (anchor + float(rng.uniform(-2.0, 2.0)))
            policy = str(rng.choice(["satisfied", "violated", "forbid_missing"]))
            constraints.append(Linear({name: coef}, op=op, rhs=rhs, missing_policy=policy))
        if kind != "freeze" and rng.random() < 0.35:
            constraints.append(
                AllowMissing(
                    name,
                    delta_miss=float(rng.uniform(0.5, 3.0)),
                    delta_from_miss=float(rng.uniform(0.5, 3.0)),
                )
            )
    return constraints


def _draw_relational(
    rng: np.random.Generator, names: tuple[str, ...], x: np.ndarray
) -> tuple[list[Constraint], set[int]]:
    """An occasional constraint tying two features together — the exact
    backend's own supported set: an order pair, a one-hot pair, or an
    implication. Returns the constraint(s) and the feature indices they claim
    (kept out of the single-feature draws above)."""
    kind = str(rng.choice(["", "", "", "", "order", "onehot", "implies"]))
    if not kind:
        return [], set()
    a, b = (int(i) for i in rng.choice(N_FEATURES, size=2, replace=False))
    if kind == "order":
        policy = str(rng.choice(["satisfied", "violated", "forbid_missing"]))
        return [_order_pair(names[a], names[b], policy)], {a, b}
    if kind == "onehot":
        x[a], x[b] = 1.0, 0.0  # the group holds at the factual
        return [OneHot((names[a], names[b]))], {a, b}
    x[a] = float(rng.integers(0, 2))
    x[b] = float(rng.integers(0, 2))
    return [Implies(Equals(names[a], 1.0), Equals(names[b], 1.0))], {a, b}


def _build_case(seed: int) -> tuple[Explainer, np.ndarray, Target, float] | None:
    """One randomized ``(Explainer, x, Target, sparsity_weight)`` problem, or
    ``None`` when the draw compiled to something contradictory (fine — every
    draw is independent, and hypothesis tries plenty of others).

    No value policies here: they snap heuristically on the genetic path
    (reverting to the raw, off-grid value when snapping would break
    feasibility, see ``Explainer._apply_value_policies``) but are a hard
    constraint on the exact search's own domains — comparing the two costs
    would be comparing different feasible sets, not testing dominance.
    """
    rng = np.random.default_rng(seed)
    ir = make_random_ir(rng, n_features=N_FEATURES, n_trees=int(rng.integers(3, 6)), depth=3)
    names = ir.feature_names
    x = np.round(rng.normal(scale=2.0, size=N_FEATURES), 2)
    for j in range(N_FEATURES):
        if rng.random() < 0.2:
            x[j] = math.nan

    relational, claimed = _draw_relational(rng, names, x)
    constraints = _draw_single_feature_constraints(rng, names, x, claimed) + relational
    try:
        compile_constraints(constraints, names)
    except Exception:
        return None

    scores = [raw_score(ir, rng.normal(scale=3.0, size=N_FEATURES)) for _ in range(60)]
    lo = float(np.percentile(scores, float(rng.choice([40.0, 60.0, 80.0]))))
    lam = float(rng.choice([0.0, 0.05, 0.1]))

    try:
        exp = Explainer(ir, normalizers=np.ones(N_FEATURES), constraints=constraints)
    except Exception:
        return None
    return exp, x, Target.raw(op=">=", value=lo), lam


def _explain_both(
    exp: Explainer, x: np.ndarray, target: Target, lam: float, warm_start: bool
) -> tuple[object, object]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TreecfWarning)
        genetic = exp.explain(x, target, backend="genetic", sparsity_weight=lam, seed=0)
        exact = exp.explain(
            x, target, backend="exact", sparsity_weight=lam, seed=0, warm_start=warm_start,
        )
    return genetic, exact


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(min_value=0, max_value=1_000_000))
def test_exact_dominates_genetic_warm_start_false(seed: int) -> None:
    case = _build_case(seed)
    if case is None:
        return
    exp, x, target, lam = case
    genetic, exact = _explain_both(exp, x, target, lam, warm_start=False)
    if not isinstance(genetic, Counterfactual):
        return  # nothing to dominate

    assert isinstance(exact, Counterfactual), (
        f"seed {seed}: genetic found distance={genetic.distance} but "
        f"exact returned {exact!r}"
    )
    assert exact.distance <= genetic.distance + 1e-9, (
        f"seed {seed}: exact={exact.distance} worse than genetic={genetic.distance}"
    )


# One fixed, known-feasible case exercised with warm_start=True: the property
# is the same, but the exact search now starts from a genetic incumbent
# instead of an empty one, which is worth checking on its own path.
_WARM_START_TRUE_SEED = 7


def test_exact_dominates_genetic_warm_start_true() -> None:
    case = _build_case(_WARM_START_TRUE_SEED)
    assert case is not None
    exp, x, target, lam = case
    genetic, exact = _explain_both(exp, x, target, lam, warm_start=True)
    assert isinstance(genetic, Counterfactual)
    assert isinstance(exact, Counterfactual)
    assert exact.distance <= genetic.distance + 1e-9
