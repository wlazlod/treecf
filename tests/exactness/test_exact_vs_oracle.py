"""Randomized exact-search versus brute-force-oracle comparison.

Every case draws a small ensemble, a factual (sometimes with missing values), a
constraint set, an optional isolation forest, and optional value policies, then
asks both solvers for the cheapest counterfactual. The oracle enumerates the
whole grid, so its verdict is ground truth: the two must agree on whether a
counterfactual exists at all, agree on its cost to within a relative 1e-12, and
the exact backend's row must survive an independent re-check of the score and
the constraints.

Value policies are only drawn for features the constraints have not pinned to a
single value. On a pinned feature the two solvers deliberately disagree: the
exact backend treats the pinned value as authoritative and never snaps it,
while the oracle snaps every non-keep candidate. That difference is settled
policy (see ``tests/backends/test_exact.py``), not a bug for this suite to
re-litigate.

Some cases also draw a constraint that ties two features together, and those
cases are compared differently, because the two solvers no longer search the
same set of rows.

The oracle puts each feature on one point per cell — the point nearest to the
factual — and nothing else. An order pair (``a <= b``) can need a value that is
not any cell's nearest point: the exact backend may move both features onto a
shared value inside their cells, which the oracle cannot express. So on those
draws the exact backend may legitimately come back cheaper, and it may find a
counterfactual where the oracle found none. What still holds is the other
direction: anything the oracle can build the exact backend can build too, so an
oracle answer is an upper bound on the exact answer.

A one-hot group restricts its members to 0 and 1, and the exact backend offers
each of them only those two values (plus its unchanged factual) while the
oracle keeps offering every cell's nearest point. Neither set contains the
other, so those draws are compared only where the disagreement cannot bite:
when the oracle's own answer keeps every one-hot member on 0 or 1, it is a row
the exact backend could have built, and it has to do at least as well.

An implication restricts nothing, so the exact backend offers its features
every candidate the oracle does and one more — the value the implication could
demand of the consequence. Those draws are compared like order pairs: the
oracle's answer is an upper bound, and the exact backend may beat it.

Whatever the exact backend returns is re-verified from scratch on every draw,
against the score, the constraints and the plausibility bound, and draws
without any cross-feature constraint keep the original strict agreement.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest

from treecf.api import Grid, ValuePolicy
from treecf.backends.exact import ExactResult, solve_exact
from treecf.constraints import (
    AllowMissing,
    Equals,
    Freeze,
    Implies,
    Linear,
    Monotone,
    OneHot,
    Range,
    compile_constraints,
)
from treecf.constraints.compile import CompiledConstraints
from treecf.constraints.objects import Constraint
from treecf.ir.evaluate import raw_score
from treecf.ir.model import EnsembleIR

from ..conftest import make_random_ir
from .brute_force import solve_brute_force

FloatArray = npt.NDArray[np.float64]

SEEDS = tuple(range(40))


@dataclass(frozen=True)
class _Case:
    """One randomized problem, in the argument shape both solvers accept.

    ``relational`` says which cross-feature constraint the draw added, if any:
    ``""`` for none, ``"order"`` for an order pair, ``"onehot"`` or
    ``"implies"`` for a binary group. It selects how strictly the two solvers
    may be compared, for the reasons in the module docstring.
    """

    seed: int
    ir: EnsembleIR
    x: FloatArray
    interval: tuple[float, float]
    compiled: CompiledConstraints
    sigma: FloatArray
    weights: FloatArray
    lam: float
    plausibility: tuple[EnsembleIR, float] | None
    value_policies: Mapping[str, ValuePolicy] | None
    relational: str


def _draw_interval(
    rng: np.random.Generator, ir: EnsembleIR, n_features: int
) -> tuple[float, float]:
    """A target a random point reaches only some of the time, so that both
    feasible and infeasible cases show up across the seeds."""
    scores = np.array([raw_score(ir, rng.normal(scale=2.0, size=n_features)) for _ in range(200)])
    quantile = float(rng.choice([40.0, 60.0, 80.0, 95.0]))
    lo = float(np.percentile(scores, quantile))
    if rng.random() < 0.3:  # a two-sided window, occasionally
        hi = float(np.percentile(scores, min(quantile + 25.0, 100.0)))
        return lo, hi
    return lo, math.inf


def _draw_constraints(
    rng: np.random.Generator, names: tuple[str, ...], x: FloatArray
) -> tuple[Constraint, ...]:
    constraints: list[Constraint] = []
    for j, name in enumerate(names):
        anchor = 0.0 if math.isnan(x[j]) else float(x[j])
        kind = str(
            rng.choice(["none", "none", "freeze", "range", "equals", "monotone", "linear"])
        )
        if kind == "freeze":
            constraints.append(Freeze(name))
        elif kind == "range":
            width = float(rng.uniform(0.5, 3.0))
            shift = float(rng.uniform(-1.5, 1.5))
            constraints.append(Range(name, anchor + shift - width, anchor + shift + width))
        elif kind == "equals":
            constraints.append(Equals(name, float(rng.integers(0, 2))))
        elif kind == "monotone":
            constraints.append(
                Monotone(name, "increase" if rng.random() < 0.5 else "decrease")
            )
        elif kind == "linear":
            coef = float(rng.choice([-2.0, -1.0, 1.0, 2.0]))
            op = str(rng.choice(["<=", ">=", "=="]))
            rhs = coef * (anchor + float(rng.uniform(-2.0, 2.0)))
            policy = str(rng.choice(["satisfied", "violated", "forbid_missing"]))
            constraints.append(
                Linear({name: coef}, op=op, rhs=rhs, missing_policy=policy)
            )
        if kind != "freeze" and rng.random() < 0.4:
            constraints.append(
                AllowMissing(
                    name,
                    delta_miss=float(rng.uniform(0.5, 3.0)),
                    delta_from_miss=float(rng.uniform(0.5, 3.0)),
                )
            )
    return tuple(constraints)


def _order_pair(rng: np.random.Generator, a: str, b: str) -> Linear:
    policy = str(rng.choice(["satisfied", "violated", "forbid_missing"]))
    return Linear({a: 1.0, b: -1.0}, op="<=", rhs=0.0, missing_policy=policy)


def _draw_relational(
    rng: np.random.Generator, names: tuple[str, ...], x: FloatArray
) -> tuple[tuple[Constraint, ...], str]:
    """An occasional constraint tying two features together.

    A one-hot group is only drawn over features whose factual value is 0 or 1 —
    anything else would leave the group unsatisfiable at the factual for
    reasons that have nothing to do with the search. Writing those values into
    ``x`` also clears any missing value drawn there.

    Three of the kinds combine constraints on purpose. ``"onehot+order"`` puts
    an order pair across a one-hot member and an outside feature, and
    ``"implies+order"`` puts one across the consequence of an implication and
    an outside feature; both are shapes that used to have the search throw a
    repair away and call the result a proof. ``"chain"`` draws two order pairs
    sharing a feature, the shape a pair-at-a-time repair cannot always settle.
    """
    kind = str(
        rng.choice(
            [
                "",
                "",
                "",
                "",
                "",
                "order",
                "onehot",
                "implies",
                "onehot+order",
                "implies+order",
                "chain",
            ]
        )
    )
    if not kind:
        return (), ""
    picks = [int(i) for i in rng.choice(len(names), size=min(3, len(names)), replace=False)]
    a, b, c = (picks + picks)[:3]
    if kind == "order":
        return (_order_pair(rng, names[a], names[b]),), kind
    if kind == "chain":
        # a <= b and b <= c: repairing either pair can break the other
        return (
            _order_pair(rng, names[a], names[b]),
            _order_pair(rng, names[b], names[c]),
        ), kind
    if kind == "onehot":
        x[a], x[b] = 1.0, 0.0  # the group holds at the factual
        return (OneHot((names[a], names[b])),), kind
    if kind == "onehot+order":
        x[a], x[b] = 1.0, 0.0
        return (
            OneHot((names[a], names[b])),
            _order_pair(rng, names[a], names[c]),
        ), kind
    x[a] = float(rng.integers(0, 2))
    x[b] = float(rng.integers(0, 2))
    implication = Implies(Equals(names[a], 1.0), Equals(names[b], 1.0))
    if kind == "implies+order":
        # the pair runs across the consequence, whose legal value the search
        # only learns from the implication
        return (implication, _order_pair(rng, names[c], names[b])), kind
    return (implication,), kind


def _draw_value_policies(
    rng: np.random.Generator,
    names: tuple[str, ...],
    compiled: CompiledConstraints,
    x: FloatArray,
) -> Mapping[str, ValuePolicy] | None:
    if rng.random() >= 0.35:
        return None
    lo, hi = compiled.instance_bounds(x)[:2]
    policies: dict[str, ValuePolicy] = {}
    for j, name in enumerate(names):
        if lo[j] == hi[j]:  # pinned: the two solvers snap it differently, by design
            continue
        if rng.random() < 0.5:
            policies[name] = (
                "integer"
                if rng.random() < 0.5
                else Grid(step=float(rng.choice([0.5, 1.0, 2.0])), anchor=0.0)
            )
    return policies or None


def _draw_case(seed: int) -> _Case:
    rng = np.random.default_rng(seed)
    n_features = 3
    ir = make_random_ir(rng, n_features=n_features, n_trees=4, depth=3)
    names = ir.feature_names

    x = np.round(rng.normal(scale=2.0, size=n_features), 2)
    for j in range(n_features):  # missing-value patterns in the factual
        if rng.random() < 0.2:
            x[j] = math.nan

    plausibility: tuple[EnsembleIR, float] | None = None
    if rng.random() < 0.4:
        if_ir = make_random_ir(rng, n_features=n_features, n_trees=3, depth=3)
        if_scores = np.array(
            [raw_score(if_ir, rng.normal(scale=2.0, size=n_features)) for _ in range(200)]
        )
        # permissive: most of the space clears the bound, a minority does not
        plausibility = (if_ir, float(np.percentile(if_scores, 25.0)))

    relational, kind = _draw_relational(rng, names, x)
    compiled = compile_constraints(_draw_constraints(rng, names, x) + relational, names)
    return _Case(
        seed=seed,
        ir=ir,
        x=x,
        interval=_draw_interval(rng, ir, n_features),
        compiled=compiled,
        sigma=np.round(rng.uniform(0.5, 3.0, n_features), 2),
        weights=np.round(rng.uniform(0.5, 2.0, n_features), 2),
        lam=float(rng.choice([0.0, 0.0, 0.25])),
        plausibility=plausibility,
        value_policies=_draw_value_policies(rng, names, compiled, x),
        relational=kind,
    )


def _solve(case: _Case) -> ExactResult:
    return solve_exact(
        case.ir,
        case.x,
        case.interval,
        case.compiled,
        case.sigma,
        case.weights,
        case.lam,
        value_policies=case.value_policies,
        plausibility=case.plausibility,
        time_budget_s=1e9,  # fixtures must never be decided by the wall clock
    )


def _verify(case: _Case, row: FloatArray) -> list[str]:
    """Re-check a returned row from scratch: the target, the constraints and
    the plausibility bound, none of it taken from the solver's own word."""
    problems: list[str] = []
    lo_t, hi_t = case.interval
    score = raw_score(case.ir, row)
    if not lo_t <= score <= hi_t:
        problems.append(f"seed {case.seed}: returned row scores {score!r}, outside the target")
    if not bool(case.compiled.check_matrix(row[np.newaxis, :], case.x)[0]):
        problems.append(f"seed {case.seed}: returned row violates the constraints")
    if case.plausibility is not None:
        if_ir, min_total_path = case.plausibility
        if raw_score(if_ir, row) < min_total_path:
            problems.append(f"seed {case.seed}: returned row is below the plausibility bound")
    return problems


def _may_leave_the_space_unsettled(case: _Case) -> bool:
    """True when the draw holds an order pair at all.

    Any repair that comes to nothing leaves its completion undecided, and the
    search says so rather than reporting the empty hand as proof. For a plain
    pair it still remembers what that completion would have cost at least, so
    the certificate survives whenever the incumbent is already that cheap —
    but whether it does is a property of the instance, not of the draw, so a
    `completed` of False is a documented answer for any pair-carrying case
    here. Everything the oracle can build was enumerated either way, so the
    comparisons below hold regardless.
    """
    return any(len(lin.indices) == 2 for lin in case.compiled.linears)


def _binary_valued(case: _Case, row: FloatArray | None) -> bool:
    """True when ``row`` puts every binary feature on 0 or 1 — the case where
    the exact backend could have built the same row."""
    if row is None:
        return False
    return all(row[f] in (0.0, 1.0) for f in case.compiled.binary_features)


class TestExactVersusOracle:
    def test_verdicts_and_objectives_agree_on_every_seed(self) -> None:
        problems: list[str] = []
        feasible = 0
        for seed in SEEDS:
            case = _draw_case(seed)
            oracle = solve_brute_force(
                case.ir,
                case.x,
                case.interval,
                case.compiled,
                case.sigma,
                case.weights,
                case.lam,
                plausibility=case.plausibility,
                value_policies=case.value_policies,
            )
            result = solve_exact(
                case.ir,
                case.x,
                case.interval,
                case.compiled,
                case.sigma,
                case.weights,
                case.lam,
                value_policies=case.value_policies,
                plausibility=case.plausibility,
                time_budget_s=1e9,
            )
            feasible += int(oracle.feasible)

            if result.stats["completed"] is not True and not _may_leave_the_space_unsettled(
                case
            ):
                problems.append(f"seed {seed}: search did not complete")
                continue
            if result.x_cf is not None:
                problems.extend(_verify(case, result.x_cf))
            tolerance = 1e-12 * max(1.0, oracle.objective)

            if case.relational in ("onehot", "onehot+order"):
                # One-hot members: the two solvers offer different values for
                # them, so a like-for-like comparison is only possible where
                # the oracle's own answer stays on 0/1 there. Every value in
                # such a row is one the exact backend can build too, so it has
                # to do at least as well.
                if not oracle.feasible or not _binary_valued(case, oracle.x_cf):
                    continue
                if result.x_cf is None:
                    problems.append(f"seed {seed}: oracle found a row the exact search missed")
                    continue
                assert result.distance is not None
                if result.distance > oracle.objective + tolerance:
                    problems.append(
                        f"seed {seed}: exact={result.distance!r} is worse than "
                        f"oracle={oracle.objective!r} on a row it could have built"
                    )
                continue
            if case.relational in ("order", "chain", "implies", "implies+order"):
                # the exact backend may beat the oracle here, never lose to it
                if oracle.feasible and result.x_cf is None:
                    problems.append(f"seed {seed}: oracle found a row the exact search missed")
                elif oracle.feasible:
                    assert result.distance is not None
                    if result.distance > oracle.objective + tolerance:
                        problems.append(
                            f"seed {seed}: exact={result.distance!r} is worse than "
                            f"oracle={oracle.objective!r}"
                        )
                continue

            if (result.x_cf is not None) != oracle.feasible:
                problems.append(
                    f"seed {seed}: feasibility disagreement — "
                    f"exact={result.x_cf is not None}, oracle={oracle.feasible}"
                )
                continue
            if result.x_cf is None:
                assert result.distance is None
                continue
            assert result.distance is not None
            if abs(result.distance - oracle.objective) > tolerance:
                problems.append(
                    f"seed {seed}: objective disagreement — "
                    f"exact={result.distance!r}, oracle={oracle.objective!r}"
                )

        assert not problems, "\n".join(problems)
        # a suite that is all-feasible or all-infeasible would prove very little
        assert 5 <= feasible <= len(SEEDS) - 5, f"{feasible}/{len(SEEDS)} feasible"


class TestSolverDeterminism:
    # one seed per draw kind: 0 and 3 plain, 39 an order pair, 22 a chain of
    # two, 1 a one-hot group crossed by an order pair, 12 a plain one-hot
    # group, 11 an implication crossed by an order pair, 18 a plain implication
    @pytest.mark.parametrize("seed", [0, 3, 39, 22, 1, 12, 11, 18])
    def test_same_inputs_twice_give_the_same_row_and_node_count(self, seed: int) -> None:
        case = _draw_case(seed)
        first = _solve(case)
        second = _solve(case)
        if first.x_cf is None:
            assert second.x_cf is None
        else:
            assert second.x_cf is not None
            assert np.array_equal(first.x_cf, second.x_cf, equal_nan=True)
        assert first.stats == second.stats
        assert first.distance == second.distance
        assert first.snapped == second.snapped
