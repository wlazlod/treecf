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

One-hot groups, implications and order pairs are absent by design — the exact
backend does not search over them yet.
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
    Linear,
    Monotone,
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

SEEDS = tuple(range(25))


@dataclass(frozen=True)
class _Case:
    """One randomized problem, in the argument shape both solvers accept."""

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

    compiled = compile_constraints(_draw_constraints(rng, names, x), names)
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

            if result.stats["completed"] is not True:
                problems.append(f"seed {seed}: search did not complete")
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
            tolerance = 1e-12 * max(1.0, oracle.objective)
            if abs(result.distance - oracle.objective) > tolerance:
                problems.append(
                    f"seed {seed}: objective disagreement — "
                    f"exact={result.distance!r}, oracle={oracle.objective!r}"
                )
                continue
            lo_t, hi_t = case.interval
            score = raw_score(case.ir, result.x_cf)
            if not lo_t <= score <= hi_t:
                problems.append(f"seed {seed}: returned row scores {score!r}, outside the target")
            if not bool(case.compiled.check_matrix(result.x_cf[np.newaxis, :], case.x)[0]):
                problems.append(f"seed {seed}: returned row violates the constraints")
            if case.plausibility is not None:
                if_ir, min_total_path = case.plausibility
                if raw_score(if_ir, result.x_cf) < min_total_path:
                    problems.append(f"seed {seed}: returned row is below the plausibility bound")

        assert not problems, "\n".join(problems)
        # a suite that is all-feasible or all-infeasible would prove very little
        assert 5 <= feasible <= len(SEEDS) - 5, f"{feasible}/{len(SEEDS)} feasible"


class TestSolverDeterminism:
    @pytest.mark.parametrize("seed", [0, 3, 7, 11, 17])
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
