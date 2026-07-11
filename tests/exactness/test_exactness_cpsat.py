"""CP-SAT optimum must match the brute-force oracle (spec §12.2) — the M1 gate."""

from __future__ import annotations

import math

import numpy as np
import pytest

from treecf import Explainer, Freeze, Monotone, Range, Target
from treecf.api import Counterfactual, Infeasible
from treecf.constraints.compile import compile_constraints
from treecf.ir.evaluate import raw_score
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree

from ..conftest import make_random_ir
from .brute_force import solve_brute_force

pytest.importorskip("ortools")

J_TOL = 1e-5


def _stump() -> EnsembleIR:
    nodes = (
        Node(0, 0, 1.0, SplitOp.LT, True, 1, 2, None),
        Node(1, None, None, None, None, None, None, -1.0),
        Node(2, None, None, None, None, None, None, 1.0),
    )
    return EnsembleIR(
        trees=(Tree(nodes=nodes),),
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=2,
        feature_names=("a", "b"),
        meta={},
    )


class TestHandCheckable:
    def test_minimal_move_to_flip_stump(self) -> None:
        exp = Explainer(_stump(), normalizers=np.ones(2))
        res = exp.explain(np.array([0.0, 5.0]), target=Target.raw(op=">=", value=0.5))
        assert isinstance(res, Counterfactual)
        # must move feature a to the [1.0, inf) cell; nearest point is exactly 1.0
        assert res.x_cf[0] == pytest.approx(1.0)
        assert res.x_cf[1] == 5.0
        assert res.distance == pytest.approx(1.0)
        assert res.n_changed == 1
        assert res.proof == "optimal"

    def test_frozen_feature_makes_it_infeasible(self) -> None:
        exp = Explainer(_stump(), normalizers=np.ones(2), constraints=[Freeze("a")])
        res = exp.explain(np.array([0.0, 5.0]), target=Target.raw(op=">=", value=0.5))
        assert isinstance(res, Infeasible)

    def test_factual_already_in_target_costs_zero(self) -> None:
        exp = Explainer(_stump(), normalizers=np.ones(2))
        res = exp.explain(np.array([2.0, 5.0]), target=Target.raw(op=">=", value=0.5))
        assert isinstance(res, Counterfactual)
        assert res.distance == 0.0
        assert res.n_changed == 0


class TestRandomizedExactness:
    @pytest.mark.parametrize("seed", range(50))
    def test_cpsat_matches_brute_force(self, seed: int) -> None:
        rng = np.random.default_rng(1000 + seed)
        ir = make_random_ir(
            rng,
            n_features=int(rng.integers(2, 5)),
            n_trees=int(rng.integers(2, 7)),
            depth=int(rng.integers(2, 4)),
        )
        p = ir.n_features
        x = rng.normal(scale=2.0, size=p)
        sigma = np.ones(p)
        weights = np.ones(p)
        lam = float(rng.choice([0.0, 0.1]))

        # Target interval from sampled score percentiles: sometimes tight, sometimes empty.
        scores = [raw_score(ir, rng.normal(scale=3.0, size=p)) for _ in range(60)]
        lo_t = float(np.percentile(scores, rng.uniform(40, 80)))
        hi_t = lo_t + float(rng.choice([0.05, 0.5, math.inf]))

        constraints = _random_constraints(rng, ir.feature_names, x)
        compiled = compile_constraints(constraints, ir.feature_names)
        oracle = solve_brute_force(
            ir, x, (lo_t, hi_t), compiled, sigma, weights, lam=lam
        )

        exp = Explainer(ir, normalizers=sigma, constraints=constraints)
        res = exp.explain(
            x,
            target=Target.raw(range=(lo_t, hi_t)) if math.isfinite(hi_t)
            else Target.raw(op=">=", value=lo_t),
            sparsity_weight=lam,
        )

        if oracle.feasible:
            assert isinstance(res, Counterfactual), f"oracle J={oracle.objective}, got {res}"
            assert res.proof == "optimal"
            score = raw_score(ir, res.x_cf)
            assert lo_t <= score <= hi_t
            j_cpsat = res.distance + lam * res.n_changed
            assert j_cpsat == pytest.approx(oracle.objective, abs=J_TOL), (
                f"J_cpsat={j_cpsat} vs J_oracle={oracle.objective}"
            )
        else:
            assert isinstance(res, Infeasible)


def _random_constraints(
    rng: np.random.Generator, names: tuple[str, ...], x: np.ndarray
) -> list[Freeze | Monotone | Range]:
    constraints: list[Freeze | Monotone | Range] = []
    for name, x_j in zip(names, x, strict=True):
        roll = rng.random()
        if roll < 0.15:
            constraints.append(Freeze(name))
        elif roll < 0.3:
            constraints.append(
                Monotone(name, "increase" if rng.random() < 0.5 else "decrease")
            )
        elif roll < 0.45:
            lo = x_j - float(rng.uniform(0.5, 3.0))
            constraints.append(Range(name, lo, lo + float(rng.uniform(1.0, 5.0))))
    return constraints
