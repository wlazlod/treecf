"""NaN as a first-class counterfactual value (spec §4.2, §12.6)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from treecf import AllowMissing, Counterfactual, Explainer, Linear, Target, constraint
from treecf.constraints.compile import compile_constraints
from treecf.ir.evaluate import raw_score
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree

from .conftest import make_random_ir
from .exactness.brute_force import solve_brute_force


def _nan_stump(missing_left: bool = False) -> EnsembleIR:
    """Split at 1.0 on feature a: left leaf -1, right leaf +1; NaN routes per flag."""
    nodes = (
        Node(0, 0, 1.0, SplitOp.LT, missing_left, 1, 2, None),
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


class TestValueToNaN:
    def test_cheap_delta_prefers_nan_flip(self) -> None:
        exp = Explainer(
            _nan_stump(),
            normalizers=np.ones(2),
            constraints=[AllowMissing("a", delta_miss=0.3)],
        )
        res = exp.explain(np.array([0.0, 0.0]), target=Target.raw(op=">=", value=0.5), seed=0)
        assert isinstance(res, Counterfactual)
        assert math.isnan(res.x_cf[0])  # NaN routes right, reaching +1
        assert res.distance == pytest.approx(0.3)
        assert res.n_changed == 1

    def test_expensive_delta_prefers_value_move(self) -> None:
        exp = Explainer(
            _nan_stump(),
            normalizers=np.ones(2),
            constraints=[AllowMissing("a", delta_miss=5.0)],
        )
        res = exp.explain(np.array([0.0, 0.0]), target=Target.raw(op=">=", value=0.5), seed=0)
        assert isinstance(res, Counterfactual)
        assert res.x_cf[0] == pytest.approx(1.0)
        assert res.distance == pytest.approx(1.0)

    def test_without_allow_missing_nan_never_appears(self) -> None:
        exp = Explainer(_nan_stump(), normalizers=np.ones(2))
        res = exp.explain(np.array([0.0, 0.0]), target=Target.raw(op=">=", value=0.5), seed=0)
        assert isinstance(res, Counterfactual)
        assert not np.isnan(res.x_cf).any()


class TestNaNFactual:
    def test_staying_missing_costs_nothing(self) -> None:
        exp = Explainer(
            _nan_stump(missing_left=False),
            normalizers=np.ones(2),
            constraints=[AllowMissing("a", delta_miss=0.3)],
        )
        # NaN routes right (+1) already: target satisfied unchanged
        res = exp.explain(np.array([np.nan, 0.0]), target=Target.raw(op=">=", value=0.5), seed=0)
        assert isinstance(res, Counterfactual)
        assert math.isnan(res.x_cf[0])
        assert res.distance == 0.0 and res.n_changed == 0

    def test_leaving_missing_pays_delta_from(self) -> None:
        exp = Explainer(
            _nan_stump(missing_left=True),  # NaN routes left (-1): must take a value
            normalizers=np.ones(2),
            constraints=[AllowMissing("a", delta_miss=0.3, delta_from_miss=0.7)],
        )
        res = exp.explain(np.array([np.nan, 0.0]), target=Target.raw(op=">=", value=0.5), seed=0)
        assert isinstance(res, Counterfactual)
        assert res.x_cf[0] >= 1.0
        assert res.distance == pytest.approx(0.7)

    def test_nan_factual_without_allow_missing_stays_fixed(self) -> None:
        exp = Explainer(_nan_stump(missing_left=True), normalizers=np.ones(2))
        # NaN fixed on the -1 side; only feature b exists but has no splits -> infeasible
        res = exp.explain(np.array([np.nan, 0.0]), target=Target.raw(op=">=", value=0.5), seed=0)
        assert not isinstance(res, Counterfactual)


class TestMissingPolicy:
    def test_forbid_missing_blocks_the_nan_shortcut(self) -> None:
        exp = Explainer(
            _nan_stump(),
            normalizers=np.ones(2),
            constraints=[
                AllowMissing("a", delta_miss=0.3),
                Linear(
                    coefficients={"a": 1.0, "b": -1.0},
                    op="<=",
                    rhs=0.0,
                    missing_policy="forbid_missing",
                ),
            ],
        )
        res = exp.explain(np.array([0.0, 5.0]), target=Target.raw(op=">=", value=0.5), seed=0)
        assert isinstance(res, Counterfactual)
        assert not math.isnan(res.x_cf[0])  # NaN was cheaper but is forbidden
        assert res.x_cf[0] == pytest.approx(1.0)

    def test_satisfied_policy_lets_nan_bypass_linear(self) -> None:
        exp = Explainer(
            _nan_stump(),
            normalizers=np.ones(2),
            constraints=[
                AllowMissing("a", delta_miss=0.3),
                constraint("a >= 100"),  # impossible for values; vacuous when NaN
            ],
        )
        res = exp.explain(np.array([0.0, 0.0]), target=Target.raw(op=">=", value=0.5), seed=0)
        assert isinstance(res, Counterfactual)
        assert math.isnan(res.x_cf[0])


class TestRandomizedWithNaN:
    @pytest.mark.parametrize("seed", range(15))
    def test_ga_brackets_oracle_with_allow_missing(self, seed: int) -> None:
        rng = np.random.default_rng(7000 + seed)
        ir = make_random_ir(rng, n_features=3, n_trees=3, depth=3)
        x = rng.normal(scale=2.0, size=3)
        if rng.random() < 0.5:
            x[int(rng.integers(0, 3))] = np.nan
        allow = [
            AllowMissing(ir.feature_names[j], delta_miss=float(rng.uniform(0.1, 2.0)))
            for j in range(3)
            if rng.random() < 0.6
        ]
        scores = [raw_score(ir, rng.normal(scale=3.0, size=3)) for _ in range(40)]
        lo_t = float(np.percentile(scores, 60))

        compiled = compile_constraints(allow, ir.feature_names)
        oracle = solve_brute_force(
            ir, x, (lo_t, math.inf), compiled, np.ones(3), np.ones(3), lam=0.05
        )
        exp = Explainer(ir, normalizers=np.ones(3), constraints=allow)
        res = exp.explain(x, target=Target.raw(op=">=", value=lo_t), sparsity_weight=0.05, seed=0)

        if oracle.feasible:
            assert isinstance(res, Counterfactual), f"oracle J={oracle.objective}, got {res}"
            j_ga = res.distance + 0.05 * res.n_changed
            # heuristic engine: never beats the brute-force optimum, lands close
            assert j_ga >= oracle.objective - 1e-9
            assert j_ga <= oracle.objective + 0.5
        else:
            assert not isinstance(res, Counterfactual)
