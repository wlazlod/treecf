"""Genetic backend behavior (spec §8.2): feasibility-first, deterministic, numpy-only."""

from __future__ import annotations

import math

import numpy as np
import pytest

from treecf import AllowMissing, Counterfactual, Explainer, Freeze, Infeasible, Monotone, Target
from treecf.ir.evaluate import raw_score
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree


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


class TestBasics:
    def test_finds_feasible_solution_with_heuristic_proof(self) -> None:
        exp = Explainer(_stump(), normalizers=np.ones(2))
        res = exp.explain(
            np.array([0.0, 0.0]),
            target=Target.raw(op=">=", value=0.5),
            backend="genetic",
            seed=1,
        )
        assert isinstance(res, Counterfactual)
        assert res.proof == "heuristic"
        assert res.score_raw >= 0.5
        assert res.x_cf[1] == 0.0  # no reason to touch b

    def test_respects_freeze_and_monotone(self) -> None:
        exp = Explainer(
            _stump(),
            normalizers=np.ones(2),
            constraints=[Freeze("b"), Monotone("a", "increase")],
        )
        res = exp.explain(
            np.array([0.0, 3.0]),
            target=Target.raw(op=">=", value=0.5),
            backend="genetic",
            seed=1,
        )
        assert isinstance(res, Counterfactual)
        assert res.x_cf[0] >= 0.0
        assert res.x_cf[1] == 3.0

    def test_infeasible_reports_heuristic_exhaustion(self) -> None:
        exp = Explainer(_stump(), normalizers=np.ones(2), constraints=[Freeze("a")])
        res = exp.explain(
            np.array([0.0, 0.0]),
            target=Target.raw(op=">=", value=0.5),
            backend="genetic",
            seed=1,
        )
        assert isinstance(res, Infeasible)
        assert "heuristic" in res.reason

    def test_same_seed_is_deterministic(self) -> None:
        exp = Explainer(_stump(), normalizers=np.ones(2))
        x = np.array([0.0, 0.0])
        target = Target.raw(op=">=", value=0.5)
        r1 = exp.explain(x, target=target, backend="genetic", seed=7)
        r2 = exp.explain(x, target=target, backend="genetic", seed=7)
        assert isinstance(r1, Counterfactual) and isinstance(r2, Counterfactual)
        np.testing.assert_array_equal(r1.x_cf, r2.x_cf)

    def test_nan_flip_available_to_genetic(self) -> None:
        exp = Explainer(
            _stump(),
            normalizers=np.ones(2),
            constraints=[AllowMissing("a", delta_miss=0.1)],
        )
        # NaN routes left (missing_left=True) giving -1... so target <= -0.5 via NaN is cheap
        res = exp.explain(
            np.array([5.0, 0.0]),
            target=Target.raw(op="<=", value=-0.5),
            backend="genetic",
            seed=3,
        )
        assert isinstance(res, Counterfactual)
        assert math.isnan(res.x_cf[0]) or res.x_cf[0] < 1.0


class TestCrossBackendSoundness:
    """Spec §12.3: GA solutions verify in float and approach CP-SAT optima."""

    @pytest.mark.parametrize("seed", range(20))
    def test_ga_close_to_cpsat_on_toy_suite(self, seed: int) -> None:
        pytest.importorskip("ortools")
        from tests.conftest import make_random_ir

        rng = np.random.default_rng(3000 + seed)
        ir = make_random_ir(rng, n_features=3, n_trees=4, depth=3)
        x = rng.normal(scale=2.0, size=3)
        scores = [raw_score(ir, rng.normal(scale=3.0, size=3)) for _ in range(40)]
        lo_t = float(np.percentile(scores, 60))
        target = Target.raw(op=">=", value=lo_t)

        exp = Explainer(ir, normalizers=np.ones(3))
        exact = exp.explain(x, target=target, backend="cpsat")
        heur = exp.explain(x, target=target, backend="genetic", seed=seed)

        if isinstance(exact, Counterfactual):
            assert isinstance(heur, Counterfactual), "GA missed a CP-SAT-feasible case"
            assert heur.score_raw >= lo_t  # float-verified by the API already
            # CP-SAT's conservative fixed-point rounding may cost up to ~1/K per
            # feature, so "never beats the optimum" carries the same 1e-5 slack as
            # the exactness suite.
            assert heur.distance >= exact.distance - 1e-5
            assert heur.distance <= exact.distance + 1.0  # and lands reasonably close
