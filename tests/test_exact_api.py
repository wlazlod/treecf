"""Exact backend API integration: dispatch, warm start, kwargs, certified infeasibility."""

from __future__ import annotations

import numpy as np
import pytest

from treecf import (
    ConstraintValidationError,
    Counterfactual,
    Explainer,
    Grid,
    Infeasible,
    Target,
    TreecfWarning,
    constraint,
)
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree
from treecf.plausibility import Plausibility


def _leaf(i: int, v: float) -> Node:
    return Node(i, None, None, None, None, None, None, v)


def _stump(feature: int, threshold: float, right_value: float) -> Tree:
    return Tree(
        nodes=(
            Node(0, feature, threshold, SplitOp.LT, True, 1, 2, None),
            _leaf(1, 0.0),
            _leaf(2, right_value),
        )
    )


def _ir() -> EnsembleIR:
    """Three independent levers worth 1.0 / 0.8 / 0.6 on features a/b/c; max raw score 2.4."""
    return EnsembleIR(
        trees=(_stump(0, 1.0, 1.0), _stump(1, 1.0, 0.8), _stump(2, 1.0, 0.6)),
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=3,
        feature_names=("a", "b", "c"),
        meta={},
    )


def _single_feature_ir(threshold: float) -> EnsembleIR:
    """One feature 'amount', one stump."""
    return EnsembleIR(
        trees=(_stump(0, threshold, 1.0),),
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=1,
        feature_names=("amount",),
        meta={},
    )


@pytest.fixture()
def exp() -> Explainer:
    return Explainer(_ir(), normalizers=np.ones(3))


X0 = np.zeros(3)


class TestExactOnlyKwargs:
    @pytest.mark.parametrize(
        "kwargs", [{"warm_start": False}, {"node_budget": 100}, {"gap": 0.1}]
    )
    def test_non_default_value_with_other_backend_raises(
        self, exp: Explainer, kwargs: dict[str, object]
    ) -> None:
        with pytest.raises(ValueError, match="only valid with backend='exact'"):
            exp.explain(X0, Target.raw(op=">=", value=0.5), backend="genetic", seed=0, **kwargs)

    @pytest.mark.parametrize("backend", ["genetic", "python"])
    def test_documented_defaults_with_other_backend_are_accepted(
        self, exp: Explainer, backend: str
    ) -> None:
        result = exp.explain(
            X0, Target.raw(op=">=", value=0.5), backend=backend, seed=0,
            warm_start=True, node_budget=2_000_000, gap=0.0,
        )
        assert isinstance(result, Counterfactual | Infeasible)

    def test_exact_backend_accepts_non_default_values(self, exp: Explainer) -> None:
        result = exp.explain(
            X0, Target.raw(op=">=", value=0.5), backend="exact", seed=0,
            warm_start=False, node_budget=1000, gap=0.05,
        )
        assert isinstance(result, Counterfactual | Infeasible)


class TestWarmStart:
    def test_node_budget_one_returns_the_warm_start_row_unchanged(self, exp: Explainer) -> None:
        target = Target.raw(op=">=", value=1.5)  # needs at least two levers
        genetic = exp.explain(X0, target, backend="genetic", seed=0)
        assert isinstance(genetic, Counterfactual)

        with pytest.warns(TreecfWarning, match="exhausted"):
            exact = exp.explain(
                X0, target, backend="exact", seed=0,
                warm_start=True, node_budget=1, time_budget_s=5.0,
            )
        assert isinstance(exact, Counterfactual)
        assert np.array_equal(exact.x_cf, genetic.x_cf, equal_nan=True)
        assert exact.proof == "heuristic"
        assert exact.solver_stats["completed"] is False
        assert exact.solver_stats["warm_start_used"] is True

    def test_warm_start_false_skips_the_genetic_pass(self, exp: Explainer) -> None:
        target = Target.raw(op=">=", value=1.5)
        with pytest.warns(TreecfWarning, match="exhausted"):
            result = exp.explain(
                X0, target, backend="exact", seed=0,
                warm_start=False, node_budget=1, time_budget_s=5.0,
            )
        assert isinstance(result, Infeasible)
        assert result.solver_stats["warm_start_used"] is False


class TestFactualAlreadyInTarget:
    def test_immediate_optimal_with_zero_nodes(self, exp: Explainer) -> None:
        target = Target.raw(op="<=", value=0.5)  # raw_score(X0) == 0.0 already satisfies
        result = exp.explain(X0, target, backend="exact", seed=0)
        assert isinstance(result, Counterfactual)
        assert result.proof == "optimal"
        assert result.distance == 0.0
        assert result.n_changed == 0
        assert result.solver_stats["nodes_expanded"] == 0


class TestBands:
    def test_mixes_counterfactual_and_certified_infeasible(self, exp: Explainer) -> None:
        target = Target.bands({"reachable": (0.9, 1.0), "unreachable": (3.0, 10.0)}, space="raw")
        result = exp.explain(X0, target, backend="exact", seed=0)
        assert isinstance(result, dict)
        assert isinstance(result["reachable"], Counterfactual)
        assert isinstance(result["unreachable"], Infeasible)
        assert result["unreachable"].proof == "certified"


def test_exact_backend_with_plausibility_smoke() -> None:
    if_ir = EnsembleIR(
        trees=(_stump(0, 0.5, 3.0),),
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=3,
        feature_names=("a", "b", "c"),
        meta={"max_samples": 16.0},
    )
    plaus = Plausibility(if_ir=if_ir, max_anomaly_score=0.99)  # loose bound: always plausible
    exp = Explainer(_ir(), normalizers=np.ones(3), plausibility=plaus)
    target = Target.raw(op=">=", value=0.5)
    result = exp.explain(X0, target, backend="exact", seed=0, time_budget_s=10.0)
    assert isinstance(result, Counterfactual | Infeasible)
    if isinstance(result, Counterfactual):
        assert plaus.anomaly_score(result.x_cf) <= plaus.max_anomaly_score + 1e-9


def test_exact_backend_value_policy_end_to_end() -> None:
    # warm_start=False: an untied warm-started row would report no snapping of
    # its own (the backend that produced it already applied the policy), so
    # this checks the exact search's own domain-tracked ``snapped`` instead.
    exp = Explainer(
        _single_feature_ir(2.5), normalizers=np.ones(1), value_policy={"amount": "integer"}
    )
    result = exp.explain(
        np.array([0.0]), Target.raw(op=">=", value=0.5), backend="exact", seed=0,
        warm_start=False,
    )
    assert isinstance(result, Counterfactual)
    assert result.x_cf[0] == 3.0  # optimal 2.5 snapped to the nearest integer in [2.5, inf)
    assert result.snapped == {"amount": True}


def test_exact_backend_grid_value_policy() -> None:
    exp = Explainer(
        _single_feature_ir(1000.3), normalizers=np.ones(1),
        value_policy={"amount": Grid(step=50.0)},
    )
    result = exp.explain(
        np.array([0.0]), Target.raw(op=">=", value=0.5), backend="exact", seed=0,
        warm_start=False,
    )
    assert isinstance(result, Counterfactual)
    assert result.x_cf[0] == 1050.0  # 1000 is below the threshold; next grid point inside
    assert result.snapped == {"amount": True}


def test_explain_coalitions_backend_exact(exp: Explainer) -> None:
    target = Target.raw(op=">=", value=0.9)  # unreachable via "a" or "b" alone
    result = exp.explain_coalitions(
        X0, target, {"first": ["a"], "rest": ["b", "c"]}, backend="exact", seed=0
    )
    assert isinstance(result["first"], Counterfactual)
    assert set(result["first"].changes) <= {"a"}
    assert isinstance(result["rest"], Counterfactual | Infeasible)
    if isinstance(result["rest"], Counterfactual):
        assert set(result["rest"].changes) <= {"b", "c"}


def test_explain_batch_exact_matches_per_row_explain(exp: Explainer) -> None:
    target = Target.raw(op=">=", value=0.9)  # unique cheapest solution: "a" alone
    X = np.zeros((2, 3))
    batch = exp.explain_batch(X, target, backend="exact", seed=0, allow_exact_batch=True)
    assert len(batch) == 2
    for i, row in enumerate(X):
        single = exp.explain(row, target, backend="exact", seed=0)
        assert isinstance(single, Counterfactual)
        record = batch.for_id(i)[0]
        assert record.feasible
        assert record.x_cf is not None
        assert np.array_equal(record.x_cf, single.x_cf, equal_nan=True)
        assert record.distance == single.distance


class TestValidationErrorsPropagate:
    def test_multi_feature_linear_names_genetic_fallback(self) -> None:
        exp = Explainer(_ir(), normalizers=np.ones(3), constraints=[constraint("a + b >= 1.5")])
        x = np.array([2.0, 0.0, 0.0])  # satisfies the constraint: no factual-violation warning
        target = Target.raw(op=">=", value=0.5)
        with pytest.raises(ConstraintValidationError, match="genetic"):
            exp.explain(x, target, backend="exact", seed=0)

    def test_callable_value_policy_names_genetic_fallback(self) -> None:
        exp = Explainer(
            _ir(), normalizers=np.ones(3), value_policy={"a": lambda v: float(np.ceil(v))}
        )
        target = Target.raw(op=">=", value=0.5)
        with pytest.raises(ConstraintValidationError, match="genetic"):
            exp.explain(X0, target, backend="exact", seed=0)


class TestProofTaxonomy:
    def test_documented_verbatim_on_dataclass_docstrings(self) -> None:
        assert '{"heuristic", "optimal", "optimal_within_gap"}' in (Counterfactual.__doc__ or "")
        assert '{"search_exhausted", "certified"}' in (Infeasible.__doc__ or "")

    @pytest.mark.parametrize(
        ("backend", "target_value"),
        [
            ("genetic", 0.9),
            ("python", 0.9),
            ("exact", 0.9),
            ("exact", 10.0),  # unreachable: max raw score is 2.4
        ],
    )
    def test_produced_results_only_carry_documented_proof_values(
        self, exp: Explainer, backend: str, target_value: float
    ) -> None:
        counterfactual_proofs = {"heuristic", "optimal", "optimal_within_gap"}
        infeasible_proofs = {"search_exhausted", "certified"}
        target = Target.raw(op=">=", value=target_value)
        result = exp.explain(X0, target, backend=backend, seed=0)
        if isinstance(result, Counterfactual):
            assert result.proof in counterfactual_proofs
        else:
            assert isinstance(result, Infeasible)
            assert result.proof in infeasible_proofs
