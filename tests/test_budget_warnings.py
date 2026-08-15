"""Honest warnings for a degraded exact-search result (``stats["completed"] is False``).

A ``completed=False`` result has two different, non-overlapping causes -- the
budget genuinely ran out, or a conservative constraint repair withdrew the
optimality certificate without touching the budget -- and the warning must
never claim the wrong one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from treecf import Counterfactual, Explainer, Infeasible, Target, TreecfWarning, constraint
from treecf.api import _gap_parenthetical
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree


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


@pytest.fixture()
def exp() -> Explainer:
    return Explainer(_ir(), normalizers=np.ones(3))


X0 = np.zeros(3)


class TestGapParenthetical:
    """``_gap_parenthetical`` in isolation: the omission rules are exact, and an
    end-to-end scenario that forces a finite, positive lower bound mid-search
    is not straightforward to construct, so this covers the formula directly."""

    def test_present_when_lower_bound_is_finite_and_positive(self) -> None:
        assert _gap_parenthetical(1.5, 1.0) == " (lower bound 1, gap ≤ 50.0%)"

    def test_omitted_when_no_row(self) -> None:
        assert _gap_parenthetical(None, 1.0) == ""

    def test_omitted_when_lower_bound_is_zero(self) -> None:
        assert _gap_parenthetical(1.0, 0.0) == ""

    def test_omitted_when_lower_bound_is_negative(self) -> None:
        assert _gap_parenthetical(1.0, -1.0) == ""

    def test_omitted_when_lower_bound_is_infinite(self) -> None:
        assert _gap_parenthetical(1.0, math.inf) == ""


class TestExhaustionBodies:
    """node_budget=1 exhausts on the very first assignment; the two bodies
    differ only by whether the warm start had already produced a row."""

    def test_incumbent_exists_uses_body_a(self, exp: Explainer) -> None:
        target = Target.raw(op=">=", value=1.5)  # needs at least two levers
        with pytest.warns(TreecfWarning, match="exhausted") as record:
            result = exp.explain(
                X0, target, backend="exact", seed=0,
                warm_start=True, node_budget=1, time_budget_s=5.0,
            )
        assert isinstance(result, Counterfactual)
        assert result.solver_stats["completed"] is False
        assert len(record) == 1
        message = str(record[0].message)
        assert "the result is the best found, not proven optimal" in message
        assert "raise node_budget/time_budget_s" in message

    def test_no_incumbent_uses_body_b(self, exp: Explainer) -> None:
        target = Target.raw(op=">=", value=1.5)
        with pytest.warns(TreecfWarning, match="exhausted") as record:
            result = exp.explain(
                X0, target, backend="exact", seed=0,
                warm_start=False, node_budget=1, time_budget_s=5.0,
            )
        assert isinstance(result, Infeasible)
        assert result.solver_stats["completed"] is False
        assert len(record) == 1
        message = str(record[0].message)
        assert "without finding a feasible counterfactual" in message
        assert "NOT a certified infeasibility" in message


class TestWithdrawalBody:
    """Declaring the same order pair twice trips the conservative repair's
    "several pairs sharing features" fallback -- a completion is set aside
    although the whole budget was never touched."""

    def test_duplicate_order_pair_withdraws_without_exhaustion(self) -> None:
        withdrawing = Explainer(
            _ir(), normalizers=np.ones(3),
            constraints=[constraint("a <= b"), constraint("a <= b")],
        )
        target = Target.raw(op=">=", value=0.5)
        with pytest.warns(TreecfWarning) as record:
            result = withdrawing.explain(
                X0, target, backend="exact", seed=0,
                warm_start=False, node_budget=2_000_000, time_budget_s=10.0,
            )
        assert isinstance(result, Counterfactual)
        assert result.proof == "heuristic"
        stats = result.solver_stats
        assert stats["completed"] is False
        assert stats["nodes_expanded"] < 100  # budget (2_000_000) nowhere near touched
        assert len(record) == 1
        message = str(record[0].message)
        # the cardinal rule: a withdrawal must never be spelled as an exhaustion
        assert "exhausted" not in message
        assert "withdrew its optimality certificate" in message
        assert "not proven cheapest" in message


class TestSeedClause:
    """Appended iff ``seed is None`` and the warm start actually produced the
    incumbent the exact search used; both polarities exercised at node_budget=1
    where the warm start is guaranteed to have run and to have been used."""

    def test_present_when_seed_none_and_warm_start_used(self, exp: Explainer) -> None:
        target = Target.raw(op=">=", value=1.5)
        with pytest.warns(TreecfWarning, match="exhausted") as record:
            result = exp.explain(
                X0, target, backend="exact", seed=None,
                warm_start=True, node_budget=1, time_budget_s=5.0,
            )
        assert isinstance(result, Counterfactual)
        assert result.solver_stats["warm_start_used"] is True
        assert "Warm start was unseeded" in str(record[0].message)

    def test_absent_when_seed_given(self, exp: Explainer) -> None:
        target = Target.raw(op=">=", value=1.5)
        with pytest.warns(TreecfWarning, match="exhausted") as record:
            result = exp.explain(
                X0, target, backend="exact", seed=0,
                warm_start=True, node_budget=1, time_budget_s=5.0,
            )
        assert isinstance(result, Counterfactual)
        assert result.solver_stats["warm_start_used"] is True
        assert "Warm start was unseeded" not in str(record[0].message)


class TestNeverWarns:
    """completed/certified/genetic results never carry this warning; the whole
    suite's ``filterwarnings = ["error"]`` is the real net, this just names
    the three cases explicitly."""

    def test_completed_counterfactual_never_warns(
        self, exp: Explainer, recwarn: pytest.WarningsRecorder
    ) -> None:
        target = Target.raw(op=">=", value=0.5)
        result = exp.explain(X0, target, backend="exact", seed=0)
        assert isinstance(result, Counterfactual)
        assert result.solver_stats["completed"] is True
        assert not any(issubclass(w.category, TreecfWarning) for w in recwarn.list)

    def test_certified_infeasible_never_warns(
        self, exp: Explainer, recwarn: pytest.WarningsRecorder
    ) -> None:
        target = Target.raw(op=">=", value=10.0)  # unreachable: max raw score is 2.4
        result = exp.explain(X0, target, backend="exact", seed=0)
        assert isinstance(result, Infeasible)
        assert result.proof == "certified"
        assert not any(issubclass(w.category, TreecfWarning) for w in recwarn.list)

    def test_genetic_backend_never_warns(
        self, exp: Explainer, recwarn: pytest.WarningsRecorder
    ) -> None:
        target = Target.raw(op=">=", value=0.5)
        result = exp.explain(X0, target, backend="genetic", seed=0)
        assert isinstance(result, Counterfactual | Infeasible)
        assert not any(issubclass(w.category, TreecfWarning) for w in recwarn.list)


class TestAggregateWarnings:
    """bands/coalitions/batch each collapse every degraded solve in one call
    into exactly one ``TreecfWarning``."""

    def test_bands_aggregate_is_exactly_one_warning(self, exp: Explainer) -> None:
        target = Target.bands({"lo": (0.5, 0.7), "hi": (1.3, 1.5)}, space="raw")
        with pytest.warns(TreecfWarning) as record:
            result = exp.explain(
                X0, target, backend="exact", seed=0,
                warm_start=False, node_budget=1, time_budget_s=5.0,
            )
        assert isinstance(result, dict)
        assert len(record) == 1
        assert "2/2 bands" in str(record[0].message)

    def test_coalitions_aggregate_is_exactly_one_warning(self, exp: Explainer) -> None:
        target = Target.raw(op=">=", value=0.5)
        with pytest.warns(TreecfWarning) as record:
            result = exp.explain_coalitions(
                X0, target, {"c1": ["a"], "c2": ["b", "c"]}, backend="exact", seed=0,
                warm_start=False, node_budget=1, time_budget_s=5.0,
            )
        assert set(result) == {"c1", "c2"}
        assert len(record) == 1
        assert "2/2 coalitions" in str(record[0].message)

    def test_batch_aggregate_is_exactly_one_warning(self, exp: Explainer) -> None:
        target = Target.raw(op=">=", value=0.5)
        X = np.zeros((3, 3))
        with pytest.warns(TreecfWarning) as record:
            batch = exp.explain_batch(
                X, target, backend="exact", seed=0,
                warm_start=False, node_budget=1, time_budget_s=5.0,
            )
        assert len(batch) == 3
        assert len(record) == 1
        assert "3/3 rows" in str(record[0].message)
