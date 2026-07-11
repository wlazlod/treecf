"""Linear / Implies / OneHot constraints end-to-end through CP-SAT (M2)."""

from __future__ import annotations

import numpy as np
import pytest

from treecf import (
    Counterfactual,
    Equals,
    Explainer,
    Freeze,
    Implies,
    Infeasible,
    OneHot,
    Target,
    constraint,
)
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree

pytest.importorskip("ortools")


def _leaf(i: int, v: float) -> Node:
    return Node(i, None, None, None, None, None, None, v)


def _stump_tree(feature: int, threshold: float, left: float, right: float) -> Tree:
    return Tree(
        nodes=(
            Node(0, feature, threshold, SplitOp.LT, True, 1, 2, None),
            _leaf(1, left),
            _leaf(2, right),
        )
    )


def _two_feature_ir() -> EnsembleIR:
    """S = 1[a >= 1] * 1.0 + 1[b >= 1] * 0.5 (as ±half contributions plus base)."""
    return EnsembleIR(
        trees=(_stump_tree(0, 1.0, 0.0, 1.0), _stump_tree(1, 1.0, 0.0, 0.5)),
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=2,
        feature_names=("a", "b"),
        meta={},
    )


def _flags_ir() -> EnsembleIR:
    """Three binary flags split at 0.5, contributing 1.0 / 0.5 / 0.25."""
    return EnsembleIR(
        trees=(
            _stump_tree(0, 0.5, 0.0, 1.0),
            _stump_tree(1, 0.5, 0.0, 0.5),
            _stump_tree(2, 0.5, 0.0, 0.25),
        ),
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=3,
        feature_names=("flag1", "flag2", "flag3"),
        meta={},
    )


class TestLinear:
    def test_order_constraint_shapes_the_optimum(self) -> None:
        # Target needs both indicators on; a <= b forces b to move at least as far as a.
        exp = Explainer(
            _two_feature_ir(),
            normalizers=np.ones(2),
            constraints=[constraint("a <= b")],
        )
        res = exp.explain(np.array([0.0, 0.0]), target=Target.raw(op=">=", value=1.4))
        assert isinstance(res, Counterfactual)
        assert res.x_cf[0] >= 1.0 and res.x_cf[1] >= 1.0
        assert res.x_cf[0] <= res.x_cf[1]

    def test_linear_with_frozen_feature_folds_to_constant(self) -> None:
        # b frozen at 0.5; a <= b caps a at 0.5, making the target unreachable.
        exp = Explainer(
            _two_feature_ir(),
            normalizers=np.ones(2),
            constraints=[Freeze("b"), constraint("a <= b")],
        )
        res = exp.explain(np.array([0.0, 0.5]), target=Target.raw(op=">=", value=0.9))
        assert isinstance(res, Infeasible)

    def test_solution_always_satisfies_linear_in_float(self) -> None:
        exp = Explainer(
            _two_feature_ir(),
            normalizers=np.ones(2),
            constraints=[constraint("a + b <= 2.5")],
        )
        res = exp.explain(np.array([0.0, 0.0]), target=Target.raw(op=">=", value=1.4))
        assert isinstance(res, Counterfactual)
        assert res.x_cf[0] + res.x_cf[1] <= 2.5


class TestImpliesAndOneHot:
    def test_implies_propagates_flag_change(self) -> None:
        # Reaching S >= 1.0 needs flag1 = 1; Implies then forces flag2 = 1 too.
        exp = Explainer(
            _flags_ir(),
            normalizers=np.ones(3),
            constraints=[Implies(Equals("flag1", 1.0), Equals("flag2", 1.0))],
        )
        res = exp.explain(np.array([0.0, 0.0, 0.0]), target=Target.raw(op=">=", value=1.0))
        assert isinstance(res, Counterfactual)
        assert res.x_cf[0] == 1.0 and res.x_cf[1] == 1.0

    def test_onehot_keeps_exactly_one_flag(self) -> None:
        exp = Explainer(
            _flags_ir(),
            normalizers=np.ones(3),
            constraints=[OneHot(("flag1", "flag2", "flag3"))],
        )
        # start in a valid one-hot state (flag3 on); ask for the flag1 contribution
        res = exp.explain(np.array([0.0, 0.0, 1.0]), target=Target.raw(op=">=", value=1.0))
        assert isinstance(res, Counterfactual)
        assert res.x_cf[0] == 1.0
        assert res.x_cf[1] == 0.0 and res.x_cf[2] == 0.0

    def test_binary_features_only_take_zero_or_one(self) -> None:
        exp = Explainer(
            _flags_ir(),
            normalizers=np.ones(3),
            constraints=[Equals("flag3", 0.0), OneHot(("flag1", "flag2"))],
        )
        res = exp.explain(np.array([0.0, 1.0, 0.0]), target=Target.raw(op=">=", value=0.9))
        assert isinstance(res, Counterfactual)
        assert set(np.unique(res.x_cf)) <= {0.0, 1.0}
