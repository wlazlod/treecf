"""Certified infeasibility, end to end: API ``Infeasible(proof="certified")`` verdicts,
each checked against a complete-enumeration oracle so the certificate is never taken on faith.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from treecf import Equals, Explainer, Freeze, Infeasible, OneHot, Range, Target, TreecfWarning
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree

from .brute_force import solve_brute_force

_REASON_RE = re.compile(
    r"no counterfactual exists in the target interval under the given constraints "
    r"\(certified; \d+ nodes\)"
)


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


def _assert_certified(exp: Explainer, x: np.ndarray, target: Target) -> None:
    result = exp.explain(x, target, backend="exact", seed=0, time_budget_s=30.0)
    assert isinstance(result, Infeasible)
    assert result.proof == "certified"
    assert _REASON_RE.fullmatch(result.reason), result.reason
    assert result.solver_stats["completed"] is True

    interval = target.raw_interval(exp.ir.link)
    oracle = solve_brute_force(
        exp.ir, np.asarray(x, dtype=np.float64), interval,
        exp.compiled, exp.sigma, exp.weights, 0.0,
    )
    assert not oracle.feasible, "oracle disagrees: it found a feasible row"


def test_target_above_ensemble_maximum() -> None:
    # two independent levers worth 1.0 and 0.8; max raw score 1.8
    ir = EnsembleIR(
        trees=(_stump(0, 1.0, 1.0), _stump(1, 1.0, 0.8)),
        base_score=0.0, link=Link.IDENTITY, n_features=2,
        feature_names=("a", "b"), meta={},
    )
    exp = Explainer(ir, normalizers=np.ones(2))
    _assert_certified(exp, np.zeros(2), Target.raw(range=(5.0, 10.0)))


def test_freeze_on_only_influential_feature_with_unreachable_target() -> None:
    ir = EnsembleIR(
        trees=(_stump(0, 1.0, 1.0),),
        base_score=0.0, link=Link.IDENTITY, n_features=1,
        feature_names=("amount",), meta={},
    )
    exp = Explainer(ir, normalizers=np.ones(1), constraints=[Freeze("amount")])
    _assert_certified(exp, np.zeros(1), Target.raw(range=(1.0, 2.0)))


def test_equals_pins_feature_into_off_target_cell() -> None:
    # "a" alone can reach 1.0; "b" alone tops out at 0.3
    ir = EnsembleIR(
        trees=(_stump(0, 1.0, 1.0), _stump(1, 1.0, 0.3)),
        base_score=0.0, link=Link.IDENTITY, n_features=2,
        feature_names=("a", "b"), meta={},
    )
    exp = Explainer(ir, normalizers=np.ones(2), constraints=[Equals("a", 0.0)])
    _assert_certified(exp, np.zeros(2), Target.raw(range=(1.0, 2.0)))


def test_onehot_group_forces_an_off_target_branch() -> None:
    # OneHot forces exactly one of a/b on; even both branches "on" at once
    # (which the group forbids) would only sum to 0.2 + 0.1 = 0.3
    ir = EnsembleIR(
        trees=(_stump(0, 0.5, 0.2), _stump(1, 0.5, 0.1)),
        base_score=0.0, link=Link.IDENTITY, n_features=2,
        feature_names=("a", "b"), meta={},
    )
    exp = Explainer(ir, normalizers=np.ones(2), constraints=[OneHot(("a", "b"))])
    _assert_certified(exp, np.array([1.0, 0.0]), Target.raw(range=(0.5, 2.0)))


def test_factual_violating_a_constraint_still_certifies() -> None:
    ir = EnsembleIR(
        trees=(_stump(0, 2.5, 1.0),),
        base_score=0.0, link=Link.IDENTITY, n_features=1,
        feature_names=("amount",), meta={},
    )
    exp = Explainer(ir, normalizers=np.ones(1), constraints=[Range("amount", 2.0, 3.0)])
    x = np.array([0.0])  # violates Range("amount", 2.0, 3.0) by design
    target = Target.raw(range=(5.0, 10.0))  # unreachable even once "amount" moves into range

    with pytest.warns(TreecfWarning):
        result = exp.explain(x, target, backend="exact", seed=0, time_budget_s=30.0)
    assert isinstance(result, Infeasible)
    assert result.proof == "certified"
    assert _REASON_RE.fullmatch(result.reason), result.reason

    interval = target.raw_interval(exp.ir.link)
    oracle = solve_brute_force(
        exp.ir, x, interval, exp.compiled, exp.sigma, exp.weights, 0.0
    )
    assert not oracle.feasible
