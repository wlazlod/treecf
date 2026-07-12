"""Per-feature value policies: integer/grid snapping inside the chosen cell (spec §5.6)."""

from __future__ import annotations

import numpy as np
import pytest

from treecf import Counterfactual, Explainer, Grid, Range, Target
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree


def _leaf(i: int, v: float) -> Node:
    return Node(i, None, None, None, None, None, None, v)


def _ir(thresholds: list[float]) -> EnsembleIR:
    """One stump per threshold on feature 0, each contributing +1 on the right side."""
    trees = tuple(
        Tree(
            nodes=(
                Node(0, 0, t, SplitOp.LT, True, 1, 2, None),
                _leaf(1, 0.0),
                _leaf(2, 1.0),
            )
        )
        for t in thresholds
    )
    return EnsembleIR(
        trees=trees,
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=1,
        feature_names=("amount",),
        meta={},
    )


def test_integer_policy_snaps_up_within_cell() -> None:
    exp = Explainer(_ir([2.5]), normalizers=np.ones(1), value_policy={"amount": "integer"})
    res = exp.explain(np.array([0.0]), target=Target.raw(op=">=", value=0.5), seed=0)
    assert isinstance(res, Counterfactual)
    assert res.x_cf[0] == 3.0  # optimal 2.5 snapped to the nearest integer in [2.5, inf)
    assert res.snapped == {"amount": True}


def test_integer_policy_keeps_raw_when_no_integer_in_cell() -> None:
    # middle cell [2.5, 2.9) contains no integer
    exp = Explainer(_ir([2.5, 2.9]), normalizers=np.ones(1), value_policy={"amount": "integer"})
    res = exp.explain(np.array([0.0]), target=Target.raw(range=(0.5, 1.5)))
    assert isinstance(res, Counterfactual)
    assert res.x_cf[0] == pytest.approx(2.5)
    assert res.snapped == {"amount": False}


def test_grid_policy_snaps_to_step() -> None:
    exp = Explainer(
        _ir([1000.3]),
        normalizers=np.ones(1),
        value_policy={"amount": Grid(step=50.0)},
    )
    res = exp.explain(np.array([0.0]), target=Target.raw(op=">=", value=0.5), seed=0)
    assert isinstance(res, Counterfactual)
    assert res.x_cf[0] == 1050.0  # 1000 is below the threshold; next grid point inside
    assert res.snapped == {"amount": True}


def test_unchanged_features_are_not_snapped() -> None:
    exp = Explainer(_ir([2.5]), normalizers=np.ones(1), value_policy={"amount": "integer"})
    res = exp.explain(np.array([3.7]), target=Target.raw(op=">=", value=0.5), seed=0)
    assert isinstance(res, Counterfactual)
    assert res.x_cf[0] == 3.7  # already in target: no change, no snapping
    assert res.n_changed == 0


def test_snapping_never_violates_constraints() -> None:
    # Range caps at 2.7: integer snap to 3 would violate; expect raw 2.5 kept
    exp = Explainer(
        _ir([2.5]),
        normalizers=np.ones(1),
        constraints=[Range("amount", 0.0, 2.7)],
        value_policy={"amount": "integer"},
    )
    res = exp.explain(np.array([0.0]), target=Target.raw(op=">=", value=0.5), seed=0)
    assert isinstance(res, Counterfactual)
    assert res.x_cf[0] == pytest.approx(2.5)
    assert res.snapped == {"amount": False}


def test_callable_policy() -> None:
    exp = Explainer(
        _ir([2.5]),
        normalizers=np.ones(1),
        value_policy={"amount": lambda v: float(np.ceil(v))},
    )
    res = exp.explain(np.array([0.0]), target=Target.raw(op=">=", value=0.5), seed=0)
    assert isinstance(res, Counterfactual)
    assert res.x_cf[0] == 3.0
