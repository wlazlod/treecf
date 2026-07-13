"""Post-solve pruning: changes that verification proves unnecessary are reverted."""

from __future__ import annotations

import numpy as np
import pytest

from treecf import Counterfactual, Explainer, Target
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree


def _stump(feature: int, threshold: float, right_value: float) -> Tree:
    return Tree(
        nodes=(
            Node(0, feature, threshold, SplitOp.LT, True, 1, 2, None),
            Node(1, None, None, None, None, None, None, 0.0),
            Node(2, None, None, None, None, None, None, right_value),
        )
    )


def _ir() -> EnsembleIR:
    return EnsembleIR(
        trees=(_stump(0, 1.0, 1.0), _stump(1, 1.0, 0.8), _stump(2, 1.0, 0.6)),
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=3,
        feature_names=("a", "b", "c"),
        meta={},
    )


X0 = np.zeros(3)
TARGET = Target.raw(op=">=", value=0.5)
INTERVAL = TARGET.raw_interval(Link.IDENTITY)


@pytest.fixture()
def exp() -> Explainer:
    return Explainer(_ir(), normalizers=np.ones(3))


def test_residual_micro_change_is_pruned(exp: Explainer) -> None:
    # feature a crosses its threshold (necessary); b moves 0.3 without crossing
    # anything — pure cost, zero score effect: the classic GA residue
    candidate = np.array([2.0, 0.3, 0.0])
    result = exp._finalize_candidate(X0, candidate, INTERVAL, stats={})
    assert isinstance(result, Counterfactual)
    assert set(result.changes) == {"a"}
    assert result.n_changed == 1


def test_necessary_changes_survive_pruning(exp: Explainer) -> None:
    # the target >= 1.5 needs BOTH a (1.0) and b (0.8): neither may be pruned
    interval = Target.raw(op=">=", value=1.5).raw_interval(Link.IDENTITY)
    candidate = np.array([2.0, 2.0, 0.0])
    result = exp._finalize_candidate(X0, candidate, interval, stats={})
    assert isinstance(result, Counterfactual)
    assert set(result.changes) == {"a", "b"}


def test_returned_plans_are_minimal_across_seeds(exp: Explainer) -> None:
    """Every change in a returned plan is necessary: reverting it alone breaks
    verification."""
    for seed in range(10):
        result = exp.explain(X0, TARGET, seed=seed)
        assert isinstance(result, Counterfactual)
        index = {name: j for j, name in enumerate(exp.ir.feature_names)}
        for name in result.changes:
            trial = result.x_cf.copy()
            trial[index[name]] = X0[index[name]]
            assert exp._verify(X0, trial, INTERVAL) is not None, (
                f"seed {seed}: change {name!r} was unnecessary"
            )
