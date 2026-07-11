"""Diversity via iterative no-good cuts (spec §8.3, D8)."""

from __future__ import annotations

import numpy as np
import pytest

from treecf import Counterfactual, Explainer, Target
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree

pytest.importorskip("ortools")


def _leaf(i: int, v: float) -> Node:
    return Node(i, None, None, None, None, None, None, v)


def _multi_path_ir() -> EnsembleIR:
    """Three features, each alone able to reach the target (stumps of value 1 each)."""
    trees = tuple(
        Tree(
            nodes=(
                Node(0, j, 1.0, SplitOp.LT, True, 1, 2, None),
                _leaf(1, 0.0),
                _leaf(2, 1.0),
            )
        )
        for j in range(3)
    )
    return EnsembleIR(
        trees=trees,
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=3,
        feature_names=("a", "b", "c"),
        meta={},
    )


def test_distinct_change_sets() -> None:
    exp = Explainer(_multi_path_ir(), normalizers=np.array([1.0, 2.0, 3.0]))
    results = exp.explain(
        np.zeros(3),
        target=Target.raw(op=">=", value=0.5),
        n_counterfactuals=3,
    )
    assert isinstance(results, list) and len(results) == 3
    assert all(isinstance(r, Counterfactual) for r in results)
    change_sets = [frozenset(r.changes) for r in results]
    assert len(set(change_sets)) == 3  # pairwise distinct change-sets
    distances = [r.distance for r in results]
    assert distances == sorted(distances)  # non-decreasing cost
    assert change_sets[0] == frozenset({"c"})  # largest normalizer = cheapest move first


def test_diversity_exhausts_gracefully() -> None:
    exp = Explainer(_multi_path_ir(), normalizers=np.ones(3))
    results = exp.explain(
        np.zeros(3),
        target=Target.raw(op=">=", value=2.5),  # needs all three features changed
        n_counterfactuals=3,
    )
    assert isinstance(results, list)
    assert len(results) == 1  # only one change-set can reach the target
