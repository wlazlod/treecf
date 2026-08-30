"""Set-membership splits: IR construction, routing, and value validation."""

import math

import numpy as np
import pytest

from treecf._errors import TreecfError
from treecf.ir.evaluate import leaf_assignment, raw_score, raw_score_batch
from treecf.ir.model import (
    CategoricalFeature,
    EnsembleIR,
    Link,
    Node,
    SplitOp,
    Tree,
    validate_feature_matrix,
)

LEFT = -1.0
RIGHT = 1.0


def _leaf(node_id: int, value: float) -> Node:
    return Node(
        node_id=node_id,
        feature=None,
        threshold=None,
        op=None,
        missing_left=None,
        left=None,
        right=None,
        value=value,
    )


def _set_split(
    node_id: int,
    feature: int,
    categories: frozenset[int],
    left: int,
    right: int,
    missing_left: bool = True,
) -> Node:
    return Node(
        node_id=node_id,
        feature=feature,
        threshold=None,
        op=None,
        missing_left=missing_left,
        left=left,
        right=right,
        value=None,
        categories=categories,
    )


def _set_ir(
    categories: frozenset[int], cardinality: int = 4, missing_left: bool = True
) -> EnsembleIR:
    """One tree: set split on feature 0; left leaf = -1.0, right leaf = +1.0."""
    nodes = (
        _set_split(0, 0, categories, 1, 2, missing_left=missing_left),
        _leaf(1, LEFT),
        _leaf(2, RIGHT),
    )
    return EnsembleIR(
        trees=(Tree(nodes=nodes),),
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=1,
        feature_names=("occupation",),
        meta={"source": "test"},
        categorical={0: CategoricalFeature(cardinality=cardinality)},
    )


def test_numeric_ir_carries_no_categorical_metadata() -> None:
    nodes = (
        Node(0, 0, 1.0, SplitOp.LT, True, 1, 2, None),
        _leaf(1, LEFT),
        _leaf(2, RIGHT),
    )
    ir = EnsembleIR(
        trees=(Tree(nodes=nodes),),
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=1,
        feature_names=("x0",),
        meta={},
    )
    assert ir.categorical == {}
    assert ir.trees[0].nodes[0].categories is None


def test_member_code_routes_left() -> None:
    ir = _set_ir(frozenset({0, 2}))
    assert raw_score(ir, np.array([0.0])) == LEFT
    assert raw_score(ir, np.array([2.0])) == LEFT


def test_nonmember_code_routes_right() -> None:
    ir = _set_ir(frozenset({0, 2}))
    assert raw_score(ir, np.array([1.0])) == RIGHT
    assert raw_score(ir, np.array([3.0])) == RIGHT


def test_unseen_code_routes_right() -> None:
    ir = _set_ir(frozenset({0, 2}), cardinality=4)
    assert raw_score(ir, np.array([7.0])) == RIGHT


def test_nan_routes_by_missing_left() -> None:
    assert raw_score(_set_ir(frozenset({1}), missing_left=True), np.array([math.nan])) == LEFT
    assert raw_score(_set_ir(frozenset({1}), missing_left=False), np.array([math.nan])) == RIGHT


def test_non_integral_value_is_not_a_member() -> None:
    ir = _set_ir(frozenset({0, 2}))
    assert raw_score(ir, np.array([1.5])) == RIGHT
    assert raw_score(ir, np.array([2.5])) == RIGHT


def test_batch_matches_scalar_routing() -> None:
    ir = _set_ir(frozenset({0, 3}), cardinality=5)
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [7.0], [math.nan], [2.5]])
    batch = raw_score_batch(ir, X)
    scalar = np.array([raw_score(ir, row) for row in X])
    np.testing.assert_array_equal(batch, scalar)


def test_leaf_assignment_on_set_split() -> None:
    ir = _set_ir(frozenset({1}))
    assert leaf_assignment(ir, np.array([1.0])) == (1,)
    assert leaf_assignment(ir, np.array([0.0])) == (2,)


def test_membership_across_word_boundary() -> None:
    ir = _set_ir(frozenset({63, 64, 100}), cardinality=128)
    for code, expected in [(63.0, LEFT), (64.0, LEFT), (100.0, LEFT), (65.0, RIGHT)]:
        assert raw_score(ir, np.array([code])) == expected
    X = np.array([[63.0], [64.0], [100.0], [65.0], [127.0]])
    np.testing.assert_array_equal(
        raw_score_batch(ir, X), np.array([LEFT, LEFT, LEFT, RIGHT, RIGHT])
    )


def test_validate_accepts_codes_and_nan() -> None:
    ir = _set_ir(frozenset({1}), cardinality=4)
    validate_feature_matrix(ir, np.array([0.0]), where="factual")
    validate_feature_matrix(ir, np.array([3.0]), where="factual")
    validate_feature_matrix(ir, np.array([[math.nan], [2.0]]), where="background")


@pytest.mark.parametrize("bad", [2.5, -1.0, 4.0, math.inf])
def test_validate_rejects_non_codes(bad: float) -> None:
    ir = _set_ir(frozenset({1}), cardinality=4)
    with pytest.raises(TreecfError, match="occupation"):
        validate_feature_matrix(ir, np.array([bad]), where="factual")


def test_validate_ignores_numeric_features() -> None:
    nodes = (Node(0, 0, 1.0, SplitOp.LT, True, 1, 2, None), _leaf(1, LEFT), _leaf(2, RIGHT))
    ir = EnsembleIR(
        trees=(Tree(nodes=nodes),),
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=1,
        feature_names=("x0",),
        meta={},
    )
    validate_feature_matrix(ir, np.array([2.5]), where="factual")
