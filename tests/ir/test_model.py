"""IR dataclass construction and immutability."""

import dataclasses

import pytest

from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree


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


def _split(
    node_id: int, feature: int, threshold: float, op: SplitOp, left: int, right: int
) -> Node:
    return Node(
        node_id=node_id,
        feature=feature,
        threshold=threshold,
        op=op,
        missing_left=True,
        left=left,
        right=right,
        value=None,
    )


def test_construct_single_split_ensemble() -> None:
    tree = Tree(nodes=(_split(0, 0, 1.0, SplitOp.LT, 1, 2), _leaf(1, -0.5), _leaf(2, 0.5)))
    ir = EnsembleIR(
        trees=(tree,),
        base_score=0.1,
        link=Link.SIGMOID,
        n_features=1,
        feature_names=("x0",),
        meta={"source": "test"},
    )
    assert ir.trees[0].nodes[0].op is SplitOp.LT
    assert ir.trees[0].nodes[1].value == -0.5
    assert ir.link is Link.SIGMOID


def test_ir_dataclasses_are_frozen() -> None:
    leaf = _leaf(0, 1.0)
    tree = Tree(nodes=(leaf,))
    ir = EnsembleIR(
        trees=(tree,),
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=0,
        feature_names=(),
        meta={},
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        leaf.value = 2.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        tree.nodes = ()  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        ir.base_score = 1.0  # type: ignore[misc]
