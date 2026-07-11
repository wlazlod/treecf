"""Float64 evaluation of the IR — the reference semantics every backend is verified against."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree


def raw_score(ir: EnsembleIR, x: npt.NDArray[np.float64]) -> float:
    """Raw score ``S(x) = base_score + sum of leaf values`` (spec §3.1)."""
    total = ir.base_score
    for tree in ir.trees:
        total += _leaf_value(tree, x)
    return float(total)


def apply_link(link: Link, score: float) -> float:
    if link is Link.SIGMOID:
        return 1.0 / (1.0 + math.exp(-score))
    return score


def leaf_assignment(ir: EnsembleIR, x: npt.NDArray[np.float64]) -> tuple[int, ...]:
    """Leaf node_id reached in each tree — the routing fingerprint of ``x``."""
    return tuple(_leaf_node(tree, x).node_id for tree in ir.trees)


def _leaf_value(tree: Tree, x: npt.NDArray[np.float64]) -> float:
    value = _leaf_node(tree, x).value
    assert value is not None
    return value


def _leaf_node(tree: Tree, x: npt.NDArray[np.float64]) -> Node:
    node = tree.nodes[0]
    while node.feature is not None:
        assert node.threshold is not None and node.left is not None and node.right is not None
        value = float(x[node.feature])
        if math.isnan(value):
            if node.missing_left is None:
                raise ValueError(
                    f"NaN at feature {node.feature} but node {node.node_id} "
                    "defines no missing routing"
                )
            child = node.left if node.missing_left else node.right
        else:
            go_left = value < node.threshold if node.op is SplitOp.LT else value <= node.threshold
            child = node.left if go_left else node.right
        node = tree.nodes[child]
    return node
