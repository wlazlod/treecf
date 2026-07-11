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


def raw_score_batch(ir: EnsembleIR, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Vectorized ``raw_score`` over the rows of X (used by the genetic backend)."""
    n = X.shape[0]
    total = np.full(n, ir.base_score, dtype=np.float64)
    for tree in ir.trees:
        feature = np.array([-1 if nd.feature is None else nd.feature for nd in tree.nodes])
        threshold = np.array(
            [np.nan if nd.threshold is None else nd.threshold for nd in tree.nodes]
        )
        is_lt = np.array([nd.op is SplitOp.LT for nd in tree.nodes])
        miss_left = np.array([bool(nd.missing_left) for nd in tree.nodes])
        left = np.array([-1 if nd.left is None else nd.left for nd in tree.nodes])
        right = np.array([-1 if nd.right is None else nd.right for nd in tree.nodes])
        value = np.array([0.0 if nd.value is None else nd.value for nd in tree.nodes])

        idx = np.zeros(n, dtype=np.int64)
        active = feature[idx] >= 0
        while active.any():
            rows = np.flatnonzero(active)
            nodes = idx[rows]
            v = X[rows, feature[nodes]]
            nan_mask = np.isnan(v)
            go_left = np.where(
                nan_mask,
                miss_left[nodes],
                np.where(is_lt[nodes], v < threshold[nodes], v <= threshold[nodes]),
            )
            idx[rows] = np.where(go_left, left[nodes], right[nodes])
            active[rows] = feature[idx[rows]] >= 0
        total += value[idx]
    return total


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
