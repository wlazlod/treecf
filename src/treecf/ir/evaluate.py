"""Float64 evaluation of the IR — the reference semantics every backend is verified against."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree, code_goes_left


def raw_score(ir: EnsembleIR, x: npt.NDArray[np.float64]) -> float:
    """Raw score ``S(x) = base_score + sum of leaf values``."""
    total = ir.base_score
    for tree in ir.trees:
        total += _leaf_value(tree, x)
    return float(total)


def apply_link(link: Link, score: float) -> float:
    if link is Link.SIGMOID:
        # branch on sign: math.exp raises OverflowError for exponents above ~709
        if score >= 0.0:
            return 1.0 / (1.0 + math.exp(-score))
        e = math.exp(score)
        return e / (1.0 + e)
    return score


def leaf_assignment(ir: EnsembleIR, x: npt.NDArray[np.float64]) -> tuple[int, ...]:
    """Leaf node_id reached in each tree — the routing fingerprint of ``x``."""
    return tuple(_leaf_node(tree, x).node_id for tree in ir.trees)


@dataclass(frozen=True)
class TreeArrays:
    """One tree's node fields as flat arrays, ready for vectorized traversal.

    ``set_id`` is -1 for numeric splits and leaves; a set-membership split
    stores an index into the tree's bitset table (``set_offsets`` CSR into
    ``set_words``, 64 codes per word, code ``c`` at word ``c >> 6`` bit
    ``c & 63``).
    """

    feature: npt.NDArray[np.int64]
    threshold: npt.NDArray[np.float64]
    is_lt: npt.NDArray[np.bool_]
    miss_left: npt.NDArray[np.bool_]
    left: npt.NDArray[np.int64]
    right: npt.NDArray[np.int64]
    value: npt.NDArray[np.float64]
    set_id: npt.NDArray[np.int64]
    set_offsets: npt.NDArray[np.int64]
    set_words: npt.NDArray[np.uint64]


def bitset_words(categories: frozenset[int]) -> list[int]:
    """The set as little-endian 64-bit words: code ``c`` -> word c>>6, bit c&63."""
    words = [0] * ((max(categories) >> 6) + 1)
    for code in categories:
        words[code >> 6] |= 1 << (code & 63)
    return words


def prepare_tree_arrays(ir: EnsembleIR) -> tuple[TreeArrays, ...]:
    """Per-tree arrays for ``raw_score_batch_prepared`` (cacheable per IR)."""
    prepared = []
    for tree in ir.trees:
        set_id = np.full(len(tree.nodes), -1, dtype=np.int64)
        offsets = [0]
        words: list[int] = []
        for i, nd in enumerate(tree.nodes):
            if nd.categories is not None:
                set_id[i] = len(offsets) - 1
                words.extend(bitset_words(nd.categories))
                offsets.append(len(words))
        prepared.append(
            TreeArrays(
                feature=np.array([-1 if nd.feature is None else nd.feature for nd in tree.nodes]),
                threshold=np.array(
                    [np.nan if nd.threshold is None else nd.threshold for nd in tree.nodes]
                ),
                is_lt=np.array([nd.op is SplitOp.LT for nd in tree.nodes]),
                miss_left=np.array([bool(nd.missing_left) for nd in tree.nodes]),
                left=np.array([-1 if nd.left is None else nd.left for nd in tree.nodes]),
                right=np.array([-1 if nd.right is None else nd.right for nd in tree.nodes]),
                value=np.array([0.0 if nd.value is None else nd.value for nd in tree.nodes]),
                set_id=set_id,
                set_offsets=np.array(offsets, dtype=np.int64),
                set_words=np.array(words, dtype=np.uint64),
            )
        )
    return tuple(prepared)


def _set_membership(
    tree: TreeArrays, sid: npt.NDArray[np.int64], v: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    """Vectorized ``code_goes_left`` for the rows whose ``sid`` selects a bitset."""
    finite = np.isfinite(v)
    codes = np.zeros(len(v), dtype=np.int64)
    codes[finite] = v[finite].astype(np.int64)
    integral = finite & (codes == v)
    start = tree.set_offsets[sid]
    n_words = tree.set_offsets[sid + 1] - start
    in_range = integral & (codes >= 0) & (codes < n_words * 64)
    safe = np.where(in_range, codes, 0)
    word = tree.set_words[start + (safe >> 6)]
    bit = (word >> (safe & 63).astype(np.uint64)) & np.uint64(1)
    result: npt.NDArray[np.bool_] = in_range & (bit == 1)
    return result


def raw_score_batch_prepared(
    prepared: tuple[TreeArrays, ...],
    base_score: float,
    X: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Vectorized ``raw_score`` over the rows of X from pre-built tree arrays."""
    n = X.shape[0]
    total = np.full(n, base_score, dtype=np.float64)
    for tree in prepared:
        idx = np.zeros(n, dtype=np.int64)
        active = tree.feature[idx] >= 0
        while active.any():
            rows = np.flatnonzero(active)
            nodes = idx[rows]
            v = X[rows, tree.feature[nodes]]
            nan_mask = np.isnan(v)
            go_left = np.where(
                tree.is_lt[nodes], v < tree.threshold[nodes], v <= tree.threshold[nodes]
            )
            sid = tree.set_id[nodes]
            is_set = sid >= 0
            if is_set.any():
                go_left[is_set] = _set_membership(tree, sid[is_set], v[is_set])
            go_left = np.where(nan_mask, tree.miss_left[nodes], go_left)
            idx[rows] = np.where(go_left, tree.left[nodes], tree.right[nodes])
            active[rows] = tree.feature[idx[rows]] >= 0
        total += tree.value[idx]
    return total


def raw_score_batch(ir: EnsembleIR, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Vectorized ``raw_score`` over the rows of X (used by the genetic backend)."""
    return raw_score_batch_prepared(prepare_tree_arrays(ir), ir.base_score, X)


def _leaf_value(tree: Tree, x: npt.NDArray[np.float64]) -> float:
    value = _leaf_node(tree, x).value
    assert value is not None
    return value


def _leaf_node(tree: Tree, x: npt.NDArray[np.float64]) -> Node:
    node = tree.nodes[0]
    while node.feature is not None:
        assert node.left is not None and node.right is not None
        value = float(x[node.feature])
        if math.isnan(value):
            if node.missing_left is None:
                raise ValueError(
                    f"NaN at feature {node.feature} but node {node.node_id} "
                    "defines no missing routing"
                )
            child = node.left if node.missing_left else node.right
        elif node.categories is not None:
            child = node.left if code_goes_left(value, node.categories) else node.right
        else:
            assert node.threshold is not None
            go_left = value < node.threshold if node.op is SplitOp.LT else value <= node.threshold
            child = node.left if go_left else node.right
        node = tree.nodes[child]
    return node
