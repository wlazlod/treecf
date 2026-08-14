"""Score-bracket machinery for the exact backend: static per-tree summaries and
the live bracket a partial assignment narrows them to.

Split out of ``treecf.backends.exact`` for size only: ``exact``, this file,
``_exact_domains``, ``_exact_orderpairs`` and ``_exact_propagation`` are one
implementation, and the Rust mirror has to match all five bit-for-bit.

``_prepare_tree`` flattens one tree into parallel arrays plus the static
``sub_min``/``sub_max`` bracket and feature mask below each node.
``_EnsembleBounds`` holds the live per-tree and ensemble brackets under a
partial assignment: assigning a feature re-walks only the trees that split on
it, from their roots, and the ensemble bracket is then re-summed in full over
every tree in ascending index — the same order and the same additions
``raw_score`` performs — which is what keeps prune decisions identical between
this implementation and its Rust mirror.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from treecf.ir.model import EnsembleIR, SplitOp, Tree


@dataclass(frozen=True)
class _PreparedTree:
    """One tree flattened into parallel arrays plus static per-node summaries.

    ``sub_min``/``sub_max`` bracket the leaf values reachable below each node.
    ``mask`` has bit ``f`` set when feature ``f`` is split on anywhere in that
    node's subtree, so a walk can stop at any node that no current assignment
    can influence and read the static bracket instead.
    """

    feature: tuple[int, ...]  # -1 at leaves
    threshold: tuple[float, ...]
    is_lt: tuple[bool, ...]
    missing_left: tuple[bool, ...]
    left: tuple[int, ...]
    right: tuple[int, ...]
    sub_min: tuple[float, ...]
    sub_max: tuple[float, ...]
    mask: tuple[int, ...]


def _prepare_tree(tree: Tree) -> _PreparedTree:
    """Flatten one tree and fill in its static subtree brackets and feature masks."""
    n = len(tree.nodes)
    sub_min = [0.0] * n
    sub_max = [0.0] * n
    mask = [0] * n

    def visit(idx: int) -> None:
        node = tree.nodes[idx]
        if node.feature is None:
            assert node.value is not None
            sub_min[idx] = node.value
            sub_max[idx] = node.value
            return
        assert node.left is not None and node.right is not None
        visit(node.left)
        visit(node.right)
        sub_min[idx] = min(sub_min[node.left], sub_min[node.right])
        sub_max[idx] = max(sub_max[node.left], sub_max[node.right])
        # Rust mirror: Python ints are arbitrary-precision, so one int carries a
        # bit per feature however wide the model is. A mirror needs a real
        # bitset here (and for the search's assigned_mask) past 64 features.
        mask[idx] = (1 << node.feature) | mask[node.left] | mask[node.right]

    visit(0)
    return _PreparedTree(
        feature=tuple(-1 if nd.feature is None else nd.feature for nd in tree.nodes),
        threshold=tuple(0.0 if nd.threshold is None else nd.threshold for nd in tree.nodes),
        is_lt=tuple(nd.op is SplitOp.LT for nd in tree.nodes),
        # bool(None) -> route right, matching the vectorized evaluator's convention
        missing_left=tuple(bool(nd.missing_left) for nd in tree.nodes),
        left=tuple(0 if nd.left is None else nd.left for nd in tree.nodes),
        right=tuple(0 if nd.right is None else nd.right for nd in tree.nodes),
        sub_min=tuple(sub_min),
        sub_max=tuple(sub_max),
        mask=tuple(mask),
    )


class _EnsembleBounds:
    """Score bracket of one ensemble under a partial assignment.

    Holds a per-tree ``[min, max]`` over the leaves still reachable, and the
    ensemble bracket ``[score_min, score_max]`` those trees sum to. Assigning
    a feature recomputes only the trees that split on it, from their roots —
    the per-tree numbers are always full walks, never patched. The ensemble
    bracket is then re-summed in full over every tree in ascending index, the
    same order and the same additions ``raw_score`` performs, which is what
    keeps prune decisions identical between this implementation and its Rust
    mirror.

    ``assigned`` and ``values`` are the search's own arrays, shared by
    reference so the model and plausibility ensembles read one assignment.
    """

    def __init__(self, ir: EnsembleIR, assigned: list[bool], values: list[float]) -> None:
        self.base_score = ir.base_score
        self.trees = tuple(_prepare_tree(tree) for tree in ir.trees)
        self.assigned = assigned
        self.values = values
        on_feature: list[list[int]] = [[] for _ in range(ir.n_features)]
        for t, tree in enumerate(self.trees):
            for f in sorted(set(tree.feature)):
                if f >= 0:
                    on_feature[f].append(t)
        self.trees_on_feature = tuple(tuple(ts) for ts in on_feature)
        self.tree_min = [0.0] * len(self.trees)
        self.tree_max = [0.0] * len(self.trees)
        self.score_min = 0.0
        self.score_max = 0.0
        self.recompute(0)

    def recompute(self, assigned_mask: int) -> None:
        """Walk every tree from its root and re-sum — the from-scratch path."""
        for t, tree in enumerate(self.trees):
            low, high = self._walk(tree, 0, assigned_mask)
            self.tree_min[t] = low
            self.tree_max[t] = high
        self._resum()

    def apply(self, j: int, assigned_mask: int) -> tuple[tuple[int, float, float], ...]:
        """Refresh the trees that split on feature ``j``; return their old brackets."""
        frame: list[tuple[int, float, float]] = []
        for t in self.trees_on_feature[j]:
            frame.append((t, self.tree_min[t], self.tree_max[t]))
            low, high = self._walk(self.trees[t], 0, assigned_mask)
            self.tree_min[t] = low
            self.tree_max[t] = high
        self._resum()
        return tuple(frame)

    def restore(self, frame: tuple[tuple[int, float, float], ...]) -> None:
        """Put back the brackets an ``apply`` replaced."""
        for t, low, high in frame:
            self.tree_min[t] = low
            self.tree_max[t] = high
        self._resum()

    def _resum(self) -> None:
        low = self.base_score
        high = self.base_score
        for t in range(len(self.trees)):
            # written out rather than accumulated with += so that no reader
            # mistakes this for an incremental update: it is a full re-sum
            low = low + self.tree_min[t]
            high = high + self.tree_max[t]
        self.score_min = low
        self.score_max = high

    def _walk(self, tree: _PreparedTree, idx: int, assigned_mask: int) -> tuple[float, float]:
        if tree.mask[idx] & assigned_mask == 0:
            return tree.sub_min[idx], tree.sub_max[idx]
        f = tree.feature[idx]  # a set mask bit means this node is a split
        if self.assigned[f]:
            value = self.values[f]
            if math.isnan(value):
                child = tree.left[idx] if tree.missing_left[idx] else tree.right[idx]
            elif tree.is_lt[idx]:
                child = tree.left[idx] if value < tree.threshold[idx] else tree.right[idx]
            else:
                child = tree.left[idx] if value <= tree.threshold[idx] else tree.right[idx]
            return self._walk(tree, child, assigned_mask)
        left_min, left_max = self._walk(tree, tree.left[idx], assigned_mask)
        right_min, right_max = self._walk(tree, tree.right[idx], assigned_mask)
        # Rust mirror: on a 0.0 / -0.0 tie Python's min/max return the first
        # argument while f64::min/max return -0.0. The two compare equal, so no
        # prune or sum can differ — but a harness comparing stored brackets by
        # raw bits would see it.
        return min(left_min, right_min), max(left_max, right_max)
