"""Routing-atomic cells per feature.

Cells are built directly from the ``(threshold, op)`` pairs stored in the IR;
LT and LE at the same threshold produce a singleton cell for the threshold
value itself. Normalizing operators via ``nextafter`` is forbidden.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from treecf.ir.model import EnsembleIR, SplitOp


@dataclass(frozen=True)
class Cell:
    """Interval of feature values within which every tree routes identically."""

    lo: float
    hi: float
    lo_open: bool
    hi_open: bool

    def contains(self, x: float) -> bool:
        above = x > self.lo if self.lo_open else x >= self.lo
        below = x < self.hi if self.hi_open else x <= self.hi
        return above and below

    def nearest_to(self, x: float) -> float:
        """Point of the cell closest to ``x``.

        Open bounds step one FLOAT32 ulp inside (falling back to a float64 ulp
        for cells narrower than that): GBDT libraries compare in float32, so a
        float64-ulp neighbour of a threshold would collapse onto it natively
        and route the other way — the counterfactual must stay distinguishable
        from the threshold in the deployed model, not just in the IR.
        """
        if self.contains(x):
            return x
        if x <= self.lo:
            if not self.lo_open:
                return self.lo
            stepped = float(np.nextafter(np.float32(self.lo), np.float32(math.inf)))
            if self.contains(stepped):
                return stepped
            return float(np.nextafter(self.lo, math.inf))
        if not self.hi_open:
            return self.hi
        stepped = float(np.nextafter(np.float32(self.hi), np.float32(-math.inf)))
        if self.contains(stepped):
            return stepped
        return float(np.nextafter(self.hi, -math.inf))


def build_cells(pairs: Iterable[tuple[float, SplitOp]]) -> tuple[Cell, ...]:
    """Partition the real line into routing-atomic cells for one feature."""
    ops_at: dict[float, set[SplitOp]] = {}
    for threshold, op in pairs:
        ops_at.setdefault(threshold, set()).add(op)

    cells: list[Cell] = []
    lo, lo_open = -math.inf, True
    for threshold in sorted(ops_at):
        ops = ops_at[threshold]
        if ops == {SplitOp.LT, SplitOp.LE}:
            cells.append(Cell(lo=lo, hi=threshold, lo_open=lo_open, hi_open=True))
            cells.append(Cell(lo=threshold, hi=threshold, lo_open=False, hi_open=False))
            lo, lo_open = threshold, True
        elif ops == {SplitOp.LE}:
            cells.append(Cell(lo=lo, hi=threshold, lo_open=lo_open, hi_open=False))
            lo, lo_open = threshold, True
        else:  # LT only
            cells.append(Cell(lo=lo, hi=threshold, lo_open=lo_open, hi_open=True))
            lo, lo_open = threshold, False
    cells.append(Cell(lo=lo, hi=math.inf, lo_open=lo_open, hi_open=True))
    return tuple(cells)


def feature_cells(*irs: EnsembleIR) -> tuple[tuple[Cell, ...], ...]:
    """Cells per feature across all given ensembles (model + optional isolation forest)."""
    n_features = irs[0].n_features
    if any(ir.n_features != n_features for ir in irs):
        raise ValueError("all ensembles must share the same feature space")
    pairs: list[list[tuple[float, SplitOp]]] = [[] for _ in range(n_features)]
    for ir in irs:
        for tree in ir.trees:
            for node in tree.nodes:
                # set-membership splits carry no threshold; their features
                # partition into category blocks instead of interval cells
                if node.feature is not None and node.categories is None:
                    assert node.threshold is not None and node.op is not None
                    pairs[node.feature].append((node.threshold, node.op))
    return tuple(build_cells(feature_pairs) for feature_pairs in pairs)


def category_blocks(*irs: EnsembleIR) -> dict[int, tuple[tuple[int, ...], ...]]:
    """Category blocks per categorical feature across all given ensembles.

    Two codes share a block iff no split in any ensemble separates them:
    set-membership splits partition by membership, and numeric splits on a
    categorical feature (an isolation forest trained on raw codes) partition
    by threshold side. Blocks are ordered by their smallest member code, block
    members ascend, and a block's representative is its smallest member.
    """
    cardinality: dict[int, int] = {}
    for ir in irs:
        for j, info in ir.categorical.items():
            if cardinality.setdefault(j, info.cardinality) != info.cardinality:
                raise ValueError(f"ensembles disagree on the cardinality of feature {j}")

    set_splits: dict[int, list[frozenset[int]]] = {j: [] for j in cardinality}
    numeric_splits: dict[int, list[tuple[float, SplitOp]]] = {j: [] for j in cardinality}
    for ir in irs:
        for tree in ir.trees:
            for node in tree.nodes:
                if node.feature is None:
                    continue
                if node.categories is not None:
                    if node.feature not in cardinality:
                        raise ValueError(
                            f"set split on feature {node.feature} which no ensemble "
                            "declares categorical"
                        )
                    set_splits[node.feature].append(node.categories)
                elif node.feature in cardinality:
                    assert node.threshold is not None and node.op is not None
                    numeric_splits[node.feature].append((node.threshold, node.op))

    blocks: dict[int, tuple[tuple[int, ...], ...]] = {}
    for j, k in sorted(cardinality.items()):
        signatures: list[tuple[bool, ...]] = []
        for code in range(k):
            sig = [code in members for members in set_splits[j]]
            for threshold, op in numeric_splits[j]:
                sig.append(code < threshold if op is SplitOp.LT else code <= threshold)
            signatures.append(tuple(sig))
        groups: dict[tuple[bool, ...], list[int]] = {}
        for code in range(k):
            groups.setdefault(signatures[code], []).append(code)
        # codes scan ascending, so insertion order == ascending smallest member
        blocks[j] = tuple(tuple(members) for members in groups.values())
    return blocks


def cell_index(cells: tuple[Cell, ...], x: float) -> int:
    """Index of the unique cell containing ``x``."""
    for i, cell in enumerate(cells):
        if cell.contains(x):
            return i
    raise ValueError(f"value {x!r} not covered by cells (should be impossible)")
