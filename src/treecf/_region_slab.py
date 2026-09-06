"""The emptiness search behind maximal recourse regions.

Fast region growth stops a side the moment the conservative interval bracket
of the extended box leaves the target. That bound is sound but loose: the
bracket of a box is wider than the set of scores its points really reach.
The maximal mode asks the sharper question — does the *extension slab*
(the feature moved into its next routing cell, every other coordinate
ranging over the box already certified) contain a single point that leaves
the target or falls below the plausibility floor? — and answers it with a
budgeted search over sub-boxes.

A sub-box holds each free numeric feature to a run of its routing cells and
each free categorical feature to a run of its category blocks. Its bracket
comes from the same interval routing the region oracle uses. A sub-box whose
bracket lies inside the target is empty of violators; one whose bracket lies
entirely outside contains nothing else, so the clamp of the counterfactual
into it is a witness (re-scored before it is believed); anything in between
is split on the feature with the most tree nodes still straddled, the half
holding the counterfactual's clamp first. The search completes empty, finds a
witness, or gives up when its node budget is spent or a bracket cannot be
soundly computed — and only the first two outcomes settle a side.

Linear constraints need no search: the region oracle checks them at their
worst corner, and a failing corner is itself a witness. The Rust core mirrors
this search node for node.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from treecf.aim.cells import Cell, cell_index
from treecf.backends._exact_orderpairs import _achievable_bounds, _intersect_cell
from treecf.ir.evaluate import raw_score
from treecf.ir.model import EnsembleIR

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

# outcomes of one emptiness search
EMPTY = "empty"
WITNESS = "witness"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class SlabProblem:
    """One emptiness question, fully specified.

    ``lo``/``hi`` are the closed box to search (the slab feature already
    narrowed to its slab); ``cat_sets`` the certified code sets of the
    tracked categorical coordinates, with the slab feature, when
    categorical, already narrowed to the block under test; ``free`` the
    numeric coordinates allowed to range (never the slab feature, never a
    degenerate one); ``free_cats`` the categorical coordinates whose set
    holds more than one block; ``blocks`` each categorical coordinate's
    blocks as member tuples, in canonical order.
    """

    ir: EnsembleIR
    if_ir: EnsembleIR | None
    min_total_path: float
    interval: tuple[float, float]
    grids: Sequence[tuple[Cell, ...]]
    x_cf: FloatArray
    is_nan: BoolArray
    lo: FloatArray
    hi: FloatArray
    cat_sets: Mapping[int, frozenset[int]]
    free: tuple[int, ...]
    free_cats: tuple[int, ...]
    blocks: Mapping[int, Sequence[tuple[int, ...]]]


def _position(cells: tuple[Cell, ...], value: float, upper: bool) -> int:
    """The cell holding ``value``; an infinite endpoint means the outermost cell."""
    if math.isinf(value):
        return len(cells) - 1 if upper else 0
    return cell_index(cells, value)


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


class _SlabSearch:
    """The depth-first search itself; one instance per emptiness question."""

    def __init__(self, problem: SlabProblem, limit: int) -> None:
        from treecf.regions import _tree_interval_bracket

        self.p = problem
        self.limit = limit
        self.nodes = 0
        self._bracket = _tree_interval_bracket
        # per free numeric feature, the closed run of cell positions it
        # currently ranges over; per free categorical feature, the run of
        # block positions among the blocks its certified set covers
        self.num_range: dict[int, tuple[int, int]] = {}
        for k in problem.free:
            cells = problem.grids[k]
            self.num_range[k] = (
                _position(cells, float(problem.lo[k]), upper=False),
                _position(cells, float(problem.hi[k]), upper=True),
            )
        self.cat_blocks: dict[int, list[tuple[int, ...]]] = {}
        self.cat_range: dict[int, tuple[int, int]] = {}
        for c in problem.free_cats:
            covered = [
                tuple(code for code in members if code in problem.cat_sets[c])
                for members in problem.blocks[c]
            ]
            covered = [members for members in covered if members]
            self.cat_blocks[c] = covered
            self.cat_range[c] = (0, len(covered) - 1)

    # -- the box a node stands for ----------------------------------------

    def _box(self) -> tuple[FloatArray, FloatArray, dict[int, set[int]]]:
        lo = self.p.lo.copy()
        hi = self.p.hi.copy()
        for k, (a, b) in self.num_range.items():
            cells = self.p.grids[k]
            head = _intersect_cell(cells[a], float(self.p.lo[k]), float(self.p.hi[k]))
            tail = _intersect_cell(cells[b], float(self.p.lo[k]), float(self.p.hi[k]))
            # the box's own endpoints are achievable values inside their
            # cells, so the intersections are never empty
            assert head is not None and tail is not None
            lo[k] = _achievable_bounds(head)[0]
            hi[k] = _achievable_bounds(tail)[1]
        cat_sets: dict[int, set[int]] = {c: set(codes) for c, codes in self.p.cat_sets.items()}
        for c, (a, b) in self.cat_range.items():
            cat_sets[c] = {code for members in self.cat_blocks[c][a : b + 1] for code in members}
        return lo, hi, cat_sets

    def _brackets(
        self, lo: FloatArray, hi: FloatArray, cat_sets: dict[int, set[int]]
    ) -> tuple[tuple[float, float] | None, tuple[float, float] | None, list[int]]:
        """Model and plausibility brackets of a box plus, per feature, how
        many tree nodes its interval straddles (both children live)."""
        straddles = [0] * len(self.p.x_cf)
        total = self._ensemble(self.p.ir, lo, hi, cat_sets, straddles)
        if_total: tuple[float, float] | None = None
        if self.p.if_ir is not None:
            if_total = self._ensemble(self.p.if_ir, lo, hi, cat_sets, straddles)
        return total, if_total, straddles

    def _ensemble(
        self,
        ir: EnsembleIR,
        lo: FloatArray,
        hi: FloatArray,
        cat_sets: dict[int, set[int]],
        straddles: list[int],
    ) -> tuple[float, float] | None:
        total_min = ir.base_score
        total_max = ir.base_score
        for tree in ir.trees:
            bracket = self._bracket(tree.nodes, 0, lo, hi, self.p.is_nan, cat_sets, straddles)
            if bracket is None:
                return None
            total_min = total_min + bracket[0]
            total_max = total_max + bracket[1]
        return total_min, total_max

    def _witness(
        self, lo: FloatArray, hi: FloatArray, cat_sets: dict[int, set[int]]
    ) -> FloatArray | None:
        """The counterfactual clamped into the box, believed only after it
        re-scores outside the target or below the plausibility floor."""
        point = self.p.x_cf.copy()
        for k in range(len(point)):
            if self.p.is_nan[k]:
                continue
            if k in cat_sets:
                code = int(point[k])
                point[k] = float(code if code in cat_sets[k] else min(cat_sets[k]))
                continue
            point[k] = _clamp(float(point[k]), float(lo[k]), float(hi[k]))
        try:
            score = raw_score(self.p.ir, point)
            lo_t, hi_t = self.p.interval
            if score < lo_t or score > hi_t:
                return point
            if self.p.if_ir is not None and raw_score(self.p.if_ir, point) < self.p.min_total_path:
                return point
        except ValueError:
            return None
        return None

    # -- the search ----------------------------------------------------------

    def run(self) -> tuple[str, FloatArray | None]:
        return self._visit()

    def _visit(self) -> tuple[str, FloatArray | None]:
        if self.nodes >= self.limit:
            return UNKNOWN, None
        self.nodes += 1
        lo, hi, cat_sets = self._box()
        total, if_total, straddles = self._brackets(lo, hi, cat_sets)
        if total is None or (self.p.if_ir is not None and if_total is None):
            return UNKNOWN, None
        lo_t, hi_t = self.p.interval
        clean = lo_t <= total[0] and total[1] <= hi_t
        violates = total[1] < lo_t or total[0] > hi_t
        if if_total is not None:
            clean = clean and if_total[0] >= self.p.min_total_path
            violates = violates or if_total[1] < self.p.min_total_path
        if clean:
            return EMPTY, None
        if violates:
            point = self._witness(lo, hi, cat_sets)
            return (WITNESS, point) if point is not None else (UNKNOWN, None)

        # refine the feature with the most straddled tree nodes, lowest index on ties
        target = -1
        best = 0
        for k in self.p.free:
            a, b = self.num_range[k]
            if b > a and straddles[k] > best:
                target, best = k, straddles[k]
        for c in self.p.free_cats:
            a, b = self.cat_range[c]
            if b > a and straddles[c] > best:
                target, best = c, straddles[c]
        if target < 0:
            return UNKNOWN, None  # every coordinate atomic yet undecided: cannot happen
        if target in self.num_range:
            a, b = self.num_range[target]
            mid = (a + b) // 2
            rep = _position(
                self.p.grids[target],
                _clamp(float(self.p.x_cf[target]), float(lo[target]), float(hi[target])),
                upper=False,
            )
            first, second = ((a, mid), (mid + 1, b)) if rep <= mid else ((mid + 1, b), (a, mid))
            for child in (first, second):
                self.num_range[target] = child
                outcome, point = self._visit()
                if outcome != EMPTY:
                    self.num_range[target] = (a, b)
                    return outcome, point
            self.num_range[target] = (a, b)
            return EMPTY, None
        a, b = self.cat_range[target]
        mid = (a + b) // 2
        code = int(self.p.x_cf[target])
        rep = next(
            (pos for pos in range(a, b + 1) if code in self.cat_blocks[target][pos]), a
        )
        first, second = ((a, mid), (mid + 1, b)) if rep <= mid else ((mid + 1, b), (a, mid))
        for child in (first, second):
            self.cat_range[target] = child
            outcome, point = self._visit()
            if outcome != EMPTY:
                self.cat_range[target] = (a, b)
                return outcome, point
        self.cat_range[target] = (a, b)
        return EMPTY, None


def search_slab(problem: SlabProblem, limit: int) -> tuple[str, FloatArray | None, int]:
    """Answer one emptiness question within ``limit`` nodes.

    Returns the outcome (``"empty"``, ``"witness"`` or ``"unknown"``), the
    witness point when there is one, and the number of sub-boxes visited.
    """
    if limit <= 0:
        return UNKNOWN, None, 0
    search = _SlabSearch(problem, limit)
    outcome, point = search.run()
    return outcome, point, search.nodes
