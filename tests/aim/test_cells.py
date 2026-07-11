"""Cell construction (spec §5.1): half-open cells with exact side assignment, no nextafter."""

from __future__ import annotations

import math

import numpy as np
import pytest

from treecf.aim.cells import Cell, build_cells, cell_index, feature_cells
from treecf.ir.evaluate import leaf_assignment
from treecf.ir.model import SplitOp

from ..conftest import make_random_ir


class TestBuildCells:
    def test_single_lt_threshold(self) -> None:
        cells = build_cells([(1.0, SplitOp.LT)])
        # x < 1.0 | x >= 1.0
        assert cells == (
            Cell(lo=-math.inf, hi=1.0, lo_open=True, hi_open=True),
            Cell(lo=1.0, hi=math.inf, lo_open=False, hi_open=True),
        )

    def test_single_le_threshold(self) -> None:
        cells = build_cells([(1.0, SplitOp.LE)])
        # x <= 1.0 | x > 1.0
        assert cells == (
            Cell(lo=-math.inf, hi=1.0, lo_open=True, hi_open=False),
            Cell(lo=1.0, hi=math.inf, lo_open=True, hi_open=True),
        )

    def test_same_threshold_with_both_ops_yields_singleton(self) -> None:
        cells = build_cells([(1.0, SplitOp.LT), (1.0, SplitOp.LE)])
        # x < 1.0 | x == 1.0 | x > 1.0
        assert cells == (
            Cell(lo=-math.inf, hi=1.0, lo_open=True, hi_open=True),
            Cell(lo=1.0, hi=1.0, lo_open=False, hi_open=False),
            Cell(lo=1.0, hi=math.inf, lo_open=True, hi_open=True),
        )

    def test_duplicate_pairs_are_deduplicated(self) -> None:
        assert build_cells([(1.0, SplitOp.LT)] * 3) == build_cells([(1.0, SplitOp.LT)])

    def test_no_thresholds_single_unbounded_cell(self) -> None:
        assert build_cells([]) == (
            Cell(lo=-math.inf, hi=math.inf, lo_open=True, hi_open=True),
        )


class TestCellPointOps:
    def test_contains_respects_openness(self) -> None:
        c = Cell(lo=0.0, hi=1.0, lo_open=True, hi_open=False)
        assert not c.contains(0.0)
        assert c.contains(1.0)
        assert c.contains(0.5)

    def test_nearest_to_inside_is_identity(self) -> None:
        c = Cell(lo=0.0, hi=1.0, lo_open=False, hi_open=True)
        assert c.nearest_to(0.5) == 0.5

    def test_nearest_to_closed_bound_is_bound(self) -> None:
        c = Cell(lo=0.0, hi=1.0, lo_open=False, hi_open=True)
        assert c.nearest_to(-3.0) == 0.0

    def test_nearest_to_open_bound_steps_inside(self) -> None:
        c = Cell(lo=0.0, hi=1.0, lo_open=True, hi_open=True)
        below = c.nearest_to(-3.0)
        above = c.nearest_to(9.0)
        assert below > 0.0 and c.contains(below)
        assert above < 1.0 and c.contains(above)

    def test_cell_index_partitions_the_line(self) -> None:
        cells = build_cells([(0.0, SplitOp.LT), (1.0, SplitOp.LE), (1.0, SplitOp.LT)])
        for x in (-5.0, 0.0, 0.5, 1.0, 1.1, 42.0):
            idx = cell_index(cells, x)
            assert cells[idx].contains(x)
            assert sum(c.contains(x) for c in cells) == 1


class TestRoutingAtomicity:
    """Gating property (§5.1/§5.3): every point of a cell routes to the same leaves."""

    @pytest.mark.parametrize("seed", range(10))
    def test_points_within_cells_share_leaf_assignment(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        ir = make_random_ir(rng, n_features=3, n_trees=4, depth=3)
        per_feature = feature_cells(ir)
        for _ in range(50):
            x = np.empty(3)
            reference = np.empty(3)
            for j, cells in enumerate(per_feature):
                cell = cells[rng.integers(0, len(cells))]
                reference[j] = _representative(cell, rng)
                x[j] = _representative(cell, rng)
            assert leaf_assignment(ir, x) == leaf_assignment(ir, reference)


def _representative(cell: Cell, rng: np.random.Generator) -> float:
    lo = cell.lo if math.isfinite(cell.lo) else cell.hi - 10.0
    hi = cell.hi if math.isfinite(cell.hi) else cell.lo + 10.0
    if not math.isfinite(lo):  # fully unbounded cell
        lo, hi = -10.0, 10.0
    if lo == hi:
        return lo
    for _ in range(100):
        v = float(rng.uniform(lo, hi))
        if cell.contains(v):
            return v
    return cell.nearest_to((lo + hi) / 2.0)
