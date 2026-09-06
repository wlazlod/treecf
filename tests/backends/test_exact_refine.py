"""The per-feature cell hierarchy behind the coarse-to-fine exact search."""

from __future__ import annotations

import math

from treecf.aim.cells import Cell, build_cells
from treecf.backends._exact_bounds import _prepare_tree
from treecf.backends._exact_domains import _State
from treecf.backends._exact_refine import (
    _Alt,
    build_hierarchy,
    children_of,
    count_unresolved,
)
from treecf.ir.model import Node, SplitOp, Tree

INF = math.inf


def _cells(n: int = 10) -> tuple[Cell, ...]:
    """``n`` routing cells: (-inf, 1), [1, 2), ..., [n-1, inf)."""
    return build_cells([(float(t), SplitOp.LT) for t in range(1, n)])


def _states(cell_indices: list[int]) -> list[_State]:
    """One state per listed cell, the point nearest a factual of 0, cost-sorted."""
    return [_State(value=float(i), cost=float(i), cell_idx=i, is_nan=False) for i in cell_indices]


def _positions(alts: tuple[_Alt, ...], hier) -> list[tuple[int, int]]:
    out = []
    for alt in alts:
        if alt.node is None:
            pos = hier.cells_used.index(alt.state.cell_idx)
            out.append((pos, pos))
        else:
            node = hier.nodes[alt.node]
            out.append((node.first, node.last))
    return out


class TestBuilder:
    def test_initial_alternatives_are_the_depth_three_frontier(self) -> None:
        hier = build_hierarchy(_states(list(range(10))), _cells(), -INF, INF)
        assert hier is not None
        assert hier.cells_used == tuple(range(10))
        assert _positions(hier.initial, hier) == [
            (0, 1), (2, 2), (3, 3), (4, 4), (5, 6), (7, 7), (8, 8), (9, 9),
        ]

    def test_node_cost_and_representative_come_from_the_cheapest_state(self) -> None:
        hier = build_hierarchy(_states(list(range(10))), _cells(), -INF, INF)
        assert hier is not None
        root = hier.nodes[0]
        assert (root.first, root.last) == (0, 9)
        assert root.cost == 0.0 and root.rep == 0 and root.n_states == 10
        pair = next(hier.nodes[a.node] for a in hier.initial if a.node is not None)
        assert (pair.first, pair.last) == (0, 1)
        assert pair.cost == 0.0 and pair.rep == 0

    def test_cells_missing_from_the_domain_vanish_and_hulls_span_the_gap(self) -> None:
        states = _states([0, 1, 2, 5, 6, 7, 8, 9])
        hier = build_hierarchy(states, _cells(), -INF, INF)
        assert hier is not None
        assert hier.cells_used == (0, 1, 2, 5, 6, 7, 8, 9)
        # eight cells: the depth-three frontier is every leaf, hence atomic
        assert all(alt.node is None for alt in hier.initial)
        assert len(hier.initial) == 8
        left = hier.nodes[hier.nodes[0].left]
        assert (left.first, left.last) == (0, 3)
        assert left.iv == (-INF, 6.0, True, True)  # hull of (-inf, 1) ... [5, 6)
        assert left.span[0] == -INF and left.span[1] < 6.0

    def test_instance_bounds_clamp_the_hull(self) -> None:
        hier = build_hierarchy(_states(list(range(8))), _cells(), 0.5, 7.5)
        assert hier is not None
        root = hier.nodes[0]
        assert root.iv == (0.5, 7.5, False, False)
        assert root.span == (0.5, 7.5)

    def test_single_cell_leaf_with_several_states_refines_into_them(self) -> None:
        states = [
            _State(value=0.0, cost=0.0, cell_idx=0, is_nan=False),
            _State(value=2.0, cost=2.0, cell_idx=2, is_nan=False),
            _State(value=2.5, cost=2.5, cell_idx=2, is_nan=False),  # a demanded value
        ]
        hier = build_hierarchy(states, _cells(), -INF, INF)
        assert hier is not None
        leaf_alt = next(a for a in hier.initial if a.state.cell_idx == 2)
        assert leaf_alt.node is not None and hier.nodes[leaf_alt.node].n_states == 2
        kids = children_of(hier, leaf_alt.node)
        assert [k.state_idx for k in kids] == [1, 2]
        assert all(k.node is None for k in kids)

    def test_nan_state_is_an_atomic_alternative_sorted_last_on_ties(self) -> None:
        states = [*_states([0, 1, 2]), _State(value=math.nan, cost=0.0, cell_idx=10, is_nan=True)]
        hier = build_hierarchy(states, _cells(), -INF, INF)
        assert hier is not None
        assert hier.initial[0].state.cell_idx == 0
        assert hier.initial[1].state.is_nan and hier.initial[1].node is None

    def test_children_lead_with_the_side_holding_the_representative(self) -> None:
        hier = build_hierarchy(_states(list(range(10))), _cells(), -INF, INF)
        assert hier is not None
        first, second = children_of(hier, 0)
        assert (hier.nodes[first.node].first, hier.nodes[first.node].last) == (0, 4)
        assert (hier.nodes[second.node].first, hier.nodes[second.node].last) == (5, 9)
        assert first.state_idx == 0 and second.state_idx == 5

    def test_one_used_cell_means_no_hierarchy(self) -> None:
        assert build_hierarchy(_states([3]), _cells(), -INF, INF) is None
        two_in_one = [
            _State(value=3.0, cost=3.0, cell_idx=3, is_nan=False),
            _State(value=3.5, cost=3.5, cell_idx=3, is_nan=False),
        ]
        assert build_hierarchy(two_in_one, _cells(), -INF, INF) is None


def _leaf(i: int, v: float) -> Node:
    return Node(i, None, None, None, None, None, None, v)


def _tree() -> Tree:
    """``f0 < 5`` at the root, ``f1 < 2`` below its right child."""
    return Tree(
        nodes=(
            Node(0, 0, 5.0, SplitOp.LT, True, 1, 2, None),
            _leaf(1, 0.0),
            Node(2, 1, 2.0, SplitOp.LT, True, 3, 4, None),
            _leaf(3, 1.0),
            _leaf(4, 2.0),
        )
    )


class TestUnresolvedNodes:
    def _count(self, ranges, assigned, values):
        mask = sum(1 << f for f, r in enumerate(ranges) if r is not None)
        return count_unresolved((_prepare_tree(_tree()),), 2, ranges, mask, assigned, values)

    def test_a_straddled_split_counts_for_its_feature(self) -> None:
        counts = self._count([(0.0, 10.0, False, False), None], [True, False], [0.0, 0.0])
        assert counts == [1, 0]

    def test_both_features_straddling_count_separately(self) -> None:
        counts = self._count(
            [(0.0, 10.0, False, False), (0.0, 5.0, False, False)], [True, True], [0.0, 0.0]
        )
        assert counts == [1, 1]

    def test_a_resolved_split_is_not_counted_and_only_its_child_is_visited(self) -> None:
        counts = self._count(
            [(6.0, 10.0, False, False), (0.0, 5.0, False, False)], [True, True], [0.0, 0.0]
        )
        assert counts == [0, 1]
        counts = self._count(
            [(0.0, 4.0, False, False), (0.0, 5.0, False, False)], [True, True], [0.0, 0.0]
        )
        assert counts == [0, 0]  # the left leaf hides the f1 split entirely

    def test_scalar_assignment_routes_and_never_counts(self) -> None:
        counts = self._count([None, (0.0, 5.0, False, False)], [True, True], [7.0, 0.0])
        assert counts == [0, 1]
        counts = self._count([None, (0.0, 5.0, False, False)], [True, True], [1.0, 0.0])
        assert counts == [0, 0]
