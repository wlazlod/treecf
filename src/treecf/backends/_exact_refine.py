"""The cell hierarchy behind the coarse-to-fine exact search.

Split out of ``treecf.backends.exact`` for size only: it is part of the same
implementation as the other five exact-backend files, and the Rust mirror
has to match it bit-for-bit.

The classic search assigns one candidate state at a time, and a numeric
feature with many thresholds has many states — the breadth of that alphabet
is what makes wide models expensive to certify. The coarse-to-fine search
instead assigns *ranges* of routing cells first and descends only where the
score bound forces it. This file builds the per-feature hierarchy those
ranges come from and answers the one question the search asks of a full
range assignment: which feature still has the most tree nodes left
unresolved, so that refining it does the most to tighten the bracket.

A hierarchy is a balanced binary tree over the cells the feature's states
survive in after presolve: every node covers a contiguous run of those
cells, split recursively at the median. A node is held to the hull of its
cells, intersected with the instance bounds, and costs what its cheapest
state costs — the states are cost-sorted, so that is the first one in
classic order, which also serves as the node's representative value. A
feature's own candidate list is what the hierarchy is built from, so every
range the search can hold a feature to contains at least one candidate the
classic search would have tried, and refining all the way down reaches
exactly the classic states.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from treecf.aim.cells import Cell
from treecf.backends._exact_bounds import _PreparedTree, _RangeIv
from treecf.backends._exact_domains import _State
from treecf.backends._exact_orderpairs import _achievable_bounds, _intersect_cell
from treecf.ir.model import code_goes_left

# how many ranges a feature starts the search with, at most: the deepest
# level of its hierarchy holding no more nodes than this
_INITIAL_RANGES = 8


@dataclass(frozen=True)
class _HierNode:
    """One node of a feature's hierarchy: a contiguous run of used cells.

    ``first``/``last`` are positions into ``_Hier.cells_used``; ``iv`` is the
    hull of the run's cells intersected with the instance bounds, with the
    outer cells' openness; ``span`` is the achievable ``(lo, hi)`` of that
    hull, what an order-pair bound reads; ``rep`` is the index (into the
    feature's domain) of the cheapest state inside, whose cost is ``cost``.
    """

    first: int
    last: int
    iv: _RangeIv
    span: tuple[float, float]
    cost: float
    rep: int
    n_states: int
    left: int | None
    right: int | None


@dataclass(frozen=True)
class _Alt:
    """One alternative the search can try on a feature: an atomic state
    (``node`` is None) or a hierarchy node it is held to, represented by that
    node's cheapest state."""

    state: _State
    state_idx: int
    node: int | None

    @property
    def cost(self) -> float:
        return self.state.cost

    @property
    def is_range(self) -> bool:
        return self.node is not None


@dataclass(frozen=True)
class _Hier:
    states: tuple[_State, ...]  # the feature's domain the hierarchy was built from
    cells_used: tuple[int, ...]
    state_ids_by_cell: dict[int, tuple[int, ...]]  # cell index -> state indices, classic order
    nodes: tuple[_HierNode, ...]  # node 0 is the root
    initial: tuple[_Alt, ...]  # the alternatives the feature starts the search with


def build_hierarchy(
    states: Sequence[_State], cells: Sequence[Cell], lo_b: float, hi_b: float
) -> _Hier | None:
    """The hierarchy of a feature's (presolved, cost-sorted) domain, or
    ``None`` when its states occupy a single cell — then every state is an
    atomic alternative and there is nothing to range over."""
    by_cell: dict[int, list[int]] = {}
    for idx, state in enumerate(states):
        if not state.is_nan:
            by_cell.setdefault(state.cell_idx, []).append(idx)
    cells_used = tuple(sorted(by_cell))
    if len(cells_used) < 2:
        return None
    state_ids_by_cell = {c: tuple(ids) for c, ids in by_cell.items()}

    def clamped(position: int) -> Cell:
        cell = cells[cells_used[position]]
        iv = _intersect_cell(cell, lo_b, hi_b)
        # a state's own cell always meets the bounds its value satisfies; the
        # fallback only keeps the builder total
        return cell if iv is None else iv

    nodes: list[_HierNode | None] = []

    def build(first: int, last: int) -> int:
        node_id = len(nodes)
        nodes.append(None)  # reserve the slot so ids are pre-order
        head, tail = clamped(first), clamped(last)
        iv: _RangeIv = (head.lo, tail.hi, head.lo_open, tail.hi_open)
        span = (_achievable_bounds(head)[0], _achievable_bounds(tail)[1])
        ids = [i for pos in range(first, last + 1) for i in state_ids_by_cell[cells_used[pos]]]
        rep = min(ids)
        left = right = None
        if first < last:
            mid = (first + last) // 2
            left = build(first, mid)
            right = build(mid + 1, last)
        nodes[node_id] = _HierNode(
            first=first,
            last=last,
            iv=iv,
            span=span,
            cost=states[rep].cost,
            rep=rep,
            n_states=len(ids),
            left=left,
            right=right,
        )
        return node_id

    build(0, len(cells_used) - 1)
    built = tuple(node for node in nodes if node is not None)
    assert len(built) == len(nodes)

    depth_limit = int(math.log2(_INITIAL_RANGES))
    frontier: list[_Alt] = []

    def collect(node_id: int, depth: int) -> None:
        node = built[node_id]
        if node.left is None or depth == depth_limit:
            frontier.append(_alt_for(built, states, node_id))
            return
        assert node.right is not None
        collect(node.left, depth + 1)
        collect(node.right, depth + 1)

    collect(0, 0)
    for idx, state in enumerate(states):
        if state.is_nan:
            frontier.append(_Alt(state=state, state_idx=idx, node=None))
    n_cells = len(cells)

    def key(alt: _Alt) -> tuple[float, int, float]:
        if alt.state.is_nan:
            return (alt.cost, n_cells, 0.0)
        first_cell = alt.state.cell_idx if alt.node is None else cells_used[built[alt.node].first]
        return (alt.cost, first_cell, alt.state.value)

    frontier.sort(key=key)
    return _Hier(
        states=tuple(states),
        cells_used=cells_used,
        state_ids_by_cell=state_ids_by_cell,
        nodes=built,
        initial=tuple(frontier),
    )


def _alt_for(nodes: Sequence[_HierNode], states: Sequence[_State], node_id: int) -> _Alt:
    """The alternative standing for a node: atomic when the node is a single
    cell holding a single state, a range otherwise."""
    node = nodes[node_id]
    if node.left is None and node.n_states == 1:
        return _Alt(state=states[node.rep], state_idx=node.rep, node=None)
    return _Alt(state=states[node.rep], state_idx=node.rep, node=node_id)


def children_of(hier: _Hier, node_id: int) -> tuple[_Alt, ...]:
    """What replaces a range when the search refines it: its two children,
    the one holding the representative first; for a single cell holding
    several states, those states in classic order."""
    node = hier.nodes[node_id]
    if node.left is None:
        cell = hier.cells_used[node.first]
        return tuple(
            _Alt(state=hier.states[idx], state_idx=idx, node=None)
            for idx in hier.state_ids_by_cell[cell]
        )
    assert node.right is not None
    rep_pos = hier.cells_used.index(hier.states[node.rep].cell_idx)
    left_first = rep_pos <= hier.nodes[node.left].last
    ordered = (node.left, node.right) if left_first else (node.right, node.left)
    return tuple(_alt_for(hier.nodes, hier.states, child) for child in ordered)


def count_unresolved(
    trees: Sequence[_PreparedTree],
    n_features: int,
    ranges: Sequence[_RangeIv | None],
    range_mask: int,
    assigned: Sequence[bool],
    values: Sequence[float],
) -> list[int]:
    """Per feature, how many live tree nodes a range assignment leaves
    unresolved: nodes reachable under the current routing whose threshold
    falls strictly inside the interval their feature is held to, so both
    children stay live. Refining the feature with the most of them is what
    tightens the bracket fastest."""
    counts = [0] * n_features

    def visit(tree: _PreparedTree, idx: int) -> None:
        if tree.mask[idx] & range_mask == 0:
            return  # nothing below is held to a range: nothing to resolve
        f = tree.feature[idx]
        rng = ranges[f]
        if rng is not None:
            lo, hi, lo_open, hi_open = rng
            t = tree.threshold[idx]
            if tree.is_lt[idx]:
                all_left = hi < t or (hi == t and hi_open)
                all_right = lo >= t
            else:
                all_left = hi <= t
                all_right = lo > t or (lo == t and lo_open)
            if all_left:
                visit(tree, tree.left[idx])
            elif all_right:
                visit(tree, tree.right[idx])
            else:
                counts[f] += 1
                visit(tree, tree.left[idx])
                visit(tree, tree.right[idx])
            return
        if assigned[f]:
            value = values[f]
            members = tree.categories[idx]
            if math.isnan(value):
                child = tree.left[idx] if tree.missing_left[idx] else tree.right[idx]
            elif members is not None:
                child = tree.left[idx] if code_goes_left(value, members) else tree.right[idx]
            elif tree.is_lt[idx]:
                child = tree.left[idx] if value < tree.threshold[idx] else tree.right[idx]
            else:
                child = tree.left[idx] if value <= tree.threshold[idx] else tree.right[idx]
            visit(tree, child)
            return
        visit(tree, tree.left[idx])
        visit(tree, tree.right[idx])

    for tree in trees:
        visit(tree, 0)
    return counts
