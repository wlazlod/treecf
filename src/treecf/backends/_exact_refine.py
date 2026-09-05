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
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from treecf.aim.cells import Cell
from treecf.backends._exact_bounds import _EnsembleBounds, _PreparedTree, _RangeIv
from treecf.backends._exact_domains import FloatArray, _State
from treecf.backends._exact_orderpairs import _achievable_bounds, _intersect_cell
from treecf.backends._exact_propagation import _Propagation, _PropFrame
from treecf.backends._exact_trace import TraceSample, _Trace
from treecf.ir.model import code_goes_left

# how many ranges a feature starts the search with, at most: the deepest
# level of its hierarchy holding no more nodes than this
_INITIAL_RANGES = 8


@dataclass
class _Ledger:
    """The one piece of search state a repair writes from inside ``finish``:
    the cheapest committed cost among the completions the repair had to set
    aside. Nothing derived from one of those can cost less than this, so once
    the incumbent is at least as cheap, setting them aside changed nothing."""

    dropped_floor: float


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


# ------------------------------------------------------------ the search ---


@dataclass
class _Applied:
    """What applying one alternative changed, so it can be undone: the
    bracket and propagation frames, and the feature's previous holding."""

    alt: _Alt
    model: tuple[tuple[int, float, float], ...]
    plausibility: tuple[tuple[int, float, float], ...]
    prop: _PropFrame
    prev: tuple[_RangeIv | None, tuple[float, float] | None, int | None, int, float, bool]


@dataclass
class _RFrame:
    """One level of the coarse-to-fine stack: a feature and the alternatives
    left to try on it. A feature frame introduces a feature (its ``level`` is
    the feature's position in the search order); a refine frame replaces the
    range a feature already holds by that range's children (``refine_of`` is
    the range's node, ``level`` is the number of ordered features, so nothing
    remains to be introduced below it)."""

    j: int
    alts: tuple[_Alt, ...]
    next: int
    refine_of: int | None
    level: int
    h_rest: float  # cheapest remainder below this frame's own alternative
    g_before: float  # cost committed before this frame's alternative
    applied: _Applied | None = None


@dataclass
class _RefineCtx:
    """Everything ``run_refine`` reads or drives, handed over by
    ``solve_exact`` once the shared setup (domains, presolve, order pairs,
    propagation) is in place."""

    x: FloatArray
    order: list[int]
    domains: list[list[_State]]
    hiers: list[_Hier | None]
    h_suffix: list[float]
    lo_t: float
    hi_t: float
    min_total_path: float
    gap: float
    node_budget: int
    time_budget_s: float
    start: float
    assigned: list[bool]
    values: list[float]
    ranges: list[_RangeIv | None]
    range_span: list[tuple[float, float] | None]
    picked: list[int]
    model_bounds: _EnsembleBounds
    if_bounds: _EnsembleBounds | None
    propagation: _Propagation
    ledger: _Ledger
    bounded_pairs: list[tuple[int, int]]
    accepts: Callable[[FloatArray], bool]
    finish: Callable[[FloatArray, float], FloatArray | None]
    unorderable: Callable[[], bool]
    cost_of: Callable[[FloatArray], float]
    incumbent: tuple[float, FloatArray] | None


@dataclass
class _RefineOutcome:
    incumbent_cost: float
    incumbent_row: FloatArray | None
    incumbent_states: list[_State] | None
    nodes_expanded: int
    nodes_pruned_score: int
    nodes_pruned_cost: int
    coarse_accepts: int
    refinements: int
    completed: bool
    lower_bound: float
    proof: str
    warm_start_used: bool
    trace: list[TraceSample]


def run_refine(ctx: _RefineCtx) -> _RefineOutcome:
    """The coarse-to-fine search over the hierarchies ``ctx`` carries.

    Features are introduced in the classic order, each held to one of its
    initial alternatives — a range of cells, or an atomic state where the
    feature has no hierarchy. Every push runs the classic prune checks. Once
    every feature holds something, the assignment is settled one of three
    ways: all atomic, and the classic completion path (boundary repair,
    withdrawal bookkeeping, acceptance) decides it; the ensemble bracket lies
    inside the target and the row of representatives passes the arbiter
    as-is, and that row is accepted; otherwise the range with the most
    unresolved live tree nodes is replaced by its children and the search
    descends into them.
    """
    order = ctx.order
    n_levels = len(order)
    x = ctx.x
    hiers = ctx.hiers
    h_suffix = ctx.h_suffix
    gap = ctx.gap
    assigned, values, ranges = ctx.assigned, ctx.values, ctx.ranges
    range_span, picked = ctx.range_span, ctx.picked
    model_bounds, if_bounds = ctx.model_bounds, ctx.if_bounds
    propagation, ledger = ctx.propagation, ctx.ledger

    incumbent_cost = math.inf
    incumbent_row: FloatArray | None = None
    incumbent_states: list[_State] | None = None
    warm_start_used = False
    if ctx.incumbent is not None:
        incumbent_cost = ctx.incumbent[0]
        incumbent_row = np.array(ctx.incumbent[1], dtype=np.float64)
        warm_start_used = True

    nodes_expanded = 0
    nodes_pruned_score = 0
    nodes_pruned_cost = 0
    coarse_accepts = 0
    refinements = 0
    gap_prune_fired = False
    completed = True
    assigned_mask = 0
    range_mask = 0
    range_node: list[int | None] = [None] * len(x)
    g = 0.0
    frames: list[_RFrame] = []
    trace = _Trace()

    def frontier_bound() -> float:
        bound = math.inf
        for fr in frames:
            if fr.next < len(fr.alts):
                rest = h_suffix[fr.level + 1] if fr.refine_of is None else 0.0
                bound = min(bound, fr.g_before + fr.alts[fr.next].cost + rest)
        return bound

    def sample(is_incumbent: bool) -> None:
        bound = frontier_bound()
        if gap > 0.0:
            bound = min(bound, incumbent_cost / (1.0 + gap))
        set_aside_view = 0.0 if ledger.dropped_floor == -math.inf else ledger.dropped_floor
        bound = min(bound, set_aside_view, incumbent_cost)
        cost = None if incumbent_row is None else incumbent_cost
        trace.record(nodes_expanded, cost, bound, is_incumbent=is_incumbent)

    def push_feature(level: int) -> None:
        j = order[level]
        hier = hiers[j]
        if hier is not None:
            alts = hier.initial
        else:
            alts = tuple(
                _Alt(state=st, state_idx=i, node=None) for i, st in enumerate(ctx.domains[j])
            )
        frames.append(
            _RFrame(
                j=j,
                alts=alts,
                next=0,
                refine_of=None,
                level=level,
                h_rest=h_suffix[level],
                g_before=g,
            )
        )

    def apply(fr: _RFrame, alt: _Alt) -> bool:
        nonlocal g, assigned_mask, range_mask
        j = fr.j
        prev = (ranges[j], range_span[j], range_node[j], picked[j], values[j], assigned[j])
        if alt.node is not None:
            hier = hiers[j]
            assert hier is not None
            node = hier.nodes[alt.node]
            ranges[j] = node.iv
            range_span[j] = node.span
            range_node[j] = alt.node
            range_mask |= 1 << j
            prop_frame: _PropFrame = ((), ())
            conflict = False
        else:
            ranges[j] = None
            range_span[j] = None
            range_node[j] = None
            range_mask &= ~(1 << j)
            prop_frame, conflict = propagation.apply(j, alt.state.value)
        assigned[j] = True
        values[j] = alt.state.value
        picked[j] = alt.state_idx
        assigned_mask |= 1 << j
        model_frame = model_bounds.apply(j, assigned_mask)
        if_frame = if_bounds.apply(j, assigned_mask) if if_bounds is not None else ()
        g = fr.g_before + alt.cost
        fr.applied = _Applied(alt, model_frame, if_frame, prop_frame, prev)
        return conflict

    def undo(fr: _RFrame) -> None:
        nonlocal g, assigned_mask, range_mask
        ap = fr.applied
        assert ap is not None
        j = fr.j
        propagation.restore(ap.prop)
        model_bounds.restore(ap.model)
        if if_bounds is not None:
            if_bounds.restore(ap.plausibility)
        ranges[j], range_span[j], range_node[j], picked[j], values[j], assigned[j] = ap.prev
        if ranges[j] is None:
            range_mask &= ~(1 << j)
        else:
            range_mask |= 1 << j
        if not assigned[j]:
            assigned_mask &= ~(1 << j)
        g = fr.g_before
        fr.applied = None

    def row_of_values() -> FloatArray:
        row = x.copy()
        for f in order:
            row[f] = values[f]
        return row

    def bracket_inside_target() -> bool:
        if model_bounds.score_min < ctx.lo_t or model_bounds.score_max > ctx.hi_t:
            return False
        return if_bounds is None or if_bounds.score_min >= ctx.min_total_path

    if n_levels:
        push_feature(0)
    while frames:
        fr = frames[-1]
        if fr.applied is not None:
            undo(fr)
        if fr.next >= len(fr.alts):
            frames.pop()
            continue
        if nodes_expanded >= ctx.node_budget or time.monotonic() - ctx.start > ctx.time_budget_s:
            completed = False
            break

        nodes_expanded += 1
        if nodes_expanded & (nodes_expanded - 1) == 0:
            sample(False)
        alt = fr.alts[fr.next]
        fr.next += 1
        conflict = apply(fr, alt)

        if conflict or (ctx.bounded_pairs and ctx.unorderable()):
            nodes_pruned_cost += 1
            continue
        if model_bounds.score_max < ctx.lo_t or model_bounds.score_min > ctx.hi_t:
            nodes_pruned_score += 1
            continue
        if if_bounds is not None and if_bounds.score_max < ctx.min_total_path:
            nodes_pruned_score += 1
            continue
        floor = g + (h_suffix[fr.level + 1] if fr.refine_of is None else 0.0)
        threshold = incumbent_cost if gap == 0.0 else incumbent_cost / (1.0 + gap)
        if floor >= threshold:
            nodes_pruned_cost += 1
            if incumbent_cost > floor:
                gap_prune_fired = True
            continue

        if fr.refine_of is None and fr.level + 1 < n_levels:
            push_feature(fr.level + 1)
            continue

        # every feature holds something: settle the assignment
        if range_mask == 0:
            accepted = ctx.finish(row_of_values(), g)
            if accepted is not None:
                cost = ctx.cost_of(accepted)
                if cost < incumbent_cost:
                    incumbent_cost = cost
                    incumbent_row = accepted
                    incumbent_states = [ctx.domains[f][picked[f]] for f in order]
                    sample(True)
            continue

        # Every closed box ends one of three ways: pruned by a sound bound,
        # accepted here at its true minimum feasible cost, or refined down to
        # atomic cells that the classic completion path decides. A completed
        # search therefore returns a true optimum, and a completed empty search
        # certifies infeasibility, exactly as the classic search does. The
        # representative row costs the sum of its ranges' minima, so no point
        # of the box is cheaper; when it is feasible as it stands, the box is
        # settled without looking inside.
        rep_row = row_of_values()
        if bracket_inside_target() and ctx.accepts(rep_row):
            coarse_accepts += 1
            cost = ctx.cost_of(rep_row)
            if cost < incumbent_cost:
                incumbent_cost = cost
                incumbent_row = rep_row
                incumbent_states = [ctx.domains[f][picked[f]] for f in order]
                sample(True)
            continue

        counts = count_unresolved(
            model_bounds.trees, len(x), ranges, range_mask, assigned, values
        )
        if if_bounds is not None:
            extra = count_unresolved(
                if_bounds.trees, len(x), ranges, range_mask, assigned, values
            )
            counts = [a + b for a, b in zip(counts, extra, strict=True)]
        target = -1
        best = -1
        for f in order:
            if ranges[f] is not None and counts[f] > best:
                target, best = f, counts[f]
        assert target >= 0
        node_id = range_node[target]
        assert node_id is not None
        hier = hiers[target]
        assert hier is not None
        node = hier.nodes[node_id]
        frames.append(
            _RFrame(
                j=target,
                alts=children_of(hier, node_id),
                next=0,
                refine_of=node_id,
                level=n_levels,
                h_rest=node.cost,
                g_before=g - node.cost,
            )
        )
        refinements += 1

    completed = completed and ledger.dropped_floor >= incumbent_cost
    if completed:
        lower_bound = math.inf
        if incumbent_row is not None:
            lower_bound = incumbent_cost if gap == 0.0 else incumbent_cost / (1.0 + gap)
        proof = "optimal_within_gap" if gap > 0.0 and gap_prune_fired else "optimal"
    else:
        open_view = math.inf
        for fr in frames:
            open_view = min(open_view, fr.g_before + fr.h_rest)
        set_aside_view = 0.0 if ledger.dropped_floor == -math.inf else ledger.dropped_floor
        lower_bound = min(open_view, incumbent_cost, set_aside_view)
        proof = "heuristic"
    trace.record(
        nodes_expanded,
        None if incumbent_row is None else incumbent_cost,
        lower_bound,
        is_incumbent=False,
    )
    return _RefineOutcome(
        incumbent_cost=incumbent_cost,
        incumbent_row=incumbent_row,
        incumbent_states=incumbent_states,
        nodes_expanded=nodes_expanded,
        nodes_pruned_score=nodes_pruned_score,
        nodes_pruned_cost=nodes_pruned_cost,
        coarse_accepts=coarse_accepts,
        refinements=refinements,
        completed=completed,
        lower_bound=lower_bound,
        proof=proof,
        warm_start_used=warm_start_used,
        trace=trace.samples(),
    )
