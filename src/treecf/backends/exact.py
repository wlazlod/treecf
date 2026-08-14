"""Exact backend — the branch-and-bound search over the candidate grid.

This file, ``_exact_domains``, ``_exact_orderpairs`` and ``_exact_propagation``
are one implementation, split across four files for size alone. Together they
are the Python reference for the exact backend; a Rust mirror exists and
matches all four bit-for-bit, so operation order in the arithmetic is a
compatibility contract, not a style choice. Here that governs the score
brackets the search prunes on: they are re-summed in full over the trees in
ascending index after every assignment, never patched with an incremental
delta, so the two implementations take the same prune decisions and expand the
same nodes. (The cost arithmetic lives in ``_exact_domains`` under the same
contract.)

``_exact_domains._build_domains`` turns a factual, the joint cell grid, and the
compiled constraints into a per-feature list of candidate counterfactual
states — the branching alphabet — already in the cost order that search wants
them in. ``solve_exact`` then walks that alphabet depth-first with an explicit
stack, bounding each partial assignment by the score interval the ensemble can
still reach and by the cost already spent plus the cheapest possible remainder.

Two rules go beyond that one-point-per-cell alphabet.

A pair of features tied by ``a - b <= 0`` may need a value the alphabet does
not offer: each feature's candidate is the point of its cell nearest to the
factual, and those two points can sit the wrong way round even though the two
cells overlap. The cheapest repair moves both features onto the same value
``t`` somewhere in the overlap, and because the cost is piecewise linear in
``t`` the cheapest ``t`` is one of four points: either factual value, or either
end of the overlap. Cost is not the only thing that decides a repair, though —
another constraint may leave one of the two features a single legal value — so
the values a constraint can demand of either feature are proposed as well.
Every completed assignment gets that repair pass before the arbiter sees it.

Two features whose repaired values would still break some other pair are
handled conservatively: the repairs are applied one pair at a time and the
whole completion is dropped if any pair is still broken afterwards. No wrong
row can come out of that, since ``check_matrix`` still decides every row that
is returned. But a repair that comes to nothing, for any reason at all, leaves
a completion the search never really settled, and it says so afterwards by
reporting that it did not get through the whole space — so a search that comes
back empty-handed there is not claiming that nothing exists. For the one shape
where it can still bound what such a completion was worth (two plain features,
no other pair sharing either of them) it remembers the cost committed so far
and keeps its claim when the answer it did find is already that cheap.

The one feature the repair leaves alone is one carrying a value policy: only
one point per cell is on the policy's grid to begin with, so there is nowhere
legal to move it, and a search over such a pair never claims to have settled
the space.

The other rule is propagation: assigning a feature can settle other features
outright (the trigger side of an implication, or the last free member of a
one-hot group), and a state that contradicts such a settlement is cut without
being explored. Propagation is a shortcut, never an authority — every row is
still checked in full at the end.

Only one of those two constraint kinds narrows what a feature may hold in the
first place, and ``_exact_domains`` is where that happens: a one-hot member is
restricted to 0 and 1, since nothing else can make its group sum to one, while
an implication restricts nothing and only adds a candidate.

An implication does change the grid the alphabet is drawn from, though. The
value it watches for gets a cell of its own, because a routing cell is a
statement about the trees and not about the constraints: every point of one
routes a row identically, but an implication fires on a single point of it and
is silent a hair away, so a search reading one point per cell could never find
that hair. ``_exact_domains._constraint_cells`` cuts those cells before
anything else runs.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from treecf.aim.cells import Cell
from treecf.api import ValuePolicy
from treecf.backends._exact_domains import (
    FloatArray,
    _build_domains,
    _constraint_cells,
    _cost_of_row,
    _demanded_values,
    _domain_span,
    _feature_order,
    _h_suffix,
    _State,
    _validate,
)
from treecf.backends._exact_orderpairs import (
    _achievable_bounds,
    _boundary_candidates,
    _intersect_cell,
)
from treecf.backends._exact_propagation import _Propagation, _PropFrame
from treecf.constraints.compile import CompiledConstraints
from treecf.ir.evaluate import raw_score
from treecf.ir.model import EnsembleIR, SplitOp, Tree

# the tolerance ``CompiledConstraints.check_matrix`` allows a linear constraint;
# an order pair counts as broken here exactly when the arbiter would reject it
_LINEAR_SLACK = 1e-9


@dataclass(frozen=True)
class ExactResult:
    """Outcome of an exact-backend search (the search populates these; this
    class only defines the shape). ``snapped`` is built by the search from the chosen
    states' own ``_State.snapped`` flags (feature name -> flag, for features that
    changed) — ``_build_domains`` does not know which state wins."""

    x_cf: FloatArray | None
    proof: str  # "optimal" | "optimal_within_gap" | "heuristic"
    stats: dict[str, object]
    snapped: dict[str, bool]
    distance: float | None


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


# (feature index, model bracket frame, plausibility bracket frame, cost before
# the move, propagation frame)
_Frame = tuple[
    int,
    tuple[tuple[int, float, float], ...],
    tuple[tuple[int, float, float], ...],
    float,
    _PropFrame,
]


def solve_exact(
    ir: EnsembleIR,
    x: FloatArray,
    interval: tuple[float, float],
    compiled: CompiledConstraints,
    sigma: FloatArray,
    weights: FloatArray,
    lam: float,
    value_policies: Mapping[str, ValuePolicy] | None = None,
    plausibility: tuple[EnsembleIR, float] | None = None,
    node_budget: int = 2_000_000,
    gap: float = 0.0,
    time_budget_s: float = 10.0,
    incumbent: tuple[float, FloatArray] | None = None,
) -> ExactResult:
    """Search the cell grid depth-first for the cheapest counterfactual.

    Features are assigned one at a time in a fixed order, each from its own
    list of candidate states. Two bounds cut the tree: the score bracket the
    ensemble can still reach (a partial assignment whose whole bracket falls
    outside the target can never be completed into the target), and the cost
    already committed plus the cheapest possible remainder. A full assignment
    is accepted only if the compiled constraints admit the row, its score
    re-computed in float space lands in the target, and — when configured —
    the isolation forest still calls it plausible.

    Implications and one-hot groups settle features ahead of the branching: a
    state that contradicts something an earlier assignment already settled is
    cut on the spot. A pair of features tied by ``a <= b`` cuts branches whose
    remaining values can no longer be ordered that way, and a completed
    assignment that breaks such a pair is first repaired by moving both
    features onto one shared value inside their cells. All of that only
    narrows or nudges what the arbiter is shown; the arbiter still decides.

    Args:
        ir: Model whose score must land in ``interval``.
        x: The factual row.
        interval: Closed target interval ``(lo, hi)`` on the raw score.
        compiled: Compiled constraint set; its ``check_matrix`` is the arbiter
            that decides every completed row.
        sigma: Per-feature scale divisors of the objective.
        weights: Per-feature weights of the objective.
        lam: Per-changed-feature penalty of the objective.
        value_policies: Optional per-feature snapping rules for values that move.
        plausibility: Optional ``(isolation forest, minimum total path length)``
            pair; its splits also widen the cell grid.
        node_budget: Maximum number of state assignments to attempt.
        gap: Relative optimality gap to settle for. Above zero the search may
            discard branches that could only improve on the incumbent by less
            than this fraction, and says so through the proof it reports.
        time_budget_s: Wall-clock budget, checked once per assignment.
        incumbent: Optional ``(cost, row)`` warm start from another backend,
            already costed by the caller on the same objective. The caller must
            also have verified the row: the search takes its feasibility on
            trust, prunes against its cost, and may hand it straight back.

    Returns:
        The best row found, the strength of the claim about it, the search
        counters, which features were moved onto a policy grid, and the cost
        of the returned row.

        There are two different ways to come back empty-handed, and callers
        must tell them apart by ``stats["completed"]``, not by ``proof``. An
        ``x_cf`` of None with ``completed`` True is a certificate: every
        assignment the grid allows was tried and none was feasible, so no
        counterfactual exists within the searched space — ``proof`` carries no
        meaning in that case and should be ignored. An ``x_cf`` of None with
        ``completed`` False only means the search never settled the whole
        space, so nothing is proven either way: a budget ran out, or an order
        pair was left undecided — several pairs sharing features that could
        not be repaired one at a time, or a pair whose feature carries a value
        policy.
    """
    start = time.monotonic()
    order_pairs = _validate(compiled, value_policies)
    lo_t, hi_t = interval
    if_ir = plausibility[0] if plausibility is not None else None
    min_total_path = plausibility[1] if plausibility is not None else 0.0

    def accepts(row: FloatArray) -> bool:
        """The arbiter: constraints, then the float-space score, then plausibility."""
        if not bool(compiled.check_matrix(row[np.newaxis, :], x)[0]):
            return False
        score = raw_score(ir, row)
        if not (lo_t <= score <= hi_t):
            return False
        return if_ir is None or raw_score(if_ir, row) >= min_total_path

    # (a) The factual itself: nothing is ever cheaper than not moving at all.
    if accepts(x):
        return ExactResult(
            x_cf=x.copy(),
            proof="optimal",
            stats=_stats(0, 0, 0, 0.0, gap, True, False),
            snapped={},
            distance=0.0,
        )

    grids = (
        _constraint_cells(compiled, ir)
        if if_ir is None
        else _constraint_cells(compiled, ir, if_ir)
    )
    domains = _build_domains(grids, x, compiled, sigma, weights, lam, value_policies)
    order = _feature_order(grids, compiled)
    if any(not domains[j] for j in order):
        # Contradictory constraints left a feature with no legal value at all:
        # nothing to search, and nothing can be feasible.
        return ExactResult(
            x_cf=None,
            proof="optimal",
            stats=_stats(0, 0, 0, math.inf, gap, True, False),
            snapped={},
            distance=None,
        )
    h_suffix = _h_suffix(order, domains)

    level_of = {f: level for level, f in enumerate(order)}
    # Every feature an implication, a one-hot group or an order pair mentions is
    # constraint-referenced, and _feature_order keeps all of those, so the search
    # really does get to decide each of them.
    related = {f for imp in compiled.implications for f in (imp.cond_index, imp.cons_index)}
    related.update(f for group in compiled.onehot_groups for f in group)
    related.update(f for pair in order_pairs for f in pair)
    assert related <= set(order), "a related feature was left out of the search order"

    bounds_lo, bounds_hi = compiled.instance_bounds(x)[:2]
    bounds_lo = np.where(np.isnan(bounds_lo), -math.inf, bounds_lo)
    bounds_hi = np.where(np.isnan(bounds_hi), math.inf, bounds_hi)
    spans: dict[int, tuple[float, float]] = {}
    for f in {f for pair in order_pairs for f in pair}:
        span = _domain_span(domains[f], grids[f], float(bounds_lo[f]), float(bounds_hi[f]))
        if span is not None:
            spans[f] = span
    bounded_pairs = [(a, b) for a, b in order_pairs if a in spans and b in spans]
    state_spans: dict[int, list[tuple[float, float]]] = {}
    for f in {f for pair in bounded_pairs for f in pair}:
        per_state: list[tuple[float, float]] = []
        for st in domains[f]:
            iv = _intersect_cell(grids[f][st.cell_idx], float(bounds_lo[f]), float(bounds_hi[f]))
            per_state.append((st.value, st.value) if iv is None else _achievable_bounds(iv))
        state_spans[f] = per_state

    policies_now: Mapping[str, ValuePolicy] = value_policies or {}

    def policy_active(f: int) -> bool:
        policy = policies_now.get(compiled.feature_names[f])
        return policy is not None and policy != "raw"

    # A feature with a value policy has to land on the policy's grid, so it
    # cannot be nudged to an arbitrary point inside its cell and pairs touching
    # one are never repaired. Nothing else is held back: a repair only ever
    # proposes a row, and the arbiter turns down the ones that break something.
    repairable_pairs = frozenset(
        (a, b) for a, b in order_pairs if not any(policy_active(f) for f in (a, b))
    )
    policy_bound = bool(order_pairs) and len(repairable_pairs) < len(order_pairs)
    onehot_members = {f for group in compiled.onehot_groups for f in group}
    demanded_values = _demanded_values(compiled)
    entangled_pairs = frozenset(
        pair
        for pair in order_pairs
        if any(other != pair and set(other) & set(pair) for other in order_pairs)
    )
    # When a repair comes to nothing, the cost committed so far is a floor on
    # what that completion could still have been worth — but only for the one
    # kind of pair the argument was proven for: two plain features, each
    # sitting on the point of its cell nearest to the factual, with no second
    # pair to disturb. A one-hot member sits on its group's 0 or 1 instead of
    # that nearest point, a feature some constraint can demand an exact value
    # of carries that value among its candidates, and a value policy moves the
    # candidate onto its own grid; in all three the committed cost can be
    # higher than what a repair would have cost, so it is no floor. Pairs are
    # listed in only if they qualify, never out if they look suspicious: a kind
    # of pair nobody has thought of yet lands on withdrawal, not on silence.
    g_floor_pairs = frozenset(
        pair
        for pair in order_pairs
        if pair not in entangled_pairs
        and not any(
            f in onehot_members or f in demanded_values or policy_active(f) for f in pair
        )
    )

    assigned = [False] * len(x)
    values = [0.0] * len(x)
    assigned_mask = 0
    propagation = _Propagation(compiled, domains, assigned, values)
    model_bounds = _EnsembleBounds(ir, assigned, values)
    if_bounds = _EnsembleBounds(if_ir, assigned, values) if if_ir is not None else None

    incumbent_cost = math.inf
    incumbent_row: FloatArray | None = None
    incumbent_states: list[_State] | None = None
    warm_start_used = False
    if incumbent is not None:
        # (b) A warm start from another backend. Its states are unknown, so a
        # warm winner reports no snapping of its own -- the backend that
        # produced the row already applied any value policy to it.
        incumbent_cost = incumbent[0]
        incumbent_row = np.array(incumbent[1], dtype=np.float64)
        warm_start_used = True

    nodes_expanded = 0
    nodes_pruned_score = 0
    nodes_pruned_cost = 0
    gap_prune_fired = False
    completed = True
    # cheapest committed cost among the completions the repair had to set aside;
    # nothing derived from one of those can cost less than this, so once the
    # incumbent is at least as cheap, setting them aside changed nothing
    dropped_floor = -math.inf if policy_bound else math.inf

    stack: list[int] = []  # state index chosen at each assigned level
    frames: list[_Frame] = []
    g_stack = [0.0]  # cost committed before the level of the same index
    g = 0.0
    next_state = 0

    def undo(frame: _Frame) -> None:
        nonlocal g, assigned_mask
        j, model_frame, if_frame, g_before, prop_frame = frame
        propagation.restore(prop_frame)
        model_bounds.restore(model_frame)
        if if_bounds is not None:
            if_bounds.restore(if_frame)
        assigned[j] = False
        assigned_mask &= ~(1 << j)
        g = g_before

    def reach(f: int, movable: bool) -> tuple[float, float]:
        """The values feature ``f`` can still end up holding.

        Undecided, that is every cell it might yet be put in. Decided, it is
        the cell it was put in and not the single value inside it, because a
        boundary repair may still move it anywhere in that cell — unless the
        pair cannot be repaired at all, and then the value it was given is the
        only point left.
        """
        if not assigned[f]:
            return spans[f]
        if not movable:
            return values[f], values[f]
        level = level_of[f]
        chosen = stack[level] if level < len(stack) else next_state
        return state_spans[f][chosen]

    def unorderable() -> bool:
        """True when some pair ``a <= b`` is already out of reach: the lowest
        value ``a`` can still hold is above the highest ``b`` can."""
        for pair in bounded_pairs:
            a, b = pair
            movable = pair in repairable_pairs
            if reach(a, movable)[0] > reach(b, movable)[1]:
                return True
        return False

    def intersected_cell(f: int) -> Cell | None:
        """The cell the current assignment puts ``f`` in, narrowed to its
        constraint bounds; ``None`` when ``f`` is currently missing."""
        level = level_of[f]
        chosen = stack[level] if level < len(stack) else next_state
        picked = domains[f][chosen]
        if picked.is_nan:
            return None
        return _intersect_cell(grids[f][picked.cell_idx], float(bounds_lo[f]), float(bounds_hi[f]))

    def demanded_for(a: int, b: int) -> list[float]:
        """Exact values a repair of this pair may have to land on: what an
        implication asks of either feature, and what the current assignment has
        already settled about them. Feature ``a`` before feature ``b``, its
        constraint's demands before what the search settled, so the order a
        repair tries them in never depends on how the two were found."""
        out: list[float] = []
        for f in (a, b):
            settled = propagation.forced_value[f]
            for value in (*demanded_values.get(f, ()), *(() if settled is None else (settled,))):
                if value not in out:
                    out.append(value)
        return out

    def candidates_for(a: int, b: int) -> list[float]:
        cell_a = intersected_cell(a)
        cell_b = intersected_cell(b)
        if cell_a is None or cell_b is None:
            return []
        return _boundary_candidates(
            cell_a, cell_b, float(x[a]), float(x[b]), demanded_for(a, b)
        )

    def broken(row: FloatArray, pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """The pairs this row orders the wrong way round, by the arbiter's own
        reading. A pair with a missing value on either side is left out: a
        missing value cannot be pulled onto a boundary, so the linear check and
        its missing policy have the last word on it."""
        return [
            (a, b)
            for a, b in pairs
            if not math.isnan(row[a])
            and not math.isnan(row[b])
            and float(row[a]) - float(row[b]) > _LINEAR_SLACK
        ]

    def set_aside(pairs: list[tuple[int, int]]) -> None:
        """Remember a completion the repair could not settle.

        Every repair that comes to nothing goes through here, whatever the
        reason: no candidate to try, a pair still broken afterwards, or the
        arbiter refusing all of them. What differs is only how much can still
        be said about the completion — the committed cost is a floor on what it
        could have become for the pairs listed in ``g_floor_pairs``, and
        nothing at all can be said about any other, so those withdraw outright.
        """
        nonlocal dropped_floor
        if all(pair in g_floor_pairs for pair in pairs):
            dropped_floor = min(dropped_floor, g)
        else:
            dropped_floor = -math.inf

    def finish(row: FloatArray) -> FloatArray | None:
        """The row to weigh against the incumbent, or ``None`` if there is none.

        A completed assignment usually goes straight to the arbiter. When it
        orders some pair the wrong way round, both features of that pair first
        move onto one shared value inside their cells — the cheapest such value
        that the arbiter still accepts. Moving inside a cell cannot change how
        any tree routes the row, so the repair keeps the score it was pruned
        on. Several broken pairs at once are repaired one after another, each
        on the cheapest shared value regardless of the arbiter, and the whole
        completion is dropped if any pair is left broken; that is the
        conservative corner named in the module docstring. Whenever a repair
        comes to nothing, ``set_aside`` records what that completion could
        still have been worth, so the search can say afterwards whether
        anything was left undecided.
        """
        violated = broken(row, order_pairs)
        if not violated:
            return row if accepts(row) else None
        if any(pair not in repairable_pairs for pair in violated):
            return None  # a policy-bound pair; the arbiter rejects the row anyway
        if len(violated) == 1:
            a, b = violated[0]
            best_row: FloatArray | None = None
            best_cost = math.inf
            for t in candidates_for(a, b):
                variant = row.copy()
                variant[a] = t
                variant[b] = t
                cost = _cost_of_row(x, variant, sigma, weights, lam, compiled.allow_missing)
                if cost < best_cost and accepts(variant):
                    best_cost = cost
                    best_row = variant
            if best_row is None:
                set_aside(violated)
            return best_row
        repaired = row.copy()
        for a, b in violated:
            best_t: float | None = None
            best_cost = math.inf
            for t in candidates_for(a, b):
                variant = repaired.copy()
                variant[a] = t
                variant[b] = t
                cost = _cost_of_row(x, variant, sigma, weights, lam, compiled.allow_missing)
                if cost < best_cost:
                    best_cost = cost
                    best_t = t
            if best_t is None:
                set_aside(violated)
                return None
            repaired[a] = best_t
            repaired[b] = best_t
        if broken(repaired, order_pairs) or not accepts(repaired):
            set_aside(violated)
            return None
        return repaired

    while order:
        k = len(stack)
        states = domains[order[k]]
        if next_state >= len(states):
            if not stack:
                break  # the whole space has been enumerated
            undo(frames.pop())
            g_stack.pop()
            next_state = stack.pop() + 1
            continue
        if nodes_expanded >= node_budget or time.monotonic() - start > time_budget_s:
            completed = False
            break

        nodes_expanded += 1
        state = states[next_state]
        j = order[k]
        prop_frame, conflict = propagation.apply(j, state.value)
        assigned[j] = True
        values[j] = state.value
        assigned_mask |= 1 << j
        frame: _Frame = (
            j,
            model_bounds.apply(j, assigned_mask),
            if_bounds.apply(j, assigned_mask) if if_bounds is not None else (),
            g,
            prop_frame,
        )
        g = g + state.cost

        if conflict or (bounded_pairs and unorderable()):
            # No completion below this state can satisfy the constraints, so it
            # is cut on feasibility, counted with the cost prunes (see _stats).
            nodes_pruned_cost += 1
            undo(frame)
            next_state += 1
            continue
        if model_bounds.score_max < lo_t or model_bounds.score_min > hi_t:
            nodes_pruned_score += 1
            undo(frame)
            next_state += 1
            continue
        if if_bounds is not None and if_bounds.score_max < min_total_path:
            nodes_pruned_score += 1
            undo(frame)
            next_state += 1
            continue
        floor = g + h_suffix[k + 1]
        threshold = incumbent_cost if gap == 0.0 else incumbent_cost / (1.0 + gap)
        if floor >= threshold:
            nodes_pruned_cost += 1
            if incumbent_cost > floor:
                gap_prune_fired = True  # only the widened threshold cut this branch
            undo(frame)
            next_state += 1
            continue

        if k + 1 == len(order):
            row = x.copy()
            for level, chosen in enumerate(stack):
                row[order[level]] = domains[order[level]][chosen].value
            row[j] = state.value
            accepted = finish(row)
            if accepted is not None:
                cost = _cost_of_row(x, accepted, sigma, weights, lam, compiled.allow_missing)
                if cost < incumbent_cost:
                    incumbent_cost = cost
                    incumbent_row = accepted
                    incumbent_states = [
                        domains[order[level]][chosen] for level, chosen in enumerate(stack)
                    ]
                    incumbent_states.append(state)
            undo(frame)
            next_state += 1
            continue

        stack.append(next_state)
        frames.append(frame)
        g_stack.append(g)
        next_state = 0

    completed = completed and dropped_floor >= incumbent_cost
    if completed:
        lower_bound = math.inf
        if incumbent_row is not None:
            lower_bound = incumbent_cost if gap == 0.0 else incumbent_cost / (1.0 + gap)
        proof = "optimal_within_gap" if gap > 0.0 and gap_prune_fired else "optimal"
    else:
        open_view = math.inf
        if order:
            open_view = min(g_stack[level] + h_suffix[level] for level in range(len(g_stack)))
        # a completion the repair set aside is worth at least its committed
        # cost, or — where even that does not hold — at least nothing, since
        # the objective is a sum of non-negative terms
        set_aside_view = 0.0 if dropped_floor == -math.inf else dropped_floor
        lower_bound = min(open_view, incumbent_cost, set_aside_view)
        proof = "heuristic"

    snapped: dict[str, bool] = {}
    for level, chosen_state in enumerate(incumbent_states or []):
        f = order[level]
        # a feature an order-pair repair moved no longer holds the value the
        # policy produced, so it is not reported as snapped either
        if incumbent_row is not None and incumbent_row[f] != chosen_state.value:
            continue
        if chosen_state.snapped and chosen_state.value != x[f]:
            snapped[compiled.feature_names[f]] = True

    return ExactResult(
        x_cf=incumbent_row,
        proof=proof,
        stats=_stats(
            nodes_expanded,
            nodes_pruned_score,
            nodes_pruned_cost,
            lower_bound,
            gap,
            completed,
            warm_start_used,
        ),
        snapped=snapped,
        distance=None if incumbent_row is None else incumbent_cost,
    )


def _stats(
    nodes_expanded: int,
    nodes_pruned_score: int,
    nodes_pruned_cost: int,
    lower_bound: float,
    gap: float,
    completed: bool,
    warm_start_used: bool,
) -> dict[str, object]:
    """The exact set of counters ``solve_exact`` reports.

    ``nodes_pruned_score`` counts branches the ensemble can no longer bring
    into the target (the plausibility bound counts here too, being the same
    kind of reachability test). ``nodes_pruned_cost`` counts every other cut:
    branches too expensive to beat the incumbent, and branches no constraint
    can be satisfied in — a state contradicting an implication or a one-hot
    group, or an order pair whose two features can no longer be ordered. A
    mirror of this search has to file those feasibility cuts the same way,
    since the counter set itself is fixed.
    """
    return {
        "nodes_expanded": nodes_expanded,
        "nodes_pruned_score": nodes_pruned_score,
        "nodes_pruned_cost": nodes_pruned_cost,
        "lower_bound": lower_bound,
        "gap": gap,
        "completed": completed,
        "warm_start_used": warm_start_used,
    }
