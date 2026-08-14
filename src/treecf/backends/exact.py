"""Exact backend — domains, state costs, canonical orders, branch-and-bound search.

This file is the Python reference implementation of the exact backend; a Rust
mirror lands later and must match it bit-for-bit, so operation order in the
cost arithmetic below is a compatibility contract, not a style choice — every
multiply/divide/add mirrors ``treecf.backends.genetic``'s ``objective()``
term-for-term. The same contract governs the score brackets the search prunes
on: they are re-summed in full over the trees in ascending index after every
assignment, never patched with an incremental delta, so the two
implementations take the same prune decisions and expand the same nodes.

``_build_domains`` turns a factual, the joint cell grid, and the compiled
constraints into a per-feature list of candidate counterfactual states — the
branching alphabet — already in the cost order that search wants them in.
``solve_exact`` then walks that alphabet depth-first with an explicit stack,
bounding each partial assignment by the score interval the ensemble can still
reach and by the cost already spent plus the cheapest possible remainder.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from treecf._errors import ConstraintValidationError
from treecf.aim.cells import Cell, cell_index, feature_cells
from treecf.api import ValuePolicy, _snap
from treecf.constraints.compile import CompiledConstraints
from treecf.constraints.objects import (
    AllowMissing,
    Equals,
    Freeze,
    Implies,
    Linear,
    Monotone,
    OneHot,
    Range,
)
from treecf.ir.evaluate import raw_score
from treecf.ir.model import EnsembleIR, SplitOp, Tree

FloatArray = npt.NDArray[np.float64]

_SUPPRESSING_MISSING_POLICIES = ("forbid_missing", "violated")


@dataclass(frozen=True)
class ExactResult:
    """Outcome of an exact-backend search (Task 2.3 produces these; this task
    only defines the shape). ``snapped`` is built by the search from the chosen
    states' own ``_State.snapped`` flags (feature name -> flag, for features that
    changed) — ``_build_domains`` does not know which state wins."""

    x_cf: FloatArray | None
    proof: str  # "optimal" | "optimal_within_gap" | "heuristic"
    stats: dict[str, object]
    snapped: dict[str, bool]
    distance: float | None


@dataclass(frozen=True)
class _State:
    """One candidate value a feature may take in the search.

    ``snapped`` is True only for a movement candidate whose value was produced
    by ``_snap`` under an active value policy; keep states, pinned-value states,
    and NaN states are never snapped.
    """

    value: float
    cost: float
    cell_idx: int
    is_nan: bool
    snapped: bool = False


def _sort_key(state: _State) -> tuple[float, int, float]:
    """Canonical state order: ascending cost, ties by ascending cell index (the
    NaN state's sentinel index makes it sort last among ties), remaining ties —
    only possible between the two binary states sharing one cell — by value, so
    0.0 sorts before 1.0."""
    return (state.cost, state.cell_idx, 0.0 if state.is_nan else state.value)


def _term_cost(
    x_j: float,
    r: float,
    weight_j: float,
    sigma_j: float,
    lam: float,
    to_miss: float,
    from_miss: float,
) -> float:
    """One feature's contribution to the objective, mirroring the per-feature
    term of ``genetic.objective()`` exactly — same four cases, same
    multiply-then-divide order, so results stay bit-identical across backends.
    """
    x_nan = math.isnan(x_j)
    r_nan = math.isnan(r)
    if x_nan and r_nan:
        return 0.0
    if x_nan:  # NaN -> value: priced by delta_from_miss, independent of r
        return (weight_j * from_miss) / sigma_j + lam
    if r_nan:  # value -> NaN: priced by delta_miss
        return (weight_j * to_miss) / sigma_j + lam
    if r == x_j:  # keep: no movement, no sparsity penalty
        return 0.0
    delta = abs(r - x_j)
    return lam + (weight_j * delta) / sigma_j


def _cost_of_row(
    x: FloatArray,
    row: FloatArray,
    sigma: FloatArray,
    weights: FloatArray,
    lam: float,
    allow_missing: Mapping[int, tuple[float, float]],
) -> float:
    """Full-row objective, accumulated ascending feature index like
    ``genetic.objective()``. ``row`` is an arbitrary candidate — not necessarily
    built from domain states — used for the factual incumbent, warm-start
    re-costing, and the ``distance`` a later task returns.
    """
    total = 0.0
    for j in range(len(x)):
        to_miss, from_miss = allow_missing.get(j, (0.0, 0.0))
        total += _term_cost(
            float(x[j]), float(row[j]), float(weights[j]), float(sigma[j]), lam, to_miss, from_miss
        )
    return total


def _intersect_cell(cell: Cell, lo: float, hi: float) -> Cell | None:
    """``cell`` ∩ ``[lo, hi]``; ``lo``/``hi`` are always closed bounds. ``None``
    if the intersection is empty (including a degenerate open singleton)."""
    if cell.lo > lo:
        new_lo, new_lo_open = cell.lo, cell.lo_open
    elif lo > cell.lo:
        new_lo, new_lo_open = lo, False
    else:
        new_lo, new_lo_open = cell.lo, cell.lo_open
    if cell.hi < hi:
        new_hi, new_hi_open = cell.hi, cell.hi_open
    elif hi < cell.hi:
        new_hi, new_hi_open = hi, False
    else:
        new_hi, new_hi_open = cell.hi, cell.hi_open
    if new_lo > new_hi:
        return None
    if new_lo == new_hi and (new_lo_open or new_hi_open):
        return None
    return Cell(new_lo, new_hi, new_lo_open, new_hi_open)


def _build_domains(
    grids: tuple[tuple[Cell, ...], ...],
    x: FloatArray,
    compiled: CompiledConstraints,
    sigma: FloatArray,
    weights: FloatArray,
    lam: float,
    value_policies: Mapping[str, ValuePolicy] | None,
) -> list[list[_State]]:
    """Per-feature candidate states, each list sorted in canonical order
    (ascending cost, ties by ascending cell index, NaN last among ties).

    Assumes ``_validate`` already accepted ``compiled``/``value_policies`` (no
    multi-feature Linears, no callable policies) — this function does not
    re-check either.

    A frozen feature gets a single keep state at zero cost (Freeze always pins
    ``lo == hi == x[j]`` for a non-NaN factual, so this is really a special
    case of the next rule, but it is checked directly since a frozen feature
    can never combine with ``AllowMissing``, unlike a plain pin). A feature
    pinned by some other constraint to a single point ``v`` (``lo == hi``)
    gets that exact value ``v`` as its only non-NaN state, at its normal
    movement cost (zero only if ``v`` happens to equal the factual) — the
    pinned value is authoritative and exempt from value-policy snapping,
    since constraints win over policies. ``AllowMissing`` still adds the
    missing state next to it: a pin restricts which value the feature may
    take, not whether it may go missing, exactly as the bounds check reads
    it. A NaN factual without
    ``AllowMissing`` gets a single keep-NaN state; with ``AllowMissing`` it
    additionally offers moving to the pinned value ``v``, priced by
    ``delta_from_miss``; either NaN-involving state is dropped when a
    single-feature Linear's ``missing_policy`` forbids NaN there, and a NaN
    factual that is both forced to stay NaN and forbidden from being NaN
    yields an empty domain for that feature — a certified-infeasible signal
    for the search, not an error.

    Every other feature intersects each grid cell with its bounds first
    (dropping empty intersections, preserving open/closed edges), and each
    surviving cell contributes its nearest point to the factual value as a
    candidate — except binary features, which instead keep whichever of
    0.0/1.0 the intersected cell contains, forced to that exact value. A
    value policy snaps every such movement candidate, dropping it on
    failure; the factual's own unchanged value is always available and
    exempt from snapping, so a value policy can never force a feature that
    did not need to move.
    """
    lo, hi, frozen = compiled.instance_bounds(x)
    lo = np.where(np.isnan(lo), -math.inf, lo)
    hi = np.where(np.isnan(hi), math.inf, hi)
    suppress_nan = {
        lin.indices[0]
        for lin in compiled.linears
        if len(lin.indices) == 1 and lin.missing_policy in _SUPPRESSING_MISSING_POLICIES
    }
    policies: Mapping[str, ValuePolicy] = value_policies or {}

    domains: list[list[_State]] = []
    for j in range(len(x)):
        x_j = float(x[j])
        x_nan = math.isnan(x_j)
        allow_j = j in compiled.allow_missing
        lo_j, hi_j = float(lo[j]), float(hi[j])
        pinned = lo_j == hi_j
        cells = grids[j]
        to_miss, from_miss = compiled.allow_missing.get(j, (0.0, 0.0))
        weight_j, sigma_j = float(weights[j]), float(sigma[j])

        if frozen[j]:
            idx = len(cells) if x_nan else cell_index(cells, x_j)
            domains.append([_State(x_j, 0.0, idx, x_nan)])
            continue

        if pinned:
            v = lo_j
            if not x_nan:
                # The pin fixes the only value the feature may *take*; it says
                # nothing about becoming missing, which the bounds check waves
                # through, so AllowMissing still offers that second state here.
                cost = _term_cost(x_j, v, weight_j, sigma_j, lam, to_miss, from_miss)
                pinned_states = [_State(v, cost, cell_index(cells, v), False)]
                if allow_j and j not in suppress_nan:
                    nan_cost = _term_cost(
                        x_j, math.nan, weight_j, sigma_j, lam, to_miss, from_miss
                    )
                    pinned_states.append(_State(math.nan, nan_cost, len(cells), True))
                    pinned_states.sort(key=_sort_key)
                domains.append(pinned_states)
                continue
            # A NaN factual pinned to v: staying NaN is legal only when no
            # single-feature Linear forbids it here (missing_policy); moving to
            # v is legal only under AllowMissing. Both can fail at once (a
            # forbid_missing feature with no AllowMissing and a NaN factual) --
            # that yields an empty domain here, a certified-infeasible signal
            # for the search, not an error.
            nan_states: list[_State] = []
            if j not in suppress_nan:
                nan_states.append(_State(math.nan, 0.0, len(cells), True))
            if allow_j:
                cost = _term_cost(x_j, v, weight_j, sigma_j, lam, to_miss, from_miss)
                nan_states.append(_State(v, cost, cell_index(cells, v), False))
            nan_states.sort(key=_sort_key)
            domains.append(nan_states)
            continue

        if x_nan and not allow_j:
            domains.append([_State(x_j, 0.0, 0, True)])
            continue

        name = compiled.feature_names[j]
        raw_policy = policies.get(name)
        policy = None if raw_policy is None or raw_policy == "raw" else raw_policy

        anchor = 0.0 if x_nan else x_j
        is_binary = j in compiled.binary_features

        states: list[_State] = []
        keep_added = False
        if not x_nan and lo_j <= x_j <= hi_j:
            states.append(_State(x_j, 0.0, cell_index(cells, x_j), False))
            keep_added = True

        for local_idx, cell in enumerate(cells):
            iv = _intersect_cell(cell, lo_j, hi_j)
            if iv is None:
                continue
            if is_binary:
                for val in (0.0, 1.0):
                    if not iv.contains(val):
                        continue
                    if keep_added and val == x_j:
                        continue
                    cost = _term_cost(x_j, val, weight_j, sigma_j, lam, to_miss, from_miss)
                    states.append(_State(val, cost, local_idx, False))
                continue
            r = iv.nearest_to(anchor)
            if keep_added and r == x_j:
                continue
            snapped = False
            if policy is not None:
                snapped_r = _snap(r, policy, iv.contains, lo_j, hi_j)
                if snapped_r is None:
                    continue
                r = snapped_r
                snapped = True
            cost = _term_cost(x_j, r, weight_j, sigma_j, lam, to_miss, from_miss)
            states.append(_State(r, cost, local_idx, False, snapped))

        if allow_j and j not in suppress_nan:
            nan_cost = _term_cost(x_j, math.nan, weight_j, sigma_j, lam, to_miss, from_miss)
            states.append(_State(math.nan, nan_cost, len(cells), True))

        states.sort(key=_sort_key)
        domains.append(states)

    return domains


def _referenced_feature_indices(compiled: CompiledConstraints) -> frozenset[int]:
    """Feature indices touched by any constraint in the set, of any kind."""
    index = {name: j for j, name in enumerate(compiled.feature_names)}
    refs: set[int] = set()
    for c in compiled.constraints:
        if isinstance(c, Freeze | Range | Monotone | Equals | AllowMissing):
            refs.add(index[c.feature])
        elif isinstance(c, Linear):
            refs.update(index[name] for name in c.coefficients)
        elif isinstance(c, Implies):
            refs.add(index[c.condition.feature])
            refs.add(index[c.consequence.feature])
        elif isinstance(c, OneHot):
            refs.update(index[name] for name in c.features)
    return frozenset(refs)


def _feature_order(
    grids: tuple[tuple[Cell, ...], ...], compiled: CompiledConstraints
) -> list[int]:
    """Search order: descending split count in the joint grid, ties ascending
    feature index. Features with no split anywhere in the joint grid AND no
    referencing constraint are excluded — such a feature's domain is
    trivially a single keep state, so it never needs to branch."""
    referenced = _referenced_feature_indices(compiled)
    split_counts = [len(cells) - 1 for cells in grids]
    included = [j for j in range(len(grids)) if split_counts[j] > 0 or j in referenced]
    return sorted(included, key=lambda j: (-split_counts[j], j))


def _h_suffix(order: list[int], domains: list[list[_State]]) -> list[float]:
    """Suffix sums of each ordered feature's cheapest state cost — a static
    lower-bound table: ``h_suffix[k]`` is the minimum possible remaining cost
    once features ``order[k:]`` are still undecided. Domain state lists are
    already sorted ascending by cost, so the cheapest state is index 0."""
    suffix = [0.0] * (len(order) + 1)
    for k in range(len(order) - 1, -1, -1):
        suffix[k] = suffix[k + 1] + domains[order[k]][0].cost
    return suffix


def _validate(
    compiled: CompiledConstraints, value_policies: Mapping[str, ValuePolicy] | None
) -> list[tuple[int, int]]:
    """Reject constraint/value-policy shapes the exact backend cannot yet handle.

    Single-feature Linears are accepted silently — their bound already lives
    in ``compiled.derived_ranges``, and their ``missing_policy`` still governs
    the feature's NaN state in ``_build_domains``. Multi-feature Linears in
    the canonical order-pair shape (``a - b <= 0``) are recognized and
    returned, but still rejected below — a later task lifts this restriction
    by deleting the ``if order_pairs`` block, nothing else changes. Any other
    multi-feature Linear, and any callable value policy, name
    ``backend="genetic"`` as the fallback.
    """
    order_pairs: list[tuple[int, int]] = []
    for lin in compiled.linears:
        if len(lin.indices) == 1:
            continue
        if lin.op == "<=" and lin.rhs == 0.0 and sorted(lin.coefs) == [-1.0, 1.0]:
            a = lin.indices[lin.coefs.index(1.0)]
            b = lin.indices[lin.coefs.index(-1.0)]
            order_pairs.append((a, b))
            continue
        raise ConstraintValidationError(
            f"Linear constraint over multiple features ({lin.coefficients}) is not "
            'supported by the exact backend yet; use backend="genetic".'
        )
    if order_pairs:  # a later task lifts this: delete this block to enable order pairs.
        raise ConstraintValidationError(
            "order-pair Linear constraints (feature_a <= feature_b) are recognized "
            "but the exact backend does not search over them yet; support is "
            "coming in a later task."
        )

    for name, policy in (value_policies or {}).items():
        if callable(policy):
            raise ConstraintValidationError(
                f"callable value_policy for {name!r} is not supported by the exact "
                'backend; use backend="genetic".'
            )

    return order_pairs


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


# (feature index, model bracket frame, plausibility bracket frame, cost before the move)
_Frame = tuple[
    int,
    tuple[tuple[int, float, float], ...],
    tuple[tuple[int, float, float], ...],
    float,
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
        ``completed`` False only means the node or time budget ran out first;
        the space was never exhausted and nothing is proven either way.
    """
    start = time.monotonic()
    _validate(compiled, value_policies)
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

    grids = feature_cells(ir) if if_ir is None else feature_cells(ir, if_ir)
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

    assigned = [False] * len(x)
    values = [0.0] * len(x)
    assigned_mask = 0
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

    stack: list[int] = []  # state index chosen at each assigned level
    frames: list[_Frame] = []
    g_stack = [0.0]  # cost committed before the level of the same index
    g = 0.0
    next_state = 0

    def undo(frame: _Frame) -> None:
        nonlocal g, assigned_mask
        j, model_frame, if_frame, g_before = frame
        model_bounds.restore(model_frame)
        if if_bounds is not None:
            if_bounds.restore(if_frame)
        assigned[j] = False
        assigned_mask &= ~(1 << j)
        g = g_before

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
        assigned[j] = True
        values[j] = state.value
        assigned_mask |= 1 << j
        # A propagation pass over one-hot groups and implications belongs here,
        # between the assignment and the bounds it feeds.
        frame: _Frame = (
            j,
            model_bounds.apply(j, assigned_mask),
            if_bounds.apply(j, assigned_mask) if if_bounds is not None else (),
            g,
        )
        g = g + state.cost

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
            if accepts(row):
                cost = _cost_of_row(x, row, sigma, weights, lam, compiled.allow_missing)
                if cost < incumbent_cost:
                    incumbent_cost = cost
                    incumbent_row = row
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

    if completed:
        lower_bound = math.inf
        if incumbent_row is not None:
            lower_bound = incumbent_cost if gap == 0.0 else incumbent_cost / (1.0 + gap)
        proof = "optimal_within_gap" if gap > 0.0 and gap_prune_fired else "optimal"
    else:
        open_view = math.inf
        if order:
            open_view = min(g_stack[level] + h_suffix[level] for level in range(len(g_stack)))
        lower_bound = min(open_view, incumbent_cost)
        proof = "heuristic"

    snapped: dict[str, bool] = {}
    for level, chosen_state in enumerate(incumbent_states or []):
        if chosen_state.snapped and chosen_state.value != x[order[level]]:
            snapped[compiled.feature_names[order[level]]] = True

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
    """The exact set of counters ``solve_exact`` reports."""
    return {
        "nodes_expanded": nodes_expanded,
        "nodes_pruned_score": nodes_pruned_score,
        "nodes_pruned_cost": nodes_pruned_cost,
        "lower_bound": lower_bound,
        "gap": gap,
        "completed": completed,
        "warm_start_used": warm_start_used,
    }
