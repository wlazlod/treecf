"""Certified recourse regions: widen a verified counterfactual into a sound box.

A :class:`RecourseRegion` is a per-feature interval around one already-verified
counterfactual ``x_cf``: every point inside it -- not just ``x_cf`` itself --
is provably still in the target interval, still plausible (when configured),
and still constraint-feasible. That makes the region a *certified*
neighbourhood, not a heuristic one: a user who lands anywhere inside it (a
slightly different repayment, a slightly later date) keeps every guarantee
the original counterfactual carried.

The region is built by growing a box outward from the single point ``x_cf``,
one joint-grid cell at a time, keeping only expansions a sound oracle proves
safe. The oracle itself never trusts an assumption it has not checked: an
interval-tree walk of every ensemble tree brackets the raw score the whole
box can reach, and every Linear constraint is checked at its worst corner.
Features an implication, a one-hot group, or an unsupported multi-feature
Linear could still break are pinned at ``x_cf`` and never widened at all --
conservative, but always sound.

This module works from the AIM/IR layer only (compiled constraints, the IR's
own trees), so it applies to a counterfactual from *any* backend, including
the genetic default. Computing a region costs one oracle call -- a full
interval-tree walk of every ensemble tree (and the isolation forest's, when
plausibility is configured) -- per attempted per-feature, per-direction
expansion; a feature with many joint-grid cells inside its instance bounds
costs proportionally more.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from treecf.aim.cells import Cell, cell_index
from treecf.backends._exact_domains import _constraint_cells
from treecf.backends._exact_orderpairs import _achievable_bounds, _intersect_cell
from treecf.constraints.compile import CompiledConstraints, ResolvedLinear
from treecf.ir.model import EnsembleIR, Node, SplitOp

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

_LINEAR_SLACK = 1e-9  # matches Explainer._verify / CompiledConstraints.check_matrix


@dataclass(frozen=True)
class RecourseRegion:
    """A certified box around one verified counterfactual.

    Every point ``z`` with ``lo <= z <= hi`` coordinate-wise (``z_j = x_cf_j``
    at a degenerate or NaN coordinate) is provably in-target, plausible when
    plausibility is configured, and feasible against every compiled
    constraint -- the same guarantees the counterfactual itself carries, not
    a heuristic neighbourhood around it.

    ``lo``/``hi`` cover every feature (degenerate coordinates included, as a
    single point); ``feature_intervals`` keys only the non-degenerate ones by
    name, for display. Regions are certified but neither maximal (a larger
    sound box may exist) nor monotone in the target interval (a strictly
    narrower target can still produce a strictly wider region on some
    feature: growth is greedy and order-dependent, so a feature that is
    forced to stop early frees room a later feature grows into).
    """

    lo: FloatArray
    hi: FloatArray
    feature_intervals: dict[str, tuple[float, float]]
    certified: bool  # always True in this release; the field is reserved

    def contains(self, x: FloatArray) -> bool:
        """Whether ``x`` lies inside the region, coordinate by coordinate."""
        for j in range(len(self.lo)):
            xj = float(x[j])
            if math.isnan(self.lo[j]):
                if not math.isnan(xj):
                    return False
                continue
            if math.isnan(xj) or not (self.lo[j] <= xj <= self.hi[j]):
                return False
        return True

    def describe(self) -> dict[str, str]:
        """One human-readable phrase per non-degenerate feature.

        One-sided (``"<= v"``/``">= v"``) when the other endpoint is
        infinite, two-sided (``"in [lo, hi]"``) otherwise; values formatted
        ``"{:.3g}"``.
        """
        out: dict[str, str] = {}
        for name, (lo, hi) in self.feature_intervals.items():
            if lo == -math.inf and hi == math.inf:
                out[name] = "unconstrained"
            elif lo == -math.inf:
                out[name] = f"≤ {hi:.3g}"
            elif hi == math.inf:
                out[name] = f"≥ {lo:.3g}"
            else:
                out[name] = f"in [{lo:.3g}, {hi:.3g}]"
        return out


def _is_order_pair(lin: ResolvedLinear) -> bool:
    """The canonical ``a - b <= 0`` shape the exact backend also repairs directly."""
    return lin.op == "<=" and lin.rhs == 0.0 and sorted(lin.coefs) == [-1.0, 1.0]


def _degenerate_features(
    compiled: CompiledConstraints,
    frozen: BoolArray,
    lo_b: FloatArray,
    hi_b: FloatArray,
    x_cf: FloatArray,
) -> frozenset[int]:
    """Coordinates the growth loop must never widen: pinned to ``x_cf`` as-is.

    Frozen or pinned (``lo == hi``) features, a NaN value in ``x_cf``, every
    one-hot member, every feature an Implies references (condition or
    consequence -- pointwise implication satisfaction is not proven sound to
    widen either side), and every feature a Linear references outside the two
    supported shapes (single-feature, and the canonical order pair) -- for
    those, conservatively, every referenced feature is pinned rather than
    trusting a general worst-corner argument this release has not proven.
    """
    onehot_members = {f for group in compiled.onehot_groups for f in group}
    implies_members = {
        f for imp in compiled.implications for f in (imp.cond_index, imp.cons_index)
    }
    linear_members: set[int] = set()
    for lin in compiled.linears:
        if len(lin.indices) == 1 or _is_order_pair(lin):
            continue
        linear_members.update(lin.indices)
    return frozenset(
        j
        for j in range(len(x_cf))
        if frozen[j]
        or lo_b[j] == hi_b[j]
        or math.isnan(x_cf[j])
        or j in onehot_members
        or j in implies_members
        or j in linear_members
    )


def _tree_interval_bracket(
    nodes: tuple[Node, ...], idx: int, lo: FloatArray, hi: FloatArray, is_nan: BoolArray
) -> tuple[float, float] | None:
    """``[min, max]`` leaf value reachable from ``nodes[idx]`` over the box, or
    ``None`` if the box cannot be soundly bracketed at all.

    At a split, the box's interval on that feature decides whether only the
    left child, only the right, or both are still reachable; a NaN-degenerate
    feature (fixed, never widened) instead routes by ``missing_left`` alone,
    the same single-child routing ``raw_score`` itself takes -- unless
    ``missing_left`` is undefined at that node. The counterfactual's own
    verified path never visits such a node (verification already re-scored it
    without raising), but widening an ANCESTOR feature's interval can open a
    subtree that does, for a point the region would otherwise certify though
    ``raw_score`` refuses to score it. There is no per-point re-check after a
    region ships, so guessing a side here (the exact backend's own bound
    computation may, since every row it returns is re-verified individually)
    would be an unsound shortcut; returning ``None`` instead makes the whole
    box's bracket undefined, which the oracle reads as a flat rejection.
    """
    node = nodes[idx]
    if node.feature is None:
        assert node.value is not None
        return node.value, node.value
    f = node.feature
    assert node.threshold is not None and node.left is not None and node.right is not None
    if is_nan[f]:
        if node.missing_left is None:
            return None
        child = node.left if node.missing_left else node.right
        return _tree_interval_bracket(nodes, child, lo, hi, is_nan)
    threshold = node.threshold
    lo_f, hi_f = float(lo[f]), float(hi[f])
    if node.op is SplitOp.LT:
        all_left, all_right = hi_f < threshold, lo_f >= threshold
    else:
        all_left, all_right = hi_f <= threshold, lo_f > threshold
    if all_left:
        return _tree_interval_bracket(nodes, node.left, lo, hi, is_nan)
    if all_right:
        return _tree_interval_bracket(nodes, node.right, lo, hi, is_nan)
    left = _tree_interval_bracket(nodes, node.left, lo, hi, is_nan)
    if left is None:
        return None
    right = _tree_interval_bracket(nodes, node.right, lo, hi, is_nan)
    if right is None:
        return None
    lmin, lmax = left
    rmin, rmax = right
    return min(lmin, rmin), max(lmax, rmax)


def _ensemble_bracket(
    ir: EnsembleIR, lo: FloatArray, hi: FloatArray, is_nan: BoolArray
) -> tuple[float, float] | None:
    """``[min, max]`` raw score the ensemble can reach anywhere in the box, or
    ``None`` if any tree's bracket could not be soundly computed (see
    ``_tree_interval_bracket``).

    Summed base + ascending tree index, the same order ``raw_score`` adds in.
    """
    total_min = ir.base_score
    total_max = ir.base_score
    for tree in ir.trees:
        bracket = _tree_interval_bracket(tree.nodes, 0, lo, hi, is_nan)
        if bracket is None:
            return None
        tree_min, tree_max = bracket
        total_min = total_min + tree_min
        total_max = total_max + tree_max
    return total_min, total_max


def _linear_holds(lin: ResolvedLinear, x_cf: FloatArray, lo: FloatArray, hi: FloatArray) -> bool:
    """Worst-corner feasibility of one Linear constraint over the box.

    Every degenerate feature has ``lo[j] == hi[j]``, so its term reduces to
    the fixed value ``x_cf`` already satisfies -- one formula covers both a
    genuine range on a supported single-feature/order-pair Linear and a
    Linear whose features were pinned by ``_degenerate_features`` instead,
    with no special-casing between the two.
    """
    if any(math.isnan(x_cf[j]) for j in lin.indices):
        return lin.missing_policy == "satisfied"
    lo_sum = 0.0
    hi_sum = 0.0
    for coef, j in zip(lin.coefs, lin.indices, strict=True):
        a, b = coef * lo[j], coef * hi[j]
        lo_sum += min(a, b)
        hi_sum += max(a, b)
    if lin.op == "<=":
        return hi_sum <= lin.rhs + _LINEAR_SLACK
    if lin.op == ">=":
        return lo_sum >= lin.rhs - _LINEAR_SLACK
    return lo_sum >= lin.rhs - _LINEAR_SLACK and hi_sum <= lin.rhs + _LINEAR_SLACK


def _box_feasible(
    ir: EnsembleIR,
    if_ir: EnsembleIR | None,
    min_total_path: float,
    interval: tuple[float, float],
    linears: tuple[ResolvedLinear, ...],
    x_cf: FloatArray,
    lo: FloatArray,
    hi: FloatArray,
    is_nan: BoolArray,
) -> bool:
    """The soundness oracle: True only if EVERY point of the box is provably
    still in-target, plausible, and Linear-feasible -- rejects on any doubt,
    including an ensemble bracket that could not be soundly computed at all
    (``_ensemble_bracket`` returning ``None``: an unrouted missing split some
    point of the box could reach)."""
    bracket = _ensemble_bracket(ir, lo, hi, is_nan)
    if bracket is None:
        return False
    score_min, score_max = bracket
    if score_min < interval[0] or score_max > interval[1]:
        return False
    if if_ir is not None:
        if_bracket = _ensemble_bracket(if_ir, lo, hi, is_nan)
        if if_bracket is None:
            return False
        if_min, _if_max = if_bracket
        if if_min < min_total_path:
            return False
    return all(_linear_holds(lin, x_cf, lo, hi) for lin in linears)


def _next_edge(
    cells: tuple[Cell, ...], value: float, lo_b: float, hi_b: float, upper: bool
) -> float | None:
    """The achievable far edge one joint-grid cell beyond ``value``, or ``None``.

    Finishes the cell ``value`` is already inside first, if it has not yet
    reached that cell's own achievable edge; once it has, claims the whole
    next cell. Clamped to the instance bounds throughout; ``None`` when there
    is no further cell, the next one does not intersect the bounds at all, or
    ``value`` has already grown to an unbounded (``+-inf``) edge -- infinity
    marks "no further constraint that way", not a cell any point occupies, so
    it can never itself be looked up again.
    """
    if math.isinf(value):
        return None
    idx = cell_index(cells, value)
    iv = _intersect_cell(cells[idx], lo_b, hi_b)
    assert iv is not None  # value is within the instance bounds by construction
    cur_lo, cur_hi = _achievable_bounds(iv)
    if upper:
        if cur_hi > value:
            return cur_hi
        if idx + 1 >= len(cells):
            return None
        nxt = _intersect_cell(cells[idx + 1], lo_b, hi_b)
        return None if nxt is None else _achievable_bounds(nxt)[1]
    if cur_lo < value:
        return cur_lo
    if idx - 1 < 0:
        return None
    prev = _intersect_cell(cells[idx - 1], lo_b, hi_b)
    return None if prev is None else _achievable_bounds(prev)[0]


def _try_grow(
    j: int,
    upper: bool,
    box_lo: FloatArray,
    box_hi: FloatArray,
    cells: tuple[Cell, ...],
    lo_b: float,
    hi_b: float,
    oracle: Callable[[], bool],
) -> bool:
    """Attempt one cell of growth on feature ``j`` in one direction.

    Mutates ``box_lo``/``box_hi`` in place; keeps the extension iff the
    oracle accepts the whole box with it, retracting otherwise.
    """
    current = float(box_hi[j]) if upper else float(box_lo[j])
    candidate = _next_edge(cells, current, lo_b, hi_b, upper)
    if candidate is None:
        return False
    if upper:
        box_hi[j] = candidate
    else:
        box_lo[j] = candidate
    if oracle():
        return True
    if upper:
        box_hi[j] = current
    else:
        box_lo[j] = current
    return False


def _recourse_region(
    ir: EnsembleIR,
    x: FloatArray,
    x_cf: FloatArray,
    interval: tuple[float, float],
    compiled: CompiledConstraints,
    if_ir: EnsembleIR | None,
    min_total_path: float,
) -> RecourseRegion:
    """Grow a certified box around the verified counterfactual ``x_cf``.

    ``x`` is the original factual (instance bounds -- Freeze/Monotone among
    them -- anchor at it, the same as verification does); ``interval`` is the
    raw-score interval ``x_cf`` was solved against; ``if_ir``/``min_total_path``
    come from the explainer's plausibility configuration, or ``(None, 0.0)``
    when none is configured.

    Growth proceeds one joint-grid cell at a time, per non-degenerate
    feature in ascending index order, upper endpoint before lower, each
    accepted only if the box that results still passes the soundness oracle;
    a feature closes once both directions fail in the same round, and the
    whole loop stops once a full round accepts nothing. No step reads from
    set or dict iteration order, so the result is bit-deterministic.
    """
    p = len(x_cf)
    lo_b, hi_b, frozen = compiled.instance_bounds(x)
    lo_b = np.where(np.isnan(lo_b), -math.inf, lo_b)
    hi_b = np.where(np.isnan(hi_b), math.inf, hi_b)

    degenerate = _degenerate_features(compiled, frozen, lo_b, hi_b, x_cf)
    grids = (
        _constraint_cells(compiled, ir)
        if if_ir is None
        else _constraint_cells(compiled, ir, if_ir)
    )

    box_lo = x_cf.astype(np.float64).copy()
    box_hi = x_cf.astype(np.float64).copy()
    is_nan_arr: BoolArray = np.isnan(x_cf)

    def oracle() -> bool:
        return _box_feasible(
            ir, if_ir, min_total_path, interval, compiled.linears, x_cf,
            box_lo, box_hi, is_nan_arr,
        )

    open_set = {j for j in range(p) if j not in degenerate}
    while open_set:
        still_open: set[int] = set()
        for j in sorted(open_set):
            cells = grids[j]
            lo_bj, hi_bj = float(lo_b[j]), float(hi_b[j])
            grew_up = _try_grow(j, True, box_lo, box_hi, cells, lo_bj, hi_bj, oracle)
            grew_down = _try_grow(j, False, box_lo, box_hi, cells, lo_bj, hi_bj, oracle)
            if grew_up or grew_down:
                still_open.add(j)
        open_set = still_open

    feature_intervals = {
        compiled.feature_names[j]: (float(box_lo[j]), float(box_hi[j]))
        for j in range(p)
        if j not in degenerate
    }
    return RecourseRegion(
        lo=box_lo, hi=box_hi, feature_intervals=feature_intervals, certified=True
    )
