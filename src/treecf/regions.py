"""Certified recourse regions: widen a verified counterfactual into a sound box.

A ``RecourseRegion`` is a per-feature interval around one already-verified
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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from treecf.aim.cells import Cell, category_blocks, cell_index
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
    forced to stop early frees room a later feature grows into). See
    [Certification](concepts/certification.md#regions-certified-not-maximal-not-monotone).

    Attributes
    ----------
    lo
        Lower bound per feature, same order as the model's features;
        equal to ``hi`` at a degenerate (never-widened) coordinate.
    hi
        Upper bound per feature, same order as the model's features.
    feature_intervals
        ``{feature: (lo, hi)}`` for every non-degenerate
        feature only, for display (``describe()`` renders these as
        phrases).
    certified
        Always ``True`` in this release — every region returned
        by ``Explainer.recourse_region``/``explain(..., region=True)`` is
        a sound certificate; the field is reserved for a future relaxed
        mode.
    """

    lo: FloatArray
    hi: FloatArray
    feature_intervals: dict[str, tuple[float, float]]
    certified: bool  # always True in this release; the field is reserved
    # certified category sets for categorical features: codes by feature name
    # (public), the same sets keyed by feature index (what ``contains`` reads),
    # and display names for the codes where the model carries them
    feature_categories: dict[str, tuple[int, ...]] = field(default_factory=dict)
    cat_sets: dict[int, tuple[int, ...]] = field(default_factory=dict)
    category_names: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def contains(self, x: FloatArray) -> bool:
        """Whether ``x`` lies inside the region, coordinate by coordinate.

        A degenerate coordinate (``lo == hi``, including NaN) requires ``x``
        to match it exactly; every other coordinate requires
        ``lo <= x[j] <= hi[j]``.

        Parameters
        ----------
        x
            A feature vector, same order and length as the region.

        Returns
        -------
        ``True`` iff every coordinate of ``x`` satisfies the region's
        bound.
        """
        for j in range(len(self.lo)):
            xj = float(x[j])
            if math.isnan(self.lo[j]):
                if not math.isnan(xj):
                    return False
                continue
            if j in self.cat_sets:
                if math.isnan(xj) or xj != int(xj) or int(xj) not in self.cat_sets[j]:
                    return False
                continue
            if math.isnan(xj) or not (self.lo[j] <= xj <= self.hi[j]):
                return False
        return True

    def describe(self) -> dict[str, str]:
        """One human-readable phrase per non-degenerate feature.

        One-sided (``"<= v"``/``">= v"``) when the other endpoint is
        infinite, two-sided (``"in [lo, hi]"``) otherwise, and
        ``"unconstrained"`` when both endpoints are infinite; values
        formatted ``"{:.3g}"``.

        Returns
        -------
        ``{feature: phrase}`` for every key of ``feature_intervals``.
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
        for name, codes in self.feature_categories.items():
            names = self.category_names.get(name)
            rendered = (
                ", ".join(names[c] for c in codes)
                if names is not None
                else ", ".join(str(c) for c in codes)
            )
            out[name] = f"∈ {{{rendered}}}"
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
    nodes: tuple[Node, ...],
    idx: int,
    lo: FloatArray,
    hi: FloatArray,
    is_nan: BoolArray,
    cat_sets: Mapping[int, set[int]] | None = None,
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
    assert node.left is not None and node.right is not None
    if is_nan[f]:
        if node.missing_left is None:
            return None
        child = node.left if node.missing_left else node.right
        return _tree_interval_bracket(nodes, child, lo, hi, is_nan, cat_sets)
    members: tuple[int, ...] | None = None
    if cat_sets is not None and f in cat_sets:
        members = tuple(sorted(cat_sets[f]))
    elif node.categories is not None:
        # a set split on a coordinate no set tracks: the box pins it to one code
        members = (int(lo[f]),)
    if members is not None:
        # a categorical coordinate holds a SET of codes, not an interval: route
        # each member and take a side only when every member agrees
        left_any = False
        right_any = False
        for code in members:
            if node.categories is not None:
                goes_left = code in node.categories
            elif node.op is SplitOp.LT:
                goes_left = code < node.threshold  # type: ignore[operator]
            else:
                goes_left = code <= node.threshold  # type: ignore[operator]
            if goes_left:
                left_any = True
            else:
                right_any = True
        all_left, all_right = not right_any, not left_any
    else:
        assert node.threshold is not None
        threshold = node.threshold
        lo_f, hi_f = float(lo[f]), float(hi[f])
        if node.op is SplitOp.LT:
            all_left, all_right = hi_f < threshold, lo_f >= threshold
        else:
            all_left, all_right = hi_f <= threshold, lo_f > threshold
    if all_left:
        return _tree_interval_bracket(nodes, node.left, lo, hi, is_nan, cat_sets)
    if all_right:
        return _tree_interval_bracket(nodes, node.right, lo, hi, is_nan, cat_sets)
    left = _tree_interval_bracket(nodes, node.left, lo, hi, is_nan, cat_sets)
    if left is None:
        return None
    right = _tree_interval_bracket(nodes, node.right, lo, hi, is_nan, cat_sets)
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
    cache: dict[str, object] | None = None,
) -> RecourseRegion:
    """Grow a certified box around the verified counterfactual ``x_cf``.

    ``x`` is the original factual (instance bounds -- Freeze/Monotone among
    them -- anchor at it, the same as verification does); ``interval`` is the
    raw-score interval ``x_cf`` was solved against; ``if_ir``/``min_total_path``
    come from the explainer's plausibility configuration, or ``(None, 0.0)``
    when none is configured. ``cache`` is the Explainer's marshaled-Rust-object
    cache, when one is available (see ``regions_rust.compute_region_rust``).

    Growth proceeds one joint-grid cell at a time, per non-degenerate
    feature in ascending index order, upper endpoint before lower, each
    accepted only if the box that results still passes the soundness oracle;
    a feature closes once both directions fail in the same round, and the
    whole loop stops once a full round accepts nothing. No step reads from
    set or dict iteration order, so the result is bit-deterministic.

    Dispatches rust-first: when the ``_treecf_core`` extension is importable,
    the growth loop runs in ``regions_rust.compute_region_rust`` instead of
    the pure-Python ``_grow_box`` below. The rust engine is a bit-parity
    mirror, not a heuristic stand-in -- every fixture under
    ``tests/fixtures/regions/`` proves the two produce byte-identical ``lo``/
    ``hi`` arrays, so the fallback only ever changes which engine ran, never
    what it found.
    """
    from treecf.backends.regions_rust import _rust_available, compute_region_rust

    lo_b, hi_b, frozen = compiled.instance_bounds(x)
    lo_b = np.where(np.isnan(lo_b), -math.inf, lo_b)
    hi_b = np.where(np.isnan(hi_b), math.inf, hi_b)
    degenerate = _degenerate_features(compiled, frozen, lo_b, hi_b, x_cf)
    # categorical coordinates are always pinned for the numeric machinery
    # (lo = hi = the counterfactual's code); their growth is a separate
    # channel over category blocks
    degenerate = degenerate | frozenset(ir.categorical)
    cat_candidates = _categorical_candidates(ir, if_ir, compiled, frozen, x_cf)

    if _rust_available():
        box_lo, box_hi, grown_sets = compute_region_rust(
            ir, x_cf, interval, compiled, lo_b, hi_b, degenerate, if_ir, min_total_path,
            cat_candidates, cache=cache,
        )
    else:
        box_lo, box_hi, grown_sets = _grow_box(
            ir, x_cf, interval, compiled, if_ir, min_total_path, degenerate, lo_b, hi_b,
            cat_candidates,
        )

    feature_intervals = {
        compiled.feature_names[j]: (float(box_lo[j]), float(box_hi[j]))
        for j in range(len(x_cf))
        if j not in degenerate
    }
    cat_sets = {j: tuple(sorted(members)) for j, members in sorted(grown_sets.items())}
    feature_categories = {
        compiled.feature_names[j]: codes for j, codes in cat_sets.items()
    }
    category_names = {
        compiled.feature_names[j]: ir.categorical[j].categories
        for j in cat_sets
        if ir.categorical[j].categories is not None
    }
    return RecourseRegion(
        lo=box_lo,
        hi=box_hi,
        feature_intervals=feature_intervals,
        certified=True,
        feature_categories=feature_categories,
        cat_sets=cat_sets,
        category_names=category_names,  # type: ignore[arg-type]
    )


def _categorical_candidates(
    ir: EnsembleIR,
    if_ir: EnsembleIR | None,
    compiled: CompiledConstraints,
    frozen: BoolArray,
    x_cf: FloatArray,
) -> dict[int, list[tuple[int, ...]]]:
    """Per growable categorical feature, its blocks' admissible members.

    Growth adds one block's admissible members at a time; a frozen feature, a
    NaN counterfactual coordinate, or a feature whose allowed set admits no
    code outside the counterfactual's own has nothing to grow. Block order is
    the canonical ascending-smallest-member order.
    """
    if not ir.categorical:
        return {}
    blocks = category_blocks(ir) if if_ir is None else category_blocks(ir, if_ir)
    candidates: dict[int, list[tuple[int, ...]]] = {}
    for j in sorted(ir.categorical):
        if frozen[j] or math.isnan(x_cf[j]):
            continue
        allowed = compiled.allowed_categories.get(j)
        per_block: list[tuple[int, ...]] = []
        for block in blocks[j]:
            members = tuple(
                c for c in block if allowed is None or c in allowed
            )
            if members:
                per_block.append(members)
        if per_block:
            candidates[j] = per_block
    return candidates


def _grow_box(
    ir: EnsembleIR,
    x_cf: FloatArray,
    interval: tuple[float, float],
    compiled: CompiledConstraints,
    if_ir: EnsembleIR | None,
    min_total_path: float,
    degenerate: frozenset[int],
    lo_b: FloatArray,
    hi_b: FloatArray,
    cat_candidates: dict[int, list[tuple[int, ...]]] | None = None,
) -> tuple[FloatArray, FloatArray, dict[int, set[int]]]:
    """Pure-Python growth loop -- the reference ``_recourse_region`` falls
    back to when the Rust extension is unavailable, and the fixture-golden
    freeze in ``tests/exactness/test_exact_golden.py`` pins directly."""
    p = len(x_cf)
    grids = (
        _constraint_cells(compiled, ir)
        if if_ir is None
        else _constraint_cells(compiled, ir, if_ir)
    )

    box_lo = x_cf.astype(np.float64).copy()
    box_hi = x_cf.astype(np.float64).copy()
    is_nan_arr: BoolArray = np.isnan(x_cf)
    cat_candidates = cat_candidates or {}
    cat_sets: dict[int, set[int]] = {j: {int(x_cf[j])} for j in sorted(cat_candidates)}

    # Per-tree brackets are cached between growth attempts: widening one
    # feature can only change the brackets of trees that split on it, so only
    # those are re-walked. The ensemble total is still re-summed in full over
    # every tree in ascending index — the same additions `_ensemble_bracket`
    # performs — so the accept/reject decisions are identical, just cheaper.
    model_cache = [
        _tree_interval_bracket(tree.nodes, 0, box_lo, box_hi, is_nan_arr, cat_sets)
        for tree in ir.trees
    ]
    if_cache = (
        []
        if if_ir is None
        else [
            _tree_interval_bracket(tree.nodes, 0, box_lo, box_hi, is_nan_arr, cat_sets)
            for tree in if_ir.trees
        ]
    )
    model_on, if_on = _trees_on_feature(ir, p), (
        [] if if_ir is None else _trees_on_feature(if_ir, p)
    )

    def total(
        base: float, cache: list[tuple[float, float] | None]
    ) -> tuple[float, float] | None:
        total_min = base
        total_max = base
        for bracket in cache:
            if bracket is None:
                return None
            total_min = total_min + bracket[0]
            total_max = total_max + bracket[1]
        return total_min, total_max

    def make_oracle(j: int) -> Callable[[], bool]:
        def oracle() -> bool:
            saved = [(t, model_cache[t]) for t in model_on[j]]
            for t in model_on[j]:
                model_cache[t] = _tree_interval_bracket(
                    ir.trees[t].nodes, 0, box_lo, box_hi, is_nan_arr, cat_sets
                )
            saved_if: list[tuple[int, tuple[float, float] | None]] = []
            if if_ir is not None:
                saved_if = [(t, if_cache[t]) for t in if_on[j]]
                for t in if_on[j]:
                    if_cache[t] = _tree_interval_bracket(
                        if_ir.trees[t].nodes, 0, box_lo, box_hi, is_nan_arr, cat_sets
                    )

            def retract() -> None:
                for t, bracket in saved:
                    model_cache[t] = bracket
                for t, bracket in saved_if:
                    if_cache[t] = bracket

            bracket = total(ir.base_score, model_cache)
            if bracket is None or bracket[0] < interval[0] or bracket[1] > interval[1]:
                retract()
                return False
            if if_ir is not None:
                if_bracket = total(if_ir.base_score, if_cache)
                if if_bracket is None or if_bracket[0] < min_total_path:
                    retract()
                    return False
            if not all(
                _linear_holds(lin, x_cf, box_lo, box_hi) for lin in compiled.linears
            ):
                retract()
                return False
            return True

        return oracle

    open_set = {j for j in range(p) if j not in degenerate}
    open_cats = set(cat_candidates)
    while open_set or open_cats:
        still_open: set[int] = set()
        for j in sorted(open_set):
            cells = grids[j]
            lo_bj, hi_bj = float(lo_b[j]), float(hi_b[j])
            oracle_j = make_oracle(j)
            grew_up = _try_grow(j, True, box_lo, box_hi, cells, lo_bj, hi_bj, oracle_j)
            grew_down = _try_grow(j, False, box_lo, box_hi, cells, lo_bj, hi_bj, oracle_j)
            if grew_up or grew_down:
                still_open.add(j)
        open_set = still_open
        # after the numeric pass: one block at a time per categorical feature,
        # ascending feature index, ascending block order
        still_open_cats: set[int] = set()
        for j in sorted(open_cats):
            oracle_j = make_oracle(j)
            grew_cat = False
            for members in cat_candidates[j]:
                new_members = [c for c in members if c not in cat_sets[j]]
                if not new_members:
                    continue
                cat_sets[j].update(new_members)
                if oracle_j():
                    grew_cat = True
                else:
                    cat_sets[j].difference_update(new_members)
            if grew_cat:
                still_open_cats.add(j)
        open_cats = still_open_cats

    return box_lo, box_hi, cat_sets


def _trees_on_feature(ir: EnsembleIR, n_features: int) -> list[list[int]]:
    """Per feature, the ascending tree indices that split on it."""
    on_feature: list[list[int]] = [[] for _ in range(n_features)]
    for t, tree in enumerate(ir.trees):
        features = sorted(
            {node.feature for node in tree.nodes if node.feature is not None}
        )
        for f in features:
            on_feature[f].append(t)
    return on_feature
