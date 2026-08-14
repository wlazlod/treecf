"""Exact backend foundations — domains, state costs, canonical orders, validation.

This file is the Python reference implementation of the exact backend; a Rust
mirror lands later and must match it bit-for-bit, so operation order in the
cost arithmetic below is a compatibility contract, not a style choice — every
multiply/divide/add mirrors ``treecf.backends.genetic``'s ``objective()``
term-for-term.

No search loop lives here: ``_build_domains`` turns a factual, the joint cell
grid, and the compiled constraints into a per-feature list of candidate
counterfactual states (the branching alphabet a later depth-first search
consumes), already in the cost order that search wants them in.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from treecf._errors import ConstraintValidationError
from treecf.aim.cells import Cell, cell_index
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
    since constraints win over policies. A NaN factual without
    ``AllowMissing`` gets a single keep-NaN state; with ``AllowMissing`` it
    additionally offers moving to the pinned value ``v``, priced by
    ``delta_from_miss``.

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
            domains.append([_State(x_j, 0.0, 0, x_nan)])
            continue

        if pinned:
            v = lo_j
            if not x_nan:
                cost = _term_cost(x_j, v, weight_j, sigma_j, lam, to_miss, from_miss)
                domains.append([_State(v, cost, cell_index(cells, v), False)])
                continue
            nan_states = [_State(math.nan, 0.0, len(cells), True)]
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
