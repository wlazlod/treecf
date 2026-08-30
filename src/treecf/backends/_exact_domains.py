"""The branching alphabet of the exact backend, and what may enter it.

Split out of ``treecf.backends.exact`` for size only: ``exact``, ``_exact_bounds``,
this file, ``_exact_orderpairs`` and ``_exact_propagation`` are one implementation,
and the Rust mirror has to match all five bit-for-bit. That makes the operation
order in the cost arithmetic here a compatibility contract rather than a style
choice — every multiply/divide/add mirrors ``treecf.backends.genetic``'s
``objective()`` term-for-term.

``_build_domains`` turns a factual, the joint cell grid and the compiled
constraints into a per-feature list of candidate counterfactual states, already
in the cost order the search wants them in; the rest settles which features the
search has to decide at all, in which order, and what the cheapest remainder
below a level can be.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from treecf._errors import ConstraintValidationError
from treecf.aim.cells import Cell, cell_index, feature_cells
from treecf.api import ValuePolicy, _snap
from treecf.backends._exact_orderpairs import _achievable_bounds, _intersect_cell
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
from treecf.ir.model import EnsembleIR

FloatArray = npt.NDArray[np.float64]

_SUPPRESSING_MISSING_POLICIES = ("forbid_missing", "violated")


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
    is_cat: bool = False,
) -> float:
    """One feature's contribution to the objective, mirroring the per-feature
    term of ``genetic.objective()`` exactly — same four cases, same
    multiply-then-divide order, so results stay bit-identical across backends.
    A categorical change (``is_cat``) costs one flat unit in place of the
    absolute code distance; NaN transitions keep their declared deltas.
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
    delta = 1.0 if is_cat else abs(r - x_j)
    return lam + (weight_j * delta) / sigma_j


def _cost_of_row(
    x: FloatArray,
    row: FloatArray,
    sigma: FloatArray,
    weights: FloatArray,
    lam: float,
    allow_missing: Mapping[int, tuple[float, float]],
    categorical: frozenset[int] = frozenset(),
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
            float(x[j]),
            float(row[j]),
            float(weights[j]),
            float(sigma[j]),
            lam,
            to_miss,
            from_miss,
            j in categorical,
        )
    return total


def _split_cell_at(cells: tuple[Cell, ...], value: float) -> tuple[Cell, ...]:
    """``cells`` with the one holding ``value`` cut into the part below it, the
    single point itself, and the part above — the same three pieces
    ``build_cells`` emits where an LT and an LE split share a threshold. The
    outer edges keep the openness they had; the two new inner edges are open,
    so the point belongs to the middle piece alone. A cell that is already that
    single point is left as it is, which makes repeated splitting harmless."""
    out: list[Cell] = []
    for cell in cells:
        if not cell.contains(value) or (cell.lo == value and cell.hi == value):
            out.append(cell)
            continue
        if cell.lo < value:
            out.append(Cell(cell.lo, value, cell.lo_open, True))
        out.append(Cell(value, value, False, False))
        if value < cell.hi:
            out.append(Cell(value, cell.hi, True, cell.hi_open))
    return tuple(out)


def _constraint_cells(
    compiled: CompiledConstraints, *irs: EnsembleIR
) -> tuple[tuple[Cell, ...], ...]:
    """The routing grid, cut finer wherever a constraint can tell two points of
    one cell apart.

    ``feature_cells`` answers a question about the trees: inside one of its
    cells every tree routes a row the same way, so one point per cell is enough
    to know the score. That is not enough to know whether a row is *allowed*.
    An implication fires on a feature holding one exact value and says nothing
    about any other, so within a single routing cell the constraint can hold at
    one point and be silent a hair away — and the search, seeing one point per
    cell, would never find the hair. So the value each implication watches for
    becomes a cell boundary of its own here: the cell holding it is cut into
    the part below, the point itself, and the part above, and the neighbours
    then offer their own nearest reachable points.

    This refines constraint geometry on top of the routing grid; it does not
    correct it. ``aim.cells`` stays as it is, because it is right about what it
    claims. Both the exact backend and the brute-force oracle build their
    candidates from this function, so the two search the same space.
    """
    grids = feature_cells(*irs)
    triggers: dict[int, set[float]] = {}
    for imp in compiled.implications:
        triggers.setdefault(imp.cond_index, set()).add(imp.cond_value)
    if not triggers:
        return grids
    refined = list(grids)
    for feature, values in triggers.items():
        for value in sorted(values):
            refined[feature] = _split_cell_at(refined[feature], value)
    return tuple(refined)


def _demanded_values(compiled: CompiledConstraints) -> dict[int, list[float]]:
    """Per feature, the exact values some constraint can come to demand of it.

    Two constraint kinds do that. An implication's consequence is left with
    one legal value and nothing else once its condition holds. A
    single-feature ``Linear(op="==")`` is even more direct: it demands its own
    algebraic solution ``rhs / coef`` of that one feature outright. Neither
    demand is something a cell's own nearest-to-factual point can be trusted
    to land on — the ``==`` case especially not: ``compile_constraints``
    widens that constraint's derived range by the ``check_matrix`` slack (see
    ``compile.py``) so the bound never *excludes* a candidate the slacked
    check would admit, but that same widening means the range's own edges,
    which is what a plain nearest-point search would reach for, can sit a
    float-ulp *outside* what the check actually accepts. Demanding the exact
    solution sidesteps the edge entirely.

    The candidate states a feature is given, the values an order-pair repair
    may propose for it, and the oracle's own candidate options (`brute_force
    .solve_brute_force`) are all built from this, so all three agree on what
    is worth offering. Ascending per feature, so the order in which those
    values get tried is fixed.
    """
    demanded: dict[int, set[float]] = {}
    for imp in compiled.implications:
        demanded.setdefault(imp.cons_index, set()).add(imp.cons_value)
    for lin in compiled.linears:
        if len(lin.indices) == 1 and lin.op == "==":
            demanded.setdefault(lin.indices[0], set()).add(lin.rhs / lin.coefs[0])
    return {f: sorted(values) for f, values in demanded.items()}


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
    re-check either. ``grids`` must be the grid ``_constraint_cells`` builds,
    not ``feature_cells``' own: every state's ``cell_idx`` points into whatever
    grid is passed here, so the caller has to search the same one.

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
    ``AllowMissing`` gets a single keep-NaN state, or none at all when a
    single-feature Linear also forbids missing values there; with
    ``AllowMissing`` it
    additionally offers moving to the pinned value ``v``, priced by
    ``delta_from_miss``; either NaN-involving state is dropped when a
    single-feature Linear's ``missing_policy`` forbids NaN there, and a NaN
    factual that is both forced to stay NaN and forbidden from being NaN
    yields an empty domain for that feature — a certified-infeasible signal
    for the search, not an error.

    Every other feature intersects each grid cell with its bounds first
    (dropping empty intersections, preserving open/closed edges), and each
    surviving cell contributes its nearest point to the factual value as a
    candidate. A value policy snaps every such movement candidate, dropping
    it on failure; the factual's own unchanged value is always available and
    exempt from snapping, so a value policy can never force a feature that
    did not need to move.

    Two constraint kinds change that last rule.

    A one-hot member holds nothing but 0 or 1 — every other value fails the
    group's own sum — so each of its surviving cells contributes whichever of
    0.0/1.0 that cell holds, at that exact value, instead of a nearest point.

    An implication does *not* narrow its features that way. Its condition only
    fires on an exact value, and any other value leaves the implication with
    nothing to say, so the ordinary nearest-point candidates are all legal
    there and are kept. What the consequence side does get is one extra
    candidate: the exact value the implication would demand of it, in whatever
    cell holds that value. Without it a triggered implication would have
    nothing legal left to offer, and rows that meet it would be out of reach.
    Like a pinned value, that candidate comes from a constraint and is exempt
    from value-policy snapping.

    A single-feature ``Linear(op="==")`` gets the same extra candidate, its
    own algebraic solution (``_demanded_values``), for the same reason: the
    derived range this constraint also contributes to ``lo``/``hi`` is
    deliberately widened past the exact solution (never narrowed — it must
    never exclude a candidate the arbiter's own slack admits), so the ordinary
    nearest-point candidate for whichever cell holds that widened edge is not
    guaranteed to still be inside the arbiter's own, unwidened, tolerance.
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
    onehot_members = {f for group in compiled.onehot_groups for f in group}
    demanded = _demanded_values(compiled)

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
            # Nothing lets this feature become a value, so staying missing is
            # the only state it has -- and even that one goes away when a
            # single-feature Linear forbids missing values here, leaving an
            # empty domain, the same certified-infeasible signal the pinned
            # branch above produces for the same contradiction.
            if j in suppress_nan:
                domains.append([])
            else:
                domains.append([_State(x_j, 0.0, len(cells), True)])
            continue

        name = compiled.feature_names[j]
        raw_policy = policies.get(name)
        policy = None if raw_policy is None or raw_policy == "raw" else raw_policy

        anchor = 0.0 if x_nan else x_j
        is_binary = j in onehot_members
        demanded_here = demanded.get(j, [])

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
            added_here: list[float] = []
            for val in demanded_here:
                # a value some implication may demand of this feature: legal
                # wherever it lands, and never snapped, since the constraint
                # that asks for it outranks any value policy
                if not iv.contains(val) or (keep_added and val == x_j):
                    continue
                cost = _term_cost(x_j, val, weight_j, sigma_j, lam, to_miss, from_miss)
                states.append(_State(val, cost, local_idx, False))
                added_here.append(val)
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
            if r in added_here:
                continue  # the demanded value was this cell's nearest point too
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
    the canonical order-pair shape (``a - b <= 0``) are returned as
    ``(a, b)`` index pairs in ascending order, which is the order the search
    repairs them in. Any other multi-feature Linear, and any callable value
    policy, name ``backend="genetic"`` as the fallback.
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
    order_pairs.sort()

    for name, policy in (value_policies or {}).items():
        if callable(policy):
            raise ConstraintValidationError(
                f"callable value_policy for {name!r} is not supported by the exact "
                'backend; use backend="genetic".'
            )

    return order_pairs


def _domain_span(
    states: list[_State], cells: tuple[Cell, ...], lo_j: float, hi_j: float
) -> tuple[float, float] | None:
    """Lowest and highest value the feature can still end up holding.

    The span covers the achievable ends of every cell the feature's states
    come from, not the states' own values: a value inside one of those cells
    is exactly what an order-pair repair may put there later, so a bound built
    on the state values alone would cut off repairs that are still reachable.

    ``None`` when the feature may go missing — a missing value has no place in
    an order comparison, so no bound on it holds at all.
    """
    span_lo = math.inf
    span_hi = -math.inf
    for state in states:
        if state.is_nan:
            return None
        iv = _intersect_cell(cells[state.cell_idx], lo_j, hi_j)
        if iv is None:  # pragma: no cover - a state's own cell always survives
            continue
        cell_lo, cell_hi = _achievable_bounds(iv)
        span_lo = min(span_lo, cell_lo)
        span_hi = max(span_hi, cell_hi)
    if span_lo > span_hi:
        return None
    return span_lo, span_hi
