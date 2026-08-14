"""Cell arithmetic the order-pair rules of the exact backend are built on.

Split out of ``treecf.backends.exact`` for size only: ``exact``, this file and
``_exact_propagation`` are one implementation, and the Rust mirror has to match
all three bit-for-bit.

Everything here answers one of two questions: which values a cell, or a pair of
cells, can really hold, and which of those are worth trying when two features
tied by ``a <= b`` end up the wrong way round.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from treecf.aim.cells import Cell

if TYPE_CHECKING:  # imported for typing only; exact.py imports this module
    from treecf.backends.exact import _State


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


def _intersect_cells(first: Cell, second: Cell) -> Cell | None:
    """``first`` ∩ ``second``, keeping the tighter edge on each side and the
    open edge when both sides sit at the same value. ``None`` if the
    intersection is empty (including a degenerate open singleton)."""
    if first.lo > second.lo:
        lo, lo_open = first.lo, first.lo_open
    elif second.lo > first.lo:
        lo, lo_open = second.lo, second.lo_open
    else:
        lo, lo_open = first.lo, first.lo_open or second.lo_open
    if first.hi < second.hi:
        hi, hi_open = first.hi, first.hi_open
    elif second.hi < first.hi:
        hi, hi_open = second.hi, second.hi_open
    else:
        hi, hi_open = first.hi, first.hi_open or second.hi_open
    if lo > hi:
        return None
    if lo == hi and (lo_open or hi_open):
        return None
    return Cell(lo, hi, lo_open, hi_open)


def _achievable_bounds(cell: Cell) -> tuple[float, float]:
    """Lowest and highest value the cell can actually take.

    A closed edge is its own bound; a finite open edge steps one f32 ulp
    inside, the same step ``nearest_to`` takes, so the endpoints returned here
    are values a counterfactual may really hold. An infinite open edge stays
    infinite — no point of the cell is extreme in that direction.
    """
    lo = cell.lo
    if cell.lo_open and lo != -math.inf:
        lo = cell.nearest_to(cell.lo)
    hi = cell.hi
    if cell.hi_open and hi != math.inf:
        hi = cell.nearest_to(cell.hi)
    return lo, hi


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


def _boundary_candidates(cell_a: Cell, cell_b: Cell, x_a: float, x_b: float) -> list[float]:
    """Values worth trying when a pair ``a <= b`` has to be pulled onto the
    boundary ``a' == b' == t``.

    ``cell_a`` and ``cell_b`` are the two features' cells already intersected
    with their constraint bounds, so ``t`` has to lie in the intersection of
    the two. Summed over the pair the cost is piecewise linear in ``t`` with
    kinks only at the two factual values, so the cheapest ``t`` is one of
    those two or one of the interval's own ends. They are returned in that
    fixed order — factual of ``a``, factual of ``b``, low end, high end —
    which is what makes the choice between equally cheap repairs
    reproducible. Values outside the interval, missing values and infinite
    ends drop out, and a repeat of an earlier candidate is not offered twice.
    An empty list means the pair cannot be repaired inside these cells at all.
    """
    iv = _intersect_cells(cell_a, cell_b)
    if iv is None:
        return []
    ach_lo, ach_hi = _achievable_bounds(iv)
    out: list[float] = []
    for t in (x_a, x_b, ach_lo, ach_hi):
        if not math.isfinite(t) or not iv.contains(t) or t in out:
            continue
        out.append(t)
    return out
