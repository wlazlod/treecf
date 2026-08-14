"""Brute-force counterfactual oracle — ground truth for every backend.

Enumerates all cell combinations (guarded to <= MAX_COMBOS), places each feature at
the point of cell ∩ constraint-bounds nearest to the factual value, and minimizes
J = sum_j w_j |x'_j - x_j| / sigma_j + lambda * #changed  subject to the target interval.

A completed enumeration that finds no feasible combination is a certificate, not a
guess: every reachable candidate was tried and rejected, so ``OracleResult(feasible=False,
objective=inf, x_cf=None)`` is exhaustive proof of infeasibility over the grid. Later
exact-search tasks compare their own certified ``Infeasible`` verdicts against this.

Optional ``plausibility`` widens the per-feature grid to the joint cells of the model
and an isolation-forest ensemble (mirrors ``Explainer.plausibility``) and filters
candidates by the forest's path-length bound. Optional ``value_policies`` snaps
per-feature candidates through the same ``_snap`` the API uses, dropping any cell
whose snapped representative falls outside the intersected cell/bounds interval.

The grid comes from ``_constraint_cells``, not from ``feature_cells`` directly, so
that a value some constraint watches for is a cell of its own and its neighbours
are reachable separately. One point per routing cell is enough to know a row's
score but not whether it is allowed, and the exact backend has to be compared
against an oracle that searches the same space, not a coarser one.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from treecf.aim.cells import Cell
from treecf.api import _snap
from treecf.backends._exact_domains import _constraint_cells
from treecf.constraints.compile import CompiledConstraints
from treecf.ir.evaluate import raw_score
from treecf.ir.model import EnsembleIR

FloatArray = npt.NDArray[np.float64]

MAX_COMBOS = 200_000


@dataclass(frozen=True)
class OracleResult:
    feasible: bool
    objective: float
    x_cf: FloatArray | None


def solve_brute_force(
    ir: EnsembleIR,
    x: FloatArray,
    interval: tuple[float, float],
    compiled: CompiledConstraints,
    sigma: FloatArray,
    weights: FloatArray,
    lam: float = 0.0,
    *,
    plausibility: tuple[EnsembleIR, float] | None = None,
    value_policies: Mapping[str, object] | None = None,
) -> OracleResult:
    """``value_policies`` snaps only candidates that move a feature; the unchanged
    factual (the "keep" option) is always available and never snapped, mirroring
    the API's ``_apply_value_policies``, which snaps only changed features."""
    lo_b, hi_b, frozen = compiled.instance_bounds(x)
    lo_b = np.where(np.isnan(lo_b), -math.inf, lo_b)  # Monotone on a NaN factual: no bound
    hi_b = np.where(np.isnan(hi_b), math.inf, hi_b)
    if_ir, min_total_path = plausibility if plausibility is not None else (None, None)
    per_feature = (
        _constraint_cells(compiled, ir, if_ir)
        if if_ir is not None
        else _constraint_cells(compiled, ir)
    )
    p = ir.n_features

    # Candidate values per feature: nearest-in-(cell ∩ bounds) to x_j, plus the NaN
    # state when AllowMissing permits it; NaN factuals without AllowMissing stay NaN.
    # A value_policies entry snaps each non-NaN candidate; a cell whose snapped
    # representative leaves the cell ∩ bounds interval loses that option.
    options: list[list[float]] = []
    for j in range(p):
        allow = j in compiled.allow_missing and not frozen[j]
        if math.isnan(x[j]) and not allow:
            options.append([math.nan])
            continue
        values: list[float] = [math.nan] if allow else []
        anchor = 0.0 if math.isnan(x[j]) else x[j]
        name = ir.feature_names[j]
        policy = value_policies.get(name) if value_policies is not None else None
        if policy == "raw":
            policy = None
        lo_j, hi_j = float(lo_b[j]), float(hi_b[j])
        # The unchanged factual is always a legal, unsnapped option -- value_policies
        # only governs candidates that actually move the feature.
        keep = not math.isnan(x[j]) and lo_j <= x[j] <= hi_j
        if keep:
            values.append(x[j])
        for cell in per_feature[j]:
            v = _nearest_in_cell_and_bounds(cell, anchor, lo_b[j], hi_b[j])
            if v is None:
                continue
            if keep and v == x[j]:
                continue  # already added, exempt, above
            if policy is not None:
                v = _snap(v, policy, _cell_and_bounds(cell, lo_j, hi_j), lo_j, hi_j)
                if v is None:
                    continue
            values.append(v)
        values = _dedup_preserve_order(values)
        if not values:
            return OracleResult(feasible=False, objective=math.inf, x_cf=None)
        options.append(values)

    n_combos = math.prod(len(v) for v in options)
    if n_combos > MAX_COMBOS:
        raise ValueError(f"{n_combos} combos exceed oracle guard {MAX_COMBOS}")

    lo_t, hi_t = interval
    best = OracleResult(feasible=False, objective=math.inf, x_cf=None)
    candidate = np.empty(p, dtype=np.float64)
    for combo in itertools.product(*options):
        candidate[:] = combo
        score = raw_score(ir, candidate)
        if not (lo_t <= score <= hi_t):
            continue
        if if_ir is not None and raw_score(if_ir, candidate) < min_total_path:
            continue
        if not _relational_ok(candidate, compiled):
            continue
        objective = _objective(candidate, x, sigma, weights, lam, compiled.allow_missing)
        if objective < best.objective:
            best = OracleResult(feasible=True, objective=objective, x_cf=candidate.copy())
    return best


def _cell_and_bounds(cell: Cell, lo: float, hi: float) -> Callable[[float], bool]:
    return lambda c: cell.contains(c) and lo <= c <= hi


def _dedup_preserve_order(values: list[float]) -> list[float]:
    """Drop repeat non-NaN values (distinct cells can snap to the same float under
    a policy), keeping first-occurrence order; NaN is never deduplicated against
    itself since it can only ever appear once, as the leading AllowMissing entry."""
    seen: set[float] = set()
    deduped: list[float] = []
    for v in values:
        if math.isnan(v):
            deduped.append(v)
            continue
        if v in seen:
            continue
        seen.add(v)
        deduped.append(v)
    return deduped


def _relational_ok(candidate: FloatArray, compiled: CompiledConstraints) -> bool:
    for lin in compiled.linears:
        values = [candidate[j] for j in lin.indices]
        if any(math.isnan(v) for v in values):
            if lin.missing_policy == "satisfied":
                continue
            return False
        total = sum(c * v for c, v in zip(lin.coefs, values, strict=True))
        ok = (
            total <= lin.rhs + 1e-9
            if lin.op == "<="
            else total >= lin.rhs - 1e-9
            if lin.op == ">="
            else abs(total - lin.rhs) <= 1e-9
        )
        if not ok:
            return False
    for imp in compiled.implications:
        if (
            candidate[imp.cond_index] == imp.cond_value
            and candidate[imp.cons_index] != imp.cons_value
        ):
            return False
    return all(sum(candidate[j] for j in group) == 1.0 for group in compiled.onehot_groups)


def _nearest_in_cell_and_bounds(cell: Cell, x_j: float, lo: float, hi: float) -> float | None:
    """Nearest-to-x point of cell ∩ [lo, hi], or None if the intersection is empty."""
    v = cell.nearest_to(min(max(x_j, lo), hi))
    if lo <= v <= hi:
        return v
    return None


def _objective(
    candidate: FloatArray,
    x: FloatArray,
    sigma: FloatArray,
    weights: FloatArray,
    lam: float,
    allow_missing: dict[int, tuple[float, float]],
) -> float:
    total = 0.0
    for j in range(len(x)):
        x_nan, cf_nan = math.isnan(x[j]), math.isnan(candidate[j])
        if x_nan and cf_nan:
            continue
        if cf_nan:  # value -> NaN
            total += weights[j] * allow_missing[j][0] / sigma[j] + lam
        elif x_nan:  # NaN -> value
            total += weights[j] * allow_missing[j][1] / sigma[j] + lam
        else:
            delta = abs(candidate[j] - x[j])
            if delta > 0:
                total += weights[j] * delta / sigma[j] + lam
    return total
