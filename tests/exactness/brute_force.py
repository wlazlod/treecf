"""Brute-force counterfactual oracle — ground truth for every backend.

Enumerates all cell combinations (guarded to <= MAX_COMBOS), places each feature at
the point of cell ∩ constraint-bounds nearest to the factual value, and minimizes
J = sum_j w_j |x'_j - x_j| / sigma_j + lambda * #changed  subject to the target interval.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from treecf.aim.cells import Cell, feature_cells
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
) -> OracleResult:
    lo_b, hi_b, frozen = compiled.instance_bounds(x)
    lo_b = np.where(np.isnan(lo_b), -math.inf, lo_b)  # Monotone on a NaN factual: no bound
    hi_b = np.where(np.isnan(hi_b), math.inf, hi_b)
    per_feature = feature_cells(ir)
    p = ir.n_features

    # Candidate values per feature: nearest-in-(cell ∩ bounds) to x_j, plus the NaN
    # state when AllowMissing permits it; NaN factuals without AllowMissing stay NaN.
    options: list[list[float]] = []
    for j in range(p):
        allow = j in compiled.allow_missing and not frozen[j]
        if math.isnan(x[j]) and not allow:
            options.append([math.nan])
            continue
        values: list[float] = [math.nan] if allow else []
        anchor = 0.0 if math.isnan(x[j]) else x[j]
        for cell in per_feature[j]:
            v = _nearest_in_cell_and_bounds(cell, anchor, lo_b[j], hi_b[j])
            if v is not None:
                values.append(v)
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
        if not _relational_ok(candidate, compiled):
            continue
        objective = _objective(candidate, x, sigma, weights, lam, compiled.allow_missing)
        if objective < best.objective:
            best = OracleResult(feasible=True, objective=objective, x_cf=candidate.copy())
    return best


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
