"""Constrained genetic backend (spec §8.2) — numpy only, feasibility-first (Deb ranking).

Works in float space: scores via the vectorized IR evaluator, constraints via the
compiler's vectorized check/repair pair (§7.3). Never claims optimality; the API
labels its results ``proof="heuristic"``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from treecf.aim.cells import feature_cells
from treecf.constraints.compile import CompiledConstraints
from treecf.ir.evaluate import raw_score_batch
from treecf.ir.model import EnsembleIR

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class GeneticResult:
    x_cf: FloatArray | None
    stats: dict[str, object]


def solve_genetic(
    ir: EnsembleIR,
    x: FloatArray,
    interval: tuple[float, float],
    compiled: CompiledConstraints,
    sigma: FloatArray,
    weights: FloatArray,
    lam: float,
    background: FloatArray | None = None,
    plausibility: tuple[EnsembleIR, float] | None = None,
    seed: int | None = None,
    population: int = 80,
    max_generations: int = 200,
    stall_generations: int = 30,
    time_budget_s: float = 10.0,
) -> GeneticResult:
    rng = np.random.default_rng(seed)
    p = ir.n_features
    lo_t, hi_t = interval

    _lo_b, _hi_b, frozen = compiled.instance_bounds(x)
    fixed = frozen | (np.isnan(x) & ~np.isin(np.arange(p), list(compiled.allow_missing)))
    mutable = ~fixed
    can_be_nan = np.zeros(p, dtype=bool)
    for j in compiled.allow_missing:
        if not fixed[j]:
            can_be_nan[j] = True

    # Per-feature candidate pool: nearest point of every cell to the factual value.
    cells = feature_cells(ir)
    anchor = np.where(np.isnan(x), 0.0, x)
    pools = [
        np.array([c.nearest_to(anchor[j]) for c in cells[j]]) if mutable[j] else np.empty(0)
        for j in range(p)
    ]

    def objective(X: FloatArray) -> FloatArray:
        total = np.zeros(len(X))
        for j in range(p):
            x_nan = math.isnan(x[j])
            col = X[:, j]
            col_nan = np.isnan(col)
            if j in compiled.allow_missing:
                to_miss, from_miss = compiled.allow_missing[j]
            else:
                to_miss = from_miss = 0.0
            if x_nan:
                changed = ~col_nan
                total += changed * (weights[j] * from_miss / sigma[j] + lam)
            else:
                went_nan = col_nan
                moved = ~col_nan & (col != x[j])
                delta = np.where(moved, np.abs(np.nan_to_num(col) - x[j]), 0.0)
                total += went_nan * (weights[j] * to_miss / sigma[j] + lam)
                total += moved * lam + weights[j] * delta / sigma[j]
        return total

    def rank_keys(X: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Deb ranking: (tier, key). Tier 0 = fully feasible (key = J)."""
        scores = raw_score_batch(ir, X)
        ok = compiled.check_matrix(X, x)
        if plausibility is not None:
            if_ir, min_total_path = plausibility
            ok &= raw_score_batch(if_ir, X) >= min_total_path
        target_ok = (scores >= lo_t) & (scores <= hi_t)
        tier = np.where(ok & target_ok, 0.0, np.where(ok, 1.0, 2.0))
        target_gap = np.maximum(0.0, lo_t - scores) + np.maximum(0.0, scores - hi_t)
        key = np.where(tier == 0.0, objective(X), np.nan_to_num(target_gap, posinf=1e18))
        return tier, key

    def make_population(n: int) -> FloatArray:
        X = np.tile(x, (n, 1))
        for i in range(n):
            k = int(rng.integers(1, max(2, int(mutable.sum()) + 1)))
            picks = rng.choice(
                np.flatnonzero(mutable), size=min(k, int(mutable.sum())), replace=False
            )
            for pick in picks:
                jj = int(pick)
                X[i, jj] = _mutate_value(rng, x[jj], pools[jj], sigma[jj], can_be_nan[jj])
        return X

    # Initialization: single-feature cell moves + random multi-feature perturbations
    seeds: list[FloatArray] = [x.copy()]
    for pick in np.flatnonzero(mutable):
        jj = int(pick)
        for value in pools[jj]:
            row = x.copy()
            row[jj] = value
            seeds.append(row)
        if can_be_nan[jj]:
            row = x.copy()
            row[jj] = math.nan
            seeds.append(row)
    if background is not None and len(background) > 0:
        idx = rng.integers(0, len(background), size=min(20, len(background)))
        for row_bg in background[idx]:
            row = x.copy()
            mask = mutable & (rng.random(p) < 0.5)
            row[mask] = row_bg[mask]
            seeds.append(row)
    pop = np.vstack([np.vstack(seeds), make_population(max(population - len(seeds), 10))])
    pop = compiled.repair_matrix(pop, x)
    pop[:, fixed] = x[fixed]

    best: FloatArray | None = None
    best_j = math.inf
    stall = 0
    start = time.monotonic()
    generations = 0

    for generation in range(max_generations):
        generations = generation + 1
        tier, key = rank_keys(pop)
        order = np.lexsort((key, tier))
        pop = pop[order]
        tier, key = tier[order], key[order]

        if tier[0] == 0.0 and key[0] < best_j - 1e-12:
            best, best_j = pop[0].copy(), float(key[0])
            stall = 0
        else:
            stall += 1
        if stall >= stall_generations or time.monotonic() - start > time_budget_s:
            break

        elite = pop[: max(4, population // 8)]
        children: list[FloatArray] = []
        while len(children) < population - len(elite):
            a, b = pop[rng.integers(0, max(2, len(pop) // 2), size=2)]
            mask = rng.random(p) < 0.5
            child = np.where(mask, a, b)
            for pick in np.flatnonzero(mutable):
                jj = int(pick)
                roll = rng.random()
                if roll < 0.15:
                    child[jj] = _mutate_value(rng, child[jj], pools[jj], sigma[jj], can_be_nan[jj])
                elif roll < 0.30:
                    child[jj] = x[jj]  # revert-to-factual: drives sparsity (L0)
            children.append(child)
        pop = np.vstack([elite, np.vstack(children)])
        pop = compiled.repair_matrix(pop, x)
        pop[:, fixed] = x[fixed]

    stats: dict[str, object] = {
        "generations": generations,
        "wall_time_s": time.monotonic() - start,
        "population": population,
    }
    return GeneticResult(x_cf=best, stats=stats)


def _mutate_value(
    rng: np.random.Generator,
    current: float,
    pool: FloatArray,
    sigma_j: float,
    nan_allowed: bool,
) -> float:
    roll = rng.random()
    if nan_allowed and roll < 0.15:
        return math.nan
    if len(pool) > 0 and roll < 0.6:
        return float(pool[rng.integers(0, len(pool))])
    base = 0.0 if math.isnan(current) else current
    return float(base + rng.normal(scale=max(sigma_j, 1e-9)))
