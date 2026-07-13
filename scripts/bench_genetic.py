"""Benchmark gate: Python (numpy) GA vs Rust GA.

- same seeds per config, infinite time budget (stall/max-gen stopping only)
- only the solve call is timed; marshaling/Explainer setup excluded; 1 warmup each
- backends interleaved per seed on the same process/machine
- reports median/p95 wall time and normalized us-per-individual-evaluation
- gate: Rust >= 2x median speedup on the LARGE scenario at default rayon threads

Run:            uv run python scripts/bench_genetic.py
Single-thread:  RAYON_NUM_THREADS=1 uv run python scripts/bench_genetic.py --headline-only
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from treecf import AllowMissing, Freeze, Monotone, Range, constraint
from treecf.aim.cells import feature_cells
from treecf.backends.genetic import solve_genetic
from treecf.backends.genetic_rust import solve_genetic_rust
from treecf.constraints.compile import compile_constraints
from treecf.ir.evaluate import raw_score
from treecf.ir.parsers import parse_model
from treecf.plausibility import Plausibility

GA = {"population": 80, "max_generations": 200, "stall_generations": 30, "time_budget_s": 1e9}


def build_xgb(n_trees: int, depth: int, p: int, seed: int):
    import xgboost as xgb

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(20000, p)) * rng.uniform(0.1, 100, size=p)
    y = (X @ rng.normal(size=p) + rng.logistic(scale=2.0, size=20000) > 0).astype(float)
    clf = xgb.XGBClassifier(n_estimators=n_trees, max_depth=depth, random_state=0, n_jobs=4)
    clf.fit(X, y)
    return parse_model(clf), X


def constraint_mix(names, X):
    cs = [Freeze(names[i]) for i in range(5)]
    cs += [Monotone(names[5 + i], "increase") for i in range(3)]
    cs += [
        Range(names[8 + i], float(np.nanmin(X[:, 8 + i])), float(np.nanmax(X[:, 8 + i])))
        for i in range(3)
    ]
    cs.append(constraint(f"{names[11]} <= {names[12]}"))
    cs += [AllowMissing(names[13 + i], delta_miss=1.0) for i in range(2)]
    return cs


def gen1_rows(ir, compiled, x, population):
    p = ir.n_features
    _, _, frozen = compiled.instance_bounds(x)
    fixed = frozen | (np.isnan(x) & ~np.isin(np.arange(p), list(compiled.allow_missing)))
    cells = feature_cells(ir)
    n_seeds = 1 + sum(
        len(cells[j]) + (1 if (j in compiled.allow_missing and not fixed[j]) else 0)
        for j in np.flatnonzero(~fixed)
    )
    n_seeds += 20  # background mixes
    return n_seeds + max(population - n_seeds, 10)


def run_config(tag, ir, X, constraints, plaus, population, seeds):
    p = ir.n_features
    names = ir.feature_names
    compiled = compile_constraints(constraints, names)
    sigma = np.maximum(np.nanstd(X, axis=0), 1e-6)
    weights = np.ones(p)
    x = X[0].astype(float)
    scores = [raw_score(ir, X[i]) for i in range(200)]
    lo_t = float(np.percentile(scores, 75))
    plb = None if plaus is None else (plaus.if_ir, plaus.min_total_path)
    ga = dict(GA, population=population)
    cache: dict[str, object] = {}

    def py_run(seed):
        return solve_genetic(
            ir, x, (lo_t, np.inf), compiled, sigma, weights, lam=0.05,
            background=X[:2000], plausibility=plb, seed=seed, **ga,
        )

    def rs_run(seed):
        return solve_genetic_rust(
            ir, x, (lo_t, np.inf), compiled, sigma, weights, lam=0.05,
            background=X[:2000], plausibility=plb, seed=seed, cache=cache, **ga,
        )

    py_run(seeds[0])  # warmups (excluded)
    rs_run(seeds[0])

    py_t, rs_t, py_evals, rs_evals = [], [], [], []
    g1 = gen1_rows(ir, compiled, x, population)
    for seed in seeds:
        t0 = time.perf_counter()
        r_py = py_run(seed)
        py_t.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        r_rs = rs_run(seed)
        rs_t.append(time.perf_counter() - t0)
        py_evals.append(g1 + (int(r_py.stats["generations"]) - 1) * population)
        rs_evals.append(g1 + (int(r_rs.stats["generations"]) - 1) * population)

    def stats(ts, evals):
        med = float(np.median(ts))
        p95 = float(np.percentile(ts, 95))
        per_eval = 1e6 * float(np.sum(ts) / np.sum(evals))
        return med, p95, per_eval

    py = stats(py_t, py_evals)
    rs = stats(rs_t, rs_evals)
    speedup = py[0] / rs[0]
    print(
        f"{tag:44s} py median {py[0]:8.3f}s p95 {py[1]:8.3f}s {py[2]:7.1f}us/eval | "
        f"rust median {rs[0]:8.3f}s p95 {rs[1]:8.3f}s {rs[2]:7.1f}us/eval | "
        f"speedup {speedup:5.2f}x"
    )
    return {"tag": tag, "py": py, "rs": rs, "speedup": speedup, "n_seeds": len(seeds)}


def main() -> None:
    headline_only = "--headline-only" in sys.argv
    threads = os.environ.get("RAYON_NUM_THREADS", "default")
    print(f"rayon threads: {threads}")
    results = []

    large_ir, Xl = build_xgb(300, 6, 50, seed=2)
    results.append(
        run_config("LARGE 300t/d6/50f pop80 bare [HEADLINE]", large_ir, Xl, [], None, 80,
                   list(range(30)))
    )
    if not headline_only:
        med_ir, Xm = build_xgb(100, 6, 20, seed=1)
        results.append(run_config("medium 100t/d6/20f pop80 bare", med_ir, Xm, [], None, 80,
                                  list(range(20))))
        from sklearn.ensemble import IsolationForest

        iso = IsolationForest(n_estimators=100, random_state=0).fit(Xl[:5000])
        plaus = Plausibility.isolation_forest(iso, 0.6)
        mix = constraint_mix(large_ir.feature_names, Xl)
        results.append(
            run_config("large +mix+plaus pop80", large_ir, Xl, mix, plaus, 80, list(range(10)))
        )
        results.append(
            run_config("large bare pop200", large_ir, Xl, [], None, 200, list(range(10)))
        )

    headline = results[0]
    verdict = "PASS (>=2x)" if headline["speedup"] >= 2.0 else "FAIL (<2x)"
    print(f"\nGATE on headline: speedup {headline['speedup']:.2f}x -> {verdict}")


if __name__ == "__main__":
    main()
