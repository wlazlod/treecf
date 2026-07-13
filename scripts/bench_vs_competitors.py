# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "treecf",
#     "xgboost",
#     "scikit-learn",
#     "pandas",
#     "dice-ml",
#     "NICEx",
# ]
# ///
"""Benchmark treecf against DiCE and NICE on tree-ensemble counterfactuals.

Protocol (identical for every method):
- one XGBoost classifier and one set of declined rows per scenario;
- the goal is the class flip (probability < 0.5) — the only target every
  library can express natively;
- per-instance wall time is measured around the generate call only; each
  method's one-time setup (Explainer construction, DiCE Data/Model objects,
  NICE fit) happens outside the timer;
- validity is re-checked against the model by this script, never taken from
  the library;
- treecf runs WITHOUT constraints so no method solves a harder problem;
- the batch section measures whole-dataset throughput: treecf's parallel
  ``explain_batch`` in one call vs looping each library row by row.

Run:  uv run scripts/bench_vs_competitors.py [--json results.json]

alibi's counterfactual methods are deliberately not included: they are
TensorFlow-based and, on a non-differentiable ensemble, fall back to numerical
gradients (measured separately at ~72 s/instance on the medium scenario —
three orders of magnitude outside this table).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CUTOFF = 0.5

CREDIT_NAMES = [
    "income_monthly", "utilization", "n_active_loans", "n_loans_total",
    "max_dpd_30d", "max_dpd_12m", "months_since_last_delinq", "age",
]


def make_credit_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """8 features, 6000 rows: the docs' synthetic credit population (no NaN)."""
    rng = np.random.default_rng(42)
    n = 6000
    income = rng.lognormal(8.3, 0.5, n).round(-1)
    utilization = rng.beta(2, 3, n).round(3)
    n_total = rng.poisson(4, n).astype(float) + 1
    n_active = np.minimum(np.floor(n_total * rng.beta(3, 2, n)), n_total)
    dpd_12m = np.floor(rng.exponential(6, n)) * (rng.random(n) < 0.4)
    dpd_30d = np.floor(dpd_12m * rng.beta(2, 4, n))
    months = rng.exponential(14, n).round(0)
    age = rng.integers(21, 75, n).astype(float)
    X = np.column_stack([income, utilization, n_active, n_total,
                         dpd_30d, dpd_12m, months, age])
    risk = (-0.9 * np.log(income / 4000) + 2.2 * utilization
            + 0.35 * dpd_30d + 0.15 * dpd_12m - 0.02 * months - 0.015 * age)
    y = (risk + rng.logistic(scale=0.8, size=n) > np.median(risk)).astype(int)
    return X, y, CREDIT_NAMES


def make_wide_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """50 features, 20000 rows: the headline large scenario."""
    rng = np.random.default_rng(2)
    p = 50
    X = rng.normal(size=(20000, p)) * rng.uniform(0.1, 100, size=p)
    y = (X @ rng.normal(size=p) + rng.logistic(scale=2.0, size=20000) > 0).astype(int)
    return X, y, [f"f{j:02d}" for j in range(p)]


SCENARIOS = [
    {"name": "medium (120 trees, depth 4, 8 features)", "data": make_credit_data,
     "n_estimators": 120, "max_depth": 4, "n_instances": 100, "n_batch": 500},
    {"name": "large (300 trees, depth 6, 50 features)", "data": make_wide_data,
     "n_estimators": 300, "max_depth": 6, "n_instances": 50, "n_batch": 200},
]


def evaluate_per_instance(
    name: str, clf: object, sigma: np.ndarray, rows: np.ndarray,
    runner: Callable[[np.ndarray], np.ndarray | None],
) -> dict[str, object]:
    times: list[float] = []
    n_changed: list[int] = []
    l1: list[float] = []
    valid = 0
    for x in rows:
        t0 = time.perf_counter()
        try:
            cf = runner(x)
        except Exception:  # a library failing on an instance is a result
            cf = None
        times.append(time.perf_counter() - t0)
        if cf is None:
            continue
        cf = np.asarray(cf, dtype=float).ravel()
        if clf.predict_proba(cf.reshape(1, -1))[0, 1] < CUTOFF:  # type: ignore[attr-defined]
            valid += 1
            changed = ~np.isclose(cf, x, rtol=0, atol=1e-12)
            n_changed.append(int(changed.sum()))
            l1.append(float((np.abs(cf - x) / sigma).sum()))
    return {
        "method": name,
        "valid": f"{valid}/{len(rows)}",
        "median_s": round(float(np.median(times)), 4),
        "p95_s": round(float(np.percentile(times, 95)), 4),
        "mean_changed": round(float(np.mean(n_changed)), 2) if n_changed else None,
        "mean_L1_sigma": round(float(np.mean(l1)), 2) if l1 else None,
    }


Rows = list[dict[str, object]]


def run_scenario(
    spec: dict[str, object], checkpoint: Callable[[Rows, Rows], None]
) -> dict[str, object]:
    import dice_ml
    import xgboost as xgb
    from nice import NICE

    from treecf import Counterfactual, Explainer, Target
    from treecf.objective import fit_normalizers

    X, y, names = spec["data"]()  # type: ignore[operator]
    clf = xgb.XGBClassifier(n_estimators=spec["n_estimators"], max_depth=spec["max_depth"],
                            random_state=0, n_jobs=4)
    clf.fit(X, y)
    proba = clf.predict_proba(X)[:, 1]
    declined = np.flatnonzero(proba > CUTOFF)
    rows = X[declined[: spec["n_instances"]]]
    batch_rows = X[declined[: spec["n_batch"]]]
    sigma = fit_normalizers(X)

    exp = Explainer(clf, background=X)
    target = Target.probability(range=(0.0, CUTOFF))
    exp.explain(rows[0], target, seed=0)  # warm-up: marshaling + cell cache

    def run_treecf(x: np.ndarray) -> np.ndarray | None:
        res = exp.explain(x, target, seed=0)
        return res.x_cf if isinstance(res, Counterfactual) else None

    frame = pd.DataFrame(X, columns=names)
    frame["outcome"] = y
    d = dice_ml.Data(dataframe=frame, continuous_features=names, outcome_name="outcome")
    m = dice_ml.Model(model=clf, backend="sklearn")

    def dice_runner(method: str) -> Callable[[np.ndarray], np.ndarray | None]:
        dice = dice_ml.Dice(d, m, method=method)

        def run(x: np.ndarray) -> np.ndarray | None:
            query = pd.DataFrame([x], columns=names)
            out = dice.generate_counterfactuals(query, total_CFs=1, desired_class=0,
                                                verbose=False)
            cfs = out.cf_examples_list[0].final_cfs_df
            if cfs is None or len(cfs) == 0:
                return None
            return cfs[names].iloc[0].to_numpy()

        return run

    nice_exp = NICE(
        X_train=X, predict_fn=lambda a: clf.predict_proba(a), y_train=y,
        cat_feat=[], num_feat=list(range(len(names))), distance_metric="HEOM",
        num_normalization="minmax", optimization="sparsity", justified_cf=True,
    )

    def run_nice(x: np.ndarray) -> np.ndarray | None:
        return nice_exp.explain(x.reshape(1, -1))

    runners: list[tuple[str, Callable[[np.ndarray], np.ndarray | None]]] = [
        ("treecf", run_treecf),
        ("DiCE (random)", dice_runner("random")),
        ("DiCE (genetic)", dice_runner("genetic")),
        ("DiCE (kdtree)", dice_runner("kdtree")),
        ("NICE (sparsity)", run_nice),
    ]
    per_instance = []
    for method_name, runner in runners:
        row = evaluate_per_instance(method_name, clf, sigma, rows, runner)
        print(row, flush=True)
        per_instance.append(row)
        checkpoint(per_instance, [])  # a killed run keeps every finished method

    # --- batch throughput: whole dataset, rows per second ---
    def timed_loop(runner: Callable[[np.ndarray], np.ndarray | None]) -> float:
        t0 = time.perf_counter()
        for x in batch_rows:
            with contextlib.suppress(Exception):
                runner(x)
        return time.perf_counter() - t0

    n = len(batch_rows)
    print(f"batch throughput over {n} rows...", flush=True)
    t0 = time.perf_counter()
    exp.explain_batch(batch_rows, target, seed=0)
    batch_wall = time.perf_counter() - t0
    throughput: list[dict[str, object]] = [
        {"method": "treecf explain_batch (one call)", "wall_s": round(batch_wall, 2),
         "rows_per_s": round(n / batch_wall, 1)},
    ]
    checkpoint(per_instance, throughput)
    loop_runners: list[tuple[str, Callable[[np.ndarray], np.ndarray | None]]] = [
        ("treecf explain loop", run_treecf),
        ("NICE loop", run_nice),
        ("DiCE (random) loop", dice_runner("random")),
    ]
    for loop_name, runner in loop_runners:
        wall = timed_loop(runner)
        row = {"method": loop_name, "wall_s": round(wall, 2),
               "rows_per_s": round(n / wall, 1)}
        print(row, flush=True)
        throughput.append(row)
        checkpoint(per_instance, throughput)

    return {"scenario": spec["name"], "n_instances": len(rows), "n_batch": n,
            "per_instance": per_instance, "batch_throughput": throughput}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=None, help="write results to this path")
    parser.add_argument("--only", default=None, choices=("medium", "large"),
                        help="run a single scenario")
    args = parser.parse_args()

    results: list[dict[str, object]] = []

    def dump() -> None:
        if args.json:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=2)

    for spec in SCENARIOS:
        if args.only and not str(spec["name"]).startswith(args.only):
            continue
        print(f"\n=== {spec['name']} ===", flush=True)
        partial = {"scenario": spec["name"], "n_instances": spec["n_instances"],
                   "n_batch": spec["n_batch"], "per_instance": [], "batch_throughput": []}
        results.append(partial)

        def checkpoint(per_instance: list[dict[str, object]],
                       throughput: list[dict[str, object]], partial: dict = partial) -> None:
            partial["per_instance"] = per_instance
            partial["batch_throughput"] = throughput
            dump()  # a killed run keeps everything finished so far

        results[-1] = run_scenario(spec, checkpoint)
        dump()

    if args.json:
        print(f"\nwritten: {args.json}", flush=True)


if __name__ == "__main__":
    main()
