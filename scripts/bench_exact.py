"""Benchmark: exact backend (rust) vs genetic (rust) wall time and gap.

- same seeds per config; only the ``Explainer.explain`` call is timed
  (Explainer construction/background fitting excluded); 1 warmup each
- exact backend run twice per seed: warm_start on and off
- genetic backend run once per seed for comparison
- gap per seed: (genetic.distance - exact.distance) / max(1, exact.distance) —
  how much cost the heuristic leaves on the table relative to the proved optimum
- reports median/p95 wall time, median/max gap, median nodes_expanded, proof mix
- reporting only: no wall-clock-dependent assertions, budgets are documented
  below and generous so a slower machine degrades gracefully, not silently

Time budgets (fixed, not measured): exact gets EXACT_TIME_BUDGET_S wall-clock
and EXACT_NODE_BUDGET nodes per solve; genetic gets GENETIC_TIME_BUDGET_S
(effectively unused — it stops on stall/max-gen first, same GA config as
bench_genetic.py).

Run:         uv run python scripts/bench_exact.py
Full:        uv run python scripts/bench_exact.py --full
Scale:       uv run python scripts/bench_exact.py --matrix
Categorical: uv run python scripts/bench_exact.py --categorical
JSON:        add --json PATH to save every scenario's numbers with machine
             and version stamps (the docs benchmark page reads this file)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from bench_genetic import build_xgb

from treecf import Counterfactual, Explainer, Target
from treecf.ir.evaluate import raw_score

EXACT_TIME_BUDGET_S = 5.0
EXACT_NODE_BUDGET = 2_000_000
GENETIC_TIME_BUDGET_S = 5.0


def run_scenario(tag: str, ir, X, seeds: list[int]) -> dict[str, object]:
    exp = Explainer(ir, background=X[:2000])
    x = X[0].astype(float)
    scores = [raw_score(ir, X[i]) for i in range(200)]
    lo_t = float(np.percentile(scores, 75))
    target = Target.raw(op=">=", value=lo_t)

    def genetic_run(seed: int):
        return exp.explain(x, target, backend="genetic", time_budget_s=GENETIC_TIME_BUDGET_S,
                            seed=seed)

    def exact_run(seed: int, warm_start: bool):
        return exp.explain(
            x, target, backend="exact", time_budget_s=EXACT_TIME_BUDGET_S, seed=seed,
            warm_start=warm_start, node_budget=EXACT_NODE_BUDGET,
        )

    genetic_run(seeds[0])  # warmups (excluded)
    exact_run(seeds[0], True)
    exact_run(seeds[0], False)

    gen_t, on_t, off_t = [], [], []
    gaps, nodes, proofs = [], [], []
    for seed in seeds:
        t0 = time.perf_counter()
        r_gen = genetic_run(seed)
        gen_t.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        r_on = exact_run(seed, True)
        on_t.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        exact_run(seed, False)  # cold timing only; warm-start run below is the reported result
        off_t.append(time.perf_counter() - t0)

        if isinstance(r_on, Counterfactual):
            nodes.append(int(r_on.solver_stats["nodes_expanded"]))
            proofs.append(r_on.proof)
            if isinstance(r_gen, Counterfactual):
                gaps.append(
                    (r_gen.distance - r_on.distance) / max(1.0, r_on.distance)
                )

    def stats(ts: list[float]) -> tuple[float, float]:
        return float(np.median(ts)), float(np.percentile(ts, 95))

    gen_med, gen_p95 = stats(gen_t)
    on_med, on_p95 = stats(on_t)
    off_med, off_p95 = stats(off_t)
    gap_med = float(np.median(gaps)) if gaps else float("nan")
    gap_max = float(np.max(gaps)) if gaps else float("nan")
    nodes_med = float(np.median(nodes)) if nodes else float("nan")
    proof_mix = {p: proofs.count(p) for p in sorted(set(proofs))}

    print(
        f"{tag:38s} genetic median {gen_med:7.3f}s p95 {gen_p95:7.3f}s | "
        f"exact(warm) median {on_med:7.3f}s p95 {on_p95:7.3f}s | "
        f"exact(cold) median {off_med:7.3f}s p95 {off_p95:7.3f}s"
    )
    print(
        f"{'':38s} gap median {gap_med:6.2%} max {gap_max:6.2%} | "
        f"nodes_expanded median {nodes_med:9.0f} | proof mix {proof_mix}"
    )
    return {
        "tag": tag, "genetic": (gen_med, gen_p95), "exact_warm": (on_med, on_p95),
        "exact_cold": (off_med, off_p95), "gap_median": gap_med, "gap_max": gap_max,
        "nodes_expanded_median": nodes_med, "proof_mix": proof_mix,
    }


CERTIFICATION_BUDGET_S = 60.0
CERTIFICATION_NODE_BUDGET = 500_000_000  # wall time is the binding budget here


def run_certification_cell(tag: str, ir, X, seeds: list[int]) -> dict[str, object]:
    """Median time-to-certificate and certified fractions within 10 s / 60 s."""
    from treecf import Infeasible

    exp = Explainer(ir, background=X[:1000])
    x = X[0].astype(float)
    scores = [raw_score(ir, X[i]) for i in range(min(len(X), 200))]
    target = Target.raw(op=">=", value=float(np.percentile(scores, 75)))

    times: list[float] = []
    certified: list[bool] = []
    for seed in seeds:
        t0 = time.perf_counter()
        result = exp.explain(
            x, target, backend="exact", time_budget_s=CERTIFICATION_BUDGET_S,
            seed=seed, warm_start=True, node_budget=CERTIFICATION_NODE_BUDGET,
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        if isinstance(result, Counterfactual):
            certified.append(result.proof in ("optimal", "optimal_within_gap"))
        else:
            certified.append(isinstance(result, Infeasible) and result.proof == "certified")

    certified_times = [t for t, ok in zip(times, certified, strict=True) if ok]
    median_certified = float(np.median(certified_times)) if certified_times else float("nan")
    fast = sum(1 for t, ok in zip(times, certified, strict=True) if ok and t <= 10.0)
    within_10 = fast / len(seeds)
    within_60 = sum(1 for ok in certified if ok) / len(seeds)
    print(
        f"{tag:38s} time-to-certificate median {median_certified:7.3f}s | "
        f"certified <=10s {within_10:5.0%} | <=60s {within_60:5.0%}"
    )
    return {
        "tag": tag,
        "kind": "certification",
        "median_time_to_certificate_s": median_certified,
        "certified_within_10s": within_10,
        "certified_within_60s": within_60,
        "n_seeds": len(seeds),
    }


def build_lgb_categorical(n_estimators: int, depth: int, seed: int):
    """A credit-shaped mixed model: numeric levers plus native categorical
    features of cardinality 3 / 8 / 15, trained with LightGBM."""
    import lightgbm as lgb

    rng = np.random.default_rng(seed)
    n = 4000
    numeric = np.column_stack(
        [
            rng.normal(loc=3000.0, scale=1500.0, size=n),  # income-like
            rng.exponential(scale=0.4, size=n),  # utilization-like
            np.floor(rng.exponential(scale=2.0, size=n)),  # count-like
            rng.normal(loc=8.0, scale=4.0, size=n),  # tenure-like
        ]
    )
    cats = np.column_stack(
        [
            rng.integers(0, 3, size=n),
            rng.integers(0, 8, size=n),
            rng.integers(0, 15, size=n),
        ]
    ).astype(np.float64)
    X = np.column_stack([numeric, cats])
    signal = (
        (numeric[:, 0] - 3000.0) / 1500.0
        - numeric[:, 1]
        + (cats[:, 0] == 1)
        + 0.5 * (cats[:, 1] % 2)
        - 0.3 * (cats[:, 2] % 3 == 0)
    )
    y = (signal + rng.logistic(scale=0.7, size=n) > 0).astype(int)
    booster = lgb.train(
        {
            "objective": "binary", "num_leaves": 2**depth, "max_depth": depth,
            "learning_rate": 0.15, "seed": seed, "deterministic": True,
            "verbose": -1, "min_data_per_group": 1,
        },
        lgb.Dataset(X, label=y, categorical_feature=[4, 5, 6], free_raw_data=False),
        num_boost_round=n_estimators,
    )
    from treecf.ir.parsers import parse_model

    return parse_model(booster), X


def _machine_stamp() -> dict[str, object]:
    import platform

    import treecf

    stamp: dict[str, object] = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "system": platform.system(),
        "cpu_count": os.cpu_count(),
        "treecf": treecf.__version__,
    }
    for lib in ("xgboost", "lightgbm"):
        try:
            module = __import__(lib)
            stamp[lib] = module.__version__
        except ImportError:
            stamp[lib] = None
    return stamp


def main() -> None:
    full = "--full" in sys.argv
    matrix = "--matrix" in sys.argv
    categorical = "--categorical" in sys.argv
    json_path = None
    if "--json" in sys.argv:
        json_path = sys.argv[sys.argv.index("--json") + 1]
    print(f"cpu count: {os.cpu_count()} (sequential exact engine; rayon threads irrelevant here)")

    results = []
    small_ir, Xs = build_xgb(30, 4, 8, seed=1)
    results.append(run_scenario("small 30t/d4/8f [HEADLINE]", small_ir, Xs, list(range(10))))

    if full:
        med_ir, Xm = build_xgb(60, 5, 12, seed=1)
        results.append(run_scenario("medium 60t/d5/12f", med_ir, Xm, list(range(10))))
        large_ir, Xl = build_xgb(300, 6, 50, seed=2)  # bench_genetic's LARGE scenario
        results.append(run_scenario("large 300t/d6/50f", large_ir, Xl, list(range(5))))

    if matrix:
        print("\nscale matrix (certification):")
        for trees in (50, 100, 200):
            for depth in (3, 5):
                for features in (8, 12, 20):
                    ir, X = build_xgb(trees, depth, features, seed=1)
                    tag = f"matrix {trees}t/d{depth}/{features}f"
                    results.append(run_certification_cell(tag, ir, X, list(range(3))))

    if categorical:
        print("\ncategorical suite (LightGBM native, cardinalities 3/8/15):")
        for trees, depth in ((50, 3), (100, 5), (200, 5)):
            ir, X = build_lgb_categorical(trees, depth, seed=3)
            tag = f"categorical {trees}t/d{depth}/4num+3cat"
            results.append(run_certification_cell(tag, ir, X, list(range(3))))

    headline = results[0]
    print(
        f"\nHEADLINE: exact(warm) median {headline['exact_warm'][0]:.3f}s vs "
        f"genetic median {headline['genetic'][0]:.3f}s, "
        f"gap median {headline['gap_median']:.2%}"
    )

    if json_path is not None:
        import json

        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump({"stamp": _machine_stamp(), "results": results}, fh, indent=1)
        print(f"saved {json_path}")


if __name__ == "__main__":
    main()
