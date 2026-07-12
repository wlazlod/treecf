"""Batch benchmark: wave-parallel explain_batch vs the sequential per-row loop.

- medium model (100 trees / depth 6 / 20 features), 100 rows, 3 plans per row
- both paths use the same derived seeds; records must be identical (asserted)
- Explainer caches are warmed before timing (marshaling excluded on both sides)

Run:            uv run python scripts/bench_batch.py
Single-thread:  RAYON_NUM_THREADS=1 uv run python scripts/bench_batch.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from treecf import Explainer, Target
from treecf.batch import BatchRecord, _row_by_lever_blocking, _row_by_seeds
from treecf.ir.evaluate import raw_score
from treecf.ir.parsers import parse_model

N_ROWS = 100
N_PER_EXAMPLE = 3
SEED = 0
TIME_BUDGET_S = 10.0


def build_xgb(n_trees: int, depth: int, p: int, seed: int):
    import xgboost as xgb

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(20000, p)) * rng.uniform(0.1, 100, size=p)
    y = (X @ rng.normal(size=p) + rng.logistic(scale=2.0, size=20000) > 0).astype(float)
    clf = xgb.XGBClassifier(n_estimators=n_trees, max_depth=depth, random_state=0, n_jobs=4)
    clf.fit(X, y)
    return parse_model(clf), X


def assert_equal(got: list[BatchRecord], want: list[BatchRecord], mode: str) -> None:
    assert len(got) == len(want), f"{mode}: record counts differ"
    for g, w in zip(got, want, strict=True):
        same = (
            g.id == w.id
            and g.k == w.k
            and g.feasible == w.feasible
            and g.seed == w.seed
            and g.blocked_lever == w.blocked_lever
            and g.changes == w.changes
            and g.distance == w.distance
        )
        assert same, f"{mode}: record mismatch for id={g.id} k={g.k}"


def main() -> None:
    threads = os.environ.get("RAYON_NUM_THREADS", "default")
    print(f"rayon threads: {threads}; rows: {N_ROWS}; n_per_example: {N_PER_EXAMPLE}")

    ir, X_bg = build_xgb(100, 6, 20, seed=1)
    X = X_bg[:N_ROWS].astype(np.float64)
    scores = [raw_score(ir, X_bg[i]) for i in range(200)]
    target = Target.raw(op=">=", value=float(np.percentile(scores, 75)))

    batch_exp = Explainer(ir, background=X_bg[:2000])
    seq_exp = Explainer(ir, background=X_bg[:2000])
    batch_exp.explain(X[0], target, seed=0)  # warm the Rust caches (excluded)
    seq_exp.explain(X[0], target, seed=0)

    # --- seeds diversity ---
    t0 = time.perf_counter()
    batch = batch_exp.explain_batch(
        X, target, n_per_example=N_PER_EXAMPLE, seed=SEED, time_budget_s=TIME_BUDGET_S
    )
    batch_wall = time.perf_counter() - t0

    t0 = time.perf_counter()
    sequential: list[BatchRecord] = []
    for i in range(N_ROWS):
        sequential.extend(
            _row_by_seeds(
                seq_exp, X[i], target, i, N_PER_EXAMPLE, "genetic", TIME_BUDGET_S, 0.0,
                master_seed=SEED * 1_000_003 + i * 1_009,
            )
        )
    seq_wall = time.perf_counter() - t0

    assert_equal(list(batch.records), sequential, "seeds")
    print(
        f"seeds          sequential {seq_wall:8.2f}s ({N_ROWS / seq_wall:6.2f} rows/s) | "
        f"batch {batch_wall:8.2f}s ({N_ROWS / batch_wall:6.2f} rows/s) | "
        f"speedup {seq_wall / batch_wall:5.2f}x"
    )

    # --- lever-blocking diversity ---
    t0 = time.perf_counter()
    batch_lb = batch_exp.explain_batch(
        X, target, n_per_example=N_PER_EXAMPLE, diversity="lever-blocking",
        seed=SEED, time_budget_s=TIME_BUDGET_S,
    )
    batch_lb_wall = time.perf_counter() - t0

    t0 = time.perf_counter()
    sequential_lb: list[BatchRecord] = []
    essential: dict[object, list[str]] = {}
    for i in range(N_ROWS):
        rows, ess = _row_by_lever_blocking(
            seq_exp, X[i], target, i, N_PER_EXAMPLE, "genetic", TIME_BUDGET_S, 0.0,
            seed=SEED,
        )
        sequential_lb.extend(rows)
        essential[i] = ess
    seq_lb_wall = time.perf_counter() - t0

    assert_equal(list(batch_lb.records), sequential_lb, "lever-blocking")
    assert batch_lb.essential_levers == essential, "lever-blocking: essential levers differ"
    print(
        f"lever-blocking sequential {seq_lb_wall:8.2f}s ({N_ROWS / seq_lb_wall:6.2f} rows/s) | "
        f"batch {batch_lb_wall:8.2f}s ({N_ROWS / batch_lb_wall:6.2f} rows/s) | "
        f"speedup {seq_lb_wall / batch_lb_wall:5.2f}x"
    )
    print("outputs identical: OK")


if __name__ == "__main__":
    main()
