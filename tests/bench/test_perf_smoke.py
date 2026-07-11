"""Performance smoke (spec §12.8): non-gating, tracked over time via -m bench."""

from __future__ import annotations

import time

import numpy as np
import pytest

from treecf import Counterfactual, Explainer, Target

xgb = pytest.importorskip("xgboost")
pytest.importorskip("ortools")


@pytest.mark.bench
def test_cpsat_solve_time_depth6_300_trees_50_features() -> None:
    rng = np.random.default_rng(0)
    n, p = 20_000, 50
    X = rng.normal(size=(n, p)) * rng.uniform(0.1, 100, size=p)
    y = (X @ rng.normal(size=p) + rng.logistic(scale=2.0, size=n) > 0).astype(float)
    clf = xgb.XGBClassifier(n_estimators=300, max_depth=6, random_state=0, n_jobs=4)
    clf.fit(X, y)

    exp = Explainer(clf, background=X[:5000])
    proba = clf.predict_proba(X[:200])[:, 1]
    cutoff = float(np.median(proba))
    instances = X[np.argsort(proba[:200])[-8:]]

    times: list[float] = []
    solved = 0
    for row in instances:
        start = time.monotonic()
        res = exp.explain(row, target=Target.probability(range=(0.0, cutoff)), time_budget_s=30.0)
        times.append(time.monotonic() - start)
        solved += isinstance(res, Counterfactual)

    median = float(np.median(times))
    p95 = float(np.percentile(times, 95))
    # Non-gating (spec §12.8, P3): numbers are tracked over time, not asserted.
    # Status 2026-07-11: the per-leaf implication encoding does not meet the
    # <1s target at 300 trees; the §5.3 AddAllowedAssignments alternative is
    # the planned v0.2 optimization. Moderate models (<=100 trees) solve in
    # well under a second (see the exactness and API suites).
    print(f"\nbench: median={median:.2f}s p95={p95:.2f}s solved={solved}/{len(instances)}")
