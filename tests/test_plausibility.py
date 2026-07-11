"""Isolation-forest plausibility as a hard constraint (spec §9)."""

from __future__ import annotations

import numpy as np
import pytest

from treecf import Counterfactual, Explainer, Target
from treecf.ir.evaluate import raw_score
from treecf.plausibility import Plausibility

from .conftest import make_synthetic

sklearn = pytest.importorskip("sklearn")
pytest.importorskip("ortools")

from sklearn.ensemble import IsolationForest  # noqa: E402

xgb = pytest.importorskip("xgboost")


@pytest.fixture(scope="module")
def setup() -> tuple[object, np.ndarray, IsolationForest]:
    X, y, _ = make_synthetic(seed=41, nan_frac=0.0)
    clf = xgb.XGBClassifier(n_estimators=15, max_depth=3, random_state=0)
    clf.fit(X, y)
    iso = IsolationForest(n_estimators=30, random_state=0).fit(X)
    return clf, X, iso


def test_isolation_forest_ir_matches_sklearn_scores(setup: tuple) -> None:
    """IR path sums must reproduce sklearn's anomaly score s(x) = 2^(-E[h]/c(n))."""
    _, X, iso = setup
    plaus = Plausibility.isolation_forest(iso, max_anomaly_score=0.6)
    sample = X[:200]
    ours = np.array([plaus.anomaly_score(row) for row in sample])
    theirs = -iso.score_samples(sample)  # sklearn returns -s(x)
    np.testing.assert_allclose(ours, theirs, atol=1e-9)


def test_counterfactual_respects_anomaly_bound(setup: tuple) -> None:
    clf, X, iso = setup
    proba = clf.predict_proba(X)[:, 1]
    idx = int(np.argmax(proba))
    cutoff = float(np.median(proba))
    theta = 0.55

    exp = Explainer(
        clf, background=X, plausibility=Plausibility.isolation_forest(iso, theta)
    )
    res = exp.explain(X[idx], target=Target.probability(range=(0.0, cutoff)))
    assert isinstance(res, Counterfactual)
    plaus = Plausibility.isolation_forest(iso, theta)
    assert plaus.anomaly_score(res.x_cf) <= theta + 1e-9


def test_genetic_backend_respects_anomaly_bound(setup: tuple) -> None:
    clf, X, iso = setup
    proba = clf.predict_proba(X)[:, 1]
    idx = int(np.argmax(proba))
    cutoff = float(np.median(proba))
    theta = 0.55

    exp = Explainer(
        clf, background=X, plausibility=Plausibility.isolation_forest(iso, theta)
    )
    res = exp.explain(
        X[idx], target=Target.probability(range=(0.0, cutoff)), backend="genetic", seed=2
    )
    if isinstance(res, Counterfactual):
        plaus = Plausibility.isolation_forest(iso, theta)
        assert plaus.anomaly_score(res.x_cf) <= theta + 1e-9


def test_impossible_theta_is_infeasible(setup: tuple) -> None:
    clf, X, iso = setup
    exp = Explainer(
        clf, background=X, plausibility=Plausibility.isolation_forest(iso, 0.01)
    )
    res = exp.explain(X[0], target=Target.probability(range=(0.0, 0.9)))
    assert not isinstance(res, Counterfactual)


def test_raw_score_is_total_path_length(setup: tuple) -> None:
    _, X, iso = setup
    plaus = Plausibility.isolation_forest(iso, 0.5)
    total = raw_score(plaus.if_ir, X[0])
    assert total > 0  # sum of depth-adjusted path lengths over trees
