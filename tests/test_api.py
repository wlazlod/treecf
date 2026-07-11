"""End-to-end vertical slice: XGBoost -> IR -> CP-SAT -> optimal counterfactual."""

from __future__ import annotations

import numpy as np
import pytest

from treecf import Counterfactual, Explainer, Freeze, Monotone, Target, TreecfError

from .conftest import make_synthetic

xgb = pytest.importorskip("xgboost")
pytest.importorskip("ortools")


@pytest.fixture(scope="module")
def credit_model() -> tuple[object, np.ndarray]:
    X, y, _ = make_synthetic(seed=42, nan_frac=0.05)
    clf = xgb.XGBClassifier(n_estimators=25, max_depth=3, random_state=0)
    clf.fit(X, y)
    return clf, X


def test_probability_target_end_to_end(credit_model: tuple[object, np.ndarray]) -> None:
    clf, X = credit_model
    proba = clf.predict_proba(X)[:, 1]
    # pick a clearly positive instance and push it under the median probability
    idx = int(np.argmax(proba))
    cutoff = float(np.median(proba))

    exp = Explainer(clf, background=X, constraints=[Freeze("f0"), Monotone("f1", "increase")])
    res = exp.explain(X[idx], target=Target.probability(range=(0.0, cutoff)))

    assert isinstance(res, Counterfactual)
    assert res.proof == "optimal"
    assert res.score_prob is not None and res.score_prob <= cutoff
    assert res.x_cf[0] == X[idx, 0]  # frozen
    if not np.isnan(X[idx, 1]):
        assert res.x_cf[1] >= X[idx, 1]  # monotone increase
    assert res.n_changed == len(res.changes) > 0
    # the native model agrees the counterfactual crosses the cutoff (float32 slack)
    native = float(clf.predict_proba(res.x_cf.reshape(1, -1))[0, 1])
    assert native <= cutoff + 1e-5


def test_missing_background_and_normalizers_raises(
    credit_model: tuple[object, np.ndarray],
) -> None:
    clf, _ = credit_model
    with pytest.raises(TreecfError, match="background"):
        Explainer(clf)


def test_unknown_backend_raises(credit_model: tuple[object, np.ndarray]) -> None:
    clf, X = credit_model
    exp = Explainer(clf, background=X)
    with pytest.raises(TreecfError, match="unknown backend"):
        exp.explain(X[0], target=Target.probability(op="<=", value=0.5), backend="magic")
