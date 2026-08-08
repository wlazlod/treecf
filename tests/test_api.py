"""End-to-end vertical slice: XGBoost -> IR -> CP-SAT -> optimal counterfactual."""

from __future__ import annotations

import numpy as np
import pytest

from treecf import (
    Counterfactual,
    Explainer,
    Freeze,
    Monotone,
    Range,
    Target,
    TreecfError,
    constraint,
)

from .conftest import make_synthetic

xgb = pytest.importorskip("xgboost")


def _stump_dump() -> dict[str, object]:
    """Tiny two-tree binary ensemble as a LightGBM-format dump (no extras needed)."""

    def split(feat: int, thr: float, left: object, right: object) -> dict[str, object]:
        return {"split_feature": feat, "threshold": thr, "decision_type": "<=",
                "missing_type": "NaN", "default_left": True,
                "left_child": left, "right_child": right}

    return {"num_tree_per_iteration": 1, "objective": "binary", "max_feature_idx": 1,
            "feature_names": ["f0", "f1"],
            "tree_info": [
                {"tree_structure": split(0, 1.0, {"leaf_value": -0.5}, {"leaf_value": 0.5})},
                {"tree_structure": split(1, 0.0, {"leaf_value": -0.25}, {"leaf_value": 0.25})},
            ]}


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
    res = exp.explain(X[idx], target=Target.probability(range=(0.0, cutoff)), seed=0)

    assert isinstance(res, Counterfactual)
    assert res.proof == "heuristic"
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
        exp.explain(X[0], target=Target.probability(op="<=", value=0.5), backend="magic", seed=0)


@pytest.mark.parametrize("backend", ["genetic", "python"])
def test_far_single_feature_linear_matches_range(backend: str) -> None:
    # review repro: constraint("f0 >= 100") must behave like Range("f0", 100, 1e9),
    # not come back Infeasible because the halfspace is many sigma away
    x = np.array([2.12, 0.0])
    target = Target.probability(range=(0.0, 1.0))
    via_range = Explainer(
        _stump_dump(), constraints=[Range("f0", 100, 1e9)], normalizers=np.ones(2)
    ).explain(x, target, backend=backend, seed=0)
    via_linear = Explainer(
        _stump_dump(), constraints=[constraint("f0 >= 100")], normalizers=np.ones(2)
    ).explain(x, target, backend=backend, seed=0)
    assert isinstance(via_range, Counterfactual)
    assert isinstance(via_linear, Counterfactual)
    assert via_linear.x_cf[0] == via_range.x_cf[0] == 100.0
