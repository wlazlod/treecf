"""sklearn parser conformance: RandomForest, GradientBoosting, HistGradientBoosting."""

from __future__ import annotations

import numpy as np
import pytest

from treecf._errors import UnsupportedModelError
from treecf.ir.model import Link
from treecf.ir.parsers import parse_model

from ..conftest import make_synthetic
from .harness import assert_conformance

sklearn = pytest.importorskip("sklearn")

from sklearn.ensemble import (  # noqa: E402
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)


def test_random_forest_classifier_probability_average() -> None:
    X, y, _ = make_synthetic(seed=21, nan_frac=0.0)
    clf = RandomForestClassifier(n_estimators=12, max_depth=4, random_state=0)
    clf.fit(X, y)
    ir = parse_model(clf)
    assert ir.link is Link.IDENTITY  # raw score IS the averaged class-1 probability
    assert_conformance(ir, X, lambda A: clf.predict_proba(A)[:, 1], n_random=3000)


def test_random_forest_regressor() -> None:
    X, _, y = make_synthetic(seed=22, nan_frac=0.0)
    reg = RandomForestRegressor(n_estimators=10, max_depth=4, random_state=0)
    reg.fit(X, y)
    ir = parse_model(reg)
    assert ir.link is Link.IDENTITY
    assert_conformance(ir, X, reg.predict, n_random=3000)


def test_gradient_boosting_classifier() -> None:
    X, y, _ = make_synthetic(seed=23, nan_frac=0.0)
    clf = GradientBoostingClassifier(n_estimators=20, max_depth=3, random_state=0)
    clf.fit(X, y)
    ir = parse_model(clf)
    assert ir.link is Link.SIGMOID
    assert_conformance(
        ir, X, lambda A: clf.predict_proba(A)[:, 1], n_random=3000, include_nan=False
    )


def test_gradient_boosting_regressor() -> None:
    X, _, y = make_synthetic(seed=24, nan_frac=0.0)
    reg = GradientBoostingRegressor(n_estimators=20, max_depth=3, random_state=0)
    reg.fit(X, y)
    ir = parse_model(reg)
    assert_conformance(ir, X, reg.predict, n_random=3000, include_nan=False)


def test_hist_gradient_boosting_classifier_with_nans() -> None:
    X, y, _ = make_synthetic(seed=25)  # NaNs exercise missing_go_to_left
    clf = HistGradientBoostingClassifier(max_iter=20, max_depth=4, random_state=0)
    clf.fit(X, y)
    ir = parse_model(clf)
    assert ir.link is Link.SIGMOID
    assert_conformance(ir, X, lambda A: clf.predict_proba(A)[:, 1], n_random=3000)


def test_hist_gradient_boosting_regressor_with_nans() -> None:
    X, _, y = make_synthetic(seed=26)
    reg = HistGradientBoostingRegressor(max_iter=20, max_depth=4, random_state=0)
    reg.fit(X, y)
    ir = parse_model(reg)
    assert_conformance(ir, X, reg.predict, n_random=3000)


def test_multiclass_forest_raises() -> None:
    X, _, _ = make_synthetic(seed=27, nan_frac=0.0)
    rng = np.random.default_rng(1)
    clf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0)
    clf.fit(X, rng.integers(0, 3, size=len(X)))
    with pytest.raises(UnsupportedModelError, match="multi"):
        parse_model(clf)


# --------------------------------------------------------------- float64 grid
# sklearn tree_ ensembles route float32(x) <= float64(threshold). The IR must
# reproduce that for EVERY float64 x — not only float32-representable probes —
# because the counterfactual search places representatives on exact float64
# boundaries. Regression for the probcal joint finding: an x_cf whose
# coordinate sat exactly on a shared threshold evaluated 3+ raw-score units
# apart between treecf (proof="optimal") and decision_function.


@pytest.mark.parametrize("subsample", [1.0, 0.6])
def test_gradient_boosting_classifier_unquantized_probes(subsample: float) -> None:
    X, y, _ = make_synthetic(seed=31, nan_frac=0.0)
    clf = GradientBoostingClassifier(
        n_estimators=30, max_depth=3, subsample=subsample, random_state=0
    )
    clf.fit(X, y)
    ir = parse_model(clf)
    assert_conformance(
        ir, X, lambda A: clf.predict_proba(A)[:, 1],
        n_random=3000, include_nan=False, quantize=False,
    )


def test_random_forest_classifier_unquantized_probes() -> None:
    X, y, _ = make_synthetic(seed=32, nan_frac=0.0)
    clf = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=0)
    clf.fit(X, y)
    ir = parse_model(clf)
    assert_conformance(
        ir, X, lambda A: clf.predict_proba(A)[:, 1],
        n_random=3000, include_nan=False, quantize=False,
    )


def test_gradient_boosting_regressor_unquantized_probes() -> None:
    X, _, y = make_synthetic(seed=33, nan_frac=0.0)
    reg = GradientBoostingRegressor(n_estimators=20, max_depth=3, random_state=0)
    reg.fit(X, y)
    ir = parse_model(reg)
    assert_conformance(ir, X, reg.predict, n_random=3000, include_nan=False, quantize=False)


def test_hist_gradient_boosting_unquantized_probes() -> None:
    # HistGradientBoosting predicts on the float64 grid directly; the
    # unquantized mode must already hold with untouched thresholds.
    X, y, _ = make_synthetic(seed=34, nan_frac=0.0)
    clf = HistGradientBoostingClassifier(max_iter=30, random_state=0)
    clf.fit(X, y)
    ir = parse_model(clf)
    assert_conformance(
        ir, X, lambda A: clf.predict_proba(A)[:, 1], n_random=3000, quantize=False
    )


def test_exact_threshold_point_routes_like_sklearn() -> None:
    # The distilled probcal finding: a point EXACTLY at a float64 split
    # threshold must produce the same raw score as decision_function.
    from treecf.ir.evaluate import raw_score

    X, y, _ = make_synthetic(seed=35, nan_frac=0.0)
    clf = GradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=0)
    clf.fit(X, y)
    ir = parse_model(clf)
    base = np.nan_to_num(X[0], nan=0.0)
    checked = 0
    for tree in ir.trees:
        for node in tree.nodes:
            if node.feature is None:
                continue
            row = base.copy()
            row[node.feature] = node.threshold
            got = raw_score(ir, row)
            want = float(clf.decision_function(row[None])[0])
            assert got == pytest.approx(want, abs=1e-9), (
                f"threshold routing mismatch at feature {node.feature}, "
                f"threshold {node.threshold!r}: IR {got} vs sklearn {want}"
            )
            checked += 1
    assert checked > 50


def _hgb_categorical_data(seed: int, cardinalities: dict[int, int]) -> tuple:
    X, y, _ = make_synthetic(seed=seed, nan_frac=0.1)
    rng = np.random.default_rng(seed)
    for j, k in cardinalities.items():
        codes = rng.integers(0, k, size=len(X)).astype(np.float64)
        codes[rng.random(len(X)) < 0.1] = np.nan
        X[:, j] = codes
        y = np.where(np.isnan(codes), y, (y + (codes % 2)) % 2)
    return X, y


def test_hist_gradient_boosting_native_categorical() -> None:
    X, y = _hgb_categorical_data(seed=31, cardinalities={1: 5, 3: 9})
    clf = HistGradientBoostingClassifier(
        max_iter=20, max_depth=4, random_state=0, categorical_features=[1, 3]
    )
    clf.fit(X, y)
    ir = parse_model(clf)
    assert set(ir.categorical) == {1, 3}
    assert any(node.categories is not None for tree in ir.trees for node in tree.nodes)
    assert_conformance(ir, X, lambda A: clf.predict_proba(A)[:, 1])


def test_hist_gradient_boosting_categorical_regressor_sparse_codes() -> None:
    # codes {0, 2, 5} only: the encoder's positions differ from the raw codes,
    # and the unseen codes 1/3/4 must route like missing values
    X, y, _ = make_synthetic(seed=32, nan_frac=0.0)
    rng = np.random.default_rng(32)
    codes = rng.choice([0.0, 2.0, 5.0], size=len(X))
    X[:, 1] = codes
    y = y + (codes == 2)
    reg = HistGradientBoostingRegressor(
        max_iter=15, max_depth=3, random_state=0, categorical_features=[1]
    )
    reg.fit(X, y)
    ir = parse_model(reg)
    assert ir.categorical[1].cardinality == 6
    assert_conformance(ir, X, reg.predict)


def test_hist_gradient_boosting_string_categories_need_a_code_map() -> None:
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(33)
    n = 500
    names = ["clerk", "manager", "nurse"]
    frame = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "occupation": pd.Categorical(rng.choice(names, size=n), categories=names),
        }
    )
    y = ((frame["occupation"].cat.codes % 2).to_numpy() ^ (frame["num"] > 0)).astype(int)
    clf = HistGradientBoostingClassifier(
        max_iter=10, max_depth=3, random_state=0, categorical_features="from_dtype"
    )
    clf.fit(frame, y)
    with pytest.raises(UnsupportedModelError, match="pass\ncategories= |categories="):
        parse_model(clf)
    from treecf.ir.parsers.sklearn import parse_sklearn

    ir = parse_sklearn(clf, categories={"occupation": names})
    assert ir.categorical[1].categories == tuple(names)
    X = np.column_stack(
        [frame["num"].to_numpy(), frame["occupation"].cat.codes.to_numpy(dtype=np.float64)]
    )

    def predict(A: np.ndarray) -> np.ndarray:
        probe = pd.DataFrame(
            {
                "num": A[:, 0],
                "occupation": pd.Categorical.from_codes(
                    np.where(np.isnan(A[:, 1]) | (A[:, 1] >= len(names)), -1, A[:, 1])
                    .astype(int),
                    categories=names,
                ),
            }
        )
        return clf.predict_proba(probe)[:, 1]

    assert_conformance(ir, X, predict, n_random=2000)
