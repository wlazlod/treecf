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
