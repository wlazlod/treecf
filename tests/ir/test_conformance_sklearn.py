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
