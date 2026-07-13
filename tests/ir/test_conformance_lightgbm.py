"""LightGBM parser conformance: LE convention, missing_type semantics."""

from __future__ import annotations

import numpy as np
import pytest

from treecf._errors import UnsupportedModelError
from treecf.ir.model import Link
from treecf.ir.parsers import parse_model

from ..conftest import make_synthetic
from .harness import assert_conformance

lgb = pytest.importorskip("lightgbm")


def _params(objective: str) -> dict[str, object]:
    return {
        "objective": objective,
        "num_leaves": 15,
        "learning_rate": 0.3,
        "seed": 7,
        "deterministic": True,
        "verbose": -1,
    }


@pytest.mark.parametrize("objective", ["binary", "regression"])
def test_booster_conformance_with_nans(objective: str) -> None:
    X, y_bin, y_cont = make_synthetic(seed=5)  # contains NaNs -> missing_type "NaN"
    y = y_bin if objective == "binary" else y_cont
    booster = lgb.train(_params(objective), lgb.Dataset(X, label=y), num_boost_round=25)
    ir = parse_model(booster)
    assert ir.link is (Link.SIGMOID if objective == "binary" else Link.IDENTITY)
    assert_conformance(ir, X, booster.predict)


def test_booster_conformance_without_nans_missing_type_none() -> None:
    """Training without NaNs yields missing_type 'None': NaN must route as 0.0."""
    X, y, _ = make_synthetic(seed=6, nan_frac=0.0)
    booster = lgb.train(_params("binary"), lgb.Dataset(X, label=y), num_boost_round=20)
    ir = parse_model(booster)
    assert_conformance(ir, X, booster.predict)  # probe matrix includes NaN patterns


def test_sklearn_wrapper_and_dump_dict() -> None:
    X, y, _ = make_synthetic(seed=8)
    clf = lgb.LGBMClassifier(n_estimators=15, num_leaves=7, random_state=0, verbose=-1)
    clf.fit(X, y)
    ir_wrapper = parse_model(clf)
    ir_dump = parse_model(clf.booster_.dump_model())
    assert ir_wrapper == ir_dump
    assert_conformance(ir_wrapper, X, lambda A: clf.predict_proba(A)[:, 1], n_random=2000)


def test_multiclass_raises() -> None:
    X, _, _ = make_synthetic(seed=9, nan_frac=0.0)
    rng = np.random.default_rng(0)
    y3 = rng.integers(0, 3, size=len(X))
    clf = lgb.LGBMClassifier(n_estimators=5, num_leaves=7, verbose=-1)
    clf.fit(X, y3)
    with pytest.raises(UnsupportedModelError, match="multi"):
        parse_model(clf)


def test_categorical_split_raises() -> None:
    X, y, _ = make_synthetic(seed=10, nan_frac=0.0)
    Xc = X.copy()
    Xc[:, 1] = np.floor(np.abs(Xc[:, 1])) % 4  # small-cardinality integer column
    booster = lgb.train(
        {**_params("binary"), "min_data_per_group": 1},
        lgb.Dataset(Xc, label=y, categorical_feature=[1], free_raw_data=False),
        num_boost_round=10,
    )
    with pytest.raises(UnsupportedModelError, match="categorical"):
        parse_model(booster)
