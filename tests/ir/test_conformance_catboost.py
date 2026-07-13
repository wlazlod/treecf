"""CatBoost parser conformance: oblivious trees expanded to binary IR trees."""

from __future__ import annotations

import numpy as np
import pytest

from treecf._errors import UnsupportedModelError
from treecf.ir.model import Link
from treecf.ir.parsers import parse_model

from ..conftest import make_synthetic
from .harness import assert_conformance

catboost = pytest.importorskip("catboost")


def _fit(objective: str, X: np.ndarray, y: np.ndarray) -> object:
    cls = catboost.CatBoostClassifier if objective == "Logloss" else catboost.CatBoostRegressor
    model = cls(
        iterations=20,
        depth=4,
        learning_rate=0.3,
        loss_function=objective,
        random_seed=7,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X, y)
    return model


def test_classifier_conformance_with_nans() -> None:
    X, y, _ = make_synthetic(seed=31)
    model = _fit("Logloss", X, y)
    ir = parse_model(model)
    assert ir.link is Link.SIGMOID
    assert_conformance(ir, X, lambda A: model.predict_proba(A)[:, 1])


def test_regressor_conformance() -> None:
    X, _, y = make_synthetic(seed=32)
    model = _fit("RMSE", X, y)
    ir = parse_model(model)
    assert ir.link is Link.IDENTITY
    assert_conformance(ir, X, model.predict)


def test_json_dump_matches_model(tmp_path: object) -> None:
    X, y, _ = make_synthetic(seed=33, nan_frac=0.0)
    model = _fit("Logloss", X, y)
    path = f"{tmp_path}/model.json"
    model.save_model(path, format="json")
    ir_from_path = parse_model(path)
    ir_from_model = parse_model(model)
    assert ir_from_path == ir_from_model


def test_multiclass_raises() -> None:
    X, _, _ = make_synthetic(seed=34, nan_frac=0.0)
    rng = np.random.default_rng(2)
    model = catboost.CatBoostClassifier(
        iterations=5, depth=3, verbose=False, allow_writing_files=False
    )
    model.fit(X, rng.integers(0, 3, size=len(X)))
    with pytest.raises(UnsupportedModelError, match="multi"):
        parse_model(model)
