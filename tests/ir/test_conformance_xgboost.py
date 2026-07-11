"""XGBoost parser conformance (spec §3.3–§3.4) including the OQ1 base_score gate."""

from __future__ import annotations

import json

import numpy as np
import pytest

from treecf._errors import UnsupportedModelError
from treecf.ir.model import Link
from treecf.ir.parsers import parse_model

from ..conftest import make_synthetic
from .harness import assert_conformance

xgb = pytest.importorskip("xgboost")


def _train_booster(objective: str, base_score: float | None = None, seed: int = 7) -> object:
    X, y_bin, y_cont = make_synthetic(seed=seed)
    y = y_bin if objective == "binary:logistic" else y_cont
    params: dict[str, object] = {
        "objective": objective,
        "max_depth": 4,
        "eta": 0.3,
        "seed": seed,
    }
    if base_score is not None:
        params["base_score"] = base_score
    dtrain = xgb.DMatrix(X, label=y)
    return xgb.train(params, dtrain, num_boost_round=30)


@pytest.mark.parametrize(
    ("objective", "base_score"),
    [
        ("binary:logistic", None),
        ("binary:logistic", 0.2),  # OQ1: explicit prob-space base_score
        ("reg:squarederror", None),
        ("reg:squarederror", 1.5),
    ],
)
def test_booster_conformance(objective: str, base_score: float | None) -> None:
    booster = _train_booster(objective, base_score)
    X, _, _ = make_synthetic(seed=7)
    ir = parse_model(booster)
    expected_link = Link.SIGMOID if objective == "binary:logistic" else Link.IDENTITY
    assert ir.link is expected_link
    assert_conformance(ir, X, lambda A: booster.predict(xgb.DMatrix(A)))


def test_json_dump_roundtrip_conformance(tmp_path: object) -> None:
    """A saved JSON dump must parse identically to the live booster (D10)."""
    booster = _train_booster("binary:logistic")
    X, _, _ = make_synthetic(seed=7)
    path = f"{tmp_path}/model.json"
    booster.save_model(path)

    ir_from_path = parse_model(path)
    with open(path, encoding="utf-8") as fh:
        ir_from_dict = parse_model(json.load(fh))

    assert ir_from_path == ir_from_dict
    assert_conformance(ir_from_path, X, lambda A: booster.predict(xgb.DMatrix(A)), n_random=2000)


def test_sklearn_wrapper_unwraps() -> None:
    X, y_bin, _ = make_synthetic(seed=11, nan_frac=0.0)
    clf = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=0)
    clf.fit(X, y_bin)
    ir = parse_model(clf)
    assert ir.link is Link.SIGMOID
    assert_conformance(ir, X, lambda A: clf.predict_proba(A)[:, 1], n_random=2000)


def test_multiclass_raises() -> None:
    X, _, _ = make_synthetic(seed=3, nan_frac=0.0)
    rng = np.random.default_rng(3)
    y3 = rng.integers(0, 3, size=len(X))
    clf = xgb.XGBClassifier(n_estimators=5, max_depth=2, random_state=0)
    clf.fit(X, y3)
    with pytest.raises(UnsupportedModelError, match="multi"):
        parse_model(clf)
