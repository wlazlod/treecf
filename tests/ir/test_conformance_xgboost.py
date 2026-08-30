"""XGBoost parser conformance including the base_score gate."""

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
        ("binary:logistic", 0.2),  # explicit prob-space base_score
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
    """A saved JSON dump must parse identically to the live booster."""
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


def _categorical_training(seed: int, cardinalities: dict[int, int]) -> tuple:
    from ..conftest import make_synthetic

    X, y, _ = make_synthetic(seed=seed, nan_frac=0.1)
    rng = np.random.default_rng(seed)
    for j, k in cardinalities.items():
        codes = rng.integers(0, k, size=len(X)).astype(np.float64)
        codes[rng.random(len(X)) < 0.1] = np.nan
        X[:, j] = codes
        y = np.where(np.isnan(codes), y, (y + (codes % 2)) % 2)
    feature_types = ["c" if j in cardinalities else "q" for j in range(X.shape[1])]
    return X, y, feature_types


def _cat_dmatrix(A: np.ndarray, feature_types: list[str]) -> object:
    return xgb.DMatrix(
        A,
        feature_names=[f"f{i}" for i in range(A.shape[1])],
        feature_types=feature_types,
        enable_categorical=True,
    )


def test_native_categorical_conformance() -> None:
    X, y, feature_types = _categorical_training(seed=21, cardinalities={1: 5, 3: 9})
    dtrain = _cat_dmatrix(X, feature_types)
    dtrain.set_label(y)
    booster = xgb.train(
        {"objective": "binary:logistic", "max_depth": 4, "tree_method": "hist", "seed": 3},
        dtrain,
        num_boost_round=20,
    )
    ir = parse_model(booster)
    assert set(ir.categorical) == {1, 3}
    assert any(node.categories is not None for tree in ir.trees for node in tree.nodes)
    assert_conformance(
        ir, X, lambda A: booster.predict(_cat_dmatrix(A, feature_types))
    )  # probes include NaN and unseen codes


def test_sklearn_wrapper_categorical_with_pandas_names() -> None:
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(23)
    n = 700
    names = ["clerk", "manager", "nurse", "smith", "guard"]
    frame = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "occupation": pd.Categorical(rng.choice(names, size=n), categories=names),
        }
    )
    y = ((frame["occupation"].cat.codes % 2).to_numpy() ^ (frame["num"] > 0)).astype(int)
    clf = xgb.XGBClassifier(
        n_estimators=10, max_depth=3, enable_categorical=True, tree_method="hist"
    )
    clf.fit(frame, y)
    ir = parse_model(clf)
    assert 1 in ir.categorical
    X = np.column_stack(
        [frame["num"].to_numpy(), frame["occupation"].cat.codes.to_numpy(dtype=np.float64)]
    )
    types = ["q", "c"]

    def predict(A: np.ndarray) -> np.ndarray:
        dm = xgb.DMatrix(
            A, feature_names=["num", "occupation"], feature_types=types,
            enable_categorical=True,
        )
        return clf.get_booster().predict(dm)

    assert_conformance(ir, X, predict, n_random=2000)
