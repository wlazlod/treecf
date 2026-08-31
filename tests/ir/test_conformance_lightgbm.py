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


def _categorical_training(
    seed: int, cardinalities: dict[int, int], nan_frac: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    X, y, _ = make_synthetic(seed=seed, nan_frac=nan_frac)
    rng = np.random.default_rng(seed)
    for j, k in cardinalities.items():
        codes = rng.integers(0, k, size=len(X)).astype(np.float64)
        if nan_frac > 0:
            codes[rng.random(len(X)) < nan_frac] = np.nan
        X[:, j] = codes
        # give the codes signal so trees actually split on them
        y = np.where(np.isnan(codes), y, (y + (codes % 2)) % 2)
    return X, y


def test_native_categorical_conformance_with_nans() -> None:
    X, y = _categorical_training(seed=11, cardinalities={1: 4, 3: 9})
    booster = lgb.train(
        {**_params("binary"), "min_data_per_group": 1},
        lgb.Dataset(X, label=y, categorical_feature=[1, 3], free_raw_data=False),
        num_boost_round=25,
    )
    ir = parse_model(booster)
    assert set(ir.categorical) == {1, 3}
    assert ir.categorical[1].cardinality >= 4
    assert any(
        node.categories is not None for tree in ir.trees for node in tree.nodes
    )
    assert_conformance(ir, X, booster.predict)  # probes include NaN and unseen codes


def test_native_categorical_conformance_without_nans() -> None:
    X, y = _categorical_training(seed=12, cardinalities={1: 6}, nan_frac=0.0)
    booster = lgb.train(
        {**_params("binary"), "min_data_per_group": 1},
        lgb.Dataset(X, label=y, categorical_feature=[1], free_raw_data=False),
        num_boost_round=20,
    )
    ir = parse_model(booster)
    assert_conformance(ir, X, booster.predict)


def test_pandas_categorical_names_are_recovered() -> None:
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(13)
    n = 600
    names = ["clerk", "manager", "nurse", "smith"]
    frame = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "occupation": pd.Categorical(rng.choice(names, size=n), categories=names),
        }
    )
    y = ((frame["occupation"].cat.codes % 2).to_numpy() ^ (frame["num"] > 0)).astype(int)
    clf = lgb.LGBMClassifier(
        n_estimators=10, num_leaves=7, min_child_samples=5, random_state=0, verbose=-1
    )
    clf.fit(frame, y)
    ir = parse_model(clf)
    assert ir.categorical[1].categories == tuple(names)
    X = np.column_stack(
        [frame["num"].to_numpy(), frame["occupation"].cat.codes.to_numpy(dtype=np.float64)]
    )
    assert_conformance(ir, X, clf.booster_.predict, n_random=2000)


def test_declared_cardinality_beyond_training_codes() -> None:
    """Codes the model never saw but the caller declares valid route out-of-set."""
    from treecf.ir.parsers.lightgbm import parse_lightgbm

    X, y = _categorical_training(seed=14, cardinalities={1: 4}, nan_frac=0.0)
    booster = lgb.train(
        {**_params("binary"), "min_data_per_group": 1},
        lgb.Dataset(X, label=y, categorical_feature=[1], free_raw_data=False),
        num_boost_round=15,
    )
    name = booster.feature_name()[1]
    ir = parse_lightgbm(booster, categories={name: [f"c{i}" for i in range(6)]})
    assert ir.categorical[1].cardinality == 6
    assert_conformance(ir, X, booster.predict)  # probes now include codes 4 and 5
