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


NAMES = ["alpha", "beta", "gamma", "delta", "eps", "zeta", "eta", "theta", "iota"]


def _categorical_frame(seed: int, k: int, n: int = 700):
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "occ": rng.choice(NAMES[:k], size=n),
        }
    )
    prob = np.linspace(0.1, 0.9, k)
    lookup = dict(zip(NAMES[:k], prob, strict=True))
    y = (rng.random(n) < np.array([lookup[c] for c in frame["occ"]])).astype(int)
    y = y ^ (frame["num"].to_numpy() > 0.5)
    return frame, y.astype(int)


def _codes_matrix(frame) -> np.ndarray:
    code_of = {name: float(i) for i, name in enumerate(NAMES)}
    return np.column_stack(
        [frame["num"].to_numpy(), np.array([code_of[c] for c in frame["occ"]])]
    )


def _native_predict(model, k: int):
    pd = pytest.importorskip("pandas")

    def predict(A: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(
            {"num": A[:, 0], "occ": [NAMES[int(c)] for c in A[:, 1]]}
        )
        return model.predict(frame, prediction_type="Probability")[:, 1]

    return predict


def _fit_cat(frame, y, **kwargs):
    from catboost import CatBoostClassifier, Pool

    pool = Pool(frame, label=y, cat_features=["occ"])
    model = CatBoostClassifier(
        iterations=15, depth=4, verbose=False, random_seed=1, boosting_type="Plain",
        **kwargs,
    )
    model.fit(pool)
    return model


@pytest.mark.parametrize("k", [4, 9])
def test_one_hot_categorical_conformance(k: int) -> None:
    from treecf.ir.parsers.catboost import parse_catboost

    frame, y = _categorical_frame(seed=41 + k, k=k)
    model = _fit_cat(frame, y, one_hot_max_size=255)
    ir = parse_catboost(model, categories={"occ": NAMES[:k]})
    assert ir.categorical and ir.link is Link.SIGMOID
    X = _codes_matrix(frame)
    assert_conformance(ir, X, _native_predict(model, k), include_nan=False)


@pytest.mark.parametrize("k", [4, 9])
def test_single_feature_statistics_conformance(k: int) -> None:
    from treecf.ir.parsers.catboost import parse_catboost

    frame, y = _categorical_frame(seed=51 + k, k=k)
    model = _fit_cat(frame, y, one_hot_max_size=2, max_ctr_complexity=1)
    ir = parse_catboost(model, categories={"occ": NAMES[:k]})
    assert any(node.categories is not None for tree in ir.trees for node in tree.nodes)
    X = _codes_matrix(frame)
    assert_conformance(ir, X, _native_predict(model, k), include_nan=False)


def test_unseen_codes_take_the_prior_route() -> None:
    """Codes declared but never trained on get the prior-only statistic."""
    from treecf.ir.parsers.catboost import parse_catboost

    frame, y = _categorical_frame(seed=61, k=4)
    model = _fit_cat(frame, y, one_hot_max_size=2, max_ctr_complexity=1)
    ir = parse_catboost(model, categories={"occ": NAMES[:6]})  # 2 unseen codes
    assert ir.categorical[1].cardinality == 6
    X = _codes_matrix(frame)
    assert_conformance(ir, X, _native_predict(model, 6), include_nan=False)


def test_hashes_match_the_model_table() -> None:
    """Our category hashing reproduces every hash the model stored."""
    import json as json_mod
    import tempfile as tempfile_mod

    from treecf.ir.parsers._catboost_cat import cat_feature_hash, signed32

    frame, y = _categorical_frame(seed=71, k=9)
    model = _fit_cat(frame, y, one_hot_max_size=255)
    with tempfile_mod.TemporaryDirectory() as d:
        path = f"{d}/m.json"
        model.save_model(path, format="json")
        with open(path, encoding="utf-8") as fh:
            dump = json_mod.load(fh)
    stored = {
        int(v)
        for f in dump["features_info"]["categorical_features"]
        for v in (f.get("values") or [])
    }
    ours = {signed32(cat_feature_hash(name)) for name in NAMES}
    assert stored and stored <= ours


def test_missing_categories_argument_raises_the_documented_error() -> None:
    from treecf._errors import ParserError

    frame, y = _categorical_frame(seed=81, k=4)
    model = _fit_cat(frame, y, one_hot_max_size=255)
    with pytest.raises(
        ParserError,
        match="categories is a required Explainer argument for CatBoost models "
        "with native categorical features",
    ):
        parse_model(model)


def test_combination_statistics_are_rejected_with_the_recipe() -> None:
    """A statistics table whose key spans several features cannot be lowered."""
    import json as json_mod
    import tempfile as tempfile_mod

    from treecf._errors import ParserError
    from treecf.ir.parsers.catboost import parse_catboost_dump

    frame, y = _categorical_frame(seed=91, k=4)
    model = _fit_cat(frame, y, one_hot_max_size=2, max_ctr_complexity=1)
    with tempfile_mod.TemporaryDirectory() as d:
        path = f"{d}/m.json"
        model.save_model(path, format="json")
        with open(path, encoding="utf-8") as fh:
            dump = json_mod.load(fh)
    assert dump["features_info"]["ctrs"], "the model must carry statistics tables"
    # a combination key: the statistic is computed over (occ, num>t) jointly,
    # which breaks per-feature routing atomicity
    dump["features_info"]["ctrs"][0]["elements"].append(
        {"float_feature_index": 0, "border": 0.0, "combination_element": "float_feature"}
    )
    with pytest.raises(ParserError, match="max_ctr_complexity=1"):
        parse_catboost_dump(dump, categories={"occ": NAMES[:4]})
