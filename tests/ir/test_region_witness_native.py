"""Maximal-region witnesses re-scored through the training library itself.

A witness is a point one routing cell past a proved region side that the
IR says leaves the target. The IR is a parity-exact view of the native
model, so the library's own raw margin at that point must leave the
target too — up to the float32 routing tolerance the conformance harness
already grants, which is why witnesses within that tolerance of the
boundary are skipped rather than asserted.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from treecf import Counterfactual, Explainer, Target, TreecfWarning
from treecf.ir.conformance import parity_tolerance
from treecf.ir.parsers import parse_model

from ..conftest import make_synthetic


def _xgboost_model(seed: int) -> tuple[object, np.ndarray]:
    xgb = pytest.importorskip("xgboost")
    X, y_bin, _ = make_synthetic(n=800, p=6, seed=seed, nan_frac=0.0)
    params = {"objective": "binary:logistic", "max_depth": 3, "eta": 0.3, "seed": seed}
    booster = xgb.train(params, xgb.DMatrix(X, label=y_bin), num_boost_round=12)

    def margin(row: np.ndarray) -> float:
        return float(booster.predict(xgb.DMatrix(row[None, :]), output_margin=True)[0])

    return (booster, margin), X


def _lightgbm_model(seed: int) -> tuple[object, np.ndarray]:
    lgb = pytest.importorskip("lightgbm")
    X, y_bin, _ = make_synthetic(n=800, p=6, seed=seed, nan_frac=0.0)
    params = {
        "objective": "binary", "num_leaves": 8, "learning_rate": 0.3, "seed": seed,
        "verbose": -1, "min_data_in_leaf": 5,
    }
    booster = lgb.train(params, lgb.Dataset(X, label=y_bin), num_boost_round=12)

    def margin(row: np.ndarray) -> float:
        return float(booster.predict(row[None, :], raw_score=True)[0])

    return (booster, margin), X


@pytest.mark.parametrize("library", ["xgboost", "lightgbm"])
def test_witnesses_leave_the_target_under_the_native_model(library: str) -> None:
    build = _xgboost_model if library == "xgboost" else _lightgbm_model
    (booster, margin), X = build(seed=11)
    ir = parse_model(booster)
    exp = Explainer(ir, normalizers=X.std(axis=0) + 1e-9)
    target = Target.probability(range=(0.6, 0.95))
    lo_t, hi_t = target.raw_interval(ir.link)
    tolerance = parity_tolerance(ir)

    checked = 0
    for row in X[:12]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", TreecfWarning)
            result = exp.explain(row, target, backend="exact", seed=0, time_budget_s=5.0)
        if not isinstance(result, Counterfactual):
            continue
        region = exp.recourse_region(
            row, result.x_cf, target, mode="maximal", budget=5_000, keep_witnesses=True
        )
        assert region.witnesses is not None
        for key, point in region.witnesses.items():
            assert not region.contains(point), key
            native = margin(point)
            if abs(native - lo_t) <= tolerance or abs(native - hi_t) <= tolerance:
                continue  # inside the float32 routing tolerance: no verdict either way
            assert native < lo_t or native > hi_t, f"{key}: native margin {native} in target"
            checked += 1
    assert checked > 0, "no witness was decisive enough to check"
