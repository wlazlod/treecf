"""The packaged credit demo: a model, a background sample, one declined row."""

from __future__ import annotations

import math

import numpy as np

from treecf import Counterfactual, Explainer, Target
from treecf.datasets import OCCUPATIONS, credit_demo
from treecf.ir.evaluate import raw_score
from treecf.ir.model import EnsembleIR

TARGET = Target.probability(range=(0.0, 0.05))


def test_shapes_names_and_determinism() -> None:
    model, X, x = credit_demo()
    assert isinstance(model, EnsembleIR)
    assert model.feature_names == (
        "income", "utilization", "dpd_12m", "tenure_months", "occupation"
    )
    assert model.categorical[4].categories == OCCUPATIONS
    assert X.shape == (400, 5)
    np.testing.assert_array_equal(x, X[1])
    _, X_again, x_again = credit_demo()
    np.testing.assert_array_equal(X, X_again)
    np.testing.assert_array_equal(x, x_again)
    _, X_other, _ = credit_demo(seed=1)
    assert not np.array_equal(X, X_other)


def test_the_row_is_declined_and_gets_a_plan() -> None:
    model, X, x = credit_demo()
    prob = 1.0 / (1.0 + math.exp(-raw_score(model, x)))
    assert prob > 0.05
    exp = Explainer(model, background=X)
    res = exp.explain(x, target=TARGET, seed=0)
    assert isinstance(res, Counterfactual)
    assert res.changes


def test_background_size_is_adjustable() -> None:
    _, X, x = credit_demo(n=50)
    assert X.shape == (50, 5)
    np.testing.assert_array_equal(x, X[1])
