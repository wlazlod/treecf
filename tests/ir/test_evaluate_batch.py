"""Vectorized population scorer must agree with the reference row evaluator."""

from __future__ import annotations

import numpy as np
import pytest

from treecf.ir.evaluate import (
    prepare_tree_arrays,
    raw_score,
    raw_score_batch,
    raw_score_batch_prepared,
)

from ..conftest import make_random_ir


@pytest.mark.parametrize("seed", range(5))
def test_batch_matches_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    ir = make_random_ir(rng, n_features=4, n_trees=5, depth=3)
    X = rng.normal(scale=3.0, size=(200, 4))
    X[rng.random(X.shape) < 0.15] = np.nan
    batch = raw_score_batch(ir, X)
    reference = np.array([raw_score(ir, row) for row in X])
    np.testing.assert_allclose(batch, reference, rtol=0, atol=1e-12)


def test_prepared_arrays_are_reusable_and_bitwise_equal() -> None:
    rng = np.random.default_rng(11)
    ir = make_random_ir(rng, n_features=4, n_trees=5, depth=3)
    X = rng.normal(scale=3.0, size=(50, 4))
    X[rng.random(X.shape) < 0.15] = np.nan
    prepared = prepare_tree_arrays(ir)
    once = raw_score_batch_prepared(prepared, ir.base_score, X)
    twice = raw_score_batch_prepared(prepared, ir.base_score, X)
    np.testing.assert_array_equal(once, raw_score_batch(ir, X))
    np.testing.assert_array_equal(once, twice)
    reference = np.array([raw_score(ir, row) for row in X])
    np.testing.assert_array_equal(once, reference)
