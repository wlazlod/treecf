"""Flat-array IR serialization: the cross-language boundary contract (migration P1)."""

from __future__ import annotations

import numpy as np
import pytest

from treecf.ir.evaluate import raw_score_batch
from treecf.ir.flatten import flatten_ir, unflatten_ir

from ..conftest import make_random_ir


@pytest.mark.parametrize("seed", range(5))
def test_round_trip_preserves_batch_scores(seed: int) -> None:
    rng = np.random.default_rng(seed)
    ir = make_random_ir(rng, n_features=4, n_trees=5, depth=3)
    flat = flatten_ir(ir)
    back = unflatten_ir(flat)
    X = rng.normal(scale=3.0, size=(300, 4))
    X[rng.random(X.shape) < 0.2] = np.nan
    np.testing.assert_array_equal(raw_score_batch(ir, X), raw_score_batch(back, X))


def test_flat_arrays_have_the_contract_dtypes() -> None:
    rng = np.random.default_rng(0)
    ir = make_random_ir(rng, n_features=3, n_trees=2, depth=2)
    flat = flatten_ir(ir)
    assert flat["feature"].dtype == np.int32
    assert flat["threshold"].dtype == np.float64
    assert flat["value"].dtype == np.float64
    assert flat["is_lt"].dtype == np.uint8
    assert flat["missing_left"].dtype == np.uint8
    assert flat["left"].dtype == np.uint32 and flat["right"].dtype == np.uint32
    assert flat["tree_roots"].dtype == np.uint32
    assert flat["link"] in ("identity", "sigmoid")
    n_nodes = len(flat["feature"])
    assert all(len(flat[k]) == n_nodes for k in ("threshold", "value", "is_lt", "missing_left"))
    # child indices are GLOBAL (tree offsets applied) and in range for internal nodes
    internal = flat["feature"] >= 0
    assert flat["left"][internal].max(initial=0) < n_nodes
    assert flat["right"][internal].max(initial=0) < n_nodes


def test_tree_roots_are_offsets_of_each_tree(  ) -> None:
    rng = np.random.default_rng(1)
    ir = make_random_ir(rng, n_features=3, n_trees=4, depth=3)
    flat = flatten_ir(ir)
    sizes = [len(t.nodes) for t in ir.trees]
    expected_roots = np.cumsum([0, *sizes[:-1]]).astype(np.uint32)
    np.testing.assert_array_equal(flat["tree_roots"], expected_roots)
