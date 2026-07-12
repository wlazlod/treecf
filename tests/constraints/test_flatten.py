"""Flat-array constraint serialization for the cross-language boundary (migration P1)."""

from __future__ import annotations

import numpy as np

from treecf.constraints import (
    AllowMissing,
    Equals,
    Freeze,
    Implies,
    Linear,
    Monotone,
    OneHot,
    Range,
    compile_constraints,
    constraint,
)
from treecf.constraints.flatten import flatten_constraints

NAMES = ("a", "b", "c", "f1", "f2", "f3")


def test_mixed_set_flattens_with_order_preserved() -> None:
    compiled = compile_constraints(
        [
            Freeze("a"),
            Range("b", -1.0, 5.0),
            Monotone("c", "decrease"),
            Equals("f1", 1.0),
            constraint("b <= c"),
            Linear({"a": 2.0, "c": -0.5}, op=">=", rhs=1.5, missing_policy="forbid_missing"),
            Implies(Equals("f2", 1.0), Equals("f3", 1.0)),
            OneHot(("f1", "f2", "f3")),
            AllowMissing("b", delta_miss=0.5, delta_from_miss=2.0),
        ],
        NAMES,
    )
    flat = flatten_constraints(compiled)

    np.testing.assert_array_equal(flat["freeze"], [0])
    np.testing.assert_array_equal(flat["range_idx"], [1])
    np.testing.assert_array_equal(flat["range_lo"], [-1.0])
    np.testing.assert_array_equal(flat["equals_idx"], [3])
    np.testing.assert_array_equal(flat["mono_idx"], [2])
    np.testing.assert_array_equal(flat["mono_dir"], [-1])

    # two linears, in declaration order, CSR-encoded
    np.testing.assert_array_equal(flat["lin_offsets"], [0, 2, 4])
    np.testing.assert_array_equal(flat["lin_indices"], [1, 2, 0, 2])
    np.testing.assert_array_equal(flat["lin_coefs"], [1.0, -1.0, 2.0, -0.5])
    np.testing.assert_array_equal(flat["lin_op"], [0, 1])  # 0 "<=", 1 ">=", 2 "=="
    np.testing.assert_array_equal(flat["lin_rhs"], [0.0, 1.5])
    np.testing.assert_array_equal(flat["lin_policy"], [0, 1])  # 0 satisfied, 1 forbid

    np.testing.assert_array_equal(flat["imp_cond_idx"], [4])
    np.testing.assert_array_equal(flat["imp_cons_idx"], [5])
    np.testing.assert_array_equal(flat["oh_offsets"], [0, 3])
    np.testing.assert_array_equal(flat["oh_indices"], [3, 4, 5])
    np.testing.assert_array_equal(flat["am_idx"], [1])
    np.testing.assert_array_equal(flat["am_to"], [0.5])
    np.testing.assert_array_equal(flat["am_from"], [2.0])
    assert flat["n_features"] == len(NAMES)


def test_empty_set_flattens_to_empty_arrays() -> None:
    flat = flatten_constraints(compile_constraints([], NAMES))
    assert len(flat["freeze"]) == 0
    np.testing.assert_array_equal(flat["lin_offsets"], [0])
    np.testing.assert_array_equal(flat["oh_offsets"], [0])
