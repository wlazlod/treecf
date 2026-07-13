"""Rust constraint check/repair must be BITWISE-equal to Python."""

from __future__ import annotations

import numpy as np
import pytest

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
)
from treecf.constraints.compile import CompiledConstraints
from treecf.constraints.flatten import flatten_constraints

from ..parity.harness import load_scenario, scenario_paths

pytestmark = pytest.mark.rust

NAMES = ("a", "b", "c", "f1", "f2", "f3")


def rust_constraints(compiled: CompiledConstraints) -> object:
    from treecf.backends.genetic_rust import _core as _load_core
    _treecf_core = _load_core()

    flat = flatten_constraints(compiled)
    return _treecf_core.RustConstraints(
        flat["n_features"],
        flat["freeze"],
        flat["range_idx"],
        flat["range_lo"],
        flat["range_hi"],
        flat["equals_idx"],
        flat["equals_val"],
        flat["mono_idx"],
        flat["mono_dir"],
        flat["lin_offsets"],
        flat["lin_indices"],
        flat["lin_coefs"],
        flat["lin_op"],
        flat["lin_rhs"],
        flat["lin_policy"],
        flat["imp_cond_idx"],
        flat["imp_cond_val"],
        flat["imp_cons_idx"],
        flat["imp_cons_val"],
        flat["oh_offsets"],
        flat["oh_indices"],
        flat["am_idx"],
        flat["am_to"],
        flat["am_from"],
    )


def assert_check_and_repair_match(
    compiled: CompiledConstraints, X: np.ndarray, x: np.ndarray
) -> None:
    rust = rust_constraints(compiled)
    X = np.ascontiguousarray(X, dtype=np.float64)
    x = np.ascontiguousarray(x, dtype=np.float64)
    np.testing.assert_array_equal(
        np.asarray(rust.check(X, x)),  # type: ignore[attr-defined]
        compiled.check_matrix(X, x),
    )
    np.testing.assert_array_equal(
        np.asarray(rust.repair(X, x)),  # type: ignore[attr-defined]
        compiled.repair_matrix(X, x),
    )


def _synthetic_sets() -> list[CompiledConstraints]:
    return [
        compile_constraints([], NAMES),
        compile_constraints(
            [Freeze("a"), Range("b", -1.0, 2.5), Monotone("c", "increase")], NAMES
        ),
        compile_constraints(
            [
                Linear({"a": 1.0, "b": -1.0}, op="<=", rhs=0.0),
                Linear({"b": 2.0, "c": 1.0}, op=">=", rhs=-3.0, missing_policy="forbid_missing"),
                Linear({"a": 1.0, "c": 1.0}, op="==", rhs=1.0),
                AllowMissing("a", delta_miss=0.5),
                AllowMissing("c", delta_miss=1.0, delta_from_miss=0.25),
            ],
            NAMES,
        ),
        compile_constraints(
            [
                Equals("f1", 1.0),
                Implies(Equals("f2", 1.0), Equals("f3", 1.0)),
                OneHot(("f1", "f2", "f3")),
            ],
            NAMES,
        ),
        compile_constraints(
            [
                Freeze("a"),
                Monotone("b", "decrease"),
                Range("c", 0.0, 1.0),
                Linear({"b": 1.0, "c": -1.0}, op="<=", rhs=0.0),
                OneHot(("f1", "f2")),
                Implies(Equals("f3", 0.0), Equals("f1", 1.0)),
                AllowMissing("b", delta_miss=0.7),
            ],
            NAMES,
        ),
    ]


@pytest.mark.parametrize("set_idx", range(5))
@pytest.mark.parametrize("seed", range(6))
def test_synthetic_sets_fuzz_bitwise(set_idx: int, seed: int) -> None:
    compiled = _synthetic_sets()[set_idx]
    rng = np.random.default_rng(100 * set_idx + seed)
    p = len(NAMES)
    x = rng.normal(scale=2.0, size=p)
    if seed % 3 == 1:
        x[int(rng.integers(0, p))] = np.nan
    x[3:] = rng.integers(0, 2, size=3).astype(float)  # binary flags region
    X = rng.normal(scale=3.0, size=(300, p))
    X[:, 3:] = rng.integers(0, 2, size=(300, 3)).astype(float)
    X[rng.random(X.shape) < 0.15] = np.nan
    # exact boundary values sprinkled in (clip/tie-break edges)
    X[0, :3] = [0.0, -0.0, 1.0]
    X[1, 3:] = [0.7, 0.7, np.nan]
    assert_check_and_repair_match(compiled, X, x)


@pytest.mark.parametrize("path", scenario_paths(), ids=[p.stem for p in scenario_paths()])
def test_fixture_constraint_sets_bitwise(path: object) -> None:
    scenario = load_scenario(path)  # type: ignore[arg-type]
    rng = np.random.default_rng(7)
    p = scenario.ir.n_features
    X = rng.normal(scale=3.0, size=(400, p))
    X[rng.random(X.shape) < 0.2] = np.nan
    assert_check_and_repair_match(scenario.compiled, X, scenario.x)
