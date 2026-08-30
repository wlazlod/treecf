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
        ac_idx=flat["ac_idx"],
        ac_offsets=flat["ac_offsets"],
        ac_words=flat["ac_words"],
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


def _proj_case(
    constraints: list[object],
    rows: list[list[float]],
    x: list[float] | None = None,
) -> None:
    p = 4
    compiled = compile_constraints(constraints, tuple(f"g{i}" for i in range(p)))  # type: ignore[arg-type]
    factual = np.zeros(p) if x is None else np.array(x, dtype=np.float64)
    X = np.array(rows, dtype=np.float64)
    assert_check_and_repair_match(compiled, X, factual)


class TestProjectionRepairBitwise:
    """Every branch of the halfspace projection must match bit-for-bit."""

    def test_le_ge_eq_violations(self) -> None:
        _proj_case(
            [
                Linear({"g0": 1.0, "g1": 1.0}, op=">=", rhs=100.0),
                Linear({"g0": 2.0, "g2": 3.0}, op="<=", rhs=-50.0),
                Linear({"g1": 1.0, "g3": 2.0}, op="==", rhs=7.0),
            ],
            [[0.0, 0.0, 0.0, 0.0], [1.5, -2.5, 3.25, 0.125], [200.0, 0.0, 0.0, 0.0]],
        )

    def test_scaled_pair_projects_instead_of_min_clip(self) -> None:
        # 2a - 2b <= 0 is NOT the canonical order pair: projection applies
        _proj_case(
            [Linear({"g0": 2.0, "g1": -2.0}, op="<=", rhs=0.0)],
            [[5.0, 1.0, 0.0, 0.0], [1.0, 5.0, 0.0, 0.0]],
        )

    def test_canonical_pair_still_min_clips(self) -> None:
        _proj_case(
            [Linear({"g0": 1.0, "g1": -1.0}, op="<=", rhs=0.0)],
            [[5.0, 1.0, 0.0, 0.0], [np.nan, 1.0, 0.0, 0.0]],
        )

    def test_frozen_coordinate_not_moved(self) -> None:
        _proj_case(
            [Freeze("g0"), Linear({"g0": 1.0, "g1": 1.0}, op=">=", rhs=100.0)],
            [[2.0, 3.0, 0.0, 0.0]],
            x=[2.0, 3.0, 0.0, 0.0],
        )

    def test_frozen_nan_factual(self) -> None:
        # Freeze on a NaN factual leaves bounds open; the NaN forcing plus the
        # frozen mask must exclude the coordinate identically on both sides
        _proj_case(
            [
                Freeze("g0"),
                AllowMissing("g1", delta_miss=1.0),
                Linear({"g0": 1.0, "g1": 1.0}, op=">=", rhs=100.0),
            ],
            [[np.nan, 3.0, 0.0, 0.0], [1.0, 3.0, 0.0, 0.0]],
            x=[np.nan, 3.0, 0.0, 0.0],
        )

    def test_pinned_coordinate_not_moved(self) -> None:
        _proj_case(
            [Equals("g0", 1.0), Linear({"g0": 1.0, "g1": 1.0}, op=">=", rhs=100.0)],
            [[1.0, 0.0, 0.0, 0.0]],
        )

    def test_nan_skip(self) -> None:
        # AllowMissing keeps the NaN through the per-feature stage, so the
        # linear stage's NaN skip is what gets exercised
        _proj_case(
            [
                AllowMissing("g0", delta_miss=1.0),
                Linear({"g0": 1.0, "g1": 1.0}, op=">=", rhs=100.0),
            ],
            [[np.nan, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        )

    def test_denom_zero_all_frozen(self) -> None:
        _proj_case(
            [
                Freeze("g0"),
                Freeze("g1"),
                Linear({"g0": 1.0, "g1": 1.0}, op=">=", rhs=100.0),
            ],
            [[1.0, 1.0, 0.0, 0.0]],
            x=[1.0, 1.0, 0.0, 0.0],
        )

    def test_clip_then_reproject_across_rounds(self) -> None:
        # projection pushes g0 past its Range; the clip re-violates the linear
        # and later rounds shift the remaining correction onto g1
        _proj_case(
            [Range("g0", -5.0, 5.0), Linear({"g0": 1.0, "g1": 1.0}, op=">=", rhs=100.0)],
            [[0.0, 0.0, 0.0, 0.0]],
        )

    def test_interacting_linears(self) -> None:
        _proj_case(
            [
                Linear({"g0": 1.0, "g1": 1.0}, op=">=", rhs=10.0),
                Linear({"g0": 1.0, "g2": -2.0}, op="<=", rhs=-3.0),
            ],
            [[0.0, 0.0, 0.0, 0.0], [5.0, 5.0, 5.0, 5.0]],
        )

    def test_derived_bound_from_single_feature_linear_clips(self) -> None:
        _proj_case(
            [Linear({"g0": 1.0}, op=">=", rhs=100.0)],
            [[2.12, 0.0, 0.0, 0.0], [150.0, 0.0, 0.0, 0.0]],
        )


class TestAllowedCategoriesConformance:
    """Membership checks and the smallest-allowed repair rule match bitwise."""

    @staticmethod
    def _compiled(allowed: tuple[int, ...]) -> CompiledConstraints:
        from treecf.constraints import AllowedCategories
        from treecf.ir.model import CategoricalFeature

        return compile_constraints(
            [AllowedCategories(NAMES[1], allowed)],
            NAMES,
            {1: CategoricalFeature(cardinality=70)},
        )

    @pytest.mark.parametrize("seed", range(5))
    def test_check_and_repair_bitwise(self, seed: int) -> None:
        compiled = self._compiled((1, 3, 65))  # a member beyond one bitset word
        rng = np.random.default_rng(seed)
        X = rng.normal(scale=3.0, size=(200, len(NAMES)))
        X[:, 1] = rng.integers(0, 72, size=200).astype(np.float64)
        X[rng.random(200) < 0.2, 1] = 1.5  # non-integral pollution
        x = np.zeros(len(NAMES))
        x[1] = 3.0
        assert_check_and_repair_match(compiled, X, x)

    def test_empty_allowed_set_matches(self) -> None:
        compiled = self._compiled(())
        X = np.zeros((3, len(NAMES)))
        X[:, 1] = [0.0, 2.0, 69.0]
        x = np.zeros(len(NAMES))
        assert_check_and_repair_match(compiled, X, x)
