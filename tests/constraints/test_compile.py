"""Compilation and validation of the full M2 constraint set."""

import dataclasses
import math

import numpy as np
import pytest

from treecf._errors import ConstraintValidationError
from treecf.constraints import (
    AllowedCategories,
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

NAMES = ("a", "b", "c", "flag1", "flag2", "flag3")


class TestValidation:
    def test_linear_unknown_feature(self) -> None:
        with pytest.raises(ConstraintValidationError, match="nope"):
            compile_constraints([constraint("a <= nope")], NAMES)

    def test_linear_bad_missing_policy(self) -> None:
        bad = Linear(coefficients={"a": 1.0}, op="<=", rhs=0.0, missing_policy="whatever")
        with pytest.raises(ConstraintValidationError, match="missing_policy"):
            compile_constraints([bad], NAMES)

    def test_linear_bad_op(self) -> None:
        with pytest.raises(ConstraintValidationError, match="op"):
            compile_constraints([Linear(coefficients={"a": 1.0}, op="<", rhs=0.0)], NAMES)

    def test_onehot_needs_two_known_features(self) -> None:
        with pytest.raises(ConstraintValidationError, match="OneHot"):
            compile_constraints([OneHot(("flag1",))], NAMES)
        with pytest.raises(ConstraintValidationError, match="ghost"):
            compile_constraints([OneHot(("flag1", "ghost"))], NAMES)

    def test_overlapping_onehot_groups_rejected(self) -> None:
        with pytest.raises(ConstraintValidationError, match="overlap"):
            compile_constraints(
                [OneHot(("flag1", "flag2")), OneHot(("flag2", "flag3"))], NAMES
            )

    def test_implies_requires_binary_values(self) -> None:
        with pytest.raises(ConstraintValidationError, match="binary"):
            compile_constraints(
                [Implies(Equals("flag1", 2.0), Equals("flag2", 0.0))], NAMES
            )

    def test_allow_missing_positive_delta(self) -> None:
        with pytest.raises(ConstraintValidationError, match="delta"):
            compile_constraints([AllowMissing("a", delta_miss=-1.0)], NAMES)

    def test_allow_missing_on_frozen_feature_rejected(self) -> None:
        with pytest.raises(ConstraintValidationError, match="frozen"):
            compile_constraints([Freeze("a"), AllowMissing("a", delta_miss=1.0)], NAMES)


class TestStructuredAccess:
    def test_groups_are_exposed(self) -> None:
        compiled = compile_constraints(
            [
                constraint("a <= b"),
                Implies(Equals("flag1", 1.0), Equals("flag2", 1.0)),
                OneHot(("flag1", "flag2", "flag3")),
                AllowMissing("c", delta_miss=2.0),
            ],
            NAMES,
        )
        assert len(compiled.linears) == 1
        assert compiled.linears[0].coefficients == {"a": 1.0, "b": -1.0}
        assert len(compiled.implications) == 1
        assert compiled.onehot_groups == ((3, 4, 5),)
        assert compiled.allow_missing == {2: (2.0, 2.0)}

    def test_equals_pins_bounds(self) -> None:
        compiled = compile_constraints([Equals("flag1", 1.0)], NAMES)
        lo, hi, _ = compiled.instance_bounds(np.zeros(len(NAMES)))
        assert lo[3] == hi[3] == 1.0


class TestDerivedRanges:
    # Derived bounds are widened by the check_matrix slack (1e-9) translated
    # into feature space (1e-9 / |coef|), so they never exclude a candidate
    # the slacked Linear check itself admits. See TestDerivedBoundSlack below
    # for exact-float-equality coverage of the widening itself.

    def test_ge_derives_lower_bound(self) -> None:
        compiled = compile_constraints([constraint("a >= 100")], NAMES)
        assert compiled.derived_ranges == (Range("a", 100.0 - 1e-9, math.inf),)
        assert len(compiled.linears) == 1  # original Linear retained

    def test_le_derives_upper_bound(self) -> None:
        compiled = compile_constraints([constraint("2*a <= 10")], NAMES)
        assert compiled.derived_ranges == (Range("a", -math.inf, 5.0 + 5e-10),)

    def test_negative_coef_flips_inequality(self) -> None:
        compiled = compile_constraints([Linear({"a": -2.0}, "<=", -10.0)], NAMES)
        assert compiled.derived_ranges == (Range("a", 5.0 - 5e-10, math.inf),)

    def test_negative_coef_ge_gives_upper_bound(self) -> None:
        compiled = compile_constraints([Linear({"a": -2.0}, ">=", -10.0)], NAMES)
        assert compiled.derived_ranges == (Range("a", -math.inf, 5.0 + 5e-10),)

    def test_eq_derives_pin(self) -> None:
        compiled = compile_constraints([Linear({"a": 2.0}, "==", 7.0)], NAMES)
        (rng,) = compiled.derived_ranges
        assert rng.lo == 3.5 - 5e-10
        assert rng.hi == 3.5 + 5e-10

    @pytest.mark.parametrize(("op", "rhs"), [("<=", 3.0), ("<=", 0.0), (">=", -1.0), ("==", 0.0)])
    def test_zero_coef_vacuous_is_dropped(self, op: str, rhs: float) -> None:
        compiled = compile_constraints([Linear({"a": 0.0}, op, rhs)], NAMES)
        assert compiled.linears == ()
        assert compiled.derived_ranges == ()

    @pytest.mark.parametrize(("op", "rhs"), [(">=", 3.0), ("<=", -1.0), ("==", 3.0)])
    def test_zero_coef_unsatisfiable_raises(self, op: str, rhs: float) -> None:
        with pytest.raises(ConstraintValidationError, match="unsatisfiable"):
            compile_constraints([Linear({"a": 0.0}, op, rhs)], NAMES)

    def test_zero_coef_unknown_feature_still_rejected(self) -> None:
        with pytest.raises(ConstraintValidationError, match="ghost"):
            compile_constraints([Linear({"ghost": 0.0}, "<=", 3.0)], NAMES)

    def test_multi_feature_linear_derives_nothing(self) -> None:
        compiled = compile_constraints([constraint("a + b >= 100")], NAMES)
        assert compiled.derived_ranges == ()

    def test_derived_bound_reaches_instance_bounds(self) -> None:
        compiled = compile_constraints([constraint("a >= 100")], NAMES)
        lo, hi, _ = compiled.instance_bounds(np.zeros(len(NAMES)))
        assert lo[0] == 100.0 - 1e-9
        assert hi[0] == math.inf


class TestDerivedBoundSlack:
    """Exact-float coverage of the check_matrix-slack widening in feature space."""

    def test_subnormal_coef_derives_no_bound(self) -> None:
        # 1e-9 / coef overflows to inf for a small enough subnormal coef: the
        # widened Range would be vacuous, so no derived range is appended at
        # all. The retained Linear (with its own 1e-9 slack) still governs
        # feasibility.
        coef = 5e-320
        assert math.isinf(1e-9 / coef)  # sanity: this coef triggers the overflow branch
        compiled = compile_constraints([Linear({"a": coef}, "<=", 0.0)], NAMES)
        assert compiled.derived_ranges == ()
        assert len(compiled.linears) == 1

    def test_tiny_normal_coef_widens_bound_exactly(self) -> None:
        coef = 1e-12
        compiled = compile_constraints([Linear({"a": coef}, "<=", 0.0)], NAMES)
        (rng,) = compiled.derived_ranges
        assert rng.lo == -math.inf
        assert rng.hi == 0.0 / coef + 1e-9 / coef

    def test_normal_coef_widens_bound_by_half_slack(self) -> None:
        compiled = compile_constraints([Linear({"a": 2.0}, "<=", 10.0)], NAMES)
        (rng,) = compiled.derived_ranges
        assert rng.lo == -math.inf
        assert rng.hi == 5.0 + 5e-10

    def test_large_coef_ulp_floor_admits_reviewer_counterexample(self) -> None:
        # For large |coef|, 1e-9/|coef| underflows below the float rounding gap
        # between rhs/coef (this bound) and coef*candidate (how check_matrix
        # evaluates it). Without a ulp-scale floor on the widening, this exact
        # candidate satisfies the Linear (coef*cand == rhs bit-for-bit) but the
        # derived bound rejects it.
        coef = 358050016645.61365
        rhs = 9373711634.780518
        bound = rhs / coef
        cand = math.nextafter(bound, math.inf)
        assert coef * cand == rhs  # exactly on the constraint, by construction

        compiled = compile_constraints([Linear({"a": coef}, "<=", rhs)], NAMES)
        (rng,) = compiled.derived_ranges
        assert cand <= rng.hi  # the widened bound admits it

        x = np.zeros(len(NAMES))
        row = np.zeros(len(NAMES))
        row[0] = cand
        X = row.reshape(1, -1)
        assert compiled.check_matrix(X, x)[0]  # with the derived bound in play

        # Parity: stripping the derived bound must not change the verdict — the
        # retained Linear constraint alone already admits this candidate.
        stripped = dataclasses.replace(compiled, derived_ranges=())
        assert stripped.check_matrix(X, x)[0]


class TestFactualViolations:
    def test_linear_message_shape(self) -> None:
        compiled = compile_constraints([constraint("a <= b")], NAMES)
        x = np.array([2.75, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert compiled.factual_violations(x) == (
            "Linear 1*a - 1*b <= 0 violated at the factual (lhs=2.75)",
        )

    def test_linear_within_slack_is_not_violated(self) -> None:
        compiled = compile_constraints([constraint("a <= b")], NAMES)
        x = np.array([1e-12, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert compiled.factual_violations(x) == ()

    def test_linear_nan_forbid_missing_reports(self) -> None:
        lin = Linear({"a": 1.0, "b": -1.0}, "<=", 0.0, missing_policy="forbid_missing")
        compiled = compile_constraints([lin], NAMES)
        x = np.array([np.nan, 0.0, 0.0, 0.0, 0.0, 0.0])
        (desc,) = compiled.factual_violations(x)
        assert "references a missing value" in desc
        assert "missing_policy=forbid_missing" in desc

    def test_linear_nan_satisfied_is_not_violated(self) -> None:
        compiled = compile_constraints([constraint("a <= b")], NAMES)
        x = np.array([np.nan, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert compiled.factual_violations(x) == ()

    def test_range_violation_and_nan_skip(self) -> None:
        compiled = compile_constraints([Range("a", 0.0, 1.0)], NAMES)
        assert compiled.factual_violations(np.array([2.0, 0, 0, 0, 0, 0.0])) != ()
        assert compiled.factual_violations(np.array([np.nan, 0, 0, 0, 0, 0.0])) == ()

    def test_freeze_monotone_derived_never_reported(self) -> None:
        compiled = compile_constraints(
            [Freeze("a"), constraint("b >= 100")], NAMES
        )
        x = np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
        # b >= 100 is violated -> exactly one report (the Linear), never the
        # derived range duplicate and never Freeze
        (desc,) = compiled.factual_violations(x)
        assert desc.startswith("Linear")

    def test_implies_and_onehot(self) -> None:
        compiled = compile_constraints(
            [
                Implies(Equals("flag1", 1.0), Equals("flag2", 1.0)),
                OneHot(("flag2", "flag3")),
            ],
            NAMES,
        )
        x = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        descs = compiled.factual_violations(x)
        assert len(descs) == 2
        assert any(d.startswith("Implies") for d in descs)
        assert any(d.startswith("OneHot") for d in descs)


class TestAllowedCategories:
    """Name resolution, rejections, membership checks, and the repair rule."""

    @staticmethod
    def _cats(cardinality: int = 4, names: tuple[str, ...] | None = None):
        from treecf.ir.model import CategoricalFeature

        return {1: CategoricalFeature(cardinality=cardinality, categories=names)}

    NAMES = ("amount", "occupation")

    def test_codes_resolve_and_intersect(self) -> None:
        compiled = compile_constraints(
            [
                AllowedCategories("occupation", (0, 1, 2)),
                AllowedCategories("occupation", (1, 2, 3)),
            ],
            self.NAMES,
            self._cats(),
        )
        assert compiled.allowed_categories == {1: frozenset({1, 2})}

    def test_names_resolve_through_categories(self) -> None:
        compiled = compile_constraints(
            [AllowedCategories("occupation", ("clerk", "manager"))],
            self.NAMES,
            self._cats(names=("clerk", "manager", "nurse", "smith")),
        )
        assert compiled.allowed_categories == {1: frozenset({0, 1})}

    def test_name_without_model_names_raises(self) -> None:
        with pytest.raises(ConstraintValidationError, match="no names"):
            compile_constraints(
                [AllowedCategories("occupation", ("clerk",))], self.NAMES, self._cats()
            )

    def test_unknown_name_and_out_of_range_code_raise(self) -> None:
        with pytest.raises(ConstraintValidationError, match="unknown category name"):
            compile_constraints(
                [AllowedCategories("occupation", ("astronaut",))],
                self.NAMES,
                self._cats(names=("clerk", "manager", "nurse", "smith")),
            )
        with pytest.raises(ConstraintValidationError, match=r"outside \[0, 4\)"):
            compile_constraints(
                [AllowedCategories("occupation", (7,))], self.NAMES, self._cats()
            )

    def test_on_numeric_feature_raises(self) -> None:
        with pytest.raises(ConstraintValidationError, match="numeric feature — use Range"):
            compile_constraints(
                [AllowedCategories("amount", (1,))], self.NAMES, self._cats()
            )

    @pytest.mark.parametrize(
        "bad",
        [
            Range("occupation", 0.0, 2.0),
            Monotone("occupation", "increase"),
            Equals("occupation", 1.0),
            Linear({"occupation": 1.0}, "<=", 2.0),
            Implies(Equals("occupation", 1.0), Equals("amount", 0.0)),
            OneHot(("occupation", "amount")),
        ],
    )
    def test_interval_constraints_on_categorical_raise(self, bad) -> None:
        with pytest.raises(ConstraintValidationError, match="categorical feature"):
            compile_constraints([bad], self.NAMES, self._cats())

    def test_freeze_and_allow_missing_still_apply(self) -> None:
        compiled = compile_constraints(
            [Freeze("occupation"), AllowMissing("amount", delta_miss=0.5)],
            self.NAMES,
            self._cats(),
        )
        assert compiled.allowed_categories == {}

    def test_check_matrix_membership(self) -> None:
        compiled = compile_constraints(
            [AllowedCategories("occupation", (1, 3))], self.NAMES, self._cats()
        )
        x = np.array([0.0, 1.0])
        X = np.array(
            [[0.0, 1.0], [0.0, 3.0], [0.0, 2.0], [0.0, 1.5], [0.0, 7.0]]
        )
        np.testing.assert_array_equal(
            compiled.check_matrix(X, x), [True, True, False, False, False]
        )

    def test_repair_keeps_allowed_else_smallest(self) -> None:
        compiled = compile_constraints(
            [AllowedCategories("occupation", (1, 3))], self.NAMES, self._cats()
        )
        x = np.array([0.0, 1.0])
        X = np.array([[0.0, 3.0], [0.0, 2.0], [0.0, 0.0]])
        repaired = compiled.repair_matrix(X, x)
        np.testing.assert_array_equal(repaired[:, 1], [3.0, 1.0, 1.0])

    def test_factual_violation_wording(self) -> None:
        compiled = compile_constraints(
            [AllowedCategories("occupation", (1,))], self.NAMES, self._cats()
        )
        violations = compiled.factual_violations(np.array([0.0, 2.0]))
        assert any("factual's category not in allowed set" in v for v in violations)
        assert compiled.factual_violations(np.array([0.0, 1.0])) == ()

    def test_empty_allowed_set_compiles(self) -> None:
        compiled = compile_constraints(
            [AllowedCategories("occupation", ()), ], self.NAMES, self._cats()
        )
        assert compiled.allowed_categories == {1: frozenset()}
        ok = compiled.check_matrix(np.array([[0.0, 1.0]]), np.array([0.0, 1.0]))
        assert not ok[0]
