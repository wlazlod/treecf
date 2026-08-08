"""Compilation and validation of the full M2 constraint set."""

import math

import numpy as np
import pytest

from treecf._errors import ConstraintValidationError
from treecf.constraints import (
    AllowMissing,
    Equals,
    Freeze,
    Implies,
    Linear,
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
    def test_ge_derives_lower_bound(self) -> None:
        compiled = compile_constraints([constraint("a >= 100")], NAMES)
        assert compiled.derived_ranges == (Range("a", 100.0, math.inf),)
        assert len(compiled.linears) == 1  # original Linear retained

    def test_le_derives_upper_bound(self) -> None:
        compiled = compile_constraints([constraint("2*a <= 10")], NAMES)
        assert compiled.derived_ranges == (Range("a", -math.inf, 5.0),)

    def test_negative_coef_flips_inequality(self) -> None:
        compiled = compile_constraints([Linear({"a": -2.0}, "<=", -10.0)], NAMES)
        assert compiled.derived_ranges == (Range("a", 5.0, math.inf),)

    def test_negative_coef_ge_gives_upper_bound(self) -> None:
        compiled = compile_constraints([Linear({"a": -2.0}, ">=", -10.0)], NAMES)
        assert compiled.derived_ranges == (Range("a", -math.inf, 5.0),)

    def test_eq_derives_pin(self) -> None:
        compiled = compile_constraints([Linear({"a": 2.0}, "==", 7.0)], NAMES)
        (rng,) = compiled.derived_ranges
        assert rng.lo == rng.hi == 3.5

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
        assert lo[0] == 100.0
        assert hi[0] == math.inf
