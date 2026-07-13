"""Compilation and validation of the full M2 constraint set."""

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
