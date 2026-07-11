"""Constraint objects and their compilation to instance bounds (spec §7.1, M1 subset)."""

import numpy as np
import pytest

from treecf._errors import ConstraintValidationError
from treecf.constraints import Freeze, Monotone, Range, compile_constraints

NAMES = ("age", "income", "dpd")


class TestValidation:
    def test_unknown_feature_raises(self) -> None:
        with pytest.raises(ConstraintValidationError, match="nope"):
            compile_constraints([Freeze("nope")], NAMES)

    def test_inverted_range_raises(self) -> None:
        with pytest.raises(ConstraintValidationError, match="lo"):
            compile_constraints([Range("income", 10.0, 5.0)], NAMES)

    def test_bad_monotone_direction_raises(self) -> None:
        with pytest.raises(ConstraintValidationError, match="direction"):
            compile_constraints([Monotone("age", "sideways")], NAMES)

    def test_objects_are_frozen(self) -> None:
        import dataclasses

        f = Freeze("age")
        with pytest.raises(dataclasses.FrozenInstanceError):
            f.feature = "income"  # type: ignore[misc]


class TestInstanceBounds:
    def test_freeze_pins_value(self) -> None:
        compiled = compile_constraints([Freeze("age")], NAMES)
        lo, hi, frozen = compiled.instance_bounds(np.array([44.0, 1000.0, 2.0]))
        assert lo[0] == hi[0] == 44.0
        assert frozen[0] and not frozen[1] and not frozen[2]

    def test_range_bounds_value(self) -> None:
        compiled = compile_constraints([Range("income", 0.0, 5000.0)], NAMES)
        lo, hi, _ = compiled.instance_bounds(np.array([44.0, 1000.0, 2.0]))
        assert lo[1] == 0.0 and hi[1] == 5000.0
        assert lo[0] == -np.inf and hi[0] == np.inf

    def test_monotone_increase_sets_lower_bound_at_factual(self) -> None:
        compiled = compile_constraints([Monotone("age", "increase")], NAMES)
        lo, hi, _ = compiled.instance_bounds(np.array([44.0, 1000.0, 2.0]))
        assert lo[0] == 44.0 and hi[0] == np.inf

    def test_monotone_decrease_sets_upper_bound_at_factual(self) -> None:
        compiled = compile_constraints([Monotone("dpd", "decrease")], NAMES)
        lo, hi, _ = compiled.instance_bounds(np.array([44.0, 1000.0, 2.0]))
        assert lo[2] == -np.inf and hi[2] == 2.0

    def test_constraints_on_same_feature_intersect(self) -> None:
        compiled = compile_constraints(
            [Range("dpd", 0.0, 90.0), Monotone("dpd", "decrease")], NAMES
        )
        lo, hi, _ = compiled.instance_bounds(np.array([44.0, 1000.0, 30.0]))
        assert lo[2] == 0.0 and hi[2] == 30.0
