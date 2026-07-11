"""String constraint sugar -> Linear objects (spec §7.2)."""

from __future__ import annotations

import pytest

from treecf._errors import ConstraintParseError
from treecf.constraints import Linear, constraint

NAMES = ("max_dpd_30d", "max_dpd_12m", "n_loans", "util")


class TestGrammar:
    def test_simple_order(self) -> None:
        c = constraint("max_dpd_30d <= max_dpd_12m", feature_names=NAMES)
        assert c == Linear(
            coefficients={"max_dpd_30d": 1.0, "max_dpd_12m": -1.0}, op="<=", rhs=0.0
        )

    def test_ge_and_eq_ops(self) -> None:
        ge = constraint("n_loans >= 1", feature_names=NAMES)
        assert ge == Linear(coefficients={"n_loans": 1.0}, op=">=", rhs=1.0)
        eq = constraint("util == 0.5", feature_names=NAMES)
        assert eq == Linear(coefficients={"util": 1.0}, op="==", rhs=0.5)

    def test_coefficients_and_arithmetic(self) -> None:
        c = constraint("2*max_dpd_30d - max_dpd_12m + 3 <= n_loans + 5", feature_names=NAMES)
        assert c == Linear(
            coefficients={"max_dpd_30d": 2.0, "max_dpd_12m": -1.0, "n_loans": -1.0},
            op="<=",
            rhs=2.0,
        )

    def test_repeated_feature_coefficients_accumulate(self) -> None:
        c = constraint("util + util <= 1", feature_names=NAMES)
        assert c == Linear(coefficients={"util": 2.0}, op="<=", rhs=1.0)

    def test_negative_and_float_numbers(self) -> None:
        c = constraint("-0.5*util >= -1.5", feature_names=NAMES)
        assert c == Linear(coefficients={"util": -0.5}, op=">=", rhs=-1.5)

    def test_parse_without_names_defers_validation(self) -> None:
        # spec §10 usage: constraint("a <= b") with validation at Explainer compile time
        c = constraint("anything <= whatever")
        assert c == Linear(coefficients={"anything": 1.0, "whatever": -1.0}, op="<=", rhs=0.0)


class TestErrors:
    def test_unknown_feature_carries_caret(self) -> None:
        with pytest.raises(ConstraintParseError) as err:
            constraint("max_dpd_30d <= typo_feature", feature_names=NAMES)
        message = str(err.value)
        assert "typo_feature" in message
        assert "^" in message  # caret marks the offending token

    def test_missing_operator(self) -> None:
        with pytest.raises(ConstraintParseError, match="operator"):
            constraint("max_dpd_30d max_dpd_12m", feature_names=NAMES)

    def test_double_operator(self) -> None:
        with pytest.raises(ConstraintParseError):
            constraint("a <= b <= c", feature_names=("a", "b", "c"))

    def test_garbage_token(self) -> None:
        with pytest.raises(ConstraintParseError) as err:
            constraint("util <= 0.5 $", feature_names=NAMES)
        assert "$" in str(err.value)

    def test_empty_side(self) -> None:
        with pytest.raises(ConstraintParseError):
            constraint("<= 5", feature_names=NAMES)

    def test_no_feature_at_all(self) -> None:
        with pytest.raises(ConstraintParseError, match="feature"):
            constraint("3 <= 5", feature_names=NAMES)
