"""Target abstraction: intervals on the raw score, probability via logit (spec §6)."""

import math

import pytest

from treecf._errors import TargetError
from treecf.ir.model import Link
from treecf.targets import Target


class TestRaw:
    def test_range(self) -> None:
        t = Target.raw(range=(-1.2, 0.5))
        assert t.raw_interval(Link.IDENTITY) == (-1.2, 0.5)

    def test_op_ge(self) -> None:
        t = Target.raw(op=">=", value=2.0)
        assert t.raw_interval(Link.IDENTITY) == (2.0, math.inf)

    def test_op_le(self) -> None:
        t = Target.raw(op="<=", value=2.0)
        assert t.raw_interval(Link.IDENTITY) == (-math.inf, 2.0)

    def test_works_for_sigmoid_link_too(self) -> None:
        assert Target.raw(range=(0.0, 1.0)).raw_interval(Link.SIGMOID) == (0.0, 1.0)


class TestProbability:
    def test_range_converts_via_logit(self) -> None:
        t = Target.probability(range=(0.2, 0.8))
        lo, hi = t.raw_interval(Link.SIGMOID)
        assert lo == pytest.approx(math.log(0.2 / 0.8))
        assert hi == pytest.approx(math.log(0.8 / 0.2))

    def test_open_endpoints_map_to_infinities(self) -> None:
        lo, hi = Target.probability(range=(0.0, 0.04)).raw_interval(Link.SIGMOID)
        assert lo == -math.inf
        assert hi == pytest.approx(math.log(0.04 / 0.96))
        _, hi2 = Target.probability(op=">=", value=0.96).raw_interval(Link.SIGMOID)
        assert hi2 == math.inf

    def test_identity_link_rejected(self) -> None:
        t = Target.probability(range=(0.0, 0.5))
        with pytest.raises(TargetError, match=r"SIGMOID"):
            t.raw_interval(Link.IDENTITY)


class TestConstructionErrors:
    def test_needs_exactly_one_form(self) -> None:
        with pytest.raises(TargetError):
            Target.raw()
        with pytest.raises(TargetError):
            Target.raw(range=(0, 1), op=">=", value=0.5)

    def test_probability_bounds_validated(self) -> None:
        with pytest.raises(TargetError):
            Target.probability(range=(-0.1, 0.5))
        with pytest.raises(TargetError):
            Target.probability(range=(0.6, 0.4))

    def test_op_validated(self) -> None:
        with pytest.raises(TargetError, match="op"):
            Target.raw(op="==", value=1.0)
