"""Target abstraction: intervals on the raw score, probability via logit."""

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


def _logit(p: float) -> float:
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    return math.log(p / (1.0 - p))


class FakeAffineLogit:
    """g with logit(g(p)) = a * logit(p) + b; closed-form generalized inverse."""

    is_monotone_ = True

    def __init__(self, a: float, b: float) -> None:
        self.a = a
        self.b = b

    def forward(self, p: float) -> float:
        z = self.a * _logit(p) + self.b
        return 1.0 / (1.0 + math.exp(-z))

    def interval_inverse(
        self, lo: float, hi: float, *, space: str = "probability", buffer_logit: float = 0.0
    ) -> tuple[float, float]:
        assert space == "logit"
        lo_z = -math.inf if lo <= 0.0 else (_logit(lo) + buffer_logit - self.b) / self.a
        hi_z = math.inf if hi >= 1.0 else (_logit(hi) - buffer_logit - self.b) / self.a
        return lo_z, hi_z


class FakeStep:
    """Piecewise-constant non-decreasing map with a searchsorted-style inverse."""

    is_monotone_ = True

    def __init__(self, edges: tuple[float, ...], values: tuple[float, ...]) -> None:
        self.edges = edges  # logit-space thresholds between steps
        self.values = values  # len(edges) + 1 calibrated levels

    def interval_inverse(
        self, lo: float, hi: float, *, space: str = "probability", buffer_logit: float = 0.0
    ) -> tuple[float, float]:
        assert space == "logit"
        qualifying = [i for i, v in enumerate(self.values) if lo <= v <= hi]
        if not qualifying:
            raise ValueError(
                f"target outside calibrator range [{self.values[0]}, {self.values[-1]}]"
            )
        first, last = qualifying[0], qualifying[-1]
        lo_z = -math.inf if first == 0 else self.edges[first - 1]
        hi_z = math.inf if last == len(self.values) - 1 else self.edges[last]
        return lo_z, hi_z


class FakeNonMonotone:
    is_monotone_ = False

    def interval_inverse(
        self, lo: float, hi: float, *, space: str = "probability", buffer_logit: float = 0.0
    ) -> tuple[float, float]:
        return lo, hi


class FakeUnattainable:
    is_monotone_ = True

    def interval_inverse(
        self, lo: float, hi: float, *, space: str = "probability", buffer_logit: float = 0.0
    ) -> tuple[float, float]:
        raise ValueError("target outside calibrator range [0.011, 0.27]")


class FakeLegacy:
    is_monotone_ = True  # but no interval_inverse


class TestCalibrated:
    def test_round_trip_matches_closed_form(self) -> None:
        cal = FakeAffineLogit(a=0.8, b=-0.4)
        t = Target.calibrated(cal, range=(0.02, 0.10))
        lo, hi = t.raw_interval(Link.SIGMOID)
        assert lo == pytest.approx((_logit(0.02) - (-0.4)) / 0.8, abs=1e-12)
        assert hi == pytest.approx((_logit(0.10) - (-0.4)) / 0.8, abs=1e-12)

    def test_open_endpoints_map_to_infinities(self) -> None:
        cal = FakeAffineLogit(a=1.0, b=0.5)
        lo, hi = Target.calibrated(cal, range=(0.0, 0.02)).raw_interval(Link.SIGMOID)
        assert lo == -math.inf
        assert hi == pytest.approx(_logit(0.02) - 0.5, abs=1e-12)
        lo2, hi2 = Target.calibrated(cal, op=">=", value=0.5).raw_interval(Link.SIGMOID)
        assert hi2 == math.inf

    def test_buffer_nests_interval_strictly_inside(self) -> None:
        cal = FakeAffineLogit(a=1.0, b=0.0)
        plain = Target.calibrated(cal, range=(0.02, 0.10)).raw_interval(Link.SIGMOID)
        small = Target.calibrated(cal, range=(0.02, 0.10), buffer_logit=0.1).raw_interval(
            Link.SIGMOID
        )
        large = Target.calibrated(cal, range=(0.02, 0.10), buffer_logit=0.3).raw_interval(
            Link.SIGMOID
        )
        assert plain[0] < small[0] < large[0]
        assert large[1] < small[1] < plain[1]

    def test_non_monotone_rejected_at_construction(self) -> None:
        with pytest.raises(TargetError, match="monotone"):
            Target.calibrated(FakeNonMonotone(), range=(0.0, 0.1))

    def test_missing_is_monotone_rejected(self) -> None:
        cal = FakeAffineLogit(a=1.0, b=0.0)
        del FakeAffineLogit.is_monotone_
        try:
            with pytest.raises(TargetError, match="monotone"):
                Target.calibrated(cal, range=(0.0, 0.1))
        finally:
            FakeAffineLogit.is_monotone_ = True

    def test_legacy_object_rejected(self) -> None:
        with pytest.raises(TargetError, match="interval_inverse"):
            Target.calibrated(FakeLegacy(), range=(0.0, 0.1))

    def test_negative_buffer_rejected(self) -> None:
        with pytest.raises(TargetError, match="buffer_logit"):
            Target.calibrated(FakeAffineLogit(1.0, 0.0), range=(0.0, 0.1), buffer_logit=-0.1)

    def test_interval_bounds_validated(self) -> None:
        with pytest.raises(TargetError):
            Target.calibrated(FakeAffineLogit(1.0, 0.0), range=(-0.1, 0.5))
        with pytest.raises(TargetError):
            Target.calibrated(FakeAffineLogit(1.0, 0.0), range=(0.6, 0.4))

    def test_identity_link_rejected(self) -> None:
        t = Target.calibrated(FakeAffineLogit(1.0, 0.0), range=(0.0, 0.5))
        with pytest.raises(TargetError, match=r"SIGMOID"):
            t.raw_interval(Link.IDENTITY)

    def test_unattainable_wrapped_with_cause(self) -> None:
        t = Target.calibrated(FakeUnattainable(), range=(0.4, 0.9))
        with pytest.raises(TargetError, match=r"\[0\.011, 0\.27\]") as excinfo:
            t.raw_interval(Link.SIGMOID)
        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_step_calibrator_edges(self) -> None:
        cal = FakeStep(edges=(-2.0, 0.0), values=(0.01, 0.05, 0.20))
        t = Target.calibrated(cal, range=(0.04, 0.30))
        lo, hi = t.raw_interval(Link.SIGMOID)
        assert lo == -2.0  # left edge of the first qualifying step
        assert hi == math.inf  # last step qualifies

    def test_calibrated_bands_invert_per_band(self) -> None:
        cal = FakeAffineLogit(a=1.0, b=-0.5)
        t = Target.bands(
            {"A": (0.001, 0.02), "B": (0.02, 0.10)},
            space="calibrated",
            calibrator=cal,
            buffer_logit=0.05,
        )
        out = t.band_intervals(Link.SIGMOID)
        assert set(out) == {"A", "B"}
        # Regression (spec §3.5): band_intervals must propagate calibrator and
        # buffer; each band equals the calibrator's closed-form inverse.
        for name, (lo_p, hi_p) in {"A": (0.001, 0.02), "B": (0.02, 0.10)}.items():
            expected = cal.interval_inverse(lo_p, hi_p, space="logit", buffer_logit=0.05)
            assert out[name] == pytest.approx(expected, abs=1e-12)

    def test_bands_calibrated_requires_calibrator(self) -> None:
        with pytest.raises(TargetError, match="calibrat"):
            Target.bands({"A": (0.0, 0.1)}, space="calibrated")

    def test_existing_constructors_untouched(self) -> None:
        t = Target.probability(range=(0.2, 0.8))
        assert t.calibrator is None
        assert t.buffer_logit == 0.0
        assert t == Target.probability(range=(0.2, 0.8))
