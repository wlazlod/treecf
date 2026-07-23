"""Targets as intervals on the raw model output."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from treecf._errors import TargetError
from treecf.ir.model import Link

_OPS = (">=", "<=")


class _SupportsIntervalInverse(Protocol):
    """Duck-typed calibrator protocol for calibrated targets.

    Any object with these two members works; treecf never imports a
    calibration library. ``interval_inverse`` with
    ``space="logit"`` must return generalized-inverse bounds on the logit of
    the model probability — for a SIGMOID-link ensemble that is exactly the
    raw margin. ``lo=0.0``/``hi=1.0`` map to ``-inf``/``+inf``.
    """

    is_monotone_: bool

    def interval_inverse(
        self, lo: float, hi: float, *, space: str = "probability", buffer_logit: float = 0.0
    ) -> tuple[float, float]: ...


@dataclass(frozen=True)
class Target:
    """Closed interval target, expressed in raw-score, probability, or calibrated space.

    ``Target.bands`` builds a named ladder of intervals (rating grades);
    ``Explainer.explain`` then returns one result per band.
    """

    space: str  # "raw" | "probability" | "calibrated"
    lo: float
    hi: float
    bands_spec: tuple[tuple[str, float, float], ...] | None = None
    calibrator: _SupportsIntervalInverse | None = None
    buffer_logit: float = 0.0

    @classmethod
    def raw(
        cls,
        range: tuple[float, float] | None = None,
        op: str | None = None,
        value: float | None = None,
    ) -> Target:
        lo, hi = _interval_from(range, op, value, lo_limit=-math.inf, hi_limit=math.inf)
        return cls(space="raw", lo=lo, hi=hi)

    @classmethod
    def probability(
        cls,
        range: tuple[float, float] | None = None,
        op: str | None = None,
        value: float | None = None,
    ) -> Target:
        """Target on the model's own probability output.

        If model outputs are post-hoc calibrated downstream, this constructor
        targets the *uncalibrated* model probability; use ``Target.calibrated``
        with your calibrator instead.
        """
        lo, hi = _interval_from(range, op, value, lo_limit=0.0, hi_limit=1.0)
        if not (0.0 <= lo < hi <= 1.0):
            raise TargetError(f"probability interval [{lo}, {hi}] must lie within [0, 1]")
        return cls(space="probability", lo=lo, hi=hi)

    @classmethod
    def calibrated(
        cls,
        calibrator: _SupportsIntervalInverse,
        range: tuple[float, float] | None = None,
        op: str | None = None,
        value: float | None = None,
        *,
        buffer_logit: float = 0.0,
    ) -> Target:
        """Target on the *calibrated* probability ``g(model probability)``.

        The interval is inverted through the calibrator's generalized inverse
        lazily, at ``raw_interval`` time; the calibrator is held by reference,
        so refitting it between construction and ``explain`` is the caller's
        responsibility. ``buffer_logit`` shrinks the calibrated interval in
        logit space before inversion, making the counterfactual robust to
        future recalibration or central-tendency drift of that magnitude.
        """
        _validate_calibrator(calibrator, buffer_logit)
        lo, hi = _interval_from(range, op, value, lo_limit=0.0, hi_limit=1.0)
        if not (0.0 <= lo < hi <= 1.0):
            raise TargetError(f"calibrated interval [{lo}, {hi}] must lie within [0, 1]")
        return cls(
            space="calibrated", lo=lo, hi=hi, calibrator=calibrator, buffer_logit=buffer_logit
        )

    @classmethod
    def bands(
        cls,
        bands: dict[str, tuple[float, float]],
        space: str = "probability",
        *,
        calibrator: _SupportsIntervalInverse | None = None,
        buffer_logit: float = 0.0,
    ) -> Target:
        if space not in ("raw", "probability", "calibrated"):
            raise TargetError("bands space must be 'raw', 'probability', or 'calibrated'")
        if space == "calibrated":
            _validate_calibrator(calibrator, buffer_logit)
        if not bands:
            raise TargetError("bands must contain at least one named interval")
        spec = []
        for name, (lo, hi) in bands.items():
            if not lo < hi:
                raise TargetError(f"band {name!r}: empty interval [{lo}, {hi}]")
            if space in ("probability", "calibrated") and not (0.0 <= lo < hi <= 1.0):
                raise TargetError(f"band {name!r} must lie within [0, 1]")
            spec.append((name, float(lo), float(hi)))
        first = spec[0]
        return cls(
            space=space,
            lo=first[1],
            hi=first[2],
            bands_spec=tuple(spec),
            calibrator=calibrator,
            buffer_logit=buffer_logit,
        )

    def raw_interval(self, link: Link) -> tuple[float, float]:
        """Interval [L, U] on the raw score; probability and calibrated targets
        require the SIGMOID link."""
        if self.space == "raw":
            return self.lo, self.hi
        if self.space == "calibrated":
            if link is not Link.SIGMOID:
                raise TargetError(
                    "calibrated target requires a SIGMOID-link model; "
                    "use Target.raw for identity-link outputs"
                )
            assert self.calibrator is not None
            try:
                return self.calibrator.interval_inverse(
                    self.lo, self.hi, space="logit", buffer_logit=self.buffer_logit
                )
            except TargetError:
                raise
            except Exception as exc:
                raise TargetError(
                    f"calibrator could not invert [{self.lo}, {self.hi}]: {exc}"
                ) from exc
        if link is not Link.SIGMOID:
            raise TargetError(
                "probability target requires a SIGMOID-link model; "
                "use Target.raw for identity-link outputs"
            )
        return _logit(self.lo), _logit(self.hi)

    def band_intervals(self, link: Link) -> dict[str, tuple[float, float]]:
        assert self.bands_spec is not None
        out: dict[str, tuple[float, float]] = {}
        for name, lo, hi in self.bands_spec:
            single = Target(
                space=self.space,
                lo=lo,
                hi=hi,
                calibrator=self.calibrator,
                buffer_logit=self.buffer_logit,
            )
            out[name] = single.raw_interval(link)
        return out


def _validate_calibrator(calibrator: object, buffer_logit: float) -> None:
    if not callable(getattr(calibrator, "interval_inverse", None)):
        raise TargetError(
            "calibrator must expose interval_inverse(lo, hi, *, space, buffer_logit); "
            f"got {type(calibrator).__name__}"
        )
    if getattr(calibrator, "is_monotone_", None) is not True:
        raise TargetError(
            "calibrated targets require a monotone calibrator (is_monotone_ = True): "
            "the preimage of an interval under a non-monotone map need not be an interval"
        )
    if buffer_logit < 0.0:
        raise TargetError(f"buffer_logit must be >= 0.0, got {buffer_logit}")


def _interval_from(
    range: tuple[float, float] | None,
    op: str | None,
    value: float | None,
    lo_limit: float,
    hi_limit: float,
) -> tuple[float, float]:
    has_range = range is not None
    has_op = op is not None or value is not None
    if has_range == has_op:
        raise TargetError("specify exactly one of range=(lo, hi) or op=/value=")
    if has_range:
        assert range is not None
        lo, hi = float(range[0]), float(range[1])
        if not lo < hi:
            raise TargetError(f"empty target interval [{lo}, {hi}]")
        return lo, hi
    if op not in _OPS or value is None:
        raise TargetError(f"op must be one of {_OPS} with a value")
    return (float(value), hi_limit) if op == ">=" else (lo_limit, float(value))


def _logit(p: float) -> float:
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    return math.log(p / (1.0 - p))
