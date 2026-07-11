"""Targets as intervals on the raw model output (spec §6, D3/D9)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from treecf._errors import TargetError
from treecf.ir.model import Link

_OPS = (">=", "<=")


@dataclass(frozen=True)
class Target:
    """Closed interval target, expressed in raw-score or probability space.

    ``Target.bands`` builds a named ladder of intervals (rating grades, D9);
    ``Explainer.explain`` then returns one result per band from a single
    AIM compilation (§6).
    """

    space: str  # "raw" | "probability"
    lo: float
    hi: float
    bands_spec: tuple[tuple[str, float, float], ...] | None = None

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
        lo, hi = _interval_from(range, op, value, lo_limit=0.0, hi_limit=1.0)
        if not (0.0 <= lo < hi <= 1.0):
            raise TargetError(f"probability interval [{lo}, {hi}] must lie within [0, 1]")
        return cls(space="probability", lo=lo, hi=hi)

    @classmethod
    def bands(
        cls, bands: dict[str, tuple[float, float]], space: str = "probability"
    ) -> Target:
        if space not in ("raw", "probability"):
            raise TargetError("bands space must be 'raw' or 'probability'")
        if not bands:
            raise TargetError("bands must contain at least one named interval")
        spec = []
        for name, (lo, hi) in bands.items():
            if not lo < hi:
                raise TargetError(f"band {name!r}: empty interval [{lo}, {hi}]")
            if space == "probability" and not (0.0 <= lo < hi <= 1.0):
                raise TargetError(f"band {name!r} must lie within [0, 1]")
            spec.append((name, float(lo), float(hi)))
        first = spec[0]
        return cls(space=space, lo=first[1], hi=first[2], bands_spec=tuple(spec))

    def raw_interval(self, link: Link) -> tuple[float, float]:
        """Interval [L, U] on the raw score; probability targets require the SIGMOID link."""
        if self.space == "raw":
            return self.lo, self.hi
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
            single = Target(space=self.space, lo=lo, hi=hi)
            out[name] = single.raw_interval(link)
        return out


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
