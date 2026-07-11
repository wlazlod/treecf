"""Targets as intervals on the raw model output (spec §6, D3/D9)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from treecf._errors import TargetError
from treecf.ir.model import Link

_OPS = (">=", "<=")


@dataclass(frozen=True)
class Target:
    """Closed interval target, expressed in raw-score or probability space."""

    space: str  # "raw" | "probability"
    lo: float
    hi: float

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
