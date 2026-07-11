"""Canonical constraint objects (spec §7.1). Frozen dataclasses; validation at compile time.

M1 subset: Freeze, Monotone, Range. Linear, Implies, OneHot, AllowMissing arrive in M2.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Freeze:
    """The feature is immutable: the counterfactual keeps the factual value."""

    feature: str


@dataclass(frozen=True)
class Monotone:
    """The feature may only move in one direction from the factual value."""

    feature: str
    direction: str  # "increase" | "decrease"


@dataclass(frozen=True)
class Range:
    """Hard domain bounds for the counterfactual value (inclusive)."""

    feature: str
    lo: float
    hi: float


Constraint = Freeze | Monotone | Range
