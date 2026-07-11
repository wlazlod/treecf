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


@dataclass(frozen=True)
class Linear:
    """Linear inter-feature constraint: sum(coef * feature) op rhs (spec §7.1).

    ``missing_policy`` resolves the constraint when a referenced feature is NaN
    in the counterfactual: "satisfied" (vacuously true, the default),
    "violated"/"forbid_missing" (the counterfactual may not use NaN there).
    """

    coefficients: dict[str, float]
    op: str  # "<=" | ">=" | "=="
    rhs: float
    missing_policy: str = "satisfied"


@dataclass(frozen=True)
class Equals:
    """Binary-feature equality (used standalone or inside Implies)."""

    feature: str
    value: float


@dataclass(frozen=True)
class Implies:
    """If `condition` holds then `consequence` must hold; binary features only (v0.1)."""

    condition: Equals
    consequence: Equals


@dataclass(frozen=True)
class OneHot:
    """The listed binary columns sum to exactly one."""

    features: tuple[str, ...]


@dataclass(frozen=True)
class AllowMissing:
    """NaN is a feasible counterfactual value for this feature (spec §4.2).

    ``delta_miss`` prices the value<->NaN transition; pass ``delta_from_miss``
    for an asymmetric NaN->value cost (defaults to ``delta_miss``).
    """

    feature: str
    delta_miss: float
    delta_from_miss: float | None = None


Constraint = Freeze | Monotone | Range | Linear | Equals | Implies | OneHot | AllowMissing
