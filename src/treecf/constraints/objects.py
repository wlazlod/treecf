"""Canonical constraint objects. Frozen dataclasses; validation at compile time.

Pass these (or a ``constraint()`` string) to ``Explainer(..., constraints=[...])``.
See [Constraints](concepts/constraints.md) for what each one compiles to and
how they compose.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Freeze:
    """The feature is immutable: the counterfactual keeps the factual value.

    Attributes
    ----------
    feature
        The feature name to freeze.
    """

    feature: str


@dataclass(frozen=True)
class Monotone:
    """The feature may only move in one direction from the factual value.

    Attributes
    ----------
    feature
        The feature name to constrain.
    direction
        ``"increase"`` (the counterfactual value must be
        ``>=`` the factual) or ``"decrease"`` (``<=`` the factual).
    """

    feature: str
    direction: str  # "increase" | "decrease"


@dataclass(frozen=True)
class Range:
    """Hard domain bounds for the counterfactual value (inclusive).

    Attributes
    ----------
    feature
        The feature name to bound.
    lo
        Lower bound, inclusive.
    hi
        Upper bound, inclusive.
    """

    feature: str
    lo: float
    hi: float


@dataclass(frozen=True)
class Linear:
    """Linear inter-feature constraint: sum(coef * feature) op rhs.

    ``missing_policy`` resolves the constraint when a referenced feature is NaN
    in the counterfactual: "satisfied" (vacuously true, the default),
    "violated"/"forbid_missing" (the counterfactual may not use NaN there).
    The exact backend supports single-feature and the canonical two-feature
    order-pair shape exactly; any other multi-feature shape raises
    ``ConstraintValidationError`` naming ``backend="genetic"`` as the
    fallback — see
    [Certification](concepts/certification.md#what-the-exact-backend-does-not-certify-yet).

    Attributes
    ----------
    coefficients
        ``{feature: coefficient}`` for every feature in the
        sum; at least one entry.
    op
        ``"<="``, ``">="``, or ``"=="``.
    rhs
        The right-hand-side constant.
    missing_policy
        ``"satisfied"`` (the default — the constraint is
        vacuously satisfied when a referenced feature is NaN) or
        ``"violated"``/``"forbid_missing"`` (a NaN there fails the
        constraint, so the counterfactual may not use NaN on a referenced
        feature).
    """

    coefficients: dict[str, float]
    op: str  # "<=" | ">=" | "=="
    rhs: float
    missing_policy: str = "satisfied"


@dataclass(frozen=True)
class Equals:
    """Binary-feature equality (used standalone or inside ``Implies``).

    Attributes
    ----------
    feature
        The feature name to compare.
    value
        The value ``feature`` must equal (typically ``0.0``/``1.0``
        for a binary indicator).
    """

    feature: str
    value: float


@dataclass(frozen=True)
class Implies:
    """If ``condition`` holds then ``consequence`` must hold; binary features only.

    Attributes
    ----------
    condition
        The antecedent equality.
    consequence
        The equality ``condition`` requires when it holds.
    """

    condition: Equals
    consequence: Equals


@dataclass(frozen=True)
class OneHot:
    """The listed binary columns sum to exactly one.

    Attributes
    ----------
    features
        The mutually exclusive binary feature names; at least two.
    """

    features: tuple[str, ...]


@dataclass(frozen=True)
class AllowedCategories:
    """The categorical feature may only take the listed category codes or names.

    Entries are integer codes (``0..cardinality-1``) or display names resolved
    through the model's category names. Only valid on a categorical feature;
    several declarations on one feature intersect.

    Attributes
    ----------
    feature
        The categorical feature name to restrict.
    allowed
        The permitted codes (ints) or category names (strs).
    """

    feature: str
    allowed: tuple[int | str, ...]

    def __init__(self, feature: str, allowed: object) -> None:
        object.__setattr__(self, "feature", feature)
        items = (allowed,) if isinstance(allowed, int | str) else tuple(allowed)  # type: ignore[arg-type]
        object.__setattr__(self, "allowed", items)


@dataclass(frozen=True)
class AllowMissing:
    """NaN is a feasible counterfactual value for this feature.

    ``delta_miss`` prices the value<->NaN transition; pass ``delta_from_miss``
    for an asymmetric NaN->value cost (defaults to ``delta_miss``). See
    [Missing values](concepts/missing-values.md).

    Attributes
    ----------
    feature
        The feature name NaN is allowed on.
    delta_miss
        Distance cost of a value-to-NaN change on this feature.
    delta_from_miss
        Distance cost of a NaN-to-value change on this
        feature; defaults to ``delta_miss`` when ``None``.
    """

    feature: str
    delta_miss: float
    delta_from_miss: float | None = None


Constraint = (
    Freeze
    | Monotone
    | Range
    | Linear
    | Equals
    | Implies
    | OneHot
    | AllowMissing
    | AllowedCategories
)
