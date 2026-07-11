"""Constraint compiler (spec §7.3) — the single source of truth for constraint semantics.

Freeze/Monotone/Range/Equals compile to per-feature interval bounds given the
factual instance; Linear/Implies/OneHot/AllowMissing are exposed as structured,
index-resolved groups that the AIM builder encodes and the genetic backend (M3)
turns into vectorized check/repair pairs. No per-backend constraint logic may
exist elsewhere.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from treecf._errors import ConstraintValidationError
from treecf.constraints.objects import (
    AllowMissing,
    Constraint,
    Equals,
    Freeze,
    Implies,
    Linear,
    Monotone,
    OneHot,
    Range,
)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

_LINEAR_OPS = ("<=", ">=", "==")
_MISSING_POLICIES = ("satisfied", "violated", "forbid_missing")


@dataclass(frozen=True)
class ResolvedLinear:
    """A Linear constraint with features resolved to indices."""

    coefficients: dict[str, float]
    indices: tuple[int, ...]
    coefs: tuple[float, ...]
    op: str
    rhs: float
    missing_policy: str


@dataclass(frozen=True)
class ResolvedImplication:
    """Implies over binary features: (feature, value) => (feature, value)."""

    cond_index: int
    cond_value: float
    cons_index: int
    cons_value: float


@dataclass(frozen=True)
class CompiledConstraints:
    """Constraints resolved against a feature space."""

    feature_names: tuple[str, ...]
    constraints: tuple[Constraint, ...]
    linears: tuple[ResolvedLinear, ...] = ()
    implications: tuple[ResolvedImplication, ...] = ()
    onehot_groups: tuple[tuple[int, ...], ...] = ()
    allow_missing: dict[int, tuple[float, float]] = field(default_factory=dict)
    binary_features: frozenset[int] = frozenset()

    def instance_bounds(self, x: FloatArray) -> tuple[FloatArray, FloatArray, BoolArray]:
        """Per-feature (lo, hi, frozen) for factual ``x``; bounds are intersected."""
        p = len(self.feature_names)
        lo = np.full(p, -math.inf)
        hi = np.full(p, math.inf)
        frozen = np.zeros(p, dtype=bool)
        index = {name: j for j, name in enumerate(self.feature_names)}
        for constraint in self.constraints:
            if isinstance(constraint, Freeze):
                j = index[constraint.feature]
                lo[j] = max(lo[j], x[j])
                hi[j] = min(hi[j], x[j])
                frozen[j] = True
            elif isinstance(constraint, Range):
                j = index[constraint.feature]
                lo[j] = max(lo[j], constraint.lo)
                hi[j] = min(hi[j], constraint.hi)
            elif isinstance(constraint, Equals):
                j = index[constraint.feature]
                lo[j] = max(lo[j], constraint.value)
                hi[j] = min(hi[j], constraint.value)
            elif isinstance(constraint, Monotone):
                j = index[constraint.feature]
                if constraint.direction == "increase":
                    lo[j] = max(lo[j], x[j])
                else:
                    hi[j] = min(hi[j], x[j])
        return lo, hi, frozen


def compile_constraints(
    constraints: Sequence[Constraint], feature_names: Sequence[str]
) -> CompiledConstraints:
    """Validate the constraint set against the feature space and freeze it."""
    names = tuple(feature_names)
    index = {name: j for j, name in enumerate(names)}

    linears: list[ResolvedLinear] = []
    implications: list[ResolvedImplication] = []
    onehot_groups: list[tuple[int, ...]] = []
    allow_missing: dict[int, tuple[float, float]] = {}
    binary: set[int] = set()
    frozen_names: set[str] = set()

    def resolve(name: str, owner: str) -> int:
        if name not in index:
            raise ConstraintValidationError(f"{owner} references unknown feature {name!r}")
        return index[name]

    for c in constraints:
        kind = type(c).__name__
        if isinstance(c, Freeze):
            resolve(c.feature, kind)
            frozen_names.add(c.feature)
        elif isinstance(c, Range):
            resolve(c.feature, kind)
            if c.lo > c.hi:
                raise ConstraintValidationError(f"Range({c.feature!r}): lo {c.lo} > hi {c.hi}")
        elif isinstance(c, Monotone):
            resolve(c.feature, kind)
            if c.direction not in ("increase", "decrease"):
                raise ConstraintValidationError(
                    f"Monotone({c.feature!r}): direction must be 'increase' or 'decrease', "
                    f"got {c.direction!r}"
                )
        elif isinstance(c, Equals):
            binary.add(_validated_binary(c, index))
        elif isinstance(c, Linear):
            if c.op not in _LINEAR_OPS:
                raise ConstraintValidationError(f"Linear op must be one of {_LINEAR_OPS}")
            if c.missing_policy not in _MISSING_POLICIES:
                raise ConstraintValidationError(
                    f"Linear missing_policy must be one of {_MISSING_POLICIES}"
                )
            if not c.coefficients:
                raise ConstraintValidationError("Linear constraint has no coefficients")
            indices = tuple(resolve(name, kind) for name in c.coefficients)
            linears.append(
                ResolvedLinear(
                    coefficients=dict(c.coefficients),
                    indices=indices,
                    coefs=tuple(c.coefficients.values()),
                    op=c.op,
                    rhs=c.rhs,
                    missing_policy=c.missing_policy,
                )
            )
        elif isinstance(c, Implies):
            cond = _validated_binary(c.condition, index)
            cons = _validated_binary(c.consequence, index)
            binary.update((cond, cons))
            implications.append(
                ResolvedImplication(
                    cond_index=cond,
                    cond_value=c.condition.value,
                    cons_index=cons,
                    cons_value=c.consequence.value,
                )
            )
        elif isinstance(c, OneHot):
            if len(c.features) < 2:
                raise ConstraintValidationError("OneHot needs at least two features")
            group = tuple(resolve(name, kind) for name in c.features)
            for other in onehot_groups:
                if set(group) & set(other):
                    raise ConstraintValidationError("OneHot groups overlap")
            onehot_groups.append(group)
            binary.update(group)
        elif isinstance(c, AllowMissing):
            j = resolve(c.feature, kind)
            if c.delta_miss <= 0 or (c.delta_from_miss is not None and c.delta_from_miss <= 0):
                raise ConstraintValidationError(
                    f"AllowMissing({c.feature!r}): delta must be positive"
                )
            allow_missing[j] = (
                c.delta_miss,
                c.delta_miss if c.delta_from_miss is None else c.delta_from_miss,
            )
        else:  # pragma: no cover - exhaustive over Constraint
            raise ConstraintValidationError(f"unknown constraint type {kind}")

    for name in frozen_names:
        if index[name] in allow_missing:
            raise ConstraintValidationError(
                f"AllowMissing({name!r}) conflicts with Freeze on a frozen feature"
            )

    return CompiledConstraints(
        feature_names=names,
        constraints=tuple(constraints),
        linears=tuple(linears),
        implications=tuple(implications),
        onehot_groups=tuple(onehot_groups),
        allow_missing=allow_missing,
        binary_features=frozenset(binary),
    )


def _validated_binary(c: Equals, index: dict[str, int]) -> int:
    if c.feature not in index:
        raise ConstraintValidationError(f"Equals references unknown feature {c.feature!r}")
    if c.value not in (0.0, 1.0):
        raise ConstraintValidationError(
            f"Equals({c.feature!r}): only binary values 0/1 are supported in v0.1, "
            f"got {c.value}"
        )
    return index[c.feature]
