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

    def check_matrix(self, X: FloatArray, x: FloatArray) -> BoolArray:
        """Vectorized feasibility of candidate rows against every constraint (§7.3)."""
        n, p = X.shape
        ok = np.ones(n, dtype=bool)
        lo, hi, _ = self.instance_bounds(x)
        lo = np.where(np.isnan(lo), -math.inf, lo)
        hi = np.where(np.isnan(hi), math.inf, hi)
        nan_x = np.isnan(X)
        with np.errstate(invalid="ignore"):
            in_bounds = (lo <= X) | nan_x
            in_bounds &= (hi >= X) | nan_x
        ok &= in_bounds.all(axis=1)
        for j in range(p):
            if j not in self.allow_missing and not math.isnan(x[j]):
                ok &= ~nan_x[:, j]
            if math.isnan(x[j]) and j not in self.allow_missing:
                ok &= nan_x[:, j]
        for lin in self.linears:
            cols = X[:, list(lin.indices)]
            any_nan = np.isnan(cols).any(axis=1)
            total = np.nansum(cols * np.array(lin.coefs), axis=1)
            holds = (
                total <= lin.rhs + 1e-9
                if lin.op == "<="
                else total >= lin.rhs - 1e-9
                if lin.op == ">="
                else np.abs(total - lin.rhs) <= 1e-9
            )
            if lin.missing_policy == "satisfied":
                ok &= holds | any_nan
            else:
                ok &= holds & ~any_nan
        for imp in self.implications:
            cond = X[:, imp.cond_index] == imp.cond_value
            cons = X[:, imp.cons_index] == imp.cons_value
            ok &= ~cond | cons
        for group in self.onehot_groups:
            ok &= X[:, list(group)].sum(axis=1) == 1.0
        return ok

    def repair_matrix(self, X: FloatArray, x: FloatArray) -> FloatArray:
        """Best-effort repair hints (§7.3): clip to bounds, fix NaN legality, order pairs."""
        X = X.copy()
        p = X.shape[1]
        lo, hi, _ = self.instance_bounds(x)
        lo = np.where(np.isnan(lo), -math.inf, lo)
        hi = np.where(np.isnan(hi), math.inf, hi)
        for j in range(p):
            nan_col = np.isnan(X[:, j])
            if j not in self.allow_missing:
                if math.isnan(x[j]):
                    X[:, j] = math.nan  # fixed missing
                    continue
                X[nan_col, j] = x[j]
            valid = ~np.isnan(X[:, j])
            X[valid, j] = np.clip(X[valid, j], lo[j], hi[j])
        for lin in self.linears:
            # repair hint only for the canonical order pair a - b <= 0: clip a to b
            if lin.op == "<=" and lin.rhs == 0.0 and sorted(lin.coefs) == [-1.0, 1.0]:
                a = lin.indices[lin.coefs.index(1.0)]
                b = lin.indices[lin.coefs.index(-1.0)]
                both = ~np.isnan(X[:, a]) & ~np.isnan(X[:, b])
                X[both, a] = np.minimum(X[both, a], X[both, b])
        for imp in self.implications:
            cond = X[:, imp.cond_index] == imp.cond_value
            X[cond, imp.cons_index] = imp.cons_value
        for group in self.onehot_groups:
            cols = list(group)
            block = X[:, cols]
            winner = np.argmax(np.nan_to_num(block, nan=-1.0), axis=1)
            X[:, cols] = 0.0
            X[np.arange(len(X)), [cols[w] for w in winner]] = 1.0
        return X

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
