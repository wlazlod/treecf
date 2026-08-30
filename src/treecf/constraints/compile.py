"""Constraint compiler — the single source of truth for constraint semantics.

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
_REPAIR_ROUNDS = 3  # cyclic projection sweeps over the linear constraints


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


def _format_linear(lin: ResolvedLinear) -> str:
    parts: list[str] = []
    for name, coef in lin.coefficients.items():
        if not parts:
            parts.append(f"{coef:g}*{name}")
        elif coef < 0:
            parts.append(f"- {-coef:g}*{name}")
        else:
            parts.append(f"+ {coef:g}*{name}")
    return f"Linear {' '.join(parts)} {lin.op} {lin.rhs:g}"


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
    # per-feature bounds implied by single-feature Linear constraints; kept out of
    # ``constraints`` so recompiling a clone from ``constraints`` re-derives them
    # exactly once (idempotent)
    derived_ranges: tuple[Range, ...] = ()

    def check_matrix(self, X: FloatArray, x: FloatArray) -> BoolArray:
        """Vectorized feasibility of candidate rows against every constraint."""
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
            # exact float equality is intentional: repair writes literal 0.0/1.0,
            # and a tolerance would mask genuinely broken candidates
            ok &= X[:, list(group)].sum(axis=1) == 1.0
        return ok

    def repair_matrix(self, X: FloatArray, x: FloatArray) -> FloatArray:
        """Best-effort repair hints: clip to bounds, fix NaN legality, project linears.

        Feasibility is decided by ``check_matrix``, never assumed from repair.
        """
        X = X.copy()
        p = X.shape[1]
        lo, hi, frozen = self.instance_bounds(x)
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
        # cyclic projection: boxes intersected with halfspaces, a few sweeps
        for _ in range(_REPAIR_ROUNDS):
            for lin in self.linears:
                if lin.op == "<=" and lin.rhs == 0.0 and sorted(lin.coefs) == [-1.0, 1.0]:
                    # canonical order pair a - b <= 0: clip a to b (unchanged;
                    # moving one feature beats a split projection here)
                    a = lin.indices[lin.coefs.index(1.0)]
                    b = lin.indices[lin.coefs.index(-1.0)]
                    both = ~np.isnan(X[:, a]) & ~np.isnan(X[:, b])
                    X[both, a] = np.minimum(X[both, a], X[both, b])
                    continue
                # halfspace projection; sequential term order and the single
                # residual/denom division are float-parity requirements with Rust
                cols = list(lin.indices)
                finite = ~np.isnan(X[:, cols]).any(axis=1)
                total = np.zeros(len(X))
                for coef, j in zip(lin.coefs, cols, strict=True):
                    total += coef * X[:, j]
                residual = total - lin.rhs
                with np.errstate(invalid="ignore"):
                    if lin.op == "<=":
                        hit = residual > 0.0
                    elif lin.op == ">=":
                        hit = residual < 0.0
                    else:
                        hit = residual != 0.0
                hit &= finite
                if not hit.any():
                    continue
                movable = [k for k, j in enumerate(cols) if not frozen[j] and lo[j] < hi[j]]
                denom = 0.0
                for k in movable:
                    denom += lin.coefs[k] * lin.coefs[k]
                if denom == 0.0:
                    continue
                step = residual[hit] / denom
                for k in movable:
                    j = cols[k]
                    v = X[hit, j] - lin.coefs[k] * step
                    v = np.where(v < lo[j], lo[j], v)
                    v = np.where(v > hi[j], hi[j], v)
                    X[hit, j] = v
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
        for derived in self.derived_ranges:
            j = index[derived.feature]
            lo[j] = max(lo[j], derived.lo)
            hi[j] = min(hi[j], derived.hi)
        return lo, hi, frozen

    def factual_violations(self, x: FloatArray) -> tuple[str, ...]:
        """One human-readable description per constraint the factual violates."""
        return tuple(desc for _, desc in self._factual_violation_items(x))

    def _factual_violation_items(self, x: FloatArray) -> tuple[tuple[str, str], ...]:
        """(label, description) pairs; labels are row-independent for aggregation.

        Same 1e-9 slack as ``check_matrix``. Freeze/Monotone are vacuous at the
        factual by construction; ``derived_ranges`` duplicate a reported Linear —
        all three are skipped.
        """
        slack = 1e-9
        index = {name: j for j, name in enumerate(self.feature_names)}
        items: list[tuple[str, str]] = []
        for c in self.constraints:
            if isinstance(c, Range):
                v = float(x[index[c.feature]])
                if not math.isnan(v) and not (c.lo - slack <= v <= c.hi + slack):
                    label = f"Range({c.feature!r}, {c.lo:g}, {c.hi:g})"
                    items.append((label, f"{label} violated at the factual (value={v:g})"))
            elif isinstance(c, Equals):
                v = float(x[index[c.feature]])
                if not math.isnan(v) and not abs(v - c.value) <= slack:
                    label = f"Equals({c.feature!r}, {c.value:g})"
                    items.append((label, f"{label} violated at the factual (value={v:g})"))
            elif isinstance(c, Implies):
                cond = x[index[c.condition.feature]] == c.condition.value
                cons = x[index[c.consequence.feature]] == c.consequence.value
                if cond and not cons:
                    label = (
                        f"Implies({c.condition.feature} == {c.condition.value:g} -> "
                        f"{c.consequence.feature} == {c.consequence.value:g})"
                    )
                    items.append((label, f"{label} violated at the factual"))
            elif isinstance(c, OneHot):
                total = sum(float(x[index[f]]) for f in c.features)
                # exact equality mirrors check_matrix; a NaN sum compares unequal
                if total != 1.0:
                    label = f"OneHot({', '.join(c.features)})"
                    items.append((label, f"{label} violated at the factual (sum={total:g})"))
        for lin in self.linears:
            label = _format_linear(lin)
            vals = [float(x[j]) for j in lin.indices]
            if any(math.isnan(v) for v in vals):
                if lin.missing_policy != "satisfied":
                    items.append((
                        label,
                        f"{label} references a missing value "
                        f"(missing_policy={lin.missing_policy})",
                    ))
                continue
            total = 0.0
            for coef, v in zip(lin.coefs, vals, strict=True):
                total += coef * v
            holds = (
                total <= lin.rhs + slack
                if lin.op == "<="
                else total >= lin.rhs - slack
                if lin.op == ">="
                else abs(total - lin.rhs) <= slack
            )
            if not holds:
                items.append((label, f"{label} violated at the factual (lhs={total:g})"))
        return tuple(items)


def compile_constraints(
    constraints: Sequence[Constraint], feature_names: Sequence[str]
) -> CompiledConstraints:
    """Validate the constraint set against the feature space and freeze it."""
    names = tuple(feature_names)
    index = {name: j for j, name in enumerate(names)}

    linears: list[ResolvedLinear] = []
    derived: list[Range] = []
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
            if len(c.coefficients) == 1:
                ((lin_name, coef),) = c.coefficients.items()
                if coef == 0.0:
                    vacuous = (
                        (c.op == "<=" and c.rhs >= 0.0)
                        or (c.op == ">=" and c.rhs <= 0.0)
                        or (c.op == "==" and c.rhs == 0.0)
                    )
                    if vacuous:
                        continue  # 0 op rhs holds for every x: no linear, no bound
                    raise ConstraintValidationError(
                        f"Linear constraint over {lin_name!r} is unsatisfiable: "
                        f"0 {c.op} {c.rhs}"
                    )
                # lower the constraint into an equivalent per-feature bound (the
                # Linear itself is retained so missing_policy still governs NaN).
                # Widen by the check_matrix slack (1e-9) translated into feature
                # space, so the derived bound never excludes a candidate that the
                # slacked linear check itself admits. For large |coef| that slack
                # can shrink below the float rounding gap between coef*candidate
                # (how check_matrix evaluates it) and rhs/coef (this bound), so
                # floor the widening at a few ulp of the bound too.
                bound = c.rhs / coef
                slack_feat = max(1e-9 / abs(coef), 4.0 * math.ulp(abs(bound)))
                if math.isinf(slack_feat):
                    pass  # subnormal coef: slack is unbounded; the retained Linear still governs
                elif c.op == "==":
                    derived.append(Range(lin_name, bound - slack_feat, bound + slack_feat))
                elif (c.op == "<=") == (coef > 0.0):
                    derived.append(Range(lin_name, -math.inf, bound + slack_feat))
                else:
                    derived.append(Range(lin_name, bound - slack_feat, math.inf))
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
        derived_ranges=tuple(derived),
    )


def _validated_binary(c: Equals, index: dict[str, int]) -> int:
    if c.feature not in index:
        raise ConstraintValidationError(f"Equals references unknown feature {c.feature!r}")
    if c.value not in (0.0, 1.0):
        raise ConstraintValidationError(
            f"Equals({c.feature!r}): only binary values 0/1 are supported, "
            f"got {c.value}"
        )
    return index[c.feature]
