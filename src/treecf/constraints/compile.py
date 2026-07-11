"""Constraint compiler (spec §7.3) — the single source of truth for constraint semantics.

M1 scope: Freeze/Monotone/Range compile to per-feature interval bounds given the
factual instance. The same visitor grows AIM forms and genetic check/repair pairs
in M2/M3; no per-backend constraint logic may exist elsewhere.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from treecf._errors import ConstraintValidationError
from treecf.constraints.objects import Constraint, Freeze, Monotone, Range

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class CompiledConstraints:
    """Constraints resolved against a feature space."""

    feature_names: tuple[str, ...]
    constraints: tuple[Constraint, ...]

    def instance_bounds(self, x: FloatArray) -> tuple[FloatArray, FloatArray, BoolArray]:
        """Per-feature (lo, hi, frozen) for factual ``x``; bounds are intersected."""
        p = len(self.feature_names)
        lo = np.full(p, -math.inf)
        hi = np.full(p, math.inf)
        frozen = np.zeros(p, dtype=bool)
        index = {name: j for j, name in enumerate(self.feature_names)}
        for constraint in self.constraints:
            j = index[constraint.feature]
            if isinstance(constraint, Freeze):
                lo[j] = max(lo[j], x[j])
                hi[j] = min(hi[j], x[j])
                frozen[j] = True
            elif isinstance(constraint, Range):
                lo[j] = max(lo[j], constraint.lo)
                hi[j] = min(hi[j], constraint.hi)
            elif constraint.direction == "increase":
                lo[j] = max(lo[j], x[j])
            else:
                hi[j] = min(hi[j], x[j])
        return lo, hi, frozen


def compile_constraints(
    constraints: Sequence[Constraint], feature_names: Sequence[str]
) -> CompiledConstraints:
    """Validate the constraint set against the feature space and freeze it."""
    names = tuple(feature_names)
    known = set(names)
    for constraint in constraints:
        if constraint.feature not in known:
            raise ConstraintValidationError(
                f"{type(constraint).__name__} references unknown feature {constraint.feature!r}"
            )
        if isinstance(constraint, Range) and constraint.lo > constraint.hi:
            raise ConstraintValidationError(
                f"Range({constraint.feature!r}): lo {constraint.lo} > hi {constraint.hi}"
            )
        if isinstance(constraint, Monotone) and constraint.direction not in (
            "increase",
            "decrease",
        ):
            raise ConstraintValidationError(
                f"Monotone({constraint.feature!r}): direction must be 'increase' or "
                f"'decrease', got {constraint.direction!r}"
            )
    return CompiledConstraints(feature_names=names, constraints=tuple(constraints))
