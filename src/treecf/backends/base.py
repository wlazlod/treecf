"""Backend protocol (spec §8.4) — frozen in v0.1 so the v0.2 HiGHS adapter is a translation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt

from treecf.aim.model import AimProblem


@dataclass(frozen=True)
class BackendSolution:
    """Solver outcome in problem coordinates.

    ``values_scaled`` maps feature index -> integer value at scale K (the caller
    descales, mapping the factual anchor back to the exact float); ``status`` is
    "optimal", "feasible" (time limit hit with an incumbent), or "infeasible".
    """

    status: str
    values_scaled: dict[int, int] | None
    objective: float | None  # J units (descaled by K*Q)
    gap: float | None
    stats: dict[str, object] = field(default_factory=dict)
    missing: dict[int, bool] = field(default_factory=dict)  # feature index -> chose NaN


class Backend(Protocol):
    def solve(
        self,
        problem: AimProblem,
        time_budget_s: float,
        num_workers: int,
    ) -> BackendSolution: ...


FloatArray = npt.NDArray[np.float64]
