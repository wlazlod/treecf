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
    changed_positions: tuple[int, ...] = ()  # block positions with z = 1
    chosen_cells: tuple[tuple[int, int], ...] = ()  # (block position, cell position | -1=NaN)


@dataclass(frozen=True)
class DiversityCut:
    """No-good cut derived from a previous solution (spec §8.3).

    mode "distinct_changes" forbids repeating the exact change-set (a NaN flip
    counts as a change, OQ3); "distinct_solution" forbids the exact cell/missing
    assignment.
    """

    mode: str
    changed: tuple[int, ...] = ()
    unchanged: tuple[int, ...] = ()
    chosen: tuple[tuple[int, int], ...] = ()


class Backend(Protocol):
    def solve(
        self,
        problem: AimProblem,
        time_budget_s: float,
        num_workers: int,
    ) -> BackendSolution: ...


FloatArray = npt.NDArray[np.float64]
