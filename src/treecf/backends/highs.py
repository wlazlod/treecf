"""HiGHS MILP adapter — interface frozen in v0.1, implemented in v0.2 (spec §8.4)."""

from __future__ import annotations

from treecf.aim.model import AimProblem
from treecf.backends.base import BackendSolution


class HighsBackend:
    def solve(
        self,
        problem: AimProblem,
        time_budget_s: float,
        num_workers: int,
    ) -> BackendSolution:
        raise NotImplementedError(
            "the HiGHS backend is planned for v0.2; use backend='cpsat' or 'genetic'"
        )
