"""Plausibility as a hard isolation-forest constraint.

Anomaly score ``s(x) = 2 ** (-E[h(x)] / c(n))`` with the forest parsed through
the same IR (leaf value = depth-adjusted path length). The bound ``s(x') <= theta``
is linear in leaf indicators: ``sum_t h_t(x') >= -T * c(n) * log2(theta)``.

Cost note: the IF trees join cell construction and add one boolean per IF leaf —
roughly doubling model size for a typical forest. ``plausibility=None`` costs nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from treecf._errors import TreecfError
from treecf.ir.evaluate import raw_score
from treecf.ir.model import EnsembleIR

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Plausibility:
    if_ir: EnsembleIR
    max_anomaly_score: float

    @classmethod
    def isolation_forest(
        cls, model_or_ir: object, max_anomaly_score: float = 0.55
    ) -> Plausibility:
        if not 0.0 < max_anomaly_score < 1.0:
            raise TreecfError("max_anomaly_score must lie in (0, 1)")
        if isinstance(model_or_ir, EnsembleIR):
            if_ir = model_or_ir
        else:
            from treecf.ir.parsers.sklearn import parse_isolation_forest

            if_ir = parse_isolation_forest(model_or_ir)
        return cls(if_ir=if_ir, max_anomaly_score=max_anomaly_score)

    @property
    def normalizer(self) -> float:
        """c(n) for the forest's subsample size."""
        from treecf.ir.parsers.sklearn import _avg_path

        return _avg_path(float(self.if_ir.meta["max_samples"]))  # type: ignore[arg-type]

    @property
    def min_total_path(self) -> float:
        """Feasibility bound: sum_t h_t(x') >= -T * c(n) * log2(theta)."""
        n_trees = len(self.if_ir.trees)
        return -n_trees * self.normalizer * math.log2(self.max_anomaly_score)

    def anomaly_score(self, x: FloatArray) -> float:
        total = raw_score(self.if_ir, np.asarray(x, dtype=np.float64))
        mean_path = total / len(self.if_ir.trees)
        return float(2.0 ** (-mean_path / self.normalizer))
