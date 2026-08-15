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
    """A hard isolation-forest bound keeping counterfactuals inside the data manifold.

    Construct through ``Plausibility.isolation_forest`` rather than the
    constructor directly. Pass the result as ``Explainer(...,
    plausibility=...)``; every returned counterfactual then also satisfies
    ``anomaly_score(x_cf) <= max_anomaly_score``, enforced as a hard
    constraint by every backend. Cannot be combined with ``AllowMissing`` or a
    NaN-containing factual (isolation forests define no NaN routing). See
    [Plausibility](concepts/plausibility.md).

    Attributes:
        if_ir: The isolation forest, parsed through the same tree IR the
            model uses.
        max_anomaly_score: The upper bound on ``anomaly_score``; lower values
            are stricter (closer to the training distribution).
    """

    if_ir: EnsembleIR
    max_anomaly_score: float

    @classmethod
    def isolation_forest(
        cls, model_or_ir: object, max_anomaly_score: float = 0.55
    ) -> Plausibility:
        """Build a ``Plausibility`` bound from a fitted isolation forest.

        Args:
            model_or_ir: A native isolation-forest model (currently
                sklearn's ``IsolationForest``) or an already-parsed
                ``EnsembleIR``.
            max_anomaly_score: Upper bound on the isolation-forest anomaly
                score, in ``(0, 1)``; lower is a stricter plausibility
                requirement. Defaults to ``0.55``.

        Returns:
            A ``Plausibility`` wrapping the parsed forest.

        Raises:
            TreecfError: If ``max_anomaly_score`` is not in ``(0, 1)``.
        """
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
        """The average path length ``c(n)`` for the forest's subsample size ``n``.

        Standard isolation-forest normalizer, derived from ``if_ir``'s
        ``max_samples`` metadata; used to turn a raw total path length into
        the ``[0, 1]`` anomaly score.
        """
        from treecf.ir.parsers.sklearn import _avg_path

        return _avg_path(float(self.if_ir.meta["max_samples"]))  # type: ignore[arg-type]

    @property
    def min_total_path(self) -> float:
        """The feasibility bound compiled into every backend's plausibility check.

        Equivalent to ``max_anomaly_score`` re-expressed as a lower bound on
        the summed depth-adjusted path length across every tree: a
        counterfactual is plausible iff its total path length is at least
        this value.
        """
        n_trees = len(self.if_ir.trees)
        return -n_trees * self.normalizer * math.log2(self.max_anomaly_score)

    def anomaly_score(self, x: FloatArray) -> float:
        """The isolation-forest anomaly score at ``x``, in ``[0, 1]``.

        ``2 ** (-mean_path / normalizer)``: close to ``1`` for a point the
        forest isolates in very few splits (anomalous), close to ``0`` for
        one that takes many (typical). A point is plausible under this bound
        iff its score is ``<= max_anomaly_score``.

        Args:
            x: A feature vector, aligned to the forest's feature order.

        Returns:
            The anomaly score at ``x``.
        """
        total = raw_score(self.if_ir, np.asarray(x, dtype=np.float64))
        mean_path = total / len(self.if_ir.trees)
        return float(2.0 ** (-mean_path / self.normalizer))
