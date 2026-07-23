"""Rating-ladder bands: one result per named interval via the genetic engine."""

from __future__ import annotations

import math

import numpy as np

from treecf import Counterfactual, Explainer, Target
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree


def _leaf(i: int, v: float) -> Node:
    return Node(i, None, None, None, None, None, None, v)


def _ladder_ir() -> EnsembleIR:
    """Two stumps on one feature: raw score steps 0 -> 1 -> 2 as x crosses 1 and 2."""
    trees = tuple(
        Tree(
            nodes=(
                Node(0, 0, t, SplitOp.LT, True, 1, 2, None),
                _leaf(1, 0.0),
                _leaf(2, 1.0),
            )
        )
        for t in (1.0, 2.0)
    )
    return EnsembleIR(
        trees=trees,
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=1,
        feature_names=("score_driver",),
        meta={},
    )


def test_bands_return_per_band_results() -> None:
    exp = Explainer(_ladder_ir(), normalizers=np.ones(1))
    ladder = exp.explain(
        np.array([0.0]),
        target=Target.bands({"C": (0.5, 1.5), "B": (1.5, 2.5)}, space="raw"),
        seed=0,
    )
    assert isinstance(ladder, dict)
    assert set(ladder) == {"C", "B"}
    assert all(isinstance(v, Counterfactual) for v in ladder.values())
    assert ladder["C"].distance < ladder["B"].distance  # better grade costs more


def test_unreachable_band_reports_infeasible() -> None:
    exp = Explainer(_ladder_ir(), normalizers=np.ones(1))
    ladder = exp.explain(
        np.array([0.0]),
        target=Target.bands({"B": (1.5, 2.5), "impossible": (5.0, 9.0)}, space="raw"),
        seed=0,
    )
    assert isinstance(ladder, dict)
    assert isinstance(ladder["B"], Counterfactual)
    assert not isinstance(ladder["impossible"], Counterfactual)


def _sigmoid_ladder_ir() -> EnsembleIR:
    """The same two stumps, but with a SIGMOID link: margins 0 -> 1 -> 2."""
    trees = tuple(
        Tree(
            nodes=(
                Node(0, 0, t, SplitOp.LT, True, 1, 2, None),
                _leaf(1, 0.0),
                _leaf(2, 1.0),
            )
        )
        for t in (1.0, 2.0)
    )
    return EnsembleIR(
        trees=trees,
        base_score=0.0,
        link=Link.SIGMOID,
        n_features=1,
        feature_names=("score_driver",),
        meta={},
    )


class _AffineCal:
    """logit(g(p)) = logit(p) - 2: the model overestimates by 2 log-odds."""

    is_monotone_ = True

    def forward(self, p: float) -> float:
        z = math.log(p / (1.0 - p)) - 2.0
        return 1.0 / (1.0 + math.exp(-z))

    def interval_inverse(
        self, lo: float, hi: float, *, space: str = "probability", buffer_logit: float = 0.0
    ) -> tuple[float, float]:
        assert space == "logit"
        lo_z = -math.inf if lo <= 0.0 else math.log(lo / (1.0 - lo)) + buffer_logit + 2.0
        hi_z = math.inf if hi >= 1.0 else math.log(hi / (1.0 - hi)) - buffer_logit + 2.0
        return lo_z, hi_z


def test_calibrated_target_end_to_end() -> None:
    # Start at margin 2 (model p = 0.88, calibrated p = 0.5); demand calibrated
    # PD <= 0.2, which only margin 0 satisfies: g(sigmoid(0)) = sigmoid(-2) = 0.119.
    cal = _AffineCal()
    exp = Explainer(_sigmoid_ladder_ir(), normalizers=np.ones(1))
    res = exp.explain(
        np.array([5.0]),
        target=Target.calibrated(cal, op="<=", value=0.2),
        seed=0,
    )
    assert isinstance(res, Counterfactual)
    assert res.score_prob is not None
    # The preimage identity holds through the whole pipeline: the counterfactual's
    # CALIBRATED value lands inside the requested interval.
    assert cal.forward(res.score_prob) <= 0.2
