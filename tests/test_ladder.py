"""Rating-ladder bands: one compilation, N solves (spec §6, §12.7)."""

from __future__ import annotations

import numpy as np
import pytest

import treecf.api as api_module
from treecf import Counterfactual, Explainer, Target
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree

pytest.importorskip("ortools")


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


def test_bands_return_per_band_results_with_one_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}
    original = api_module.build_problem

    def counting(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(api_module, "build_problem", counting)

    exp = Explainer(_ladder_ir(), normalizers=np.ones(1))
    ladder = exp.explain(
        np.array([0.0]),
        target=Target.bands({"C": (0.5, 1.5), "B": (1.5, 2.5)}, space="raw"),
    )
    assert set(ladder) == {"C", "B"}
    assert all(isinstance(v, Counterfactual) for v in ladder.values())
    assert ladder["C"].distance < ladder["B"].distance  # better grade costs more
    assert calls["n"] == 1  # §12.7: one AIM compilation for the whole ladder


def test_unreachable_band_reports_infeasible() -> None:
    exp = Explainer(_ladder_ir(), normalizers=np.ones(1))
    ladder = exp.explain(
        np.array([0.0]),
        target=Target.bands({"B": (1.5, 2.5), "impossible": (5.0, 9.0)}, space="raw"),
    )
    assert isinstance(ladder["B"], Counterfactual)
    assert not isinstance(ladder["impossible"], Counterfactual)
