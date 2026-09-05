"""Behavior of the coarse-to-fine exact search on models built to exercise it."""

from __future__ import annotations

import math

import numpy as np
import pytest

from treecf.backends.exact import solve_exact
from treecf.constraints import compile_constraints
from treecf.ir.evaluate import raw_score
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree

from ..conftest import make_random_ir


def _leaf(i: int, v: float) -> Node:
    return Node(i, None, None, None, None, None, None, v)


def _stump(feature: int, threshold: float, right_value: float) -> Tree:
    return Tree(
        nodes=(
            Node(0, feature, threshold, SplitOp.LT, True, 1, 2, None),
            _leaf(1, 0.0),
            _leaf(2, right_value),
        )
    )


def _wide_ir(n_thresholds: int = 40) -> EnsembleIR:
    """One feature split at forty thresholds: a staircase of forty cells."""
    trees = tuple(_stump(0, float(t), 0.1) for t in range(1, n_thresholds + 1))
    return EnsembleIR(
        trees=(*trees, _stump(1, 0.5, 0.05)),
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=2,
        feature_names=("a", "b"),
        meta={},
    )


def _solve(ir: EnsembleIR, x, interval, search: str, **kw):
    compiled = compile_constraints([], ir.feature_names)
    n = ir.n_features
    return solve_exact(ir, x, interval, compiled, np.ones(n), np.ones(n), 0.0, search=search, **kw)


class TestRefineOnAStaircase:
    def test_refine_proves_the_same_optimum_with_range_level_accepts(self) -> None:
        ir = _wide_ir()
        x = np.array([0.0, 0.0])
        interval = (2.05, math.inf)  # needs a >= 21 (twenty-one steps of 0.1) or b's 0.05
        classic = _solve(ir, x, interval, "classic")
        refine = _solve(ir, x, interval, "refine")
        assert classic.proof == "optimal" and refine.proof == "optimal"
        assert refine.stats["search"] == "refine"
        assert refine.stats["completed"] is True
        assert refine.distance == pytest.approx(classic.distance, rel=1e-12)
        assert refine.stats["refinements"] > 0
        assert refine.stats["coarse_accepts"] >= 1
        assert refine.x_cf is not None
        assert raw_score(ir, refine.x_cf) >= interval[0]

    def test_refine_expands_fewer_nodes_than_classic_on_the_staircase(self) -> None:
        ir = _wide_ir()
        x = np.array([0.0, 0.0])
        interval = (2.05, math.inf)
        classic = _solve(ir, x, interval, "classic")
        refine = _solve(ir, x, interval, "refine")
        assert refine.stats["nodes_expanded"] < classic.stats["nodes_expanded"]

    def test_refine_certifies_infeasibility_like_classic(self) -> None:
        ir = _wide_ir()
        x = np.array([0.0, 0.0])
        refine = _solve(ir, x, (100.0, math.inf), "refine")
        assert refine.x_cf is None and refine.stats["completed"] is True

    def test_budget_exhaustion_reports_a_sound_lower_bound(self) -> None:
        ir = _wide_ir()
        x = np.array([0.0, 0.0])
        refine = _solve(ir, x, (2.05, math.inf), "refine", node_budget=2)
        assert refine.stats["completed"] is False
        assert refine.stats["nodes_expanded"] == 2
        assert refine.stats["lower_bound"] <= 21.0 + 1e-9
        assert refine.stats["trace"][-1][2] == refine.stats["lower_bound"]


class TestRefineArguments:
    def test_unknown_search_mode_is_rejected(self) -> None:
        ir = _wide_ir(3)
        with pytest.raises(ValueError, match="search"):
            _solve(ir, np.zeros(2), (0.5, 1.0), "bogus")

    def test_random_models_agree_with_classic_on_cost(self) -> None:
        for seed in range(8):
            rng = np.random.default_rng(seed)
            ir = make_random_ir(rng, n_features=4, n_trees=6, depth=3)
            x = rng.normal(size=4)
            classic = _solve(ir, x, (0.3, 0.9), "classic")
            refine = _solve(ir, x, (0.3, 0.9), "refine")
            assert (classic.x_cf is None) == (refine.x_cf is None), seed
            if classic.distance is not None:
                assert refine.distance == pytest.approx(classic.distance, rel=1e-12), seed
