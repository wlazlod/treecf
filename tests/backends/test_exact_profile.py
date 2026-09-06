"""The search-space profile: what the exact backend would have to enumerate."""

from __future__ import annotations

import math

import numpy as np
import pytest

from treecf import Explainer, Freeze, Target, TreecfWarning
from treecf.aim.cells import category_blocks
from treecf.backends._exact_domains import _build_domains, _constraint_cells, _feature_order
from treecf.backends._exact_profile import search_space_profile
from treecf.constraints import compile_constraints

from ..conftest import make_random_ir, make_random_mixed_ir


def _problem(seed: int = 0, constraints=()):
    rng = np.random.default_rng(seed)
    ir = make_random_ir(rng, n_features=4, n_trees=5, depth=3)
    x = rng.normal(size=4)
    compiled = compile_constraints(list(constraints), ir.feature_names)
    return ir, x, compiled


class TestProfileTotals:
    def test_domain_sizes_agree_with_the_domain_machinery(self) -> None:
        ir, x, compiled = _problem()
        profile = search_space_profile(ir, x, compiled, np.ones(4), np.ones(4), 0.0, None, None)
        grids = _constraint_cells(compiled, ir)
        blocks = category_blocks(ir)
        domains = _build_domains(grids, x, compiled, np.ones(4), np.ones(4), 0.0, None, blocks)
        order = _feature_order(grids, compiled, blocks)
        for j, name in enumerate(ir.feature_names):
            entry = profile["features"][name]
            assert entry["kind"] == "numeric"
            assert entry["atomic"] == len(grids[j])
            assert entry["domain"] == len(domains[j])
            assert entry["frozen"] is False
            assert entry["influential"] == (j in order)
        assert profile["influential_features"] == len(order)
        expected = sum(math.log10(len(domains[j])) for j in order)
        assert profile["log10_states"] == pytest.approx(expected)
        assert "presolve_certified" not in profile

    def test_categorical_features_count_blocks(self) -> None:
        rng = np.random.default_rng(3)
        ir = make_random_mixed_ir(rng, n_features=4, n_trees=4, depth=3, categorical={1: 5})
        x = np.array([0.0, 2.0, 0.0, 0.0])
        compiled = compile_constraints([], ir.feature_names, ir.categorical)
        profile = search_space_profile(ir, x, compiled, np.ones(4), np.ones(4), 0.0, None, None)
        entry = profile["features"][ir.feature_names[1]]
        assert entry["kind"] == "categorical"
        assert entry["atomic"] == len(category_blocks(ir)[1])

    def test_freezing_shrinks_the_space(self) -> None:
        ir, x, compiled = _problem()
        full = search_space_profile(ir, x, compiled, np.ones(4), np.ones(4), 0.0, None, None)
        _, _, frozen = _problem(constraints=[Freeze(n) for n in ir.feature_names[:3]])
        small = search_space_profile(ir, x, frozen, np.ones(4), np.ones(4), 0.0, None, None)
        assert small["log10_states"] < full["log10_states"]
        assert small["features"][ir.feature_names[0]]["frozen"] is True
        assert small["features"][ir.feature_names[0]]["domain"] == 1

    def test_target_adds_presolved_sizes(self) -> None:
        ir, x, compiled = _problem()
        profile = search_space_profile(
            ir, x, compiled, np.ones(4), np.ones(4), 0.0, None, None, interval=(0.2, 0.9)
        )
        assert profile["presolve_certified"] is False
        for entry in profile["features"].values():
            assert entry["presolved"] <= entry["domain"]
        assert profile["log10_states_presolved"] <= profile["log10_states"]
        unreachable = search_space_profile(
            ir, x, compiled, np.ones(4), np.ones(4), 0.0, None, None, interval=(1e6, 1e7)
        )
        assert unreachable["presolve_certified"] is True


class TestExplainerSurface:
    def test_public_method_and_target(self) -> None:
        ir, x, _ = _problem()
        exp = Explainer(ir, normalizers=np.ones(4))
        profile = exp.search_profile(x)
        assert set(profile) >= {"features", "influential_features", "log10_states"}
        with_target = exp.search_profile(x, Target.raw(op=">=", value=0.5))
        assert "presolve_certified" in with_target

    def test_exhaustion_warning_names_the_search_space(self) -> None:
        ir, x, _ = _problem()
        exp = Explainer(ir, normalizers=np.ones(4))
        with pytest.warns(TreecfWarning, match=r"search space ≈ 10\^") as record:
            exp.explain(
                x, Target.raw(op=">=", value=0.5), backend="exact", seed=0,
                warm_start=False, node_budget=1, time_budget_s=5.0,
            )
        assert "see Explainer.search_profile" in str(record[0].message)
