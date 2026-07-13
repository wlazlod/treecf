"""Public-API smoke for backend='genetic-rust'."""

from __future__ import annotations

import math

import numpy as np
import pytest

from treecf import AllowMissing, Counterfactual, Explainer, Freeze, Infeasible, Monotone, Target
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree

from ..parity.harness import load_scenario, run_python, scenario_paths

pytestmark = pytest.mark.rust


def _stump() -> EnsembleIR:
    nodes = (
        Node(0, 0, 1.0, SplitOp.LT, True, 1, 2, None),
        Node(1, None, None, None, None, None, None, -1.0),
        Node(2, None, None, None, None, None, None, 1.0),
    )
    return EnsembleIR(
        trees=(Tree(nodes=nodes),),
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=2,
        feature_names=("a", "b"),
        meta={},
    )


def test_finds_feasible_solution_with_heuristic_proof() -> None:
    exp = Explainer(_stump(), normalizers=np.ones(2))
    res = exp.explain(
        np.array([0.0, 0.0]),
        target=Target.raw(op=">=", value=0.5),
        backend="genetic-rust",
        seed=1,
    )
    assert isinstance(res, Counterfactual)
    assert res.proof == "heuristic"
    assert res.solver_stats.get("backend") == "rust"
    assert res.score_raw >= 0.5
    assert res.x_cf[1] == 0.0


def test_respects_freeze_and_monotone() -> None:
    exp = Explainer(
        _stump(),
        normalizers=np.ones(2),
        constraints=[Freeze("b"), Monotone("a", "increase")],
    )
    res = exp.explain(
        np.array([0.0, 3.0]),
        target=Target.raw(op=">=", value=0.5),
        backend="genetic-rust",
        seed=1,
    )
    assert isinstance(res, Counterfactual)
    assert res.x_cf[0] >= 0.0
    assert res.x_cf[1] == 3.0


def test_infeasible_reports_heuristic_exhaustion() -> None:
    exp = Explainer(_stump(), normalizers=np.ones(2), constraints=[Freeze("a")])
    res = exp.explain(
        np.array([0.0, 0.0]),
        target=Target.raw(op=">=", value=0.5),
        backend="genetic-rust",
        seed=1,
    )
    assert isinstance(res, Infeasible)
    assert "heuristic" in res.reason


def test_same_seed_is_deterministic_and_cache_is_reused() -> None:
    exp = Explainer(_stump(), normalizers=np.ones(2))
    x = np.array([0.0, 0.0])
    target = Target.raw(op=">=", value=0.5)
    r1 = exp.explain(x, target=target, backend="genetic-rust", seed=7)
    r2 = exp.explain(x, target=target, backend="genetic-rust", seed=7)
    assert isinstance(r1, Counterfactual) and isinstance(r2, Counterfactual)
    np.testing.assert_array_equal(r1.x_cf, r2.x_cf)
    assert exp._rust_cache  # marshaled objects retained across calls


def test_nan_flip_available() -> None:
    exp = Explainer(
        _stump(),
        normalizers=np.ones(2),
        constraints=[AllowMissing("a", delta_miss=0.1)],
    )
    res = exp.explain(
        np.array([5.0, 0.0]),
        target=Target.raw(op="<=", value=-0.5),
        backend="genetic-rust",
        seed=3,
    )
    assert isinstance(res, Counterfactual)
    assert math.isnan(res.x_cf[0]) or res.x_cf[0] < 1.0


def test_stage_a_battery_through_public_api() -> None:
    """Every parity scenario solved via Explainer with the rust backend and verified."""
    for path in scenario_paths():
        scenario = load_scenario(path)
        exp = Explainer(
            scenario.ir,
            normalizers=scenario.sigma,
            constraints=list(scenario.compiled.constraints),
        )
        exp.background = scenario.background
        if scenario.if_ir is not None:
            # fixtures carry the raw (if_ir, min_total_path) bound; inject it directly
            bound = (scenario.if_ir, float(scenario.min_total_path))  # type: ignore[arg-type]
            exp._plausibility_bound = lambda bound=bound: bound  # type: ignore[method-assign]
        res = exp.explain(
            scenario.x,
            target=Target.raw(range=(scenario.interval[0], scenario.interval[1]))
            if math.isfinite(scenario.interval[0]) and math.isfinite(scenario.interval[1])
            else Target.raw(op=">=", value=scenario.interval[0])
            if math.isfinite(scenario.interval[0])
            else Target.raw(op="<=", value=scenario.interval[1]),
            backend="genetic-rust",
            seed=0,
            sparsity_weight=scenario.lam,
        )
        python_record = run_python(scenario, 0)
        if python_record["feasible"]:
            assert isinstance(res, Counterfactual), f"{scenario.name}: rust found nothing"
        # Counterfactual results pass Explainer._verify by construction (API verifies)
