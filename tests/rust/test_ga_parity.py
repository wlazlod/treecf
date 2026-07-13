"""Rust GA statistical parity vs the recorded distributional baselines.

Bitwise comparison is meaningless across RNGs; the gates are:
- feasibility rate within 3 percentage points,
- final J: two-sample KS statistic under the alpha=0.01 critical value and
  median within max(2%, 0.02) of the Python median,
- generations median within +-20% (floor 5),
- every Rust counterfactual passes the same float-space checks Python enforces
  (target, constraints, plausibility) — absolute, not distributional.
"""

from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest

from treecf.ir.evaluate import raw_score

from ..parity.harness import (
    Scenario,
    load_scenario,
    n_changed,
    objective_j,
    scenario_paths,
)
from .test_constraints_conformance import rust_constraints
from .test_ir_conformance import rust_ensemble

pytestmark = pytest.mark.rust

PATHS = scenario_paths()


def run_rust(scenario: Scenario, seed: int) -> dict[str, object]:
    from treecf.backends.genetic_rust import _core as _load_core
    _treecf_core = _load_core()

    ens = rust_ensemble(scenario.ir)
    cons = rust_constraints(scenario.compiled)
    if_ens = rust_ensemble(scenario.if_ir) if scenario.if_ir is not None else None
    x_cf, generations = _treecf_core.solve_genetic_raw(
        ens,
        cons,
        np.ascontiguousarray(scenario.x),
        scenario.interval[0],
        scenario.interval[1],
        np.ascontiguousarray(scenario.sigma),
        np.ascontiguousarray(scenario.weights),
        scenario.lam,
        background=(
            np.ascontiguousarray(scenario.background)
            if scenario.background is not None
            else None
        ),
        if_ensemble=if_ens,
        min_total_path=scenario.min_total_path,
        seed=seed,
        population=int(scenario.ga["population"]),
        max_generations=int(scenario.ga["max_generations"]),
        stall_generations=int(scenario.ga["stall_generations"]),
        time_budget_s=float(scenario.ga["time_budget_s"]),
    )
    if x_cf is None:
        return {"feasible": False, "j": None, "generations": generations, "x_cf": None}
    x_cf = np.asarray(x_cf)
    return {
        "feasible": True,
        "j": objective_j(scenario, x_cf),
        "n_changed": n_changed(scenario.x, x_cf),
        "generations": generations,
        "x_cf": x_cf,
    }


def ks_statistic(a: list[float], b: list[float]) -> float:
    a_sorted = np.sort(np.asarray(a))
    b_sorted = np.sort(np.asarray(b))
    grid = np.concatenate([a_sorted, b_sorted])
    cdf_a = np.searchsorted(a_sorted, grid, side="right") / len(a_sorted)
    cdf_b = np.searchsorted(b_sorted, grid, side="right") / len(b_sorted)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _verify_soundness(scenario: Scenario, x_cf: np.ndarray) -> None:
    score = raw_score(scenario.ir, x_cf)
    assert scenario.interval[0] <= score <= scenario.interval[1]
    ok = scenario.compiled.check_matrix(x_cf.reshape(1, -1), scenario.x)
    assert bool(ok[0])
    if scenario.if_ir is not None:
        assert scenario.min_total_path is not None
        assert raw_score(scenario.if_ir, x_cf) >= scenario.min_total_path


@pytest.mark.parametrize("path", PATHS, ids=[p.stem for p in PATHS])
def test_statistical_parity_and_soundness(path: object) -> None:
    scenario = load_scenario(path)  # type: ignore[arg-type]
    seeds = [int(s) for s in scenario.dist_seeds]

    rust = [run_rust(scenario, seed) for seed in seeds]
    for record in rust:
        if record["feasible"]:
            _verify_soundness(scenario, record["x_cf"])  # type: ignore[arg-type]

    py_feasible = np.asarray(scenario.dist["feasible"], dtype=bool)
    rs_feasible = np.asarray([r["feasible"] for r in rust], dtype=bool)
    assert abs(rs_feasible.mean() - py_feasible.mean()) <= 0.03, (
        f"feasibility {rs_feasible.mean():.2%} vs Python {py_feasible.mean():.2%}"
    )

    py_j = [j for j in scenario.dist["j"] if j is not None]
    rs_j = [r["j"] for r in rust if r["feasible"]]
    if py_j and rs_j:
        d = ks_statistic(py_j, rs_j)  # type: ignore[arg-type]
        n, m = len(py_j), len(rs_j)
        critical = 1.628 * math.sqrt((n + m) / (n * m))  # alpha = 0.01
        med_py, med_rs = float(np.median(py_j)), float(np.median(rs_j))
        tol = max(0.02 * abs(med_py), 0.02)
        assert d <= critical, f"KS D={d:.3f} > {critical:.3f} (medians {med_py:.4f}/{med_rs:.4f})"
        assert abs(med_rs - med_py) <= max(tol, d * 0.0 + tol), (
            f"J median {med_rs:.4f} vs Python {med_py:.4f}"
        )

    gens_py = float(np.median(scenario.dist["generations"]))
    gens_rs = float(np.median([r["generations"] for r in rust]))
    assert abs(gens_rs - gens_py) <= max(0.2 * gens_py, 5.0), (
        f"generations median {gens_rs} vs Python {gens_py}"
    )


@pytest.mark.parametrize("seed", range(10))
def test_oracle_bracket(seed: int) -> None:
    """Rust GA never beats the brute-force optimum and lands close."""
    from treecf.backends.genetic_rust import _core as _load_core
    _treecf_core = _load_core()

    from treecf.constraints import compile_constraints

    from ..conftest import make_random_ir
    from ..exactness.brute_force import solve_brute_force

    rng = np.random.default_rng(5000 + seed)
    ir = make_random_ir(rng, n_features=3, n_trees=4, depth=3)
    x = rng.normal(scale=2.0, size=3)
    scores = [raw_score(ir, rng.normal(scale=3.0, size=3)) for _ in range(40)]
    lo_t = float(np.percentile(scores, 60))

    compiled = compile_constraints([], ir.feature_names)
    oracle = solve_brute_force(
        ir, x, (lo_t, math.inf), compiled, np.ones(3), np.ones(3), lam=0.0
    )
    x_cf, _gens = _treecf_core.solve_genetic_raw(
        rust_ensemble(ir),
        rust_constraints(compiled),
        np.ascontiguousarray(x),
        lo_t,
        float("inf"),
        np.ones(3),
        np.ones(3),
        0.0,
        seed=seed,
        time_budget_s=1e9,
    )

    if oracle.feasible:
        assert x_cf is not None, "Rust GA missed an oracle-feasible case"
        x_cf = np.asarray(x_cf)
        assert raw_score(ir, x_cf) >= lo_t
        distance = float(np.sum(np.abs(x_cf - x)))
        assert distance >= oracle.objective - 1e-9  # never beats the optimum
        assert distance <= oracle.objective + 1.0


def test_thread_count_invariance() -> None:
    """Same seed must give bitwise-identical results at 1, 2, and 4 rayon threads."""
    snippet = r"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from tests.parity.harness import load_scenario, scenario_paths
from tests.rust.test_ga_parity import run_rust
scenario = load_scenario(scenario_paths()[0])
record = run_rust(scenario, 42)
print(json.dumps([None if record["x_cf"] is None else np.asarray(record["x_cf"]).tolist(),
                  record["generations"]]))
"""
    import os
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    outputs = []
    for threads in ("1", "2", "4"):
        env = dict(os.environ, RAYON_NUM_THREADS=threads)
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            check=True,
            env=env,
            cwd=str(repo_root),
        )
        outputs.append(result.stdout.strip())
    assert outputs[0] == outputs[1] == outputs[2]
