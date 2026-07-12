"""Batch GA entry point: bitwise equality with per-task single solves.

`solve_genetic_batch_raw` fans independently seeded searches out with rayon;
every task must be bitwise-identical to the corresponding `solve_genetic_raw`
call, and the whole batch must be invariant to the rayon thread count (which
also exercises the inner-parallel heuristic at the Python level).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ..parity.harness import Scenario, load_scenario, scenario_paths
from .test_constraints_conformance import rust_constraints
from .test_ir_conformance import rust_ensemble

pytestmark = pytest.mark.rust

PATHS = scenario_paths()


def _tasks_for(scenario: Scenario) -> tuple[np.ndarray, list[tuple[int, int]]]:
    x_rows = np.stack([scenario.x, np.roll(scenario.x, 1)])
    return x_rows, [(0, 11), (0, 12), (1, 13), (1, 11)]


def run_batch(scenario: Scenario) -> list[tuple[object, int]]:
    from treecf.backends.genetic_rust import _core as _load_core

    _treecf_core = _load_core()
    ens = rust_ensemble(scenario.ir)
    cons = rust_constraints(scenario.compiled)
    if_ens = rust_ensemble(scenario.if_ir) if scenario.if_ir is not None else None
    x_rows, tasks = _tasks_for(scenario)
    x_cf, feasible, generations = _treecf_core.solve_genetic_batch_raw(
        ens,
        cons,
        np.ascontiguousarray(x_rows),
        np.asarray([row for row, _ in tasks], dtype=np.uint64),
        np.asarray([seed for _, seed in tasks], dtype=np.uint64),
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
        population=int(scenario.ga["population"]),
        max_generations=int(scenario.ga["max_generations"]),
        stall_generations=int(scenario.ga["stall_generations"]),
        time_budget_s=1e9,
    )
    return [
        (np.asarray(x_cf[t]) if feasible[t] else None, int(generations[t]))
        for t in range(len(tasks))
    ]


@pytest.mark.parametrize("path", PATHS, ids=[p.stem for p in PATHS])
def test_batch_matches_looped_single_solves(path: object) -> None:
    from treecf.backends.genetic_rust import _core as _load_core

    scenario = load_scenario(path)  # type: ignore[arg-type]
    _treecf_core = _load_core()
    ens = rust_ensemble(scenario.ir)
    cons = rust_constraints(scenario.compiled)
    if_ens = rust_ensemble(scenario.if_ir) if scenario.if_ir is not None else None
    x_rows, tasks = _tasks_for(scenario)

    batch = run_batch(scenario)
    for (row, seed), (batch_cf, batch_gens) in zip(tasks, batch, strict=True):
        single_cf, single_gens = _treecf_core.solve_genetic_raw(
            ens,
            cons,
            np.ascontiguousarray(x_rows[row]),
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
            time_budget_s=1e9,
        )
        assert batch_gens == single_gens
        if single_cf is None:
            assert batch_cf is None
        else:
            assert batch_cf is not None
            assert np.array_equal(batch_cf, np.asarray(single_cf), equal_nan=True)


def test_batch_thread_count_invariance() -> None:
    """The batch entry must be bitwise-identical at 1, 2, and 4 rayon threads."""
    snippet = r"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from tests.parity.harness import load_scenario, scenario_paths
from tests.rust.test_ga_batch import run_batch
scenario = load_scenario(scenario_paths()[0])
results = run_batch(scenario)
print(json.dumps([[None if cf is None else cf.tolist(), gens] for cf, gens in results]))
"""
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
