"""Golden per-seed regression: the Python GA is frozen for the migration's duration.

If one of these fails, the Python GA's behavior changed — either revert the
change or regenerate fixtures DELIBERATELY via scripts/gen_parity_fixtures.py
(and say so in the commit message).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from .harness import decode_floats, load_scenario, run_python, scenario_paths

PATHS = scenario_paths()
assert PATHS, "parity fixtures missing — run scripts/gen_parity_fixtures.py"


@pytest.mark.parametrize("path", PATHS, ids=[p.stem for p in PATHS])
def test_golden_seeds_reproduce_exactly(path: Path) -> None:
    scenario = load_scenario(path)
    for expected in scenario.golden:
        actual = run_python(scenario, int(expected["seed"]))
        assert actual["feasible"] == expected["feasible"], expected["seed"]
        assert actual["generations"] == expected["generations"], expected["seed"]
        if not expected["feasible"]:
            continue
        assert actual["n_changed"] == expected["n_changed"], expected["seed"]
        assert actual["j"] == pytest.approx(expected["j"], abs=0.0), expected["seed"]
        expected_x = np.asarray(decode_floats(expected["x_cf"]), dtype=np.float64)
        actual_x = np.asarray(decode_floats(actual["x_cf"]), dtype=np.float64)
        np.testing.assert_array_equal(actual_x, expected_x)


def test_distributional_summaries_are_present() -> None:
    for path in PATHS:
        scenario = load_scenario(path)
        n = len(scenario.dist_seeds)
        assert n == 200
        assert len(scenario.dist["feasible"]) == n
        assert len(scenario.dist["generations"]) == n
        feasible_js = [j for j in scenario.dist["j"] if j is not None]
        if any(scenario.dist["feasible"]):
            assert feasible_js and all(math.isfinite(j) for j in feasible_js)
