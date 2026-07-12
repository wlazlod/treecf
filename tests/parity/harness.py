"""Stage A parity harness (migration P1).

Scenario fixtures are JSON files under tests/fixtures/parity/ carrying the
flat-array ensemble/constraint contract (shared with the Rust core), the
factual instance, GA parameters, golden per-seed records (Python-vs-Python
regression) and 200-seed distributional summaries (Rust-vs-Python parity).

JSON portability: NaN and infinities are encoded as null / "inf" / "-inf"
because serde_json (the Rust consumer) rejects non-standard JSON literals.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from treecf._json import decode_floats, encode_floats
from treecf.backends.genetic import solve_genetic
from treecf.constraints import (
    AllowMissing,
    Equals,
    Freeze,
    Implies,
    Linear,
    Monotone,
    OneHot,
    Range,
    compile_constraints,
)
from treecf.constraints.compile import CompiledConstraints
from treecf.constraints.objects import Constraint
from treecf.ir.flatten import unflatten_ir
from treecf.ir.model import EnsembleIR

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "parity"

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Scenario:
    name: str
    ir: EnsembleIR
    if_ir: EnsembleIR | None
    min_total_path: float | None
    x: FloatArray
    sigma: FloatArray
    weights: FloatArray
    lam: float
    interval: tuple[float, float]
    compiled: CompiledConstraints
    background: FloatArray | None
    ga: dict[str, Any]
    golden: list[dict[str, Any]]
    dist: dict[str, list[Any]]
    dist_seeds: list[int]


def build_constraints(descriptors: list[dict[str, Any]]) -> list[Constraint]:
    out: list[Constraint] = []
    for d in descriptors:
        kind = d["type"]
        if kind == "Freeze":
            out.append(Freeze(d["feature"]))
        elif kind == "Monotone":
            out.append(Monotone(d["feature"], d["direction"]))
        elif kind == "Range":
            out.append(Range(d["feature"], d["lo"], d["hi"]))
        elif kind == "Equals":
            out.append(Equals(d["feature"], d["value"]))
        elif kind == "Linear":
            out.append(
                Linear(
                    coefficients=dict(d["coefficients"]),
                    op=d["op"],
                    rhs=d["rhs"],
                    missing_policy=d.get("missing_policy", "satisfied"),
                )
            )
        elif kind == "Implies":
            out.append(
                Implies(
                    Equals(d["cond_feature"], d["cond_value"]),
                    Equals(d["cons_feature"], d["cons_value"]),
                )
            )
        elif kind == "OneHot":
            out.append(OneHot(tuple(d["features"])))
        elif kind == "AllowMissing":
            out.append(
                AllowMissing(
                    d["feature"], d["delta_miss"], d.get("delta_from_miss")
                )
            )
        else:
            raise ValueError(f"unknown constraint descriptor type {kind!r}")
    return out


def _decode_array(values: Any) -> FloatArray:
    return np.asarray(decode_floats(values), dtype=np.float64)


def load_scenario(path: Path) -> Scenario:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    ir = unflatten_ir(_decode_ensemble(data["ensemble"]))
    if_ir = (
        unflatten_ir(_decode_ensemble(data["if_ensemble"]))
        if data.get("if_ensemble")
        else None
    )
    constraints = build_constraints(data["constraints"])
    compiled = compile_constraints(constraints, ir.feature_names)
    interval_raw = decode_floats(data["interval"])
    return Scenario(
        name=data["name"],
        ir=ir,
        if_ir=if_ir,
        min_total_path=data.get("min_total_path"),
        x=_decode_array(data["x"]),
        sigma=_decode_array(data["sigma"]),
        weights=_decode_array(data["weights"]),
        lam=float(data["lam"]),
        interval=(float(interval_raw[0]), float(interval_raw[1])),
        compiled=compiled,
        background=_decode_array(data["background"]) if data.get("background") else None,
        ga=data["ga"],
        golden=data["golden"],
        dist=data["dist"],
        dist_seeds=data["dist_seeds"],
    )


def _decode_ensemble(raw: dict[str, Any]) -> dict[str, Any]:
    out = dict(raw)
    for key in ("threshold", "value"):
        out[key] = _decode_array(raw[key])
    for key, dtype in (
        ("feature", np.int32),
        ("is_lt", np.uint8),
        ("missing_left", np.uint8),
        ("left", np.uint32),
        ("right", np.uint32),
        ("tree_roots", np.uint32),
    ):
        out[key] = np.asarray(raw[key], dtype=dtype)
    return out


def run_python(scenario: Scenario, seed: int) -> dict[str, Any]:
    """One GA run through the current Python backend -> comparable record."""
    plausibility = None
    if scenario.if_ir is not None:
        assert scenario.min_total_path is not None
        plausibility = (scenario.if_ir, float(scenario.min_total_path))
    result = solve_genetic(
        scenario.ir,
        scenario.x,
        scenario.interval,
        scenario.compiled,
        scenario.sigma,
        scenario.weights,
        lam=scenario.lam,
        background=scenario.background,
        plausibility=plausibility,
        seed=seed,
        population=int(scenario.ga["population"]),
        max_generations=int(scenario.ga["max_generations"]),
        stall_generations=int(scenario.ga["stall_generations"]),
        time_budget_s=float(scenario.ga["time_budget_s"]),
    )
    if result.x_cf is None:
        return {
            "seed": seed,
            "feasible": False,
            "x_cf": None,
            "j": None,
            "n_changed": None,
            "generations": int(result.stats["generations"]),  # type: ignore[call-overload]
        }
    return {
        "seed": seed,
        "feasible": True,
        "x_cf": encode_floats(result.x_cf),
        "j": objective_j(scenario, result.x_cf),
        "n_changed": n_changed(scenario.x, result.x_cf),
        "generations": int(result.stats["generations"]),  # type: ignore[call-overload]
    }


def objective_j(scenario: Scenario, x_cf: FloatArray) -> float:
    """J = sum_j w_j * d_j / sigma_j + lam * #changed — same pricing as the GA."""
    total = 0.0
    allow = scenario.compiled.allow_missing
    for j in range(len(scenario.x)):
        x_nan, cf_nan = math.isnan(scenario.x[j]), math.isnan(x_cf[j])
        if (x_nan and cf_nan) or scenario.x[j] == x_cf[j]:
            continue
        if cf_nan:
            delta = allow[j][0]
        elif x_nan:
            delta = allow[j][1]
        else:
            delta = abs(x_cf[j] - scenario.x[j])
        total += scenario.weights[j] * delta / scenario.sigma[j] + scenario.lam
    return float(total)


def n_changed(x: FloatArray, x_cf: FloatArray) -> int:
    changed = 0
    for j in range(len(x)):
        x_nan, cf_nan = math.isnan(x[j]), math.isnan(x_cf[j])
        if (x_nan and cf_nan) or x[j] == x_cf[j]:
            continue
        changed += 1
    return changed


def scenario_paths() -> list[Path]:
    return sorted(FIXTURES_DIR.glob("*.json"))
