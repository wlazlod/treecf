"""Golden exact-backend fixture loading — shared by the generator and the tests.

Fixtures are JSON files under ``tests/fixtures/exact/`` carrying the exact
backend's full input contract (flat-array ensemble, constraint descriptors in
the parity harness's format, value-policy descriptors, an optional pinned
warm-start incumbent) plus the golden output the Python backend produced for
those inputs. ``time_budget_s`` is always ``1e9`` in every fixture — the
budget must never be what decides ``completed``, or the freeze would be
machine-dependent.

This module is the one place that knows the JSON layout, so the Rust parity
harness (Task 2.8) can import it too instead of re-deriving the schema.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from treecf._json import decode_floats, encode_floats
from treecf.api import Grid, ValuePolicy
from treecf.backends.exact import ExactResult, solve_exact
from treecf.constraints.compile import CompiledConstraints, compile_constraints
from treecf.ir.flatten import flatten_ir, unflatten_ir
from treecf.ir.model import EnsembleIR

from ..parity.harness import build_constraints

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "exact"
REGION_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "regions"

FloatArray = npt.NDArray[np.float64]


def encode_value_policy(policy: ValuePolicy) -> Any:
    """A fixture never carries a callable policy (unsupported by the exact backend)."""
    if isinstance(policy, Grid):
        return {"type": "grid", "step": policy.step, "anchor": policy.anchor}
    if callable(policy):
        raise TypeError("callable value policies are not fixture-representable")
    return {"type": policy}


def decode_value_policy(descriptor: Mapping[str, Any]) -> ValuePolicy:
    kind = descriptor["type"]
    if kind == "grid":
        return Grid(step=float(descriptor["step"]), anchor=float(descriptor["anchor"]))
    return kind  # "raw" | "integer"


def encode_value_policies(policies: Mapping[str, ValuePolicy]) -> dict[str, Any]:
    return {name: encode_value_policy(policy) for name, policy in policies.items()}


def decode_value_policies(descriptors: Mapping[str, Any]) -> dict[str, ValuePolicy]:
    return {name: decode_value_policy(desc) for name, desc in descriptors.items()}


def encode_ensemble(ir: EnsembleIR) -> dict[str, Any]:
    flat = flatten_ir(ir)
    out: dict[str, Any] = {}
    for key, value in flat.items():
        out[key] = encode_floats(value) if isinstance(value, np.ndarray) else value
    return out


def decode_ensemble(raw: Mapping[str, Any]) -> EnsembleIR:
    flat = dict(raw)
    flat["threshold"] = np.asarray(decode_floats(raw["threshold"]), dtype=np.float64)
    flat["value"] = np.asarray(decode_floats(raw["value"]), dtype=np.float64)
    for key, dtype in (
        ("feature", np.int32),
        ("is_lt", np.uint8),
        ("missing_left", np.uint8),
        ("left", np.uint32),
        ("right", np.uint32),
        ("tree_roots", np.uint32),
    ):
        flat[key] = np.asarray(raw[key], dtype=dtype)
    return unflatten_ir(flat)


@dataclass(frozen=True)
class ExactFixture:
    """One frozen exact-backend problem plus the golden answer for it."""

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
    value_policies: dict[str, ValuePolicy]
    node_budget: int
    gap: float
    time_budget_s: float
    incumbent: tuple[float, FloatArray] | None
    golden_x_cf: FloatArray | None
    golden_distance: float | None
    golden_proof: str
    golden_nodes_expanded: int
    golden_nodes_pruned_score: int
    golden_nodes_pruned_cost: int
    golden_completed: bool


def build_fixture_payload(
    name: str,
    ir: EnsembleIR,
    x: FloatArray,
    interval: tuple[float, float],
    constraint_descriptors: list[dict[str, Any]],
    *,
    sigma: FloatArray | None = None,
    weights: FloatArray | None = None,
    lam: float = 0.0,
    if_ir: EnsembleIR | None = None,
    min_total_path: float | None = None,
    value_policies: Mapping[str, ValuePolicy] | None = None,
    node_budget: int = 2_000_000,
    gap: float = 0.0,
    incumbent: tuple[float, FloatArray] | None = None,
) -> dict[str, Any]:
    """Inputs -> the fixture dict, minus the ``golden`` block (the caller solves and adds it)."""
    p = ir.n_features
    sigma = np.ones(p) if sigma is None else sigma
    weights = np.ones(p) if weights is None else weights
    value_policies = dict(value_policies or {})
    return {
        "name": name,
        "ensemble": encode_ensemble(ir),
        "if_ensemble": encode_ensemble(if_ir) if if_ir is not None else None,
        "min_total_path": min_total_path,
        "x": encode_floats(x.astype(np.float64)),
        "sigma": encode_floats(sigma.astype(np.float64)),
        "weights": encode_floats(weights.astype(np.float64)),
        "lam": lam,
        "interval": encode_floats(list(interval)),
        "constraints": constraint_descriptors,
        "value_policies": encode_value_policies(value_policies),
        "node_budget": node_budget,
        "gap": gap,
        "time_budget_s": 1e9,  # ALWAYS: completion must never depend on the wall clock
        "incumbent": (
            None
            if incumbent is None
            else {"cost": encode_floats(incumbent[0]), "row": encode_floats(incumbent[1])}
        ),
    }


def solve_payload(payload: Mapping[str, Any]) -> ExactResult:
    """Run the Python exact backend over a payload dict built by ``build_fixture_payload``."""
    fixture = _fixture_from_payload(payload, golden=None)
    return run_fixture(fixture)


def golden_block(result: ExactResult) -> dict[str, Any]:
    return {
        "x_cf": None if result.x_cf is None else encode_floats(result.x_cf),
        "distance": None if result.distance is None else encode_floats(result.distance),
        "proof": result.proof,
        "nodes_expanded": result.stats["nodes_expanded"],
        "nodes_pruned_score": result.stats["nodes_pruned_score"],
        "nodes_pruned_cost": result.stats["nodes_pruned_cost"],
        "completed": result.stats["completed"],
    }


def fixture_paths() -> list[Path]:
    return sorted(FIXTURES_DIR.glob("*.json"))


def load_fixture(path: Path) -> ExactFixture:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return _fixture_from_payload(payload, golden=payload["golden"])


def _fixture_from_payload(
    payload: Mapping[str, Any], golden: Mapping[str, Any] | None
) -> ExactFixture:
    ir = decode_ensemble(payload["ensemble"])
    if_ir = decode_ensemble(payload["if_ensemble"]) if payload.get("if_ensemble") else None
    constraints = build_constraints(payload["constraints"])
    compiled = compile_constraints(constraints, ir.feature_names)
    interval_raw = decode_floats(payload["interval"])
    incumbent_raw = payload.get("incumbent")
    incumbent = (
        None
        if incumbent_raw is None
        else (
            float(decode_floats(incumbent_raw["cost"])),
            np.asarray(decode_floats(incumbent_raw["row"]), dtype=np.float64),
        )
    )
    golden = golden or {}
    golden_x_cf = golden.get("x_cf")
    golden_distance = golden.get("distance")
    return ExactFixture(
        name=payload["name"],
        ir=ir,
        if_ir=if_ir,
        min_total_path=payload.get("min_total_path"),
        x=np.asarray(decode_floats(payload["x"]), dtype=np.float64),
        sigma=np.asarray(decode_floats(payload["sigma"]), dtype=np.float64),
        weights=np.asarray(decode_floats(payload["weights"]), dtype=np.float64),
        lam=float(payload["lam"]),
        interval=(float(interval_raw[0]), float(interval_raw[1])),
        compiled=compiled,
        value_policies=decode_value_policies(payload["value_policies"]),
        node_budget=int(payload["node_budget"]),
        gap=float(payload["gap"]),
        time_budget_s=float(payload["time_budget_s"]),
        incumbent=incumbent,
        golden_x_cf=(
            None
            if golden_x_cf is None
            else np.asarray(decode_floats(golden_x_cf), dtype=np.float64)
        ),
        golden_distance=None if golden_distance is None else float(decode_floats(golden_distance)),
        golden_proof=golden.get("proof", ""),
        golden_nodes_expanded=int(golden.get("nodes_expanded", 0)),
        golden_nodes_pruned_score=int(golden.get("nodes_pruned_score", 0)),
        golden_nodes_pruned_cost=int(golden.get("nodes_pruned_cost", 0)),
        golden_completed=bool(golden.get("completed", False)),
    )


def run_fixture(fixture: ExactFixture) -> ExactResult:
    """Run the Python exact backend over a loaded/built fixture's inputs."""
    plausibility = None
    if fixture.if_ir is not None:
        assert fixture.min_total_path is not None
        plausibility = (fixture.if_ir, fixture.min_total_path)
    return solve_exact(
        fixture.ir,
        fixture.x,
        fixture.interval,
        fixture.compiled,
        fixture.sigma,
        fixture.weights,
        fixture.lam,
        value_policies=fixture.value_policies,
        plausibility=plausibility,
        node_budget=fixture.node_budget,
        gap=fixture.gap,
        time_budget_s=fixture.time_budget_s,
        incumbent=fixture.incumbent,
    )


@dataclass(frozen=True)
class RegionFixture:
    """One frozen region-growth problem plus the golden ``lo``/``hi`` for it.

    Unlike ``ExactFixture``, ``x_cf`` is a pre-verified counterfactual baked
    into the fixture at generation time (found by whichever backend the
    scenario names), not re-derived by loading the fixture -- region growth
    only ever widens an already-verified point.
    """

    name: str
    ir: EnsembleIR
    if_ir: EnsembleIR | None
    min_total_path: float | None
    x: FloatArray
    x_cf: FloatArray
    interval: tuple[float, float]
    compiled: CompiledConstraints
    golden_lo: FloatArray
    golden_hi: FloatArray


def build_region_fixture_payload(
    name: str,
    ir: EnsembleIR,
    x: FloatArray,
    x_cf: FloatArray,
    interval: tuple[float, float],
    constraint_descriptors: list[dict[str, Any]],
    *,
    if_ir: EnsembleIR | None = None,
    min_total_path: float | None = None,
) -> dict[str, Any]:
    """Inputs -> the region fixture dict, minus the ``golden`` block."""
    return {
        "name": name,
        "ensemble": encode_ensemble(ir),
        "if_ensemble": encode_ensemble(if_ir) if if_ir is not None else None,
        "min_total_path": min_total_path,
        "x": encode_floats(x.astype(np.float64)),
        "x_cf": encode_floats(x_cf.astype(np.float64)),
        "interval": encode_floats(list(interval)),
        "constraints": constraint_descriptors,
    }


def region_golden_block(lo: FloatArray, hi: FloatArray) -> dict[str, Any]:
    return {"lo": encode_floats(lo), "hi": encode_floats(hi)}


def region_fixture_paths() -> list[Path]:
    return sorted(REGION_FIXTURES_DIR.glob("*.json"))


def load_region_fixture(path: Path) -> RegionFixture:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return _region_fixture_from_payload(payload, golden=payload["golden"])


def _region_fixture_from_payload(
    payload: Mapping[str, Any], golden: Mapping[str, Any] | None
) -> RegionFixture:
    ir = decode_ensemble(payload["ensemble"])
    if_ir = decode_ensemble(payload["if_ensemble"]) if payload.get("if_ensemble") else None
    constraints = build_constraints(payload["constraints"])
    compiled = compile_constraints(constraints, ir.feature_names)
    interval_raw = decode_floats(payload["interval"])
    golden = golden or {}
    golden_lo = golden.get("lo")
    golden_hi = golden.get("hi")
    return RegionFixture(
        name=payload["name"],
        ir=ir,
        if_ir=if_ir,
        min_total_path=payload.get("min_total_path"),
        x=np.asarray(decode_floats(payload["x"]), dtype=np.float64),
        x_cf=np.asarray(decode_floats(payload["x_cf"]), dtype=np.float64),
        interval=(float(interval_raw[0]), float(interval_raw[1])),
        compiled=compiled,
        golden_lo=(
            np.empty(0) if golden_lo is None else np.asarray(decode_floats(golden_lo))
        ),
        golden_hi=(
            np.empty(0) if golden_hi is None else np.asarray(decode_floats(golden_hi))
        ),
    )


def solve_region_payload(payload: Mapping[str, Any]) -> tuple[FloatArray, FloatArray]:
    """Run the pure-Python region growth over a payload dict built by
    ``build_region_fixture_payload`` (no ``golden`` block needed)."""
    fixture = _region_fixture_from_payload(payload, golden=None)
    return run_region_fixture(fixture)


def region_degenerate_and_bounds(
    fixture: RegionFixture,
) -> tuple[frozenset[int], FloatArray, FloatArray]:
    """The ``(degenerate, lo_b, hi_b)`` triple every region-growth caller
    (Python or the Rust marshaling wrapper) needs, computed once here so
    fixture-driven tests never re-derive it differently from one another."""
    lo_b, hi_b, frozen = fixture.compiled.instance_bounds(fixture.x)
    lo_b = np.where(np.isnan(lo_b), -np.inf, lo_b)
    hi_b = np.where(np.isnan(hi_b), np.inf, hi_b)
    from treecf.regions import _degenerate_features

    degenerate = _degenerate_features(fixture.compiled, frozen, lo_b, hi_b, fixture.x_cf)
    return degenerate, lo_b, hi_b


def run_region_fixture(fixture: RegionFixture) -> tuple[FloatArray, FloatArray]:
    """Run the pure-Python growth loop (bypasses the rust-first dispatch in
    ``treecf.regions._recourse_region``) over a loaded fixture."""
    from treecf.regions import _grow_box

    degenerate, lo_b, hi_b = region_degenerate_and_bounds(fixture)
    min_total_path = fixture.min_total_path if fixture.min_total_path is not None else 0.0
    return _grow_box(
        fixture.ir, fixture.x_cf, fixture.interval, fixture.compiled,
        fixture.if_ir, min_total_path, degenerate, lo_b, hi_b,
    )


def diff_region_golden(fixture: RegionFixture, lo: FloatArray, hi: FloatArray) -> list[str]:
    """Byte-exact comparison, ``float`` bits via ``encode_floats``. Empty = match."""
    problems: list[str] = []
    got_lo, want_lo = encode_floats(lo), encode_floats(fixture.golden_lo)
    if got_lo != want_lo:
        problems.append(f"lo: golden={want_lo!r} got={got_lo!r}")
    got_hi, want_hi = encode_floats(hi), encode_floats(fixture.golden_hi)
    if got_hi != want_hi:
        problems.append(f"hi: golden={want_hi!r} got={got_hi!r}")
    return problems


def diff_golden(fixture: ExactFixture, result: ExactResult) -> list[str]:
    """Byte-exact comparison, ``float`` bits via ``encode_floats``. Empty = match."""
    problems: list[str] = []
    got_x_cf = None if result.x_cf is None else encode_floats(result.x_cf)
    want_x_cf = None if fixture.golden_x_cf is None else encode_floats(fixture.golden_x_cf)
    if got_x_cf != want_x_cf:
        problems.append(f"x_cf: golden={want_x_cf!r} got={got_x_cf!r}")
    got_distance = None if result.distance is None else encode_floats(result.distance)
    want_distance = (
        None if fixture.golden_distance is None else encode_floats(fixture.golden_distance)
    )
    if got_distance != want_distance:
        problems.append(f"distance: golden={want_distance!r} got={got_distance!r}")
    if result.proof != fixture.golden_proof:
        problems.append(f"proof: golden={fixture.golden_proof!r} got={result.proof!r}")
    for key, want in (
        ("nodes_expanded", fixture.golden_nodes_expanded),
        ("nodes_pruned_score", fixture.golden_nodes_pruned_score),
        ("nodes_pruned_cost", fixture.golden_nodes_pruned_cost),
        ("completed", fixture.golden_completed),
    ):
        got = result.stats[key]
        if got != want:
            problems.append(f"{key}: golden={want!r} got={got!r}")
    return problems
