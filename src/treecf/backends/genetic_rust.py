"""Rust genetic backend wrapper — the default engine.

Same contract as ``solve_genetic``; the heavy loop runs in the `_treecf_core`
extension (built locally with maturin, GIL released). Statistical parity with
the Python GA is established by the parity test suite; results carry
``proof="heuristic"`` exactly like the Python backend.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from treecf._errors import MissingExtraError
from treecf.backends.genetic import GeneticResult
from treecf.constraints.compile import CompiledConstraints
from treecf.constraints.flatten import flatten_constraints
from treecf.ir.flatten import flatten_ir
from treecf.ir.model import EnsembleIR

FloatArray = npt.NDArray[np.float64]

_BUILD_HINT = (
    "the treecf._treecf_core extension is missing; reinstall treecf from a wheel, "
    "or in a dev checkout run: uv sync (maturin builds the extension)"
)


def _core() -> Any:
    try:
        import treecf._treecf_core as _treecf_core
    except ImportError as exc:
        raise MissingExtraError(_BUILD_HINT) from exc
    return _treecf_core


def build_rust_ensemble(ir: EnsembleIR) -> Any:
    flat = flatten_ir(ir)
    return _core().RustEnsemble(
        flat["feature"],
        flat["threshold"],
        flat["is_lt"],
        flat["missing_left"],
        flat["left"],
        flat["right"],
        flat["value"],
        flat["tree_roots"],
        flat["base_score"],
        flat["link"],
        flat["n_features"],
    )


def build_rust_constraints(compiled: CompiledConstraints) -> Any:
    flat = flatten_constraints(compiled)
    return _core().RustConstraints(
        flat["n_features"],
        flat["freeze"],
        flat["range_idx"],
        flat["range_lo"],
        flat["range_hi"],
        flat["equals_idx"],
        flat["equals_val"],
        flat["mono_idx"],
        flat["mono_dir"],
        flat["lin_offsets"],
        flat["lin_indices"],
        flat["lin_coefs"],
        flat["lin_op"],
        flat["lin_rhs"],
        flat["lin_policy"],
        flat["imp_cond_idx"],
        flat["imp_cond_val"],
        flat["imp_cons_idx"],
        flat["imp_cons_val"],
        flat["oh_offsets"],
        flat["oh_indices"],
        flat["am_idx"],
        flat["am_to"],
        flat["am_from"],
    )


def solve_genetic_rust(
    ir: EnsembleIR,
    x: FloatArray,
    interval: tuple[float, float],
    compiled: CompiledConstraints,
    sigma: FloatArray,
    weights: FloatArray,
    lam: float,
    background: FloatArray | None = None,
    plausibility: tuple[EnsembleIR, float] | None = None,
    seed: int | None = None,
    population: int = 80,
    max_generations: int = 200,
    stall_generations: int = 30,
    time_budget_s: float = 10.0,
    cache: dict[str, Any] | None = None,
) -> GeneticResult:
    """Drop-in for ``solve_genetic``; ``cache`` (e.g. on the Explainer) avoids
    re-marshaling the ensembles and constraints on every call."""
    core = _core()
    cache = cache if cache is not None else {}
    if "ensemble" not in cache:
        cache["ensemble"] = build_rust_ensemble(ir)
    if "constraints" not in cache:
        cache["constraints"] = build_rust_constraints(compiled)
    if_ens = None
    min_total_path = None
    if plausibility is not None:
        if "if_ensemble" not in cache:
            cache["if_ensemble"] = build_rust_ensemble(plausibility[0])
        if_ens = cache["if_ensemble"]
        min_total_path = float(plausibility[1])

    x_cf, generations = core.solve_genetic_raw(
        cache["ensemble"],
        cache["constraints"],
        np.ascontiguousarray(x, dtype=np.float64),
        float(interval[0]),
        float(interval[1]),
        np.ascontiguousarray(sigma, dtype=np.float64),
        np.ascontiguousarray(weights, dtype=np.float64),
        float(lam),
        background=(
            np.ascontiguousarray(background, dtype=np.float64)
            if background is not None
            else None
        ),
        if_ensemble=if_ens,
        min_total_path=min_total_path,
        seed=seed,
        population=population,
        max_generations=max_generations,
        stall_generations=stall_generations,
        time_budget_s=time_budget_s,
    )
    stats: dict[str, object] = {"generations": generations, "backend": "rust"}
    return GeneticResult(
        x_cf=None if x_cf is None else np.asarray(x_cf, dtype=np.float64), stats=stats
    )


def solve_genetic_batch_rust(
    ir: EnsembleIR,
    X: FloatArray,
    tasks: Sequence[tuple[int, int]],
    interval: tuple[float, float],
    compiled: CompiledConstraints,
    sigma: FloatArray,
    weights: FloatArray,
    lam: float,
    background: FloatArray | None = None,
    plausibility: tuple[EnsembleIR, float] | None = None,
    population: int = 80,
    max_generations: int = 200,
    stall_generations: int = 30,
    time_budget_s: float = 10.0,
    cache: dict[str, Any] | None = None,
) -> list[GeneticResult]:
    """Batch counterpart of ``solve_genetic_rust``: one independent, seeded
    search per ``(row_index, seed)`` task, run in parallel inside the extension
    (GIL released). Results follow task order and are bitwise-identical to
    per-task ``solve_genetic_rust`` calls."""
    core = _core()
    cache = cache if cache is not None else {}
    if "ensemble" not in cache:
        cache["ensemble"] = build_rust_ensemble(ir)
    if "constraints" not in cache:
        cache["constraints"] = build_rust_constraints(compiled)
    if_ens = None
    min_total_path = None
    if plausibility is not None:
        if "if_ensemble" not in cache:
            cache["if_ensemble"] = build_rust_ensemble(plausibility[0])
        if_ens = cache["if_ensemble"]
        min_total_path = float(plausibility[1])

    task_row = np.asarray([row for row, _ in tasks], dtype=np.uint64)
    task_seed = np.asarray([seed for _, seed in tasks], dtype=np.uint64)
    x_cf, feasible, generations = core.solve_genetic_batch_raw(
        cache["ensemble"],
        cache["constraints"],
        np.ascontiguousarray(X, dtype=np.float64),
        task_row,
        task_seed,
        float(interval[0]),
        float(interval[1]),
        np.ascontiguousarray(sigma, dtype=np.float64),
        np.ascontiguousarray(weights, dtype=np.float64),
        float(lam),
        background=(
            np.ascontiguousarray(background, dtype=np.float64)
            if background is not None
            else None
        ),
        if_ensemble=if_ens,
        min_total_path=min_total_path,
        population=population,
        max_generations=max_generations,
        stall_generations=stall_generations,
        time_budget_s=time_budget_s,
    )
    return [
        GeneticResult(
            x_cf=np.asarray(x_cf[t], dtype=np.float64) if feasible[t] else None,
            stats={"generations": int(generations[t]), "backend": "rust"},
        )
        for t in range(len(tasks))
    ]
