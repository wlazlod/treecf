"""Rust exact backend wrapper — result-identical to ``solve_exact``.

Same contract as ``solve_exact``; the branch-and-bound search runs in the
`_treecf_core` extension (built locally with maturin, GIL released for the
search itself). Bit-parity with the Python reference is established by
``tests/rust/test_exact_parity.py`` over the golden fixture set; results
carry the same ``proof``/``stats``/``snapped`` contract the Python backend
does, byte for byte on every fixture.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from treecf._errors import ConstraintValidationError, MissingExtraError
from treecf.api import Grid, ValuePolicy
from treecf.backends.exact import ExactResult
from treecf.backends.genetic_rust import build_rust_constraints, build_rust_ensemble
from treecf.constraints.compile import CompiledConstraints
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


def _rust_available() -> bool:
    """Probe only — never raises. ``Explainer._explain_exact`` uses this to
    decide rust-first vs. the Python reference; ``_core()`` still raises
    ``MissingExtraError`` for any caller that expects the extension."""
    try:
        import treecf._treecf_core  # noqa: F401
    except ImportError:
        return False
    return True


def encode_value_policies(
    value_policy: Mapping[str, ValuePolicy], feature_names: Sequence[str]
) -> tuple[npt.NDArray[np.uint8], FloatArray, FloatArray]:
    """Per-feature ``(code, step, anchor)``: ``0=raw`` (unset), ``1=integer``,
    ``2=grid``. A callable policy can never legally reach ``solve_exact``
    either: its own ``_validate`` rejects one before the search runs, so this
    raises the identical exception type here, before any marshaling, rather
    than letting the rust path fail differently. (Message text is never
    compared across languages, only the exception type and that it fires.)
    """
    n = len(feature_names)
    code = np.zeros(n, dtype=np.uint8)
    step = np.zeros(n, dtype=np.float64)
    anchor = np.zeros(n, dtype=np.float64)
    for j, name in enumerate(feature_names):
        policy = value_policy.get(name, "raw")
        if callable(policy):
            raise ConstraintValidationError(
                f"callable value_policy for {name!r} is not supported by the exact "
                'backend; use backend="genetic".'
            )
        if policy == "raw":
            continue
        if policy == "integer":
            code[j] = 1
            continue
        assert isinstance(policy, Grid)
        code[j] = 2
        step[j] = policy.step
        anchor[j] = policy.anchor
    return code, step, anchor


def solve_exact_rust(
    ir: EnsembleIR,
    x: FloatArray,
    interval: tuple[float, float],
    compiled: CompiledConstraints,
    sigma: FloatArray,
    weights: FloatArray,
    lam: float,
    value_policies: Mapping[str, ValuePolicy] | None = None,
    plausibility: tuple[EnsembleIR, float] | None = None,
    node_budget: int = 2_000_000,
    gap: float = 0.0,
    time_budget_s: float = 10.0,
    incumbent: tuple[float, FloatArray] | None = None,
    cache: dict[str, Any] | None = None,
) -> ExactResult:
    """Drop-in for ``solve_exact``; ``cache`` (e.g. on the ``Explainer``)
    avoids re-marshaling the ensembles and constraints on every call, exactly
    as ``solve_genetic_rust``'s does."""
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

    code, step, anchor = encode_value_policies(value_policies or {}, ir.feature_names)
    incumbent_cost = None if incumbent is None else float(incumbent[0])
    incumbent_row = (
        None if incumbent is None else np.ascontiguousarray(incumbent[1], dtype=np.float64)
    )

    try:
        x_cf, distance, proof, stats, snapped_idx = core.solve_exact_raw(
            cache["ensemble"],
            cache["constraints"],
            np.ascontiguousarray(x, dtype=np.float64),
            float(interval[0]),
            float(interval[1]),
            np.ascontiguousarray(sigma, dtype=np.float64),
            np.ascontiguousarray(weights, dtype=np.float64),
            float(lam),
            code,
            step,
            anchor,
            if_ensemble=if_ens,
            min_total_path=min_total_path,
            node_budget=node_budget,
            gap=gap,
            time_budget_s=time_budget_s,
            incumbent_cost=incumbent_cost,
            incumbent_row=incumbent_row,
        )
    except ValueError as exc:
        # solve_exact_raw raises ValueError for exactly one thing: the same
        # order-pair validation solve_exact's own _validate performs (an
        # unsupported multi-feature Linear shape) -- re-raised as the same
        # exception type the python backend uses, never compared by message
        # text. A marshaling bug on this side of the boundary (mismatched
        # array lengths, an unrecognized policy code) raises RuntimeError
        # instead and is deliberately NOT caught here: that is a bug, not a
        # user-facing constraint problem, and it should propagate as one.
        raise ConstraintValidationError(str(exc)) from exc

    (
        nodes_expanded,
        nodes_pruned_score,
        nodes_pruned_cost,
        lower_bound,
        stats_gap,
        completed,
        warm_start_used,
    ) = stats
    return ExactResult(
        x_cf=None if x_cf is None else np.asarray(x_cf, dtype=np.float64),
        proof=proof,
        stats={
            "nodes_expanded": int(nodes_expanded),
            "nodes_pruned_score": int(nodes_pruned_score),
            "nodes_pruned_cost": int(nodes_pruned_cost),
            "lower_bound": float(lower_bound),
            "gap": float(stats_gap),
            "completed": bool(completed),
            "warm_start_used": bool(warm_start_used),
        },
        snapped={ir.feature_names[int(i)]: True for i in np.asarray(snapped_idx)},
        distance=None if distance is None else float(distance),
    )
