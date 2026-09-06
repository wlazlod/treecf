"""The search-space profile: what an exact search would have to enumerate.

Built from the same pieces the exact backend builds its branching alphabet
from — the joint cell grid, the category blocks, the per-feature candidate
states after the instance bounds, and the search order — so the numbers
here are the search's own, not an estimate of them. With a target interval
the presolve filter runs too, reporting how many states it removes and
whether it certifies infeasibility outright.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from treecf.aim.cells import category_blocks
from treecf.api import ValuePolicy
from treecf.backends._exact_bounds import _EnsembleBounds
from treecf.backends._exact_domains import (
    FloatArray,
    _build_domains,
    _constraint_cells,
    _feature_order,
)
from treecf.constraints.compile import CompiledConstraints
from treecf.ir.model import EnsembleIR


def search_space_profile(
    ir: EnsembleIR,
    x: FloatArray,
    compiled: CompiledConstraints,
    sigma: FloatArray,
    weights: FloatArray,
    lam: float,
    value_policies: Mapping[str, ValuePolicy] | None,
    plausibility: tuple[EnsembleIR, float] | None,
    interval: tuple[float, float] | None = None,
) -> dict[str, object]:
    """Per-feature cell counts and domain sizes plus the totals that size a search.

    Returns ``{"features": {name: {...}}, "influential_features": int,
    "log10_states": float}`` and, with ``interval``, ``"presolved"`` per
    feature, ``"log10_states_presolved"`` and ``"presolve_certified"``.
    """
    from treecf.backends.exact import _presolve_domains

    if_ir = plausibility[0] if plausibility is not None else None
    min_total_path = plausibility[1] if plausibility is not None else 0.0
    grids = (
        _constraint_cells(compiled, ir)
        if if_ir is None
        else _constraint_cells(compiled, ir, if_ir)
    )
    blocks = category_blocks(ir) if if_ir is None else category_blocks(ir, if_ir)
    domains = _build_domains(grids, x, compiled, sigma, weights, lam, value_policies, blocks)
    order = _feature_order(grids, compiled, blocks)
    ordered = set(order)
    frozen = compiled.instance_bounds(x)[2]

    features: dict[str, dict[str, object]] = {}
    for j, name in enumerate(ir.feature_names):
        categorical = j in ir.categorical
        features[name] = {
            "kind": "categorical" if categorical else "numeric",
            "atomic": len(blocks[j]) if categorical else len(grids[j]),
            "domain": len(domains[j]),
            "frozen": bool(frozen[j]),
            "influential": j in ordered,
        }
    profile: dict[str, object] = {
        "features": features,
        "influential_features": len(order),
        "log10_states": _log10_states(domains, order),
    }
    if interval is None:
        return profile

    lo_t, hi_t = interval
    assigned = [False] * len(x)
    values = [0.0] * len(x)
    model_bounds = _EnsembleBounds(ir, assigned, values)
    if_bounds = _EnsembleBounds(if_ir, assigned, values) if if_ir is not None else None
    presolved = [list(states) for states in domains]
    _presolve_domains(
        order, presolved, model_bounds, if_bounds, min_total_path, lo_t, hi_t, assigned, values
    )
    for j, name in enumerate(ir.feature_names):
        features[name]["presolved"] = len(presolved[j])
    profile["log10_states_presolved"] = _log10_states(presolved, order)
    profile["presolve_certified"] = any(not presolved[j] for j in order)
    return profile


def _log10_states(domains: Sequence[Sequence[object]], order: list[int]) -> float:
    """``sum(log10(domain size))`` over the ordered features; ``-inf`` when a
    domain is empty, since nothing can then be enumerated at all."""
    total = 0.0
    for j in order:
        size = len(domains[j])
        if size == 0:
            return -math.inf
        total += math.log10(size)
    return float(np.float64(total))
