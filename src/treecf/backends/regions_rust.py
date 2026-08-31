"""Rust recourse-region wrapper — result-identical to the pure-Python growth
loop in ``treecf.regions``.

Same contract as ``treecf.regions._grow_box``; the oracle and growth loop run
in the ``_treecf_core`` extension (built locally with maturin, GIL released).
Bit-parity with the Python reference is established by
``tests/rust/test_region_parity.py`` over the golden fixture set.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from treecf._errors import MissingExtraError
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
    """Probe only — never raises. ``treecf.regions._recourse_region`` uses
    this to decide rust-first vs. the pure-Python growth loop."""
    try:
        import treecf._treecf_core  # noqa: F401
    except ImportError:
        return False
    return True


def _missing_defined_flags(ir: EnsembleIR) -> npt.NDArray[np.uint8]:
    """1 where a split node's ``missing_left`` is a concrete bool, 0 where it
    is ``None`` (or the node is a leaf, where the value is unused).

    Same node addressing as ``treecf.ir.flatten.flatten_ir`` (``offset +
    node.node_id``) — this is the one bit that flat encoding's own
    ``None -> 0`` collapse would otherwise erase, and the box oracle's
    unrouted-missing rejection depends on the distinction (see
    ``treecf.regions._tree_interval_bracket``).
    """
    n_nodes = sum(len(t.nodes) for t in ir.trees)
    out = np.zeros(n_nodes, dtype=np.uint8)
    offset = 0
    for tree in ir.trees:
        for node in tree.nodes:
            if node.feature is not None:
                out[offset + node.node_id] = 0 if node.missing_left is None else 1
        offset += len(tree.nodes)
    return out


def compute_region_rust(
    ir: EnsembleIR,
    x_cf: FloatArray,
    interval: tuple[float, float],
    compiled: CompiledConstraints,
    lo_b: FloatArray,
    hi_b: FloatArray,
    degenerate: frozenset[int],
    if_ir: EnsembleIR | None,
    min_total_path: float,
    cat_candidates: dict[int, list[tuple[int, ...]]] | None = None,
    cache: dict[str, Any] | None = None,
) -> tuple[FloatArray, FloatArray, dict[int, set[int]]]:
    """Drop-in for ``treecf.regions._grow_box``; ``cache`` (e.g. the
    ``Explainer``'s ``_rust_cache``) avoids re-marshaling the ensembles and
    constraints on every call, exactly as ``solve_exact_rust``'s does."""
    core = _core()
    cache = cache if cache is not None else {}
    if "ensemble" not in cache:
        cache["ensemble"] = build_rust_ensemble(ir)
    if "missing_defined" not in cache:
        cache["missing_defined"] = _missing_defined_flags(ir)
    if "constraints" not in cache:
        cache["constraints"] = build_rust_constraints(compiled)
    if_ens = None
    if_missing_defined = None
    if if_ir is not None:
        if "if_ensemble" not in cache:
            cache["if_ensemble"] = build_rust_ensemble(if_ir)
        if "if_missing_defined" not in cache:
            cache["if_missing_defined"] = _missing_defined_flags(if_ir)
        if_ens = cache["if_ensemble"]
        if_missing_defined = cache["if_missing_defined"]

    p = len(x_cf)
    open_set = np.asarray(sorted(j for j in range(p) if j not in degenerate), dtype=np.uint32)
    # the categorical growth channel, flattened: per open categorical feature,
    # its blocks' admissible members (two-level CSR, canonical block order)
    cat_candidates = cat_candidates or {}
    cat_open = sorted(cat_candidates)
    cat_feat_offsets = [0]
    cat_block_offsets = [0]
    cat_members: list[int] = []
    for j in cat_open:
        for members in cat_candidates[j]:
            cat_members.extend(members)
            cat_block_offsets.append(len(cat_members))
        cat_feat_offsets.append(len(cat_block_offsets) - 1)
    lo, hi, grown_offsets, grown_members = core.compute_region_raw(
        cache["ensemble"],
        cache["missing_defined"],
        cache["constraints"],
        np.ascontiguousarray(x_cf, dtype=np.float64),
        float(interval[0]),
        float(interval[1]),
        np.ascontiguousarray(lo_b, dtype=np.float64),
        np.ascontiguousarray(hi_b, dtype=np.float64),
        open_set,
        if_ensemble=if_ens,
        if_missing_defined=if_missing_defined,
        min_total_path=None if if_ir is None else float(min_total_path),
        cat_open=np.asarray(cat_open, dtype=np.uint32),
        cat_feat_offsets=np.asarray(cat_feat_offsets, dtype=np.uint32),
        cat_block_offsets=np.asarray(cat_block_offsets, dtype=np.uint32),
        cat_members=np.asarray(cat_members, dtype=np.uint32),
    )
    grown_offsets = np.asarray(grown_offsets, dtype=np.uint32)
    grown_members = np.asarray(grown_members, dtype=np.uint32)
    grown_sets = {
        j: {int(c) for c in grown_members[grown_offsets[k] : grown_offsets[k + 1]]}
        for k, j in enumerate(cat_open)
    }
    return (
        np.asarray(lo, dtype=np.float64),
        np.asarray(hi, dtype=np.float64),
        grown_sets,
    )
