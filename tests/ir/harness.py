"""Generic IR conformance runner, reused by every parser's conformance test."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from treecf.ir.conformance import max_parity_gap, parity_tolerance, threshold_adjacent_rows
from treecf.ir.model import EnsembleIR

FloatArray = npt.NDArray[np.float64]


def probe_matrix(
    ir: EnsembleIR,
    X_train: FloatArray,
    n_random: int = 10_000,
    seed: int = 123,
    include_nan: bool = True,
    quantize: bool = True,
) -> FloatArray:
    """Random points + training samples + NaN patterns + threshold-adjacent rows.

    ``quantize=False`` keeps the probes on the raw float64 grid and adds rows at
    the exact float64 thresholds and their float64 neighbours — the semantics a
    counterfactual search actually exercises. Sources whose native evaluation
    quantizes inputs (e.g. sklearn ``tree_`` ensembles cast X to float32) must
    encode that quantization into their IR thresholds to pass this mode.
    """
    rng = np.random.default_rng(seed)
    p = ir.n_features
    lo = np.nanmin(X_train, axis=0)
    hi = np.nanmax(X_train, axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    random_part = rng.uniform(lo - 0.5 * span, hi + 0.5 * span, size=(n_random, p))
    parts = [random_part]
    if include_nan:
        # NaN patterns: random subsets of features masked out.
        nan_part = random_part[: n_random // 4].copy()
        nan_mask = rng.random(nan_part.shape) < 0.3
        nan_part[nan_mask] = np.nan
        parts.append(nan_part)
        parts.append(X_train[rng.integers(0, len(X_train), size=min(len(X_train), 2000))])
    else:
        clean = X_train[~np.isnan(X_train).any(axis=1)]
        if len(clean):
            parts.append(clean[rng.integers(0, len(clean), size=min(len(clean), 2000))])
    base_row = np.nan_to_num(X_train[0], nan=0.0)
    parts.append(threshold_adjacent_rows(ir, base_row))
    if not quantize:
        # Exact float64 thresholds and their float64 neighbours: the boundary
        # points a counterfactual search places representatives on.
        rows = []
        for tree in ir.trees:
            for node in tree.nodes:
                if node.feature is None or node.categories is not None:
                    continue
                assert node.threshold is not None
                for value in (
                    node.threshold,
                    float(np.nextafter(node.threshold, -np.inf)),
                    float(np.nextafter(node.threshold, np.inf)),
                ):
                    row = base_row.copy()
                    row[node.feature] = value
                    rows.append(row)
        if rows:
            parts.append(np.vstack(rows))
        return _categorize(ir, np.vstack(parts), rng, include_nan)
    stacked = _categorize(ir, np.vstack(parts), rng, include_nan)
    # Quantize to the float32 grid: native GBDTs store inputs as float32, so only
    # float32-representable probes test parsing rather than input quantization.
    return stacked.astype(np.float32).astype(np.float64)


def _categorize(
    ir: EnsembleIR, X: FloatArray, rng: np.random.Generator, include_nan: bool
) -> FloatArray:
    """Overwrite categorical columns with valid codes and NaN holes.

    Random real values are not legal category codes; every categorical
    coordinate becomes an integral code in ``[0, cardinality)`` — the input
    contract the API validates — with NaN kept where the row already had one.
    Unseen-but-valid codes are probed by declaring a cardinality larger than
    the training data used (via a parser's ``categories`` argument).
    """
    for j, info in ir.categorical.items():
        col = X[:, j]
        was_nan = np.isnan(col)
        already_code = (
            np.isfinite(col) & (col == np.floor(col)) & (col >= 0.0) & (col < info.cardinality)
        )
        codes = rng.integers(0, info.cardinality, size=len(col)).astype(np.float64)
        replacement = np.where(already_code, col, codes)
        X[:, j] = np.where(was_nan & include_nan, np.nan, replacement)
    return X


def assert_conformance(
    ir: EnsembleIR,
    X_train: FloatArray,
    predict_native: Callable[[FloatArray], FloatArray],
    n_random: int = 10_000,
    include_nan: bool = True,
    quantize: bool = True,
) -> None:
    """Assert |link(S_IR(x)) - predict_native(x)| <= tolerance on the probe matrix."""
    X_eval = probe_matrix(
        ir, X_train, n_random=n_random, include_nan=include_nan, quantize=quantize
    )
    gap = max_parity_gap(ir, X_eval, predict_native(X_eval))
    tol = parity_tolerance(ir)
    assert gap <= tol, f"parity gap {gap:.3e} exceeds {tol:.0e} on {len(X_eval)} points"
