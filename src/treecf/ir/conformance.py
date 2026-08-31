"""Parity utilities between the IR and a native model.

Probe points must be float32-representable: GBDT libraries store inputs and
thresholds as float32, so a float64 probe closer to a threshold than float32
resolution would route differently in the native model for reasons unrelated
to parsing correctness.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from treecf.ir.evaluate import apply_link, raw_score
from treecf.ir.model import EnsembleIR

FloatArray = npt.NDArray[np.float64]


def max_parity_gap(ir: EnsembleIR, X: FloatArray, native_output: FloatArray) -> float:
    """Max normalized |link(S_IR(x)) - native_output(x)| over the rows of X.

    Normalized by ``max(1, |native|)`` per point: native predictions are float32,
    so their error relative to exact float64 evaluation scales with the score
    magnitude and the number of accumulated trees (see ``parity_tolerance``).
    """
    preds = np.array([apply_link(ir.link, raw_score(ir, row)) for row in X], dtype=np.float64)
    native = np.asarray(native_output, dtype=np.float64)
    return float(np.max(np.abs(preds - native) / np.maximum(1.0, np.abs(native))))


def parity_tolerance(ir: EnsembleIR) -> float:
    """Float32-aware parity bound: accumulation of T float32 leaf additions.

    Native GBDT predictors sum leaf values in float32, so parity with exact
    float64 evaluation cannot beat ~T ulps relative error; fixed float64
    tolerances like 1e-9 are unattainable against float32 outputs.
    """
    eps32 = float(np.finfo(np.float32).eps)
    return max(1e-7, 2.0 * len(ir.trees) * eps32)


def threshold_adjacent_rows(ir: EnsembleIR, base_row: FloatArray) -> FloatArray:
    """For every split: base_row with the feature at the routing boundary.

    Numeric splits probe the threshold and one float32 ulp each side; set
    splits probe every member code plus one code outside the set.
    """
    rows: list[FloatArray] = []
    for tree in ir.trees:
        for node in tree.nodes:
            if node.feature is None:
                continue
            if node.categories is not None:
                info = ir.categorical.get(node.feature)
                outside = [
                    c
                    for c in range(info.cardinality if info else max(node.categories) + 2)
                    if c not in node.categories
                ]
                for code in [*sorted(node.categories), *outside[:1]]:
                    row = base_row.copy()
                    row[node.feature] = float(code)
                    rows.append(row)
                continue
            assert node.threshold is not None
            with np.errstate(over="ignore"):  # past float32 max is fine to skip
                t32 = np.float32(node.threshold)
                candidates = (
                    float(t32),
                    float(np.nextafter(t32, np.float32(-np.inf))),
                    float(np.nextafter(t32, np.float32(np.inf))),
                )
            for value in candidates:
                if not np.isfinite(value):
                    continue
                row = base_row.copy()
                row[node.feature] = value
                rows.append(row)
    if not rows:
        return np.empty((0, ir.n_features), dtype=np.float64)
    return np.vstack(rows)
