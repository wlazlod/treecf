"""Per-feature distance normalizers: MAD -> IQR -> range -> 1 (spec §4.1).

Credit features are heavy-tailed with point masses at zero (DPD counts,
utilization), so the robust MAD comes first and the chain handles the
frequent ``median = mode = 0`` case.
"""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


class DegenerateFeatureWarning(UserWarning):
    """A feature had no usable spread; its normalizer defaults to 1.0."""


def fit_normalizers(X: FloatArray) -> FloatArray:
    """Return sigma_j > 0 per feature, ignoring NaNs in the background sample."""
    n_features = X.shape[1]
    sigma = np.empty(n_features, dtype=np.float64)
    for j in range(n_features):
        col = X[:, j]
        col = col[~np.isnan(col)]
        if len(col) == 0:
            warnings.warn(
                f"feature {j}: all values missing; normalizer defaults to 1.0",
                DegenerateFeatureWarning,
                stacklevel=2,
            )
            sigma[j] = 1.0
            continue
        mad = float(np.median(np.abs(col - np.median(col))))
        if mad > 0:
            sigma[j] = mad
            continue
        iqr = float(np.percentile(col, 75) - np.percentile(col, 25))
        if iqr > 0:
            sigma[j] = iqr
            continue
        rng = float(col.max() - col.min())
        if rng > 0:
            sigma[j] = rng
            continue
        warnings.warn(
            f"feature {j}: degenerate spread (constant); normalizer defaults to 1.0",
            DegenerateFeatureWarning,
            stacklevel=2,
        )
        sigma[j] = 1.0
    return sigma
