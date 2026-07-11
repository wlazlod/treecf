"""Shared fixtures: synthetic credit-like data with mixed scales, integer columns, NaNs."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest


def make_synthetic(
    n: int = 2000,
    p: int = 8,
    seed: int = 0,
    nan_frac: float = 0.1,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return (X, y_binary, y_continuous) with mixed feature scales and NaN holes.

    Even-indexed features are continuous with scales spanning 1e-2..1e3; odd-indexed
    features are non-negative integers (counts, DPD-style), giving point masses at 0.
    """
    rng = np.random.default_rng(seed)
    X = np.empty((n, p), dtype=np.float64)
    for j in range(p):
        scale = 10.0 ** ((j % 6) - 2)
        if j % 2 == 0:
            X[:, j] = rng.normal(loc=scale, scale=scale, size=n)
        else:
            X[:, j] = np.floor(rng.exponential(scale=3.0, size=n) * rng.integers(0, 2, size=n))
    signal = np.zeros(n)
    for j in range(p):
        col = X[:, j]
        std = col.std() or 1.0
        signal += ((-1.0) ** j) * (col - col.mean()) / std
    y_continuous = signal + rng.normal(scale=0.3, size=n)
    y_binary = (signal + rng.logistic(scale=0.5, size=n) > 0).astype(np.float64)
    if nan_frac > 0:
        mask = rng.random(X.shape) < nan_frac
        X[mask] = np.nan
    return X, y_binary, y_continuous


@pytest.fixture(scope="session")
def synthetic_data() -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]:
    return make_synthetic()
