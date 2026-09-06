"""Thresholds re-expressed for a library that casts its inputs to float32.

XGBoost and CatBoost compare ``float32(x)`` against a float32 threshold;
sklearn compares ``float32(x)`` against a float64 one. The IR evaluates in
float64, so each parser stores the float64 boundary of that cast instead:
the largest ``x`` the native model still routes left. These tests walk the
float64 neighbourhood of many thresholds and check that the IR comparison
and the native comparison agree on every probe.
"""

from __future__ import annotations

import numpy as np
import pytest

from treecf.ir.parsers._float32 import effective_le_threshold, effective_lt_threshold

_SPECIAL = [0.0, 1.0, -1.0, 0.5, 3.0, 1e-3, 1e3, 1e-30, 1e30, 2.0**-100, 2.0**100]


def _thresholds() -> list[float]:
    rng = np.random.default_rng(7)
    drawn = rng.normal(size=400) * 10.0 ** rng.integers(-6, 7, size=400)
    return [float(np.float32(t)) for t in [*_SPECIAL, *drawn.tolist()]]


def _probes(t: float) -> list[float]:
    """float64 points around ``t``: its float32 neighbours, their midpoints, and
    a few float64 ulps on either side of each of those."""
    f32 = np.float32(t)
    below = float(np.nextafter(f32, np.float32(-np.inf)))
    above = float(np.nextafter(f32, np.float32(np.inf)))
    anchors = [t, below, above, (below + t) / 2.0, (t + above) / 2.0]
    out: list[float] = []
    for a in anchors:
        out.append(a)
        lo = hi = a
        for _ in range(3):
            lo = float(np.nextafter(lo, -np.inf))
            hi = float(np.nextafter(hi, np.inf))
            out.extend((lo, hi))
    return out


@pytest.mark.parametrize("t", _thresholds())
def test_lt_boundary_matches_the_float32_cast(t: float) -> None:
    boundary = effective_lt_threshold(t)
    for x in _probes(t):
        native = bool(np.float32(x) < np.float32(t))
        assert (x < boundary) == native, (t, x)


@pytest.mark.parametrize("t", _thresholds())
def test_le_boundary_matches_the_float32_cast(t: float) -> None:
    boundary = effective_le_threshold(t)
    for x in _probes(t):
        native = bool(np.float32(x) <= np.float32(t))
        assert (x <= boundary) == native, (t, x)


def test_le_boundary_of_a_float64_threshold_matches_sklearn_semantics() -> None:
    # sklearn keeps the threshold in float64 and casts only the input
    t = 0.1  # not on the float32 grid
    boundary = effective_le_threshold(t)
    for x in _probes(t):
        assert (x <= boundary) == (float(np.float32(x)) <= t), x
