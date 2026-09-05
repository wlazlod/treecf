"""Thresholds re-expressed for libraries that cast their inputs to float32.

sklearn, XGBoost, and CatBoost all compare ``float32(x)`` against a split
threshold — the input is rounded to float32 (round-to-nearest-even) before
the comparison. The IR evaluates in float64, so a parser must not store the
library's threshold as-is: a float64 input within half a float32 ulp of it
would route one way natively and the other way in the IR. Each parser stores
the float64 *boundary* of the cast instead — the largest float64 the native
model still routes left — and then plain float64 comparison reproduces the
native routing for every input, including the boundary points a
counterfactual search places representatives on.
"""

from __future__ import annotations

import numpy as np


def effective_le_threshold(t: float) -> float:
    """The largest float64 ``T`` with ``float32(T) <= t``.

    ``x <= T`` then agrees with ``float32(x) <= t`` for every float64 ``x``.
    ``t`` may sit off the float32 grid (sklearn keeps its thresholds in
    float64) or on it (CatBoost borders).

    Construction: let ``f`` be the largest float32 with ``f <= t`` and ``s``
    its float32 successor; every ``x`` below their float64 midpoint rounds to
    ``<= f``. The midpoint itself rounds half-to-even: it belongs to the left
    side exactly when it rounds back to ``f``.
    """
    f32 = np.float32(t)
    if float(f32) > t:
        f32 = np.nextafter(f32, np.float32(-np.inf))
    succ = np.nextafter(f32, np.float32(np.inf))
    mid = (float(f32) + float(succ)) / 2.0
    if float(np.float32(mid)) == float(f32):
        return mid
    return float(np.nextafter(mid, -np.inf))


def effective_lt_threshold(t: float) -> float:
    """The float64 ``T`` with ``x < T`` iff ``float32(x) < t``, for a float32 ``t``.

    ``float32(x) < t`` is ``float32(x) <= pred(t)`` with ``pred(t)`` the
    float32 just below ``t``, so the boundary is the ``<=`` boundary of
    ``pred(t)`` moved up by one float64 ulp.
    """
    pred = float(np.nextafter(np.float32(t), np.float32(-np.inf)))
    return float(np.nextafter(effective_le_threshold(pred), np.inf))
