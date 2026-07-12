"""JSON-portable float encoding shared by batch persistence and the parity harness.

NaN -> null and ±inf -> "±inf" strings, because strict JSON (and serde_json on
the Rust side) rejects the bare literals.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def encode_floats(values: Any) -> Any:
    if isinstance(values, np.ndarray):
        return [encode_floats(v) for v in values.tolist()]
    if isinstance(values, list | tuple):
        return [encode_floats(v) for v in values]
    if isinstance(values, float):
        if math.isnan(values):
            return None
        if math.isinf(values):
            return "inf" if values > 0 else "-inf"
    return values


def decode_floats(values: Any) -> Any:
    if isinstance(values, list):
        return [decode_floats(v) for v in values]
    if values is None:
        return math.nan
    if values == "inf":
        return math.inf
    if values == "-inf":
        return -math.inf
    return values
