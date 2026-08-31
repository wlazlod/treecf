"""Numeric-model answers are release-stable: recorded results, byte for byte.

The exact fixtures under ``tests/fixtures/exact/`` may be regenerated when
node counters change, so they cannot by themselves prove that *answers*
survived a release. The literals here are recorded independently of the
fixture files and pin ``x_cf``/``distance``/``proof`` — node counts are
deliberately unpinned. The model fingerprint literal pins the canonical
encoding the audit trail depends on.
"""

from __future__ import annotations

import numpy as np
import pytest

from treecf._json import encode_floats
from treecf.audit import ir_fingerprint

from . import fixture_utils

RECORDED = {
    "01-basic-lt-le": (
        [2.79, 0.22949339736348898, 1.28000009059906],
        9.9694487189172,
        "optimal",
    ),
    "03-order-pair-boundary": ([3.0, 3.0], 6.0, "optimal"),
    "06-plausibility-pruning": ([0.0, 0.0, 5.0], 5.0, "optimal"),
}

RECORDED_FINGERPRINT = "4e004fa506fd23b3a655b562cb0627a01786fcbc1e4c4c3b1771627b6e72d778"


@pytest.mark.parametrize("name", sorted(RECORDED))
def test_recorded_numeric_answers_are_byte_identical(name: str) -> None:
    fixture = fixture_utils.load_fixture(fixture_utils.FIXTURES_DIR / f"{name}.json")
    result = fixture_utils.run_fixture(fixture)
    want_x_cf, want_distance, want_proof = RECORDED[name]
    assert result.x_cf is not None and result.distance is not None
    assert encode_floats(result.x_cf) == encode_floats(np.asarray(want_x_cf))
    assert encode_floats(result.distance) == encode_floats(want_distance)
    assert result.proof == want_proof
    # node counters are allowed to change between releases; the answer is not


def test_numeric_model_fingerprint_is_frozen() -> None:
    fixture = fixture_utils.load_fixture(fixture_utils.FIXTURES_DIR / "01-basic-lt-le.json")
    assert ir_fingerprint(fixture.ir) == RECORDED_FINGERPRINT
