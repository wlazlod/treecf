"""Golden freeze of the Python exact backend, one fixture per scenario.

Each fixture under ``tests/fixtures/exact/`` rebuilds a full ``solve_exact``
call from scratch (ensemble, constraints, value policies, an optional pinned
warm-start incumbent) and compares the result byte-identically against the
answer the Python backend gave when the fixture was generated
(``scripts/gen_exact_fixtures.py``). This is the regression freeze the Rust
port (Task 2.8) must also reproduce bit-for-bit — see ``fixture_utils.py``
for the shared JSON contract.

Regenerating fixtures is a deliberate act (rerun the generator); a failure
here means the Python exact backend's behavior changed, not the fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from . import fixture_utils

FIXTURES = fixture_utils.fixture_paths()


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_exact_backend_matches_golden_fixture(path: Path) -> None:
    fixture = fixture_utils.load_fixture(path)
    result = fixture_utils.run_fixture(fixture)
    problems = fixture_utils.diff_golden(fixture, result)
    assert not problems, f"{fixture.name}:\n" + "\n".join(problems)


def test_fixture_set_is_not_empty() -> None:
    assert len(FIXTURES) >= 11
