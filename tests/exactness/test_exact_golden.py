"""Golden freeze of the Python exact backend, one fixture per scenario.

Each fixture under ``tests/fixtures/exact/`` rebuilds a full ``solve_exact``
call from scratch (ensemble, constraints, value policies, an optional pinned
warm-start incumbent) and compares the result byte-identically against the
answer the Python backend gave when the fixture was generated
(``scripts/gen_exact_fixtures.py``). This is the regression freeze the Rust
port must also reproduce bit-for-bit — see ``fixture_utils.py``
for the shared JSON contract.

Regenerating fixtures is a deliberate act (rerun the generator); a failure
here means the Python exact backend's behavior changed, not the fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from . import fixture_utils

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import gen_exact_fixtures  # noqa: E402  (needs the sys.path insert above)

FIXTURES = fixture_utils.fixture_paths()

# The frozen scenario id set: exactly these 11, no more,
# no fewer -- a scenario silently added or dropped should fail loudly here rather
# than only nudging the `>= 11` count.
EXPECTED_FIXTURE_IDS = frozenset(
    {
        "01-basic-lt-le",
        "02-nan-both-directions",
        "03-order-pair-boundary",
        "04-onehot-implies",
        "05-pinned-features",
        "06-plausibility-pruning",
        "07-gap-within-tolerance",
        "08-warm-start-on",
        "09-warm-start-off",
        "10-certified-infeasible",
        "11-value-policies",
        "12-categorical-blocks",
        "13-categorical-allowed",
        "14-categorical-frozen-nan",
    }
)


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_exact_backend_matches_golden_fixture(path: Path) -> None:
    fixture = fixture_utils.load_fixture(path)
    result = fixture_utils.run_fixture(fixture)
    problems = fixture_utils.diff_golden(fixture, result)
    assert not problems, f"{fixture.name}:\n" + "\n".join(problems)


def test_fixture_set_matches_expected_scenarios() -> None:
    assert {p.stem for p in FIXTURES} == EXPECTED_FIXTURE_IDS


def test_fixture_generation_is_deterministic() -> None:
    """Each scenario builder called twice produces byte-identical payload
    dicts (pre-``golden``, so this isolates the *input* construction from the
    solver run) -- the double-generation determinism check the fixtures'
    whole premise as a cross-language contract depends on, run in-process
    rather than by diffing two on-disk regenerations."""
    for build in gen_exact_fixtures.SCENARIO_BUILDERS:
        first = build()
        second = build()
        assert first == second, f"{first.get('name', build.__name__)}: not deterministic"


def test_scenario_builders_cover_the_expected_fixture_ids() -> None:
    names = {build()["name"] for build in gen_exact_fixtures.SCENARIO_BUILDERS}
    assert names == EXPECTED_FIXTURE_IDS


# --------------------------------------------------------------------------
# Region fixtures: the pure-Python growth loop's own golden freeze.
# --------------------------------------------------------------------------

REGION_FIXTURES = fixture_utils.region_fixture_paths()

EXPECTED_REGION_FIXTURE_IDS = frozenset(
    {
        "region-01-genetic-widened",
        "region-02-exact-found",
        "region-03-plausibility",
        "region-04-order-pair",
    }
)


@pytest.mark.parametrize("path", REGION_FIXTURES, ids=[p.stem for p in REGION_FIXTURES])
def test_region_growth_matches_golden_fixture(path: Path) -> None:
    """Pure-Python growth loop (bypasses the rust-first dispatch), pinned
    directly -- ``tests/rust/test_region_parity.py`` covers rust-vs-python-
    vs-golden three ways at once, the same split ``test_exact_golden.py`` /
    ``test_exact_parity.py`` keep for the exact backend."""
    fixture = fixture_utils.load_region_fixture(path)
    lo, hi = fixture_utils.run_region_fixture(fixture)
    problems = fixture_utils.diff_region_golden(fixture, lo, hi)
    assert not problems, f"{fixture.name}:\n" + "\n".join(problems)


def test_region_fixture_set_matches_expected_scenarios() -> None:
    assert {p.stem for p in REGION_FIXTURES} == EXPECTED_REGION_FIXTURE_IDS


def test_region_fixture_generation_is_deterministic() -> None:
    """Each region scenario builder called twice produces byte-identical
    payload dicts (pre-``golden``) -- the same double-generation determinism
    check ``test_fixture_generation_is_deterministic`` runs for the exact
    fixtures, extended to the region scenario builders."""
    for build in gen_exact_fixtures.REGION_SCENARIO_BUILDERS:
        first = build()
        second = build()
        assert first == second, f"{first.get('name', build.__name__)}: not deterministic"


def test_region_scenario_builders_cover_the_expected_fixture_ids() -> None:
    names = {build()["name"] for build in gen_exact_fixtures.REGION_SCENARIO_BUILDERS}
    assert names == EXPECTED_REGION_FIXTURE_IDS
