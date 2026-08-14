"""Rust region growth: byte-identical parity, three ways at once.

Every fixture under ``tests/fixtures/regions/`` is solved by the pure-Python
growth loop (``treecf.regions._grow_box``) and by the rust engine
(``regions_rust.compute_region_rust``, the same wrapper
``treecf.regions._recourse_region`` dispatches to), and both are compared to
each other AND to the committed golden -- the same three-way split
``test_exact_golden.py``/``test_exact_parity.py`` keep for the exact backend.
"""

from __future__ import annotations

import numpy as np
import pytest

from treecf._json import encode_floats
from treecf.backends.regions_rust import compute_region_rust

from ..exactness import fixture_utils

pytestmark = pytest.mark.rust

_treecf_core = pytest.importorskip("treecf._treecf_core")

REGION_FIXTURES = fixture_utils.region_fixture_paths()

FloatPair = tuple[np.ndarray, np.ndarray]


def _rust_lo_hi(fixture: fixture_utils.RegionFixture) -> FloatPair:
    degenerate, lo_b, hi_b = fixture_utils.region_degenerate_and_bounds(fixture)
    min_total_path = fixture.min_total_path if fixture.min_total_path is not None else 0.0
    return compute_region_rust(
        fixture.ir, fixture.x_cf, fixture.interval, fixture.compiled, lo_b, hi_b,
        degenerate, fixture.if_ir, min_total_path,
    )


@pytest.mark.parametrize("path", REGION_FIXTURES, ids=[p.stem for p in REGION_FIXTURES])
def test_rust_matches_python_and_golden_bitwise(path) -> None:
    fixture = fixture_utils.load_region_fixture(path)
    py_lo, py_hi = fixture_utils.run_region_fixture(fixture)
    rs_lo, rs_hi = _rust_lo_hi(fixture)

    # rust vs python
    assert encode_floats(rs_lo) == encode_floats(py_lo), f"{fixture.name}: lo rust vs python"
    assert encode_floats(rs_hi) == encode_floats(py_hi), f"{fixture.name}: hi rust vs python"

    # rust vs the committed golden (same comparator test_exact_golden.py's
    # region checks use for python vs golden, so all three agree transitively)
    problems = fixture_utils.diff_region_golden(fixture, rs_lo, rs_hi)
    assert not problems, f"{fixture.name} (rust vs golden):\n" + "\n".join(problems)


@pytest.mark.parametrize("path", REGION_FIXTURES, ids=[p.stem for p in REGION_FIXTURES])
def test_rust_growth_is_deterministic_across_repeated_calls(path) -> None:
    fixture = fixture_utils.load_region_fixture(path)
    first_lo, first_hi = _rust_lo_hi(fixture)
    second_lo, second_hi = _rust_lo_hi(fixture)
    assert encode_floats(first_lo) == encode_floats(second_lo)
    assert encode_floats(first_hi) == encode_floats(second_hi)


# --------------------------------------------------- batch region smoke on the rust path ---


def test_batch_region_smoke_on_the_rust_path() -> None:
    """``explain_batch(region=True)`` with ``backend="genetic"`` (rust-first
    by default, both for the search and -- since the extension is importable
    here -- for the region growth ``treecf.regions._recourse_region``
    dispatches to) attaches a region to every feasible record."""
    from treecf import Explainer, Target
    from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree

    def leaf(i: int, v: float) -> Node:
        return Node(i, None, None, None, None, None, None, v)

    def stump(f: int, t: float, rv: float) -> Tree:
        return Tree((Node(0, f, t, SplitOp.LT, True, 1, 2, None), leaf(1, 0.0), leaf(2, rv)))

    ir = EnsembleIR(
        (stump(0, 1.0, 1.0), stump(1, 1.0, 0.8), stump(2, 1.0, 0.6)),
        0.0, Link.IDENTITY, 3, ("a", "b", "c"), {},
    )
    exp = Explainer(ir, normalizers=np.ones(3))
    X = np.zeros((2, 3))
    target = Target.raw(op=">=", value=0.9)
    batch = exp.explain_batch(X, target, backend="genetic", seed=0, region=True)
    assert len(batch) == 2
    for record in batch:
        assert record.feasible
        assert record.region is not None
        assert record.x_cf is not None
        assert record.region.contains(record.x_cf)
