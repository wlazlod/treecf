"""Rust exact backend: byte-identical parity, three ways at once.

Every fixture under ``tests/fixtures/exact/`` is solved by the Python
reference (``solve_exact``) and by the rust engine
(``exact_rust.solve_exact_rust``, the same wrapper ``Explainer._explain_exact``
dispatches to), and both are compared to each other AND to the committed
golden. ``test_exact_golden.py`` already pins python-vs-golden; this module's
job is rust-vs-python and rust-vs-golden, so together the three agree.

A second, independent comparison covers domain construction alone: the
per-feature candidate-state alphabet the search branches over, built by
``debug_domains_raw`` (test-only rust binding) and Python's own
``_build_domains``, compared state by state.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from treecf._json import encode_floats
from treecf.backends import exact_rust
from treecf.backends._exact_domains import _build_domains, _constraint_cells
from treecf.backends.genetic_rust import build_rust_constraints, build_rust_ensemble

from ..exactness import fixture_utils

pytestmark = pytest.mark.rust

_treecf_core = pytest.importorskip("treecf._treecf_core")

FIXTURES = fixture_utils.fixture_paths()


def _bits(v: float | None) -> object:
    """Bit-exact scalar comparator: ``None`` stays ``None``, every NaN
    collapses to one sentinel (payload/sign not guaranteed to match across
    languages), everything else compares by its raw IEEE-754 bytes — so a
    flipped bit anywhere, including a stray sign of zero, is caught."""
    if v is None:
        return None
    if v != v:  # NaN
        return "nan"
    return struct.pack("<d", v).hex()


def diff_exact_results(
    python_result: fixture_utils.ExactResult, rust_result: fixture_utils.ExactResult
) -> list[str]:
    """Rust vs. python, byte-exact, over everything ``ExactResult`` carries."""
    problems: list[str] = []
    got_x_cf = None if rust_result.x_cf is None else encode_floats(rust_result.x_cf)
    want_x_cf = None if python_result.x_cf is None else encode_floats(python_result.x_cf)
    if got_x_cf != want_x_cf:
        problems.append(f"x_cf: python={want_x_cf!r} rust={got_x_cf!r}")
    if _bits(rust_result.distance) != _bits(python_result.distance):
        problems.append(
            f"distance bits: python={_bits(python_result.distance)!r} "
            f"rust={_bits(rust_result.distance)!r}"
        )
    if rust_result.proof != python_result.proof:
        problems.append(f"proof: python={python_result.proof!r} rust={rust_result.proof!r}")
    if rust_result.snapped != python_result.snapped:
        problems.append(f"snapped: python={python_result.snapped!r} rust={rust_result.snapped!r}")
    # int/bool stats fields compare by plain equality; the two float fields
    # (lower_bound, gap) go through _bits(), same as distance -- plain `==`
    # would let a 0.0/-0.0 mismatch slip through unnoticed
    for key in ("nodes_expanded", "nodes_pruned_score", "nodes_pruned_cost", "completed",
                "warm_start_used"):
        if rust_result.stats[key] != python_result.stats[key]:
            problems.append(
                f"stats.{key}: python={python_result.stats[key]!r} rust={rust_result.stats[key]!r}"
            )
    for key in ("lower_bound", "gap"):
        rs_bits, py_bits = _bits(rust_result.stats[key]), _bits(python_result.stats[key])
        if rs_bits != py_bits:
            problems.append(f"stats.{key} bits: python={py_bits!r} rust={rs_bits!r}")
    return problems


def _rust_result(fixture: fixture_utils.ExactFixture) -> fixture_utils.ExactResult:
    plausibility = None
    if fixture.if_ir is not None:
        assert fixture.min_total_path is not None
        plausibility = (fixture.if_ir, fixture.min_total_path)
    return exact_rust.solve_exact_rust(
        fixture.ir,
        fixture.x,
        fixture.interval,
        fixture.compiled,
        fixture.sigma,
        fixture.weights,
        fixture.lam,
        value_policies=fixture.value_policies,
        plausibility=plausibility,
        node_budget=fixture.node_budget,
        gap=fixture.gap,
        time_budget_s=fixture.time_budget_s,
        incumbent=fixture.incumbent,
    )


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_rust_matches_python_and_golden_bitwise(path) -> None:
    fixture = fixture_utils.load_fixture(path)
    python_result = fixture_utils.run_fixture(fixture)
    rust_result = _rust_result(fixture)

    # rust vs python
    problems = diff_exact_results(python_result, rust_result)
    assert not problems, f"{fixture.name} (rust vs python):\n" + "\n".join(problems)

    # rust vs the committed golden (same comparator test_exact_golden.py uses
    # for python vs golden, so all three agree transitively)
    golden_problems = fixture_utils.diff_golden(fixture, rust_result)
    assert not golden_problems, f"{fixture.name} (rust vs golden):\n" + "\n".join(golden_problems)


# ------------------------------------------------- domain-construction parity ---


def _python_domains(fixture: fixture_utils.ExactFixture) -> list[list[object]]:
    grids = (
        _constraint_cells(fixture.compiled, fixture.ir)
        if fixture.if_ir is None
        else _constraint_cells(fixture.compiled, fixture.ir, fixture.if_ir)
    )
    return _build_domains(
        grids, fixture.x, fixture.compiled, fixture.sigma, fixture.weights, fixture.lam,
        fixture.value_policies,
    )


_DomainState = tuple[float, float, int, bool, bool]


def _rust_domains(fixture: fixture_utils.ExactFixture) -> list[list[_DomainState]]:
    ens = build_rust_ensemble(fixture.ir)
    cons = build_rust_constraints(fixture.compiled)
    if_ens = build_rust_ensemble(fixture.if_ir) if fixture.if_ir is not None else None
    code, step, anchor = exact_rust.encode_value_policies(
        fixture.value_policies, fixture.ir.feature_names
    )
    offsets, value, cost, cell_idx, is_nan, snapped = _treecf_core.debug_domains_raw(
        ens,
        cons,
        np.ascontiguousarray(fixture.x, dtype=np.float64),
        np.ascontiguousarray(fixture.sigma, dtype=np.float64),
        np.ascontiguousarray(fixture.weights, dtype=np.float64),
        float(fixture.lam),
        code,
        step,
        anchor,
        if_ensemble=if_ens,
    )
    offsets = np.asarray(offsets)
    out: list[list[_DomainState]] = []
    for f in range(len(offsets) - 1):
        lo, hi = int(offsets[f]), int(offsets[f + 1])
        out.append(
            [
                (
                    float(value[i]),
                    float(cost[i]),
                    int(cell_idx[i]),
                    bool(is_nan[i]),
                    bool(snapped[i]),
                )
                for i in range(lo, hi)
            ]
        )
    return out


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_domain_construction_matches_python_bitwise(path) -> None:
    fixture = fixture_utils.load_fixture(path)
    python_domains = _python_domains(fixture)
    rust_domains = _rust_domains(fixture)

    assert len(rust_domains) == len(python_domains), (
        f"{fixture.name}: {len(python_domains)} python features, "
        f"{len(rust_domains)} rust features"
    )
    for f, (py_states, rs_states) in enumerate(zip(python_domains, rust_domains, strict=True)):
        assert len(rs_states) == len(py_states), (
            f"{fixture.name} feature {f}: {len(py_states)} python states, "
            f"{len(rs_states)} rust states"
        )
        for k, (py_state, rs_state) in enumerate(zip(py_states, rs_states, strict=True)):
            rs_value, rs_cost, rs_cell_idx, rs_is_nan, rs_snapped = rs_state
            problems = []
            if _bits(py_state.value) != _bits(rs_value):
                problems.append(f"value: python={_bits(py_state.value)!r} rust={_bits(rs_value)!r}")
            if _bits(py_state.cost) != _bits(rs_cost):
                problems.append(f"cost: python={_bits(py_state.cost)!r} rust={_bits(rs_cost)!r}")
            if py_state.cell_idx != rs_cell_idx:
                problems.append(f"cell_idx: python={py_state.cell_idx!r} rust={rs_cell_idx!r}")
            if py_state.is_nan != rs_is_nan:
                problems.append(f"is_nan: python={py_state.is_nan!r} rust={rs_is_nan!r}")
            if py_state.snapped != rs_snapped:
                problems.append(f"snapped: python={py_state.snapped!r} rust={rs_snapped!r}")
            assert not problems, f"{fixture.name} feature {f} state {k}:\n" + "\n".join(problems)
