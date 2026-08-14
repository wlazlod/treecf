"""Exact backend foundations: domains, state costs, canonical orders, validation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from treecf._errors import ConstraintValidationError
from treecf.aim.cells import Cell
from treecf.backends.exact import (
    ExactResult,
    _build_domains,
    _cost_of_row,
    _feature_order,
    _h_suffix,
    _State,
    _term_cost,
    _validate,
)
from treecf.constraints import (
    AllowMissing,
    Equals,
    Freeze,
    Implies,
    Linear,
    Range,
    compile_constraints,
)


def _reference_objective(
    X: np.ndarray,
    x: np.ndarray,
    sigma: np.ndarray,
    weights: np.ndarray,
    lam: float,
    allow_missing: dict[int, tuple[float, float]],
) -> np.ndarray:
    """Byte-for-byte transcription of ``treecf.backends.genetic``'s nested
    ``objective()`` closure (a private local function, not importable) — the
    ground truth the exact backend's state/row costs must reproduce exactly."""
    p = X.shape[1]
    total = np.zeros(len(X))
    for j in range(p):
        x_nan = math.isnan(x[j])
        col = X[:, j]
        col_nan = np.isnan(col)
        if j in allow_missing:
            to_miss, from_miss = allow_missing[j]
        else:
            to_miss = from_miss = 0.0
        if x_nan:
            changed = ~col_nan
            total += changed * (weights[j] * from_miss / sigma[j] + lam)
        else:
            went_nan = col_nan
            moved = ~col_nan & (col != x[j])
            delta = np.where(moved, np.abs(np.nan_to_num(col) - x[j]), 0.0)
            total += went_nan * (weights[j] * to_miss / sigma[j] + lam)
            total += moved * lam + weights[j] * delta / sigma[j]
    return total


class TestExactResultShape:
    def test_is_frozen_with_expected_fields(self) -> None:
        result = ExactResult(
            x_cf=np.array([1.0]),
            proof="optimal",
            stats={"nodes_expanded": 0},
            snapped={},
            distance=0.0,
        )
        assert result.proof == "optimal"
        with pytest.raises(AttributeError):
            result.proof = "heuristic"  # type: ignore[misc]


class TestKeepOnlyDomains:
    def test_frozen_feature_is_keep_only(self) -> None:
        grids = ((Cell(-math.inf, math.inf, True, True),),)
        compiled = compile_constraints((Freeze("x0"),), ("x0",))
        x = np.array([3.0])
        domains = _build_domains(grids, x, compiled, np.ones(1), np.ones(1), 0.0, None)
        assert domains == [[_State(3.0, 0.0, 0, False)]]

    def test_frozen_feature_gets_its_real_cell_index_not_a_hardcoded_zero(self) -> None:
        cells = (
            Cell(-math.inf, 0.0, True, True),
            Cell(0.0, 4.0, False, True),
            Cell(4.0, math.inf, False, True),
        )
        grids = (cells,)
        compiled = compile_constraints((Freeze("x0"),), ("x0",))
        x = np.array([5.0])  # falls in cells[2], not cells[0]
        domains = _build_domains(grids, x, compiled, np.ones(1), np.ones(1), 0.0, None)
        assert domains == [[_State(5.0, 0.0, 2, False)]]

    def test_pinned_matching_the_factual_is_keep_at_zero_cost(self) -> None:
        grids = ((Cell(-math.inf, math.inf, True, True),),)
        compiled = compile_constraints((Range("x0", 5.0, 5.0),), ("x0",))
        x = np.array([5.0])
        domains = _build_domains(grids, x, compiled, np.ones(1), np.ones(1), 0.0, None)
        assert domains == [[_State(5.0, 0.0, 0, False)]]

    def test_pinned_off_the_factual_yields_a_forced_move_state(self) -> None:
        """The controller ruling: lo == hi == v pins the counterfactual value to
        v, not to the factual — a non-conforming factual pays the real movement
        cost to reach v, matching the oracle's ``[v, v]``-intersection behavior
        and the ``check_matrix`` bound it must satisfy."""
        grids = ((Cell(-math.inf, math.inf, True, True),),)
        compiled = compile_constraints((Range("x0", 5.0, 5.0),), ("x0",))
        x = np.array([2.0])
        sigma, weights, lam = np.array([2.0]), np.array([3.0]), 0.5
        domains = _build_domains(grids, x, compiled, sigma, weights, lam, None)
        assert len(domains[0]) == 1
        state = domains[0][0]
        assert state.value == 5.0
        row = np.array([5.0])
        expected = _reference_objective(row[np.newaxis, :], x, sigma, weights, lam, {})[0]
        assert state.cost == expected
        assert state.cost != 0.0  # a real movement, not a free keep

    def test_pinned_off_the_factual_is_exempt_from_value_policy_snapping(self) -> None:
        """Constraints are authoritative: the pinned value is never run through
        ``_snap``, even when a policy is configured for the feature."""
        grids = ((Cell(-math.inf, math.inf, True, True),),)
        compiled = compile_constraints((Range("x0", 5.5, 5.5),), ("x0",))
        x = np.array([2.0])
        domains = _build_domains(
            grids, x, compiled, np.ones(1), np.ones(1), 0.0, {"x0": "integer"}
        )
        assert domains == [[_State(5.5, 3.5, 0, False)]]  # not snapped to 5 or 6

    def test_nan_without_allow_missing_is_keep_only(self) -> None:
        grids = ((Cell(-math.inf, math.inf, True, True),),)
        compiled = compile_constraints((), ("x0",))
        x = np.array([math.nan])
        domains = _build_domains(grids, x, compiled, np.ones(1), np.ones(1), 0.0, None)
        assert len(domains[0]) == 1
        state = domains[0][0]
        assert math.isnan(state.value)
        assert state.cost == 0.0
        assert state.is_nan

    def test_pinned_nan_factual_without_allow_missing_is_keep_nan(self) -> None:
        grids = ((Cell(-math.inf, math.inf, True, True),),)
        compiled = compile_constraints((Range("x0", 5.0, 5.0),), ("x0",))
        x = np.array([math.nan])
        domains = _build_domains(grids, x, compiled, np.ones(1), np.ones(1), 0.0, None)
        assert len(domains[0]) == 1
        state = domains[0][0]
        assert state.is_nan and state.cost == 0.0

    def test_pinned_nan_factual_with_allow_missing_offers_both_states(self) -> None:
        grids = ((Cell(-math.inf, math.inf, True, True),),)
        compiled = compile_constraints(
            (Range("x0", 5.0, 5.0), AllowMissing("x0", delta_miss=1.0, delta_from_miss=6.0)),
            ("x0",),
        )
        x = np.array([math.nan])
        sigma, weights, lam = np.array([2.0]), np.array([3.0]), 0.5
        domains = _build_domains(grids, x, compiled, sigma, weights, lam, None)
        d0 = domains[0]
        assert len(d0) == 2
        assert d0[0].is_nan and d0[0].cost == 0.0  # stay missing: the cheap option
        assert d0[1].value == 5.0
        assert d0[1].cost == lam + (weights[0] * 6.0) / sigma[0]  # priced by delta_from_miss

    def test_pinned_nan_factual_forbid_missing_suppresses_the_nan_state(self) -> None:
        """Reviewer repro: a pinned feature with AllowMissing but also a
        single-feature Linear forbidding NaN there must never offer the NaN
        state, even though the factual is NaN — only the pinned-value state
        (priced by delta_from_miss) survives."""
        grids = ((Cell(-math.inf, math.inf, True, True),),)
        compiled = compile_constraints(
            (
                Range("x0", 5.0, 5.0),
                AllowMissing("x0", delta_miss=1.0, delta_from_miss=6.0),
                Linear({"x0": 1.0}, op="<=", rhs=100.0, missing_policy="forbid_missing"),
            ),
            ("x0",),
        )
        x = np.array([math.nan])
        sigma, weights, lam = np.array([2.0]), np.array([3.0]), 0.5
        domains = _build_domains(grids, x, compiled, sigma, weights, lam, None)
        d0 = domains[0]
        assert len(d0) == 1
        assert not d0[0].is_nan
        assert d0[0].value == 5.0
        assert d0[0].cost == lam + (weights[0] * 6.0) / sigma[0]

    def test_pinned_nan_factual_forced_and_forbidden_yields_empty_domain(self) -> None:
        """Contradictory sub-case: no AllowMissing forces the counterfactual to
        stay NaN (``check_matrix`` legality), but ``forbid_missing`` forbids
        NaN outright -- no value can ever satisfy both, so the domain is
        empty (a certified-infeasible signal), not an error."""
        grids = ((Cell(-math.inf, math.inf, True, True),),)
        compiled = compile_constraints(
            (
                Range("x0", 5.0, 5.0),
                Linear({"x0": 1.0}, op="<=", rhs=100.0, missing_policy="forbid_missing"),
            ),
            ("x0",),
        )
        x = np.array([math.nan])
        domains = _build_domains(grids, x, compiled, np.ones(1), np.ones(1), 0.0, None)
        assert domains == [[]]


class TestIntersectFirst:
    def test_drops_cells_with_empty_bound_intersection(self) -> None:
        cells = (
            Cell(-math.inf, 0.0, True, True),
            Cell(0.0, 5.0, False, True),
            Cell(5.0, math.inf, False, True),
        )
        grids = (cells,)
        compiled = compile_constraints((Range("x0", 10.0, 20.0),), ("x0",))
        x = np.array([-3.0])  # outside bounds -> no keep option
        domains = _build_domains(grids, x, compiled, np.ones(1), np.ones(1), 0.0, None)
        # only the [5, inf) cell survives intersection with [10, 20]
        assert len(domains[0]) == 1
        assert domains[0][0].value == 10.0

    def test_open_intersected_edge_matches_cell_nearest_to_directly(self) -> None:
        """The intersected cell's own open lo edge (untouched by the wider bound)
        must still step exactly one f32 ulp inside, same as calling
        ``Cell.nearest_to`` on it directly."""
        cell = Cell(0.0, 10.0, True, False)  # open at lo=0
        grids = ((Cell(-math.inf, 0.0, True, True), cell),)
        compiled = compile_constraints((Range("x0", -5.0, 10.0),), ("x0",))
        x = np.array([-3.0])  # within bounds (keeps its own value too) but outside `cell`
        domains = _build_domains(grids, x, compiled, np.ones(1), np.ones(1), 0.0, None)
        expected = cell.nearest_to(-3.0)
        moved = next(s for s in domains[0] if not math.isclose(s.value, -3.0))
        assert moved.value == expected
        assert moved.value > 0.0  # stepped strictly inside the open edge


class TestBinaryFeature:
    def test_wide_cell_emits_both_states_zero_before_one_on_cost_tie(self) -> None:
        wide = (Cell(-math.inf, math.inf, True, True),)
        grids = (wide, wide)
        # Implies marks both features binary without narrowing their bounds.
        compiled = compile_constraints(
            (Implies(Equals("x0", 0.0), Equals("x1", 1.0)),), ("x0", "x1")
        )
        x = np.array([0.5, 0.0])  # x0=0.5 is equidistant from 0.0 and 1.0 -> tie
        domains = _build_domains(grids, x, compiled, np.ones(2), np.ones(2), 0.0, None)
        d0 = domains[0]
        assert len(d0) == 3  # keep(0.5) + 0.0 + 1.0
        assert d0[0].value == pytest.approx(0.5) and d0[0].cost == 0.0
        assert d0[1].value == 0.0
        assert d0[2].value == 1.0
        assert d0[1].cost == d0[2].cost  # exact tie
        assert d0[1].cell_idx == d0[2].cell_idx == 0  # both from the one wide cell


class TestValuePolicySnap:
    def test_movement_dropped_but_keep_survives_a_noncomforming_factual(self) -> None:
        """The ruling: value_policies only govern candidates that move a feature.
        x0=2.5 (non-integer) keeps its own value unsnapped even under an
        "integer" policy; a genuine movement candidate from a narrow band with
        no integer inside it is dropped."""
        cells = (
            Cell(1.5, 2.0, False, True),  # no integer lies in [1.5, 2.0)
            Cell(2.0, 3.0, False, True),  # contains x0 = 2.5
            Cell(3.0, math.inf, False, True),
        )
        grids = (cells,)
        compiled = compile_constraints((), ("x0",))
        x = np.array([2.5])
        domains = _build_domains(
            grids, x, compiled, np.ones(1), np.ones(1), 0.0, {"x0": "integer"}
        )
        d0 = domains[0]
        # the keep state is never snapped; the surviving movement candidate is
        # (its value happens to already be an integer, but it still went through
        # _snap since a policy was active for a genuine movement candidate)
        assert d0 == [
            _State(2.5, 0.0, 1, False, False),
            _State(3.0, 0.5, 2, False, True),
        ]


class TestNanState:
    def test_present_only_when_allow_missing(self) -> None:
        grids = ((Cell(-math.inf, math.inf, True, True),),)
        no_allow = compile_constraints((), ("x0",))
        with_allow = compile_constraints((AllowMissing("x0", delta_miss=1.0),), ("x0",))
        x = np.array([2.0])
        domains_no = _build_domains(grids, x, no_allow, np.ones(1), np.ones(1), 0.0, None)
        domains_yes = _build_domains(grids, x, with_allow, np.ones(1), np.ones(1), 0.0, None)
        assert not any(s.is_nan for s in domains_no[0])
        assert sum(s.is_nan for s in domains_yes[0]) == 1

    def test_forbid_missing_linear_suppresses_nan_state(self) -> None:
        grids = ((Cell(-math.inf, math.inf, True, True),),)
        compiled = compile_constraints(
            (
                AllowMissing("x0", delta_miss=1.0),
                Linear({"x0": 1.0}, op="<=", rhs=100.0, missing_policy="forbid_missing"),
            ),
            ("x0",),
        )
        x = np.array([2.0])
        domains = _build_domains(grids, x, compiled, np.ones(1), np.ones(1), 0.0, None)
        assert not any(s.is_nan for s in domains[0])

    def test_nan_state_ordered_last_among_cost_ties(self) -> None:
        cells = (Cell(-math.inf, 5.0, True, True), Cell(5.0, math.inf, False, True))
        grids = (cells,)
        compiled = compile_constraints((AllowMissing("x0", delta_miss=3.0),), ("x0",))
        x = np.array([2.0])
        domains = _build_domains(grids, x, compiled, np.ones(1), np.ones(1), 0.0, None)
        d0 = domains[0]
        assert len(d0) == 3
        assert d0[0].value == 2.0 and d0[0].cost == 0.0  # keep
        assert d0[1].value == 5.0 and d0[1].cost == 3.0  # movement, ties NaN below
        assert d0[2].is_nan and d0[2].cost == 3.0  # NaN sorts after the tied movement


class TestStateCostsMatchReferenceObjective:
    def test_every_domain_state_cost_matches_singleton_row_objective(self) -> None:
        cells0 = (
            Cell(-math.inf, 0.0, True, True),
            Cell(0.0, 4.0, False, True),
            Cell(4.0, math.inf, False, True),
        )
        cells1 = (Cell(-math.inf, math.inf, True, True),)
        grids = (cells0, cells1)
        compiled = compile_constraints(
            (AllowMissing("x1", delta_miss=2.0, delta_from_miss=5.0),), ("x0", "x1")
        )
        x = np.array([1.25, math.nan])
        sigma = np.array([2.0, 4.0])
        weights = np.array([3.0, 1.5])
        lam = 0.7
        domains = _build_domains(grids, x, compiled, sigma, weights, lam, None)
        allow_missing = dict(compiled.allow_missing)
        # sanity: feature 1 must offer both "stay missing" (keep) and "become a value"
        assert sum(s.is_nan for s in domains[1]) == 1
        assert len(domains[1]) == 2
        for j, states in enumerate(domains):
            for state in states:
                row = x.copy()
                row[j] = state.value
                expected = _reference_objective(
                    row[np.newaxis, :], x, sigma, weights, lam, allow_missing
                )[0]
                assert state.cost == expected, (j, state)


class TestCostOfRow:
    def test_matches_reference_objective_exactly(self) -> None:
        x = np.array([1.25, math.nan, -3.0])
        row = np.array([4.0, 0.0, -3.0])  # feature 0 moved, 1 nan->value, 2 unchanged
        sigma = np.array([2.0, 4.0, 1.0])
        weights = np.array([3.0, 1.5, 0.5])
        lam = 0.7
        allow_missing = {1: (2.0, 5.0)}
        expected = _reference_objective(
            row[np.newaxis, :], x, sigma, weights, lam, allow_missing
        )[0]
        actual = _cost_of_row(x, row, sigma, weights, lam, allow_missing)
        assert actual == expected

    def test_term_cost_four_cases_directly(self) -> None:
        # keep
        assert _term_cost(1.0, 1.0, 3.0, 2.0, 0.5, 9.0, 9.0) == 0.0
        # NaN -> NaN (also keep)
        assert _term_cost(math.nan, math.nan, 3.0, 2.0, 0.5, 9.0, 9.0) == 0.0
        # non-NaN -> moved value
        assert _term_cost(1.0, 4.0, 3.0, 2.0, 0.5, 9.0, 9.0) == 0.5 + (3.0 * 3.0) / 2.0
        # non-NaN -> NaN
        assert _term_cost(1.0, math.nan, 3.0, 2.0, 0.5, 9.0, 9.0) == (3.0 * 9.0) / 2.0 + 0.5
        # NaN -> value, independent of the landing value
        v1 = _term_cost(math.nan, 4.0, 3.0, 2.0, 0.5, 9.0, 7.0)
        v2 = _term_cost(math.nan, -100.0, 3.0, 2.0, 0.5, 9.0, 7.0)
        assert v1 == v2 == (3.0 * 7.0) / 2.0 + 0.5


class TestFeatureOrder:
    def test_split_count_desc_ties_asc_index_excludes_untouched_unconstrained(self) -> None:
        two_splits = (
            Cell(-math.inf, 0.0, True, True),
            Cell(0.0, 1.0, False, True),
            Cell(1.0, math.inf, False, True),
        )
        no_split = (Cell(-math.inf, math.inf, True, True),)
        one_split = (Cell(-math.inf, 0.0, True, True), Cell(0.0, math.inf, False, True))
        grids = (
            two_splits,  # x0: 2 splits
            no_split,  # x1: 0 splits, unconstrained -> excluded
            one_split,  # x2: 1 split
            no_split,  # x3: 0 splits, but frozen -> included
            one_split,  # x4: 1 split, same count as x2, higher index
        )
        compiled = compile_constraints((Freeze("x3"),), ("x0", "x1", "x2", "x3", "x4"))
        order = _feature_order(grids, compiled)
        assert order == [0, 2, 4, 3]


class TestHSuffix:
    def test_suffix_sums_of_cheapest_state_cost(self) -> None:
        domains = [
            [_State(0.0, 0.0, 0, False), _State(1.0, 2.0, 1, False)],
            [_State(0.0, 1.0, 0, False), _State(1.0, 3.0, 1, False)],
            [_State(0.0, 0.5, 0, False)],
        ]
        order = [1, 0, 2]
        assert _h_suffix(order, domains) == [1.5, 0.5, 0.5, 0.0]


class TestValidate:
    def test_single_feature_linear_accepted_silently(self) -> None:
        compiled = compile_constraints((Linear({"x0": 1.0}, op="<=", rhs=5.0),), ("x0", "x1"))
        assert _validate(compiled, None) == []

    def test_order_pair_recognized_but_temporarily_rejected(self) -> None:
        compiled = compile_constraints(
            (Linear({"x0": 1.0, "x1": -1.0}, op="<=", rhs=0.0),), ("x0", "x1")
        )
        with pytest.raises(ConstraintValidationError, match=r"later task|coming"):
            _validate(compiled, None)

    def test_general_multi_feature_linear_names_genetic_fallback(self) -> None:
        compiled = compile_constraints(
            (Linear({"x0": 1.0, "x1": 2.0}, op="<=", rhs=5.0),), ("x0", "x1")
        )
        with pytest.raises(ConstraintValidationError, match=r'backend="genetic"'):
            _validate(compiled, None)

    def test_callable_value_policy_names_genetic_fallback(self) -> None:
        compiled = compile_constraints((), ("x0",))
        with pytest.raises(ConstraintValidationError, match=r'backend="genetic"'):
            _validate(compiled, {"x0": lambda v: v})
