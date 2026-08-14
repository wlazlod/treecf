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
    _EnsembleBounds,
    _feature_order,
    _h_suffix,
    _State,
    _term_cost,
    _validate,
    solve_exact,
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
from treecf.ir.evaluate import raw_score
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree

from ..conftest import make_random_ir


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

    def test_pinned_feature_with_allow_missing_still_offers_the_missing_state(self) -> None:
        """Oracle divergence, seed 22 of the randomized suite: a feature pinned
        to v by Equals/Range may still go missing when AllowMissing says so —
        the pin restricts the value it may take, and ``check_matrix`` lets a
        missing entry past the bounds check entirely. Dropping that state made
        the search return a more expensive counterfactual than the oracle's."""
        grids = ((Cell(-math.inf, math.inf, True, True),),)
        compiled = compile_constraints(
            (Range("x0", 5.0, 5.0), AllowMissing("x0", delta_miss=0.5, delta_from_miss=9.0)),
            ("x0",),
        )
        x = np.array([2.0])
        sigma, weights, lam = np.array([2.0]), np.array([3.0]), 0.5
        domains = _build_domains(grids, x, compiled, sigma, weights, lam, None)
        d0 = domains[0]
        assert len(d0) == 2
        # going missing (0.5 * 3 / 2 + 0.5 = 1.25) undercuts moving to 5.0
        assert d0[0].is_nan
        assert d0[0].cost == (weights[0] * 0.5) / sigma[0] + lam
        assert d0[1].value == 5.0
        row = np.array([math.nan])
        assert bool(compiled.check_matrix(row[np.newaxis, :], x)[0])  # the pin allows it

    def test_pinned_feature_with_allow_missing_honors_forbid_missing(self) -> None:
        """The same feature loses the missing state again once a single-feature
        Linear forbids missing values there."""
        grids = ((Cell(-math.inf, math.inf, True, True),),)
        compiled = compile_constraints(
            (
                Range("x0", 5.0, 5.0),
                AllowMissing("x0", delta_miss=0.5),
                Linear({"x0": 1.0}, op="<=", rhs=100.0, missing_policy="forbid_missing"),
            ),
            ("x0",),
        )
        x = np.array([2.0])
        domains = _build_domains(grids, x, compiled, np.ones(1), np.ones(1), 0.0, None)
        assert domains == [[_State(5.0, 3.0, 0, False)]]

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


class TestUndoDifferentialInvariant:
    """The search only refreshes the trees that split on the feature it just
    assigned, and restores saved brackets on the way back up. Both shortcuts
    must be indistinguishable from walking every tree from scratch."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
    def test_random_assign_undo_walk_matches_a_from_scratch_recompute(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        n_features = 4
        ir = make_random_ir(rng, n_features=n_features, n_trees=5, depth=4)
        assigned = [False] * n_features
        values = [0.0] * n_features
        bounds = _EnsembleBounds(ir, assigned, values)
        pool = [-3.0, -1.5, 0.0, 0.5, 2.0, math.nan]
        mask = 0
        open_frames: list[tuple[int, tuple[tuple[int, float, float], ...]]] = []

        def assert_matches_from_scratch() -> None:
            reference = _EnsembleBounds(ir, assigned, values)
            reference.recompute(mask)
            assert bounds.tree_min == reference.tree_min
            assert bounds.tree_max == reference.tree_max
            assert bounds.score_min == reference.score_min
            assert bounds.score_max == reference.score_max

        for _ in range(300):
            free = [j for j in range(n_features) if not assigned[j]]
            if free and (not open_frames or rng.random() < 0.6):
                j = int(rng.choice(free))
                assigned[j] = True
                values[j] = pool[int(rng.integers(len(pool)))]
                mask |= 1 << j
                open_frames.append((j, bounds.apply(j, mask)))
            else:
                j, frame = open_frames.pop()
                bounds.restore(frame)
                assigned[j] = False
                mask &= ~(1 << j)
            assert_matches_from_scratch()

    def test_full_assignment_bracket_collapses_onto_raw_score(self) -> None:
        """Once every split feature is assigned the bracket is a single number,
        and it is the number ``raw_score`` computes — same trees, same order."""
        rng = np.random.default_rng(11)
        ir = make_random_ir(rng, n_features=3, n_trees=4, depth=3)
        assigned = [True] * 3
        values = [0.25, -1.75, 3.0]
        bounds = _EnsembleBounds(ir, assigned, values)
        bounds.recompute(0b111)
        assert bounds.score_min == bounds.score_max
        assert bounds.score_min == raw_score(ir, np.array(values))


def _two_switch_ir() -> EnsembleIR:
    """Two features, one tree each: crossing 0.0 upward adds exactly 1.0 to the score."""
    trees = tuple(
        Tree(
            nodes=(
                Node(0, j, 0.0, SplitOp.LT, True, 1, 2, None),
                Node(1, None, None, None, None, None, None, 0.0),
                Node(2, None, None, None, None, None, None, 1.0),
            )
        )
        for j in (0, 1)
    )
    return EnsembleIR(
        trees=trees,
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=2,
        feature_names=("x0", "x1"),
        meta={"source": "test"},
    )


class TestFactualIncumbent:
    def test_in_target_factual_is_returned_without_expanding_a_node(self) -> None:
        ir = _two_switch_ir()
        x = np.array([1.0, -2.0])  # score 1.0 already
        compiled = compile_constraints((), ir.feature_names)
        result = solve_exact(
            ir, x, (1.0, 1.0), compiled, np.ones(2), np.ones(2), 0.0, time_budget_s=1e9
        )
        assert result.x_cf is not None
        np.testing.assert_array_equal(result.x_cf, x)
        assert result.distance == 0.0
        assert result.proof == "optimal"
        assert result.snapped == {}
        assert result.stats["nodes_expanded"] == 0
        assert result.stats["completed"] is True
        assert result.stats["warm_start_used"] is False
        assert result.stats["lower_bound"] == 0.0


class TestNodeBudget:
    def test_exhausted_budget_returns_the_warm_incumbent_untouched(self) -> None:
        ir = _two_switch_ir()
        x = np.array([-1.0, -2.0])
        compiled = compile_constraints((), ir.feature_names)
        warm = np.array([0.0, -2.0])
        result = solve_exact(
            ir,
            x,
            (1.0, 1.0),
            compiled,
            np.ones(2),
            np.ones(2),
            0.0,
            node_budget=1,
            time_budget_s=1e9,
            incumbent=(7.0, warm),
        )
        assert result.x_cf is not None
        np.testing.assert_array_equal(result.x_cf, warm)
        assert result.distance == 7.0
        assert result.proof == "heuristic"
        assert result.stats["completed"] is False
        assert result.stats["warm_start_used"] is True
        assert result.stats["nodes_expanded"] == 1
        # an abort reports what the open stack could still reach, not the
        # incumbent it happens to be holding
        assert result.stats["lower_bound"] == 0.0

    def test_a_budget_that_is_never_reached_still_completes(self) -> None:
        ir = _two_switch_ir()
        x = np.array([-1.0, -2.0])
        compiled = compile_constraints((), ir.feature_names)
        result = solve_exact(
            ir,
            x,
            (1.0, 1.0),
            compiled,
            np.ones(2),
            np.ones(2),
            0.0,
            node_budget=10_000,
            time_budget_s=1e9,
        )
        assert result.stats["completed"] is True
        assert result.distance == 1.0  # move x0 from -1.0 to 0.0


class TestGapSemantics:
    """``x0`` reaches the target for 1.0 and ``x1`` for 2.0. A gap wide enough
    to make 2.0 look acceptable prunes the branch that would have found 1.0."""

    def _solve(self, gap: float) -> ExactResult:
        ir = _two_switch_ir()
        x = np.array([-1.0, -2.0])
        compiled = compile_constraints((), ir.feature_names)
        return solve_exact(
            ir, x, (1.0, 1.0), compiled, np.ones(2), np.ones(2), 0.0, gap=gap, time_budget_s=1e9
        )

    def test_no_gap_finds_the_true_optimum(self) -> None:
        result = self._solve(0.0)
        assert result.proof == "optimal"
        assert result.distance == 1.0
        assert result.stats["lower_bound"] == 1.0
        assert result.stats["nodes_pruned_cost"] == 0

    def test_wide_gap_that_prunes_reports_optimal_within_gap(self) -> None:
        result = self._solve(1.0)
        assert result.proof == "optimal_within_gap"
        assert result.distance == 2.0  # the gap prune cut off the 1.0 branch
        assert result.stats["nodes_pruned_cost"] == 1
        assert result.stats["completed"] is True
        assert result.stats["lower_bound"] == 1.0
        assert result.distance is not None
        assert result.distance <= (1.0 + 1.0) * 1.0  # honors the gap it promised

    def test_narrow_gap_that_prunes_nothing_still_reports_optimal(self) -> None:
        result = self._solve(0.01)
        assert result.proof == "optimal"
        assert result.distance == 1.0
        assert result.stats["nodes_pruned_cost"] == 0


def _x0_gate_ir() -> EnsembleIR:
    """Isolation-forest stand-in: path length 0.0 below x0 = 0.0, 5.0 above it."""
    nodes = (
        Node(0, 0, 0.0, SplitOp.LT, True, 1, 2, None),
        Node(1, None, None, None, None, None, None, 0.0),
        Node(2, None, None, None, None, None, None, 5.0),
    )
    return EnsembleIR(
        trees=(Tree(nodes=nodes),),
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=2,
        feature_names=("x0", "x1"),
        meta={"source": "test"},
    )


class TestPlausibilityPrune:
    """A branch the forest can no longer call plausible is cut on the same
    footing as one that can no longer reach the target, and is counted there."""

    def _solve(self, plausibility: tuple[EnsembleIR, float] | None) -> ExactResult:
        ir = _two_switch_ir()
        x = np.array([-1.0, -2.0])
        compiled = compile_constraints((), ir.feature_names)
        return solve_exact(
            ir,
            x,
            (1.0, 1.0),
            compiled,
            np.ones(2),
            np.ones(2),
            0.0,
            plausibility=plausibility,
            time_budget_s=1e9,
        )

    def test_forest_bound_cuts_a_branch_the_target_alone_would_have_explored(self) -> None:
        without = self._solve(None)
        assert without.stats["nodes_expanded"] == 6
        assert without.stats["nodes_pruned_score"] == 2

        with_forest = self._solve((_x0_gate_ir(), 1.0))
        assert with_forest.stats["nodes_expanded"] == 4  # two fewer, one branch gone
        assert with_forest.stats["nodes_pruned_score"] == 2
        assert with_forest.stats["nodes_pruned_cost"] == 0
        assert with_forest.distance == 1.0
        assert with_forest.proof == "optimal"


class TestCertifiedInfeasible:
    def test_empty_domain_returns_completed_without_searching(self) -> None:
        ir = _two_switch_ir()
        compiled = compile_constraints(
            (
                Range("x0", 5.0, 5.0),
                Linear({"x0": 1.0}, op="<=", rhs=100.0, missing_policy="forbid_missing"),
            ),
            ir.feature_names,
        )
        x = np.array([math.nan, -2.0])  # x0 must stay missing and must not be missing
        result = solve_exact(
            ir, x, (1.0, 1.0), compiled, np.ones(2), np.ones(2), 0.0, time_budget_s=1e9
        )
        assert result.x_cf is None
        assert result.distance is None
        assert result.stats["completed"] is True
        assert result.stats["nodes_expanded"] == 0
        assert result.stats["lower_bound"] == math.inf

    def test_unreachable_target_is_enumerated_and_rejected(self) -> None:
        ir = _two_switch_ir()
        x = np.array([-1.0, -2.0])
        compiled = compile_constraints((), ir.feature_names)
        result = solve_exact(
            ir, x, (5.0, 6.0), compiled, np.ones(2), np.ones(2), 0.0, time_budget_s=1e9
        )
        assert result.x_cf is None
        assert result.distance is None
        assert result.stats["completed"] is True
        assert result.stats["lower_bound"] == math.inf
        assert result.stats["nodes_pruned_score"] > 0
