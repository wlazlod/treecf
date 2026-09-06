"""Interval routing in the bracket walker and range-aware implication forcing."""

from __future__ import annotations

import numpy as np

from treecf.aim.cells import Cell
from treecf.backends._exact_bounds import _EnsembleBounds
from treecf.backends._exact_domains import _State
from treecf.backends._exact_propagation import _Propagation
from treecf.constraints import Equals, Implies, compile_constraints
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree


def _leaf(i: int, v: float) -> Node:
    return Node(i, None, None, None, None, None, None, v)


def _stump(feature: int, threshold: float, op: SplitOp, right_value: float) -> Tree:
    return Tree(
        nodes=(
            Node(0, feature, threshold, op, True, 1, 2, None),
            _leaf(1, 0.0),
            _leaf(2, right_value),
        )
    )


def _ir() -> EnsembleIR:
    """Feature 0 splits with ``< 1``, feature 1 with ``<= 1``."""
    return EnsembleIR(
        trees=(_stump(0, 1.0, SplitOp.LT, 1.0), _stump(1, 1.0, SplitOp.LE, 0.5)),
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=2,
        feature_names=("a", "b"),
        meta={},
    )


def _bracket(feature: int, rng: tuple[float, float, bool, bool]) -> tuple[float, float]:
    assigned = [False, False]
    values = [0.0, 0.0]
    ranges: list[tuple[float, float, bool, bool] | None] = [None, None]
    bounds = _EnsembleBounds(_ir(), assigned, values, ranges)
    assigned[feature] = True
    ranges[feature] = rng
    bounds.apply(feature, 1 << feature)
    return bounds.tree_min[feature], bounds.tree_max[feature]


class TestIntervalRouting:
    def test_lt_split_resolves_left_when_the_range_ends_below_or_open_at_t(self) -> None:
        assert _bracket(0, (-np.inf, 0.5, True, False)) == (0.0, 0.0)
        assert _bracket(0, (-np.inf, 1.0, True, True)) == (0.0, 0.0)

    def test_lt_split_resolves_right_from_t_onwards(self) -> None:
        assert _bracket(0, (1.0, 5.0, False, False)) == (1.0, 1.0)
        assert _bracket(0, (1.0, 1.0, False, False)) == (1.0, 1.0)  # the singleton cell [t, t]

    def test_lt_split_straddles_when_t_is_strictly_inside(self) -> None:
        assert _bracket(0, (0.0, 2.0, False, False)) == (0.0, 1.0)
        assert _bracket(0, (-np.inf, 1.0, True, False)) == (0.0, 1.0)  # closed at t: both sides

    def test_le_split_resolves_left_up_to_and_including_t(self) -> None:
        assert _bracket(1, (0.0, 1.0, False, False)) == (0.0, 0.0)
        assert _bracket(1, (1.0, 1.0, False, False)) == (0.0, 0.0)  # [t, t] goes left under <=

    def test_le_split_resolves_right_above_t_or_open_at_t(self) -> None:
        assert _bracket(1, (2.0, 3.0, False, False)) == (0.5, 0.5)
        assert _bracket(1, (1.0, 3.0, True, False)) == (0.5, 0.5)

    def test_le_split_straddles_when_t_is_strictly_inside(self) -> None:
        assert _bracket(1, (0.0, 3.0, False, False)) == (0.0, 0.5)

    def test_scalar_routing_is_untouched_when_no_range_is_set(self) -> None:
        assigned = [True, False]
        values = [0.5, 0.0]
        bounds = _EnsembleBounds(_ir(), assigned, values)
        bounds.apply(0, 1)
        assert (bounds.tree_min[0], bounds.tree_max[0]) == (0.0, 0.0)
        assert bounds.ranges == [None, None]


def _implication_setup(
    rng: tuple[float, float, bool, bool] | None,
) -> tuple[_Propagation, list[bool], list[float]]:
    compiled = compile_constraints(
        (Implies(Equals("a", 1.0), Equals("b", 1.0)),), ("a", "b")
    )
    state = _State(value=0.0, cost=0.0, cell_idx=0, is_nan=False)
    domains = [[state], [state]]
    assigned = [False, True]
    values = [0.0, 0.0]
    ranges: list[tuple[float, float, bool, bool] | None] = [None, rng]
    return _Propagation(compiled, domains, assigned, values, ranges), assigned, values


class TestRangeAwareForce:
    def test_demand_inside_a_range_is_recorded_not_conflicted(self) -> None:
        prop, _, _ = _implication_setup((0.0, 5.0, False, False))
        _, conflict = prop.apply(0, 1.0)
        assert conflict is False
        assert prop.forced_value[1] == 1.0

    def test_demand_outside_a_range_conflicts(self) -> None:
        prop, _, _ = _implication_setup((3.0, 5.0, False, False))
        _, conflict = prop.apply(0, 1.0)
        assert conflict is True

    def test_demand_on_an_open_endpoint_conflicts(self) -> None:
        prop, _, _ = _implication_setup((1.0, 5.0, True, False))
        _, conflict = prop.apply(0, 1.0)
        assert conflict is True

    def test_recorded_demand_rejects_a_later_atomic_value(self) -> None:
        prop, _, _ = _implication_setup((0.0, 5.0, False, False))
        prop.apply(0, 1.0)
        _, conflict = prop.apply(1, 0.0)
        assert conflict is True
        _, conflict = prop.apply(1, 1.0)
        assert conflict is False

    def test_scalar_assignment_still_compares_by_value(self) -> None:
        prop, _, values = _implication_setup(None)
        values[1] = 1.0
        _, conflict = prop.apply(0, 1.0)
        assert conflict is False
        values[1] = 0.0
        _, conflict = prop.apply(0, 1.0)
        assert conflict is True


def test_cell_type_is_what_ranges_are_built_from() -> None:
    cell = Cell(0.0, 1.0, False, True)
    assert (cell.lo, cell.hi, cell.lo_open, cell.hi_open) == (0.0, 1.0, False, True)

