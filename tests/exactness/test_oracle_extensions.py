"""Oracle extensions: joint (model + isolation-forest) grid and value policies."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from treecf.aim.cells import Cell, cell_index, feature_cells
from treecf.api import Grid, _snap
from treecf.constraints import compile_constraints
from treecf.constraints.objects import Range
from treecf.ir.evaluate import raw_score
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree
from treecf.plausibility import Plausibility

from ..conftest import make_random_ir
from .brute_force import solve_brute_force


def _in_window(cell: Cell, lo: float, hi: float) -> Callable[[float], bool]:
    """Mirror ``brute_force._cell_and_bounds``: containment in cell ∩ [lo, hi]."""

    def check(c: float) -> bool:
        return cell.contains(c) and lo <= c <= hi

    return check


def _single_split_ir(
    threshold: float, left_value: float, right_value: float, is_if: bool = False
) -> EnsembleIR:
    """One-feature, one-tree IR: LT ``threshold`` -> left_value else right_value.

    ``is_if`` stamps the ``max_samples`` metadata ``Plausibility.normalizer`` requires,
    for use as a hand-built isolation-forest IR.
    """
    nodes = (
        Node(0, 0, threshold, SplitOp.LT, True, 1, 2, None),
        Node(1, None, None, None, None, None, None, left_value),
        Node(2, None, None, None, None, None, None, right_value),
    )
    meta: dict[str, object] = {"source": "test"}
    if is_if:
        meta["max_samples"] = 4.0
    return EnsembleIR(
        trees=(Tree(nodes=nodes),),
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=1,
        feature_names=("x0",),
        meta=meta,
    )


def _target_plus_dummy_split_ir(dummy_threshold: float) -> EnsembleIR:
    """Two features: x0 alone drives the score (LT 0.0 -> -1.0 else 1.0); x1 has a
    split (at ``dummy_threshold``) that creates real cells but contributes the same
    leaf value on both sides, so x1's cell never affects reachability of the target.
    """
    tree_a = Tree(
        nodes=(
            Node(0, 0, 0.0, SplitOp.LT, True, 1, 2, None),
            Node(1, None, None, None, None, None, None, -1.0),
            Node(2, None, None, None, None, None, None, 1.0),
        )
    )
    tree_b = Tree(
        nodes=(
            Node(0, 1, dummy_threshold, SplitOp.LT, True, 1, 2, None),
            Node(1, None, None, None, None, None, None, 0.0),
            Node(2, None, None, None, None, None, None, 0.0),
        )
    )
    return EnsembleIR(
        trees=(tree_a, tree_b),
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=2,
        feature_names=("x0", "x1"),
        meta={"source": "test"},
    )


def _three_cell_single_feature_ir() -> EnsembleIR:
    """Two splits on x0 -- (-inf, -1), [-1, 1), [1, inf) -- three real, distinct cells."""
    nodes = (
        Node(0, 0, -1.0, SplitOp.LT, True, 1, 2, None),
        Node(1, None, None, None, None, None, None, -1.0),
        Node(2, 0, 1.0, SplitOp.LT, True, 3, 4, None),
        Node(3, None, None, None, None, None, None, 0.0),
        Node(4, None, None, None, None, None, None, 1.0),
    )
    return EnsembleIR(
        trees=(Tree(nodes=nodes),),
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=1,
        feature_names=("x0",),
        meta={"source": "test"},
    )


class TestDedupHelper:
    def test_preserves_first_occurrence_order_and_drops_repeats(self) -> None:
        from .brute_force import _dedup_preserve_order

        assert _dedup_preserve_order([3.0, 1.0, 3.0, 2.0, 1.0]) == [3.0, 1.0, 2.0]

    def test_keeps_a_lone_leading_nan_untouched(self) -> None:
        from .brute_force import _dedup_preserve_order

        result = _dedup_preserve_order([math.nan, 5.0, 5.0])
        assert len(result) == 2
        assert math.isnan(result[0])
        assert result[1] == 5.0


class TestValuePolicyDeduplication:
    def test_collapsing_policy_deduplicates_options(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Distinct cells snapping to the same float under a policy must count once
        against MAX_COMBOS, not once per originating cell."""
        import tests.exactness.brute_force as brute_force_module

        monkeypatch.setattr(
            brute_force_module, "_snap", lambda value, policy, in_cell, lo, hi: 7.0
        )
        monkeypatch.setattr(brute_force_module, "MAX_COMBOS", 2)

        model_ir = _three_cell_single_feature_ir()
        x = np.array([-100.0])  # keeps its own cell's option (-100.0, unsnapped/exempt)
        interval = (1.0, 1.0)  # only the x0 >= 1 cell (leaf 1.0) satisfies this
        compiled = compile_constraints((), model_ir.feature_names)
        sigma = weights = np.ones(1)

        # Undeduped this is 3 candidates (keep=-100.0, plus both non-keep cells
        # mocked to snap to 7.0) -- 3 > MAX_COMBOS=2 would spuriously raise.
        # Deduped it is 2 (-100.0, 7.0), which fits.
        result = solve_brute_force(
            model_ir,
            x,
            interval,
            compiled,
            sigma,
            weights,
            value_policies={"x0": "integer"},
        )
        assert result.feasible
        assert result.x_cf is not None
        assert result.x_cf[0] == 7.0
        assert result.objective == pytest.approx(107.0)


class TestValuePolicyKeepsFactualUnsnapped:
    def test_keep_option_survives_a_non_conforming_policy(self) -> None:
        """A value_policies entry must never force a feature that didn't need to
        move; the factual value is always available, exempt from snapping."""
        model_ir = _target_plus_dummy_split_ir(dummy_threshold=2.0)
        x = np.array([-2.0, 2.3])  # x1 = 2.3 does not conform to "integer"
        interval = (1.0, 1.0)  # only reachable by moving x0 into its x0 >= 0 cell
        compiled = compile_constraints((), model_ir.feature_names)
        sigma = weights = np.ones(2)

        result = solve_brute_force(
            model_ir,
            x,
            interval,
            compiled,
            sigma,
            weights,
            value_policies={"x1": "integer"},
        )
        assert result.feasible
        assert result.x_cf is not None
        assert result.x_cf[0] == pytest.approx(0.0)
        assert result.x_cf[1] == 2.3  # kept exactly, never snapped
        # x0 alone paid for the move: no extra cost was charged for keeping x1
        assert result.objective == pytest.approx(2.0)


class TestPlausibilityFilter:
    def test_rejects_below_min_total_path_and_joint_grid_shifts_result(self) -> None:
        model_ir = _single_split_ir(threshold=0.0, left_value=-1.0, right_value=1.0)
        # IF-only split at 5.0: low path length below it, high above -- so only
        # candidates >= 5.0 clear the plausibility bound.
        if_ir = _single_split_ir(threshold=5.0, left_value=0.5, right_value=5.0, is_if=True)
        plaus = Plausibility.isolation_forest(if_ir, max_anomaly_score=0.5)
        # sanity: min_total_path sits strictly between the two IF leaf values
        assert 0.5 < plaus.min_total_path < 5.0

        x = np.array([-2.0])
        interval = (1.0, 1.0)
        compiled = compile_constraints((), model_ir.feature_names)
        sigma = weights = np.ones(1)

        model_only = solve_brute_force(model_ir, x, interval, compiled, sigma, weights)
        assert model_only.feasible
        assert model_only.x_cf is not None
        assert model_only.x_cf[0] == pytest.approx(0.0)

        joint = solve_brute_force(
            model_ir,
            x,
            interval,
            compiled,
            sigma,
            weights,
            plausibility=(plaus.if_ir, plaus.min_total_path),
        )
        assert joint.feasible
        assert joint.x_cf is not None
        # forced onto the IF-only cell edge at 5.0 -- the model-only grid never offers it
        assert joint.x_cf[0] == pytest.approx(5.0)
        assert joint.objective > model_only.objective
        assert raw_score(if_ir, joint.x_cf) >= plaus.min_total_path


class TestPlausibilityConsistency:
    def test_matches_anomaly_score_formulation(self) -> None:
        if_ir = _single_split_ir(threshold=5.0, left_value=0.5, right_value=5.0, is_if=True)
        plaus = Plausibility.isolation_forest(if_ir, max_anomaly_score=0.5)
        for value in (-2.0, 0.0, 3.0, 5.0, 10.0):
            candidate = np.array([value])
            oracle_ok = raw_score(if_ir, candidate) >= plaus.min_total_path
            anomaly_ok = plaus.anomaly_score(candidate) <= plaus.max_anomaly_score + 1e-9
            assert oracle_ok == anomaly_ok, value


class TestValuePolicies:
    def test_integer_policy_matches_api_snap(self) -> None:
        model_ir = _single_split_ir(threshold=0.0, left_value=-1.0, right_value=1.0)
        x = np.array([-2.0])
        interval = (1.0, 1.0)
        # Range keeps the factual (-2.0) out of the reachable window, so it has no
        # keep option -- the oracle must snap a genuine movement candidate (the
        # bound-clamped nearest point, 2.6).
        constraints = (Range(feature="x0", lo=2.6, hi=10.0),)
        compiled = compile_constraints(constraints, model_ir.feature_names)
        sigma = weights = np.ones(1)

        cells = feature_cells(model_ir)[0]
        right_cell = cells[cell_index(cells, 2.6)]
        expected = _snap(2.6, "integer", _in_window(right_cell, 2.6, 10.0), 2.6, 10.0)
        assert expected == 3.0

        result = solve_brute_force(
            model_ir,
            x,
            interval,
            compiled,
            sigma,
            weights,
            value_policies={"x0": "integer"},
        )
        assert result.feasible
        assert result.x_cf is not None
        assert result.x_cf[0] == expected

    def test_grid_policy_snaps_to_step(self) -> None:
        model_ir = _single_split_ir(threshold=0.0, left_value=-1.0, right_value=1.0)
        x = np.array([-2.0])
        interval = (1.0, 1.0)
        constraints = (Range(feature="x0", lo=2.6, hi=10.0),)
        compiled = compile_constraints(constraints, model_ir.feature_names)
        sigma = weights = np.ones(1)

        cells = feature_cells(model_ir)[0]
        right_cell = cells[cell_index(cells, 2.6)]
        policy = Grid(step=2.0, anchor=0.0)
        expected = _snap(2.6, policy, _in_window(right_cell, 2.6, 10.0), 2.6, 10.0)
        assert expected == 4.0  # grid point 2.0 lies below the [2.6, 10.0] window

        result = solve_brute_force(
            model_ir,
            x,
            interval,
            compiled,
            sigma,
            weights,
            value_policies={"x0": policy},
        )
        assert result.feasible
        assert result.x_cf is not None
        assert result.x_cf[0] == expected

    def test_policy_outside_interval_drops_option_and_can_make_oracle_infeasible(self) -> None:
        model_ir = _single_split_ir(threshold=0.0, left_value=-1.0, right_value=1.0)
        x = np.array([2.7])
        interval = (1.0, 1.0)
        # Narrow window: the un-snapped nearest point (2.61, clamped to the
        # window) is a feasible option, but no integer lies inside the window.
        constraints = (Range(feature="x0", lo=2.6, hi=2.61),)
        compiled = compile_constraints(constraints, model_ir.feature_names)
        sigma = weights = np.ones(1)

        unpoliced = solve_brute_force(model_ir, x, interval, compiled, sigma, weights)
        assert unpoliced.feasible
        assert unpoliced.x_cf is not None
        assert unpoliced.x_cf[0] == pytest.approx(2.61)

        policed = solve_brute_force(
            model_ir,
            x,
            interval,
            compiled,
            sigma,
            weights,
            value_policies={"x0": "integer"},
        )
        assert not policed.feasible
        assert policed.objective == math.inf
        assert policed.x_cf is None


class TestBackwardCompatibility:
    def test_positional_call_matches_explicit_none_kwargs(self) -> None:
        rng = np.random.default_rng(4242)
        ir = make_random_ir(rng, n_features=3, n_trees=4, depth=3)
        x = rng.normal(scale=2.0, size=3)
        scores = [raw_score(ir, rng.normal(scale=3.0, size=3)) for _ in range(40)]
        lo_t = float(np.percentile(scores, 60))
        compiled = compile_constraints((), ir.feature_names)
        sigma = weights = np.ones(3)

        positional = solve_brute_force(ir, x, (lo_t, math.inf), compiled, sigma, weights, 0.05)
        explicit = solve_brute_force(
            ir,
            x,
            (lo_t, math.inf),
            compiled,
            sigma,
            weights,
            0.05,
            plausibility=None,
            value_policies=None,
        )
        assert positional.feasible == explicit.feasible
        assert positional.objective == explicit.objective
        if positional.x_cf is None:
            assert explicit.x_cf is None
        else:
            assert explicit.x_cf is not None
            np.testing.assert_array_equal(positional.x_cf, explicit.x_cf)


class TestMaxCombosGuard:
    def test_still_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import tests.exactness.brute_force as brute_force_module

        monkeypatch.setattr(brute_force_module, "MAX_COMBOS", 4)
        # two 1-feature single-split IRs give 2 x 2 = 4 combos at the guard's
        # current value; a third split feature pushes it past the guard.
        rng = np.random.default_rng(1)
        ir = make_random_ir(rng, n_features=3, n_trees=4, depth=3)
        x = rng.normal(scale=2.0, size=3)
        compiled = compile_constraints((), ir.feature_names)
        sigma = weights = np.ones(3)
        with pytest.raises(ValueError, match="combos exceed oracle guard"):
            solve_brute_force(ir, x, (-math.inf, math.inf), compiled, sigma, weights)


class TestCategoricalOracle:
    """Blocks are the categorical enumeration unit; a change costs one flat unit."""

    @staticmethod
    def _mixed_ir():
        from treecf.ir.model import CategoricalFeature, EnsembleIR, Link, Node, SplitOp, Tree

        t0 = Tree(
            nodes=(
                Node(0, 0, None, None, True, 1, 2, None, categories=frozenset({2, 3})),
                Node(1, None, None, None, None, None, None, 1.0),
                Node(2, None, None, None, None, None, None, 0.0),
            )
        )
        t1 = Tree(
            nodes=(
                Node(0, 1, 0.5, SplitOp.LE, True, 1, 2, None),
                Node(1, None, None, None, None, None, None, 0.0),
                Node(2, None, None, None, None, None, None, 1.0),
            )
        )
        return EnsembleIR(
            trees=(t0, t1),
            base_score=0.0,
            link=Link.IDENTITY,
            n_features=2,
            feature_names=("occupation", "amount"),
            meta={},
            categorical={0: CategoricalFeature(cardinality=5)},
        )

    def test_oracle_enumerates_block_representatives(self) -> None:
        from treecf.constraints.compile import compile_constraints

        ir = self._mixed_ir()
        compiled = compile_constraints((), ir.feature_names)
        x = np.array([0.0, 0.0])
        result = solve_brute_force(
            ir, x, (2.0, math.inf), compiled, np.ones(2), np.ones(2), lam=0.0
        )
        assert result.feasible and result.x_cf is not None
        # only codes {2, 3} reach the target; the block representative is 2
        assert result.x_cf[0] == 2.0
        amount = result.x_cf[1]
        assert amount > 0.5
        # flat unit for the category change plus the numeric move
        assert result.objective == pytest.approx(1.0 + amount, abs=1e-12)

    def test_certified_infeasible_when_no_block_reaches(self) -> None:
        from treecf.constraints.compile import compile_constraints

        ir = self._mixed_ir()
        compiled = compile_constraints((), ir.feature_names)
        x = np.array([0.0, 0.0])
        result = solve_brute_force(
            ir, x, (3.0, math.inf), compiled, np.ones(2), np.ones(2), lam=0.0
        )
        assert not result.feasible and result.x_cf is None
