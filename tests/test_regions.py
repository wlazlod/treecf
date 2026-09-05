"""Certified recourse regions: RecourseRegion unit behavior and API wiring."""

from __future__ import annotations

import math

import numpy as np
import pytest

from treecf import (
    Counterfactual,
    Explainer,
    Freeze,
    Implies,
    Infeasible,
    Linear,
    OneHot,
    Range,
    RecourseRegion,
    Target,
    TreecfError,
)
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree


def _leaf(i: int, v: float) -> Node:
    return Node(i, None, None, None, None, None, None, v)


def _stump(feature: int, threshold: float, right_value: float) -> Tree:
    return Tree(
        nodes=(
            Node(0, feature, threshold, SplitOp.LT, True, 1, 2, None),
            _leaf(1, 0.0),
            _leaf(2, right_value),
        )
    )


def _ir() -> EnsembleIR:
    """Three independent levers worth 1.0 / 0.8 / 0.6 on features a/b/c."""
    return EnsembleIR(
        trees=(_stump(0, 1.0, 1.0), _stump(1, 1.0, 0.8), _stump(2, 1.0, 0.6)),
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=3,
        feature_names=("a", "b", "c"),
        meta={},
    )


@pytest.fixture()
def exp() -> Explainer:
    return Explainer(_ir(), normalizers=np.ones(3))


x0 = np.zeros(3)
TARGET = Target.raw(op=">=", value=0.9)  # needs "a" alone; b, c never matter for the score


# --------------------------------------------------------------------------
# RecourseRegion.describe() / .contains()
# --------------------------------------------------------------------------


class TestDescribe:
    def test_two_sided_and_upper_one_sided(self) -> None:
        region = RecourseRegion(
            lo=np.array([-math.inf, 2.0]),
            hi=np.array([0.4, 5.0]),
            feature_intervals={"utilization": (-math.inf, 0.4), "n_loans": (2.0, 5.0)},
            certified=True,
        )
        assert region.describe() == {"utilization": "≤ 0.4", "n_loans": "in [2, 5]"}

    def test_lower_one_sided(self) -> None:
        region = RecourseRegion(
            lo=np.array([1.0]), hi=np.array([math.inf]),
            feature_intervals={"a": (1.0, math.inf)}, certified=True,
        )
        assert region.describe() == {"a": "≥ 1"}

    def test_unconstrained_both_sides(self) -> None:
        region = RecourseRegion(
            lo=np.array([-math.inf]), hi=np.array([math.inf]),
            feature_intervals={"a": (-math.inf, math.inf)}, certified=True,
        )
        assert region.describe() == {"a": "unconstrained"}

    def test_only_non_degenerate_features_are_described(self) -> None:
        region = RecourseRegion(
            lo=np.array([1.0, 5.0]), hi=np.array([1.0, 8.0]),
            feature_intervals={"b": (5.0, 8.0)},  # "a" pinned, excluded on purpose
            certified=True,
        )
        assert set(region.describe()) == {"b"}

    def test_an_endpoint_that_rounds_outside_the_box_is_shown_as_strict(self) -> None:
        # a box ending one float32 ulp below 1 must not read as "1 is fine"
        just_below = float(np.nextafter(np.float32(1.0), np.float32(0.0)))
        just_above = float(np.nextafter(np.float32(2.0), np.float32(3.0)))
        region = RecourseRegion(
            lo=np.array([0.0, -math.inf, just_above, just_above]),
            hi=np.array([just_below, just_below, math.inf, 5.0]),
            feature_intervals={
                "two_sided": (0.0, just_below),
                "upper": (-math.inf, just_below),
                "lower": (just_above, math.inf),
                "both_strict": (just_above, 5.0),
            },
            certified=True,
        )
        assert region.describe() == {
            "two_sided": "in [0, 1)",
            "upper": "< 1",
            "lower": "> 2",
            "both_strict": "in (2, 5]",
        }

    def test_integer_features_are_described_on_the_integers(self) -> None:
        just_below = float(np.nextafter(np.float32(1.0), np.float32(0.0)))
        region = RecourseRegion(
            lo=np.array([0.0, -math.inf, 1.5, 2.0]),
            hi=np.array([just_below, just_below, math.inf, 4.7]),
            feature_intervals={
                "single": (0.0, just_below),
                "upper": (-math.inf, just_below),
                "lower": (1.5, math.inf),
                "span": (2.0, 4.7),
            },
            certified=True,
            integer_features=("single", "upper", "lower", "span"),
        )
        assert region.describe() == {
            "single": "= 0",
            "upper": "≤ 0",
            "lower": "≥ 2",
            "span": "in [2, 4]",
        }

    def test_explain_marks_integer_policy_features(self) -> None:
        exp = Explainer(_ir(), normalizers=np.ones(3), value_policy={"b": "integer"})
        # a must move to 1; b must stay below its split or the score overshoots
        res = exp.explain(x0, Target.raw(range=(0.9, 1.5)), seed=0, region=True)
        assert isinstance(res, Counterfactual) and res.region is not None
        assert res.region.integer_features == ("b",)
        described = res.region.describe()
        assert described["b"] == "≤ 0"
        assert described["c"] == "< 1"


class TestMaximalityFields:
    def test_defaults_claim_nothing(self) -> None:
        region = RecourseRegion(
            lo=np.array([1.0]), hi=np.array([math.inf]),
            feature_intervals={"a": (1.0, math.inf)}, certified=True,
        )
        assert region.maximal == {}
        assert region.maximal_categories == {}
        assert region.witnesses is None

    def test_describe_marks_fully_proved_features(self) -> None:
        region = RecourseRegion(
            lo=np.array([1.0, 2.0, 0.0]), hi=np.array([3.0, 5.0, 0.0]),
            feature_intervals={"a": (1.0, 3.0), "b": (2.0, 5.0)},
            certified=True,
            feature_categories={"c": (0, 1)},
            cat_sets={2: (0, 1)},
            maximal={"a": (True, True), "b": (True, False)},
            maximal_categories={"c": True},
        )
        described = region.describe()
        assert described["a"] == "in [1, 3] (maximal)"
        assert described["b"] == "in [2, 5]"
        assert described["c"] == "∈ {0, 1} (maximal)"

    def test_describe_names_data_limited_sides(self) -> None:
        region = RecourseRegion(
            lo=np.array([1.0, 0.0]), hi=np.array([3.0, 2.0]),
            feature_intervals={"a": (1.0, 3.0), "b": (0.0, 2.0)},
            certified=True,
            maximal={"a": (True, True)},
            data_limited={"a": (False, True), "b": (True, False)},
        )
        described = region.describe()
        assert described["a"] == "in [1, 3] (maximal, data-limited)"
        assert described["b"] == "in [0, 2] (data-limited)"


class TestDataBounds:
    """Without a Range on a feature, growth stops at the observed range of the
    explainer's background data instead of running to infinity."""

    @staticmethod
    def _background() -> np.ndarray:
        rng = np.random.default_rng(0)
        bg = np.column_stack([
            rng.uniform(0.0, 5.0, size=50),
            rng.uniform(0.0, 3.0, size=50),
            rng.uniform(-2.0, 2.0, size=50),
        ])
        bg[0] = [0.0, 0.0, -2.0]
        bg[1] = [5.0, 3.0, 2.0]
        return bg

    def test_sides_without_a_constraint_stop_at_the_data_range(self) -> None:
        exp = Explainer(_ir(), background=self._background())
        res = exp.explain(x0, TARGET, seed=0, region=True)
        assert isinstance(res, Counterfactual) and res.region is not None
        region = res.region
        assert region.feature_intervals["a"] == (1.0, 5.0)  # lower side: the model's split
        assert region.feature_intervals["b"] == (0.0, 3.0)
        assert region.feature_intervals["c"] == (-2.0, 2.0)
        assert region.data_limited == {
            "a": (False, True), "b": (True, True), "c": (True, True),
        }
        described = region.describe()
        assert described["a"] == "in [1, 5] (data-limited)"
        assert described["b"] == "in [0, 3] (data-limited)"

    def test_a_range_constraint_wins_over_the_data(self) -> None:
        exp = Explainer(
            _ir(), background=self._background(), constraints=[Range("b", 0.0, 10.0)]
        )
        res = exp.explain(x0, TARGET, seed=0, region=True)
        assert isinstance(res, Counterfactual) and res.region is not None
        assert res.region.feature_intervals["b"] == (0.0, 10.0)
        assert "b" not in res.region.data_limited

    def test_a_counterfactual_outside_the_data_still_lies_in_its_box(self) -> None:
        bg = self._background()
        bg[:, 1] += 10.0  # b observed in [10, 13]; the counterfactual keeps b at 0
        exp = Explainer(_ir(), background=bg)
        res = exp.explain(x0, Target.raw(range=(0.9, 1.5)), seed=0, region=True)
        assert isinstance(res, Counterfactual) and res.region is not None
        lo, hi = res.region.feature_intervals["b"]
        assert lo == 0.0 and hi < 1.0
        assert res.region.data_limited["b"] == (True, False)
        assert res.region.contains(res.x_cf)

    def test_without_background_nothing_changes(self, exp: Explainer) -> None:
        res = exp.explain(x0, TARGET, seed=0, region=True)
        assert isinstance(res, Counterfactual) and res.region is not None
        assert res.region.feature_intervals["b"] == (-math.inf, math.inf)
        assert res.region.data_limited == {}


class TestContains:
    def test_closed_box_membership(self) -> None:
        region = RecourseRegion(
            lo=np.array([0.0, 1.0]), hi=np.array([2.0, 1.0]),
            feature_intervals={"a": (0.0, 2.0)}, certified=True,
        )
        assert region.contains(np.array([0.0, 1.0]))
        assert region.contains(np.array([2.0, 1.0]))
        assert not region.contains(np.array([2.1, 1.0]))
        assert not region.contains(np.array([1.0, 1.1]))  # degenerate coord must match exactly

    def test_nan_degenerate_coordinate_requires_nan(self) -> None:
        region = RecourseRegion(
            lo=np.array([1.0, math.nan]), hi=np.array([5.0, math.nan]),
            feature_intervals={"a": (1.0, 5.0)}, certified=True,
        )
        assert region.contains(np.array([2.0, math.nan]))
        assert not region.contains(np.array([2.0, 0.0]))  # b must stay missing
        assert not region.contains(np.array([0.0, math.nan]))  # a out of range


# --------------------------------------------------------------------------
# Explainer.recourse_region
# --------------------------------------------------------------------------


class TestRecourseRegionMethod:
    def test_rejects_unverified_counterfactual(self, exp: Explainer) -> None:
        unverified = np.zeros(3)  # score 0.0 does not reach the >= 0.9 target
        with pytest.raises(TreecfError, match="unverified"):
            exp.recourse_region(x0, unverified, TARGET)

    def test_rejects_bands_target(self, exp: Explainer) -> None:
        bands = Target.bands({"grade": (0.0, 1.0)}, space="raw")
        with pytest.raises(TreecfError, match="bands"):
            exp.recourse_region(x0, x0, bands)

    def test_matches_region_true_convenience_flag(self, exp: Explainer) -> None:
        result = exp.explain(x0, TARGET, backend="genetic", seed=0)
        assert isinstance(result, Counterfactual)
        region = exp.recourse_region(x0, result.x_cf, TARGET)
        assert region.contains(result.x_cf)
        assert region.feature_intervals["a"][0] == 1.0

    def test_scoring_error_on_an_unrouted_missing_split_surfaces_as_treecf_error(self) -> None:
        """Regression guard: an adversarial ``x_cf`` whose own path
        hits a split with no missing routing defined makes ``_verify``'s
        ``raw_score`` re-check raise a raw ``ValueError`` -- ``recourse_region``
        must surface that as the ``TreecfError`` its docstring promises, not
        let the ``ValueError`` propagate uncaught."""
        root = Node(0, 0, 1.0, SplitOp.LT, None, 1, 2, None)
        left_leaf = _leaf(1, 0.0)
        f_split = Node(2, 1, 0.5, SplitOp.LT, None, 3, 4, None)  # missing_left=None
        f_leaf_lo = _leaf(3, 0.0)
        f_leaf_hi = _leaf(4, 5.0)
        tree = Tree(nodes=(root, left_leaf, f_split, f_leaf_lo, f_leaf_hi))
        ir = EnsembleIR(
            trees=(tree,), base_score=0.0, link=Link.IDENTITY, n_features=2,
            feature_names=("g", "f"), meta={},
        )
        exp = Explainer(ir, normalizers=np.ones(2))
        x = np.array([0.0, math.nan])
        adversarial = np.array([2.0, math.nan])  # g=2.0 crosses into the unrouted f-split
        target = Target.raw(range=(-1.0, 10.0))
        with pytest.raises(TreecfError, match="no missing routing"):
            exp.recourse_region(x, adversarial, target)


# --------------------------------------------------------------------------
# region=True end to end, every backend
# --------------------------------------------------------------------------


class TestMaximalModeApi:
    """The public surface of the maximal mode: the explain flags, the
    post-hoc method, and the batch path."""

    def test_explain_region_mode_maximal_reports_flags(self, exp: Explainer) -> None:
        result = exp.explain(x0, TARGET, seed=0, region=True, region_mode="maximal")
        assert isinstance(result, Counterfactual) and result.region is not None
        assert set(result.region.maximal) == set(result.region.feature_intervals)
        assert result.region.witnesses is None  # explain never keeps witnesses

    def test_region_mode_without_region_raises(self, exp: Explainer) -> None:
        with pytest.raises(ValueError, match="region=True"):
            exp.explain(x0, TARGET, seed=0, region_mode="maximal")
        with pytest.raises(ValueError, match="region=True"):
            exp.explain(x0, TARGET, seed=0, region_budget=10)

    def test_defaults_are_accepted_without_region(self, exp: Explainer) -> None:
        result = exp.explain(x0, TARGET, seed=0, region_mode="fast", region_budget=100_000)
        assert isinstance(result, Counterfactual)

    def test_unknown_mode_and_bad_budget_are_rejected(self, exp: Explainer) -> None:
        with pytest.raises(TreecfError, match="mode"):
            exp.explain(x0, TARGET, seed=0, region=True, region_mode="sloppy")
        with pytest.raises(ValueError, match="budget"):
            exp.explain(x0, TARGET, seed=0, region=True, region_budget=0)

    def test_recourse_region_keeps_witnesses_on_request(self, exp: Explainer) -> None:
        result = exp.explain(x0, TARGET, seed=0)
        assert isinstance(result, Counterfactual)
        region = exp.recourse_region(
            x0, result.x_cf, TARGET, mode="maximal", budget=500, keep_witnesses=True
        )
        assert region.witnesses is not None
        assert set(region.maximal) == set(region.feature_intervals)
        plain = exp.recourse_region(x0, result.x_cf, TARGET, mode="maximal")
        assert plain.witnesses is None
        assert plain.maximal == region.maximal

    def test_batch_records_carry_flags_but_never_witnesses(self, exp: Explainer) -> None:
        X = np.zeros((2, 3))
        batch = exp.explain_batch(X, TARGET, seed=0, region=True, region_mode="maximal")
        regions = [r.region for r in batch if r.region is not None]
        assert regions
        for region in regions:
            assert set(region.maximal) == set(region.feature_intervals)
            assert region.witnesses is None
        with pytest.raises(ValueError, match="region=True"):
            exp.explain_batch(X, TARGET, seed=0, region_mode="maximal")

    def test_coalitions_forward_the_mode(self, exp: Explainer) -> None:
        by_group = exp.explain_coalitions(
            x0, TARGET, {"first": ["a"], "rest": ["b", "c"]}, seed=0, region=True,
            region_mode="maximal",
        )
        for result in by_group.values():
            if isinstance(result, Counterfactual) and result.region is not None:
                assert set(result.region.maximal) == set(result.region.feature_intervals)


class TestRegionTrueEndToEnd:
    @pytest.mark.parametrize("backend", ["genetic", "exact"])
    def test_produces_a_region_containing_its_own_counterfactual(
        self, exp: Explainer, backend: str
    ) -> None:
        result = exp.explain(x0, TARGET, backend=backend, seed=0, region=True)
        assert isinstance(result, Counterfactual)
        assert isinstance(result.region, RecourseRegion)
        assert result.region.contains(result.x_cf)
        # "a" is the only lever the target needs; b/c never affect this score
        assert result.region.feature_intervals["a"] == (1.0, math.inf)
        assert "b" in result.region.feature_intervals
        assert "c" in result.region.feature_intervals

    def test_region_false_by_default(self, exp: Explainer) -> None:
        result = exp.explain(x0, TARGET, backend="genetic", seed=0)
        assert isinstance(result, Counterfactual)
        assert result.region is None

    def test_bands_target_regions_use_their_own_band_interval(self, exp: Explainer) -> None:
        bands = Target.bands({"reachable": (0.9, 1.0), "unreachable": (3.0, 10.0)}, space="raw")
        result = exp.explain(x0, bands, backend="exact", seed=0, region=True)
        assert isinstance(result, dict)
        reachable = result["reachable"]
        assert isinstance(reachable, Counterfactual)
        assert isinstance(reachable.region, RecourseRegion)
        assert reachable.region.contains(reachable.x_cf)
        unreachable = result["unreachable"]
        assert isinstance(unreachable, Infeasible)


# --------------------------------------------------------------------------
# Degenerate exclusions from feature_intervals
# --------------------------------------------------------------------------


class TestDegenerateFeaturesArePinned:
    def test_frozen_feature_is_excluded(self) -> None:
        exp = Explainer(_ir(), normalizers=np.ones(3), constraints=[Freeze("c")])
        result = exp.explain(x0, TARGET, backend="exact", seed=0, region=True)
        assert isinstance(result, Counterfactual)
        assert result.region is not None
        assert "c" not in result.region.feature_intervals
        assert result.region.lo[2] == result.region.hi[2] == 0.0

    def test_onehot_members_are_excluded(self) -> None:
        exp = Explainer(_ir(), normalizers=np.ones(3), constraints=[OneHot(("b", "c"))])
        x = np.array([0.0, 1.0, 0.0])
        result = exp.explain(x, TARGET, backend="exact", seed=0, region=True)
        assert isinstance(result, Counterfactual)
        assert result.region is not None
        assert "b" not in result.region.feature_intervals
        assert "c" not in result.region.feature_intervals

    def test_implies_referenced_features_are_excluded(self) -> None:
        from treecf import Equals

        exp = Explainer(
            _ir(), normalizers=np.ones(3),
            constraints=[Implies(Equals("a", 1.0), Equals("b", 1.0))],
        )
        x = np.array([1.0, 1.0, 0.0])
        target = Target.raw(op=">=", value=1.7)  # already satisfied at x
        result = exp.explain(x, target, backend="exact", seed=0, region=True)
        assert isinstance(result, Counterfactual)
        assert result.region is not None
        assert "a" not in result.region.feature_intervals
        assert "b" not in result.region.feature_intervals
        assert "c" in result.region.feature_intervals

    def test_unsupported_multi_feature_linear_pins_its_features(self) -> None:
        exp = Explainer(
            _ir(), normalizers=np.ones(3),
            constraints=[Linear({"a": 1.0, "b": 1.0}, op=">=", rhs=1.5)],
        )
        x = np.array([1.0, 1.0, 0.0])  # already satisfies "a" + "b" >= 1.5
        target = Target.raw(op=">=", value=1.7)  # already satisfied at x
        result = exp.explain(x, target, backend="genetic", seed=0, region=True)
        assert isinstance(result, Counterfactual)
        assert result.region is not None
        assert "a" not in result.region.feature_intervals
        assert "b" not in result.region.feature_intervals
        assert "c" in result.region.feature_intervals


# --------------------------------------------------------------------------
# Regression: an unrouted missing split must reject the box, not guess a side
# --------------------------------------------------------------------------


class TestUnroutedMissingSplitRejectsGrowth:
    """Reviewer probe: a model whose root splits on ``g`` (routing feature),
    whose right subtree splits on a NaN-degenerate feature ``f`` with
    ``missing_left=None`` (as every split of an sklearn-parsed model has,
    since sklearn defines no missing-value routing) -- the shape the
    counterfactual's own verified path avoids by staying left of the root
    split. Widening ``g`` past the root threshold opens that unrouted
    subtree; the box oracle must reject rather than silently pick a side
    (the exact backend's identical coercion is safe only because every row
    it returns is re-verified individually -- a region has no such per-point
    recheck, so the oracle itself must carry the strictness).
    """

    def _ir(self) -> EnsembleIR:
        root = Node(0, 0, 1.0, SplitOp.LT, None, 1, 2, None)
        left_leaf = _leaf(1, 0.0)
        f_split = Node(2, 1, 0.5, SplitOp.LT, None, 3, 4, None)  # missing_left=None
        f_leaf_lo = _leaf(3, 0.0)
        f_leaf_hi = _leaf(4, 5.0)
        tree = Tree(nodes=(root, left_leaf, f_split, f_leaf_lo, f_leaf_hi))
        return EnsembleIR(
            trees=(tree,), base_score=0.0, link=Link.IDENTITY, n_features=2,
            feature_names=("g", "f"), meta={},
        )

    def test_growth_stops_before_the_unrouted_subtree_and_every_sample_verifies(self) -> None:
        from treecf.ir.evaluate import raw_score

        ir = self._ir()
        exp = Explainer(ir, normalizers=np.ones(2))
        x_cf = np.array([0.0, math.nan])
        assert raw_score(ir, x_cf) == 0.0  # the factual's own path avoids the f-split

        # Wide enough that the (unsound) old behavior -- treating the
        # unrouted node as routing right, reaching leaf value 5.0 -- would
        # have been accepted.
        target = Target.raw(range=(-1.0, 10.0))
        region = exp.recourse_region(x_cf, x_cf, target)

        assert "g" in region.feature_intervals
        assert region.feature_intervals["g"][1] < 1.0  # never crosses into the unrouted subtree

        interval = target.raw_interval(ir.link)
        rng = np.random.default_rng(0)
        for _ in range(50):
            z = x_cf.copy()
            lo_j = max(region.lo[0], -1e6)
            hi_j = min(region.hi[0], 1e6)
            z[0] = float(rng.uniform(lo_j, hi_j))
            raw_score(ir, z)  # must not raise
            assert exp._verify(x_cf, z, interval) is None
        # the box endpoints themselves, too
        for edge in (region.lo[0], region.hi[0]):
            z = x_cf.copy()
            z[0] = edge if math.isfinite(edge) else math.copysign(1e6, edge)
            raw_score(ir, z)
            assert exp._verify(x_cf, z, interval) is None


# --------------------------------------------------------------------------
# Batch wiring (Python sequential path + the parallel/coalition paths)
# --------------------------------------------------------------------------


class TestBatchRegion:
    def test_python_backend_seeds_smoke(self, exp: Explainer) -> None:
        X = np.zeros((2, 3))
        batch = exp.explain_batch(X, TARGET, backend="python", seed=0, region=True)
        assert len(batch) == 2
        for record in batch:
            assert record.feasible
            assert record.region is not None
            assert record.x_cf is not None
            assert record.region.contains(record.x_cf)

    def test_genetic_seed_waves_smoke(self, exp: Explainer) -> None:
        X = np.zeros((2, 3))
        batch = exp.explain_batch(X, TARGET, backend="genetic", seed=0, region=True)
        for record in batch:
            assert record.feasible
            assert record.region is not None
            assert record.x_cf is not None
            assert record.region.contains(record.x_cf)

    def test_lever_blocking_smoke(self, exp: Explainer) -> None:
        target = Target.raw(op=">=", value=1.7)  # needs at least two levers
        X = np.zeros((1, 3))
        batch = exp.explain_batch(
            X, target, backend="genetic", diversity="lever-blocking",
            n_per_example=2, seed=0, region=True,
        )
        feasible = [r for r in batch if r.feasible]
        assert feasible
        for record in feasible:
            assert record.region is not None
            assert record.x_cf is not None
            assert record.region.contains(record.x_cf)

    def test_coalitions_smoke(self, exp: Explainer) -> None:
        X = np.zeros((1, 3))
        batch = exp.explain_batch(
            X, TARGET, backend="genetic", diversity="coalitions",
            coalitions={"first": ["a"], "rest": ["b", "c"]}, seed=0, region=True,
        )
        feasible = [r for r in batch if r.feasible]
        assert feasible
        for record in feasible:
            assert record.region is not None
            assert record.x_cf is not None
            assert record.region.contains(record.x_cf)

    def test_region_false_leaves_records_without_a_region(self, exp: Explainer) -> None:
        X = np.zeros((2, 3))
        batch = exp.explain_batch(X, TARGET, backend="python", seed=0)
        for record in batch:
            assert record.region is None


class TestMaximalMode:
    """The maximal mode settles every side the conservative bound stops:
    extends it when a budgeted search finds no violating point, proves it
    with a witness otherwise, and reports a side it could not decide."""

    @staticmethod
    def _xor_ir() -> EnsembleIR:
        """Score 0 on the diagonal cells (a<1, b<1) and (a>=1, b>=1), 1 off it."""
        tree3 = Tree(
            nodes=(
                Node(0, 0, 1.0, SplitOp.LT, True, 1, 2, None),
                _leaf(1, 0.0),
                Node(2, 1, 1.0, SplitOp.LT, True, 3, 4, None),
                _leaf(3, 0.0),
                _leaf(4, -2.0),
            )
        )
        return EnsembleIR(
            trees=(_stump(0, 1.0, 1.0), _stump(1, 1.0, 1.0), tree3),
            base_score=0.0,
            link=Link.IDENTITY,
            n_features=2,
            feature_names=("a", "b"),
            meta={},
        )

    def _region(self, interval, mode="maximal", budget=100_000, keep_witnesses=True,
                constraints=()):
        from treecf.constraints.compile import compile_constraints
        from treecf.regions import _recourse_region

        ir = self._xor_ir()
        compiled = compile_constraints(constraints, ir.feature_names)
        x_cf = np.zeros(2)
        return _recourse_region(
            ir, x_cf, x_cf, interval, compiled, None, 0.0,
            mode=mode, budget=budget, keep_witnesses=keep_witnesses,
        )

    def test_fast_mode_stops_where_the_bound_fails(self) -> None:
        region = self._region((-0.5, 1.5), mode="fast")
        assert region.feature_intervals["a"] == (-math.inf, math.inf)
        assert region.feature_intervals["b"][1] < 1.0
        assert region.maximal == {} and region.witnesses is None

    def test_search_extends_a_side_the_bound_rejected(self) -> None:
        region = self._region((-0.5, 1.5))
        assert region.feature_intervals == {
            "a": (-math.inf, math.inf), "b": (-math.inf, math.inf),
        }
        assert region.maximal == {"a": (True, True), "b": (True, True)}
        assert region.witnesses == {}

    def test_witnesses_prove_the_sides_that_cannot_extend(self) -> None:
        region = self._region((-0.5, 0.5))
        assert region.feature_intervals["a"][1] < 1.0
        assert region.feature_intervals["b"][1] < 1.0
        assert region.maximal == {"a": (True, True), "b": (True, True)}
        assert region.witnesses is not None
        assert set(region.witnesses) == {"a:hi", "b:hi"}
        np.testing.assert_array_equal(region.witnesses["a:hi"], [1.0, 0.0])
        np.testing.assert_array_equal(region.witnesses["b:hi"], [0.0, 1.0])
        for point in region.witnesses.values():
            assert not region.contains(point)

    def test_witnesses_are_dropped_unless_asked_for(self) -> None:
        region = self._region((-0.5, 0.5), keep_witnesses=False)
        assert region.witnesses is None
        assert region.maximal == {"a": (True, True), "b": (True, True)}

    def test_budget_leaves_a_side_unproven(self) -> None:
        tight = self._region((-0.5, 1.5), budget=2)
        assert tight.maximal["b"] == (True, False)
        assert tight.feature_intervals["b"][1] < 1.0
        enough = self._region((-0.5, 1.5), budget=3)
        assert enough.maximal["b"] == (True, True)

    def test_order_pair_corner_is_a_witness_without_any_search(self) -> None:
        from treecf.constraints import Linear

        region = self._region(
            (-0.5, 0.5), constraints=[Linear({"a": 1.0, "b": -1.0}, "<=", 0.0)],
        )
        # the corner that breaks a <= b: a just past its factual, b unchanged
        assert region.feature_intervals["a"] == (-math.inf, 0.0)
        assert region.maximal["a"] == (True, True)
        assert region.witnesses is not None
        witness = region.witnesses["a:hi"]
        assert witness[0] > witness[1] == 0.0 and witness[0] < 1.0

    def test_describe_marks_the_proved_features(self) -> None:
        region = self._region((-0.5, 0.5))
        assert all(phrase.endswith("(maximal)") for phrase in region.describe().values())

    def test_unknown_mode_and_bad_budget_are_rejected(self) -> None:
        with pytest.raises(TreecfError, match="mode"):
            self._region((-0.5, 0.5), mode="sloppy")
        with pytest.raises(ValueError, match="budget"):
            self._region((-0.5, 0.5), budget=0)


class TestCategoricalRegions:
    """Category sets are grown, rendered, and honored by membership checks."""

    @staticmethod
    def _region(constraints=(), interval_width=0.4, mode="fast"):
        from treecf.constraints.compile import compile_constraints
        from treecf.ir.evaluate import raw_score
        from treecf.ir.model import CategoricalFeature, EnsembleIR, Link, Node, Tree
        from treecf.regions import _recourse_region

        tree = Tree(
            nodes=(
                Node(0, 0, None, None, True, 1, 2, None, categories=frozenset({2, 3})),
                Node(1, None, None, None, None, None, None, 1.0),
                Node(2, None, None, None, None, None, None, 0.0),
            )
        )
        ir = EnsembleIR(
            trees=(tree,),
            base_score=0.0,
            link=Link.IDENTITY,
            n_features=1,
            feature_names=("occupation",),
            meta={},
            categorical={
                0: CategoricalFeature(
                    cardinality=5,
                    categories=("clerk", "manager", "nurse", "smith", "guard"),
                )
            },
        )
        compiled = compile_constraints(constraints, ir.feature_names, ir.categorical)
        x = np.array([2.0])
        score = raw_score(ir, x)
        interval = (score - interval_width, score + interval_width)
        return _recourse_region(
            ir, x, x, interval, compiled, None, 0.0, mode=mode, keep_witnesses=True
        )

    def test_grows_the_routing_equivalent_codes(self) -> None:
        # codes 2 and 3 share a block (both in the split set): the whole block
        # is certified; the other block scores 0 and stays out
        region = self._region(interval_width=0.4)
        assert region.feature_categories == {"occupation": (2, 3)}
        assert region.cat_sets == {0: (2, 3)}

    def test_wide_interval_admits_every_code(self) -> None:
        region = self._region(interval_width=2.0)
        assert region.feature_categories == {"occupation": (0, 1, 2, 3, 4)}

    def test_allowed_categories_excludes_blocks(self) -> None:
        from treecf.constraints import AllowedCategories

        region = self._region(
            constraints=[AllowedCategories("occupation", (1, 2))], interval_width=2.0
        )
        assert region.feature_categories == {"occupation": (1, 2)}

    def test_contains_checks_membership(self) -> None:
        region = self._region(interval_width=0.4)
        assert region.contains(np.array([3.0]))
        assert not region.contains(np.array([0.0]))
        assert not region.contains(np.array([2.5]))
        assert not region.contains(np.array([np.nan]))

    def test_describe_renders_names(self) -> None:
        region = self._region(interval_width=0.4)
        assert region.describe()["occupation"] == "∈ {nurse, smith}"

    def test_maximal_mode_proves_the_excluded_block_with_a_witness(self) -> None:
        region = self._region(interval_width=0.4, mode="maximal")
        assert region.feature_categories == {"occupation": (2, 3)}
        assert region.maximal_categories == {"occupation": True}
        assert region.witnesses is not None
        np.testing.assert_array_equal(region.witnesses["occupation:cat"], [0.0])
        assert region.describe()["occupation"] == "∈ {nurse, smith} (maximal)"

    def test_maximal_mode_with_every_block_admitted(self) -> None:
        region = self._region(interval_width=2.0, mode="maximal")
        assert region.feature_categories == {"occupation": (0, 1, 2, 3, 4)}
        assert region.maximal_categories == {"occupation": True}
        assert region.witnesses == {}
