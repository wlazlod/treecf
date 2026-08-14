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


X0 = np.zeros(3)
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
            exp.recourse_region(X0, unverified, TARGET)

    def test_rejects_bands_target(self, exp: Explainer) -> None:
        bands = Target.bands({"grade": (0.0, 1.0)}, space="raw")
        with pytest.raises(TreecfError, match="bands"):
            exp.recourse_region(X0, X0, bands)

    def test_matches_region_true_convenience_flag(self, exp: Explainer) -> None:
        result = exp.explain(X0, TARGET, backend="genetic", seed=0)
        assert isinstance(result, Counterfactual)
        region = exp.recourse_region(X0, result.x_cf, TARGET)
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


class TestRegionTrueEndToEnd:
    @pytest.mark.parametrize("backend", ["genetic", "exact"])
    def test_produces_a_region_containing_its_own_counterfactual(
        self, exp: Explainer, backend: str
    ) -> None:
        result = exp.explain(X0, TARGET, backend=backend, seed=0, region=True)
        assert isinstance(result, Counterfactual)
        assert isinstance(result.region, RecourseRegion)
        assert result.region.contains(result.x_cf)
        # "a" is the only lever the target needs; b/c never affect this score
        assert result.region.feature_intervals["a"] == (1.0, math.inf)
        assert "b" in result.region.feature_intervals
        assert "c" in result.region.feature_intervals

    def test_region_false_by_default(self, exp: Explainer) -> None:
        result = exp.explain(X0, TARGET, backend="genetic", seed=0)
        assert isinstance(result, Counterfactual)
        assert result.region is None

    def test_bands_target_regions_use_their_own_band_interval(self, exp: Explainer) -> None:
        bands = Target.bands({"reachable": (0.9, 1.0), "unreachable": (3.0, 10.0)}, space="raw")
        result = exp.explain(X0, bands, backend="exact", seed=0, region=True)
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
        result = exp.explain(X0, TARGET, backend="exact", seed=0, region=True)
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
