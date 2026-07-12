"""Batch counterfactual production: explain_batch / BatchResult."""

from __future__ import annotations

import numpy as np
import pytest

from treecf import Explainer, Target, TreecfError
from treecf.batch import BatchResult
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


X = np.zeros((4, 3))
TARGET = Target.raw(op=">=", value=0.5)  # any single lever suffices


class TestSeedsDiversity:
    def test_k_records_per_row_sorted_by_distance(self, exp: Explainer) -> None:
        batch = exp.explain_batch(X, TARGET, n_per_example=2, seed=0)
        for row_id in range(4):
            records = batch.for_id(row_id)
            assert 1 <= len(records) <= 2
            distances = [r.distance for r in records if r.feasible]
            assert distances == sorted(distances)
            assert [r.k for r in records] == list(range(len(records)))

    def test_distinct_change_sets_within_a_row(self, exp: Explainer) -> None:
        batch = exp.explain_batch(X, TARGET, n_per_example=3, seed=0)
        for row_id in range(4):
            keys = [frozenset(r.changes) for r in batch.for_id(row_id) if r.feasible]
            assert len(keys) == len(set(keys))

    def test_whole_batch_is_deterministic(self, exp: Explainer) -> None:
        b1 = exp.explain_batch(X, TARGET, n_per_example=2, seed=7)
        b2 = exp.explain_batch(X, TARGET, n_per_example=2, seed=7)
        for r1, r2 in zip(b1, b2, strict=True):
            assert r1.changes == r2.changes and r1.distance == r2.distance

    def test_custom_ids(self, exp: Explainer) -> None:
        batch = exp.explain_batch(
            X[:2], TARGET, n_per_example=1, ids=["APP-1", "APP-2"], seed=0
        )
        assert batch.for_id("APP-2")
        assert not batch.for_id("APP-3")

    def test_wave_path_matches_sequential_reference(self, exp: Explainer) -> None:
        from treecf.batch import _row_by_seeds

        batch = exp.explain_batch(X, TARGET, n_per_example=2, seed=7)
        reference = Explainer(_ir(), normalizers=np.ones(3))
        expected = []
        for i in range(len(X)):
            expected.extend(
                _row_by_seeds(
                    reference, X[i], TARGET, i, 2, "genetic", 10.0, 0.0,
                    master_seed=7 * 1_000_003 + i * 1_009,
                )
            )
        for got, want in zip(batch.records, expected, strict=True):
            assert (got.id, got.k, got.seed) == (want.id, want.k, want.seed)
            assert got.changes == want.changes
            assert got.distance == want.distance
            assert got.feasible == want.feasible

    def test_infeasible_rows_get_one_infeasible_record(self) -> None:
        from treecf import Freeze

        frozen = Explainer(
            _ir(),
            normalizers=np.ones(3),
            constraints=[Freeze("a"), Freeze("b"), Freeze("c")],
        )
        batch = frozen.explain_batch(X[:2], TARGET, n_per_example=2, seed=0)
        for row_id in range(2):
            records = batch.for_id(row_id)
            assert len(records) == 1
            assert not records[0].feasible


class TestLeverBlocking:
    def test_alternatives_are_structurally_distinct(self, exp: Explainer) -> None:
        batch = exp.explain_batch(
            X[:1], TARGET, n_per_example=3, diversity="lever-blocking", seed=0
        )
        records = [r for r in batch.for_id(0) if r.feasible]
        assert len(records) >= 2
        keys = [frozenset(r.changes) for r in records]
        assert len(keys) == len(set(keys))
        assert records[0].blocked_lever is None  # the primary plan blocks nothing
        assert all(r.blocked_lever for r in records[1:])

    def test_clone_reuses_parent_rust_ensemble(self, exp: Explainer) -> None:
        exp.explain(X[0], TARGET, seed=0)
        clone = exp._with_extra_freezes(["a"])
        assert clone._rust_cache["ensemble"] is exp._rust_cache["ensemble"]
        assert "constraints" not in clone._rust_cache

    def test_essential_levers_recorded(self) -> None:
        # single lever: blocking it makes the target unreachable -> essential
        single = EnsembleIR(
            trees=(_stump(0, 1.0, 1.0),),
            base_score=0.0,
            link=Link.IDENTITY,
            n_features=2,
            feature_names=("a", "b"),
            meta={},
        )
        exp = Explainer(single, normalizers=np.ones(2))
        batch = exp.explain_batch(
            np.zeros((1, 2)), TARGET, n_per_example=2, diversity="lever-blocking", seed=0
        )
        assert batch.essential_levers[0] == ["a"]

    def test_unknown_diversity_raises(self, exp: Explainer) -> None:
        with pytest.raises(TreecfError, match="diversity"):
            exp.explain_batch(X, TARGET, diversity="magic")


class TestPersistence:
    def test_save_load_round_trip_with_nans(self, tmp_path: object) -> None:
        from treecf import AllowMissing

        # NaN routes right on feature a (missing_left=False), so the cheap NaN flip wins
        nan_ir = EnsembleIR(
            trees=(
                Tree(
                    nodes=(
                        Node(0, 0, 1.0, SplitOp.LT, False, 1, 2, None),
                        _leaf(1, 0.0),
                        _leaf(2, 1.0),
                    )
                ),
                _stump(1, 1.0, 0.8),
                _stump(2, 1.0, 0.6),
            ),
            base_score=0.0,
            link=Link.IDENTITY,
            n_features=3,
            feature_names=("a", "b", "c"),
            meta={},
        )
        exp = Explainer(
            nan_ir,
            normalizers=np.ones(3),
            constraints=[AllowMissing("a", delta_miss=0.05)],  # NaN flip is cheapest
        )
        batch = exp.explain_batch(X[:2], TARGET, n_per_example=2, seed=0)
        assert any(
            r.feasible and np.isnan(r.x_cf).any() for r in batch
        ), "expected a NaN counterfactual to exercise encoding"

        path = f"{tmp_path}/batch.json"
        batch.save(path)
        loaded = BatchResult.load(path)
        assert len(loaded) == len(batch)
        for original, restored in zip(batch, loaded, strict=True):
            assert restored.id == original.id and restored.k == original.k
            assert restored.feasible == original.feasible
            if original.feasible:
                np.testing.assert_array_equal(restored.x_cf, original.x_cf)
                assert restored.changes.keys() == original.changes.keys()
                assert restored.distance == original.distance

    def test_to_frame_wide_columns(self, exp: Explainer) -> None:
        pd = pytest.importorskip("pandas")
        batch = exp.explain_batch(X[:2], TARGET, n_per_example=1, seed=0)
        frame = batch.to_frame()
        assert isinstance(frame, pd.DataFrame)
        for column in ("id", "k", "feasible", "distance", "cf_a", "cf_b", "cf_c"):
            assert column in frame.columns
        assert len(frame) == len(batch)


def test_bands_target_rejected(exp: Explainer) -> None:
    with pytest.raises(TreecfError, match="bands"):
        exp.explain_batch(X, Target.bands({"A": (0.1, 0.2)}, space="raw"))
