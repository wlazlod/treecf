"""Batch counterfactual production: explain_batch / BatchResult."""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from treecf import Explainer, Target, TreecfError, TreecfWarning
from treecf.batch import BatchRecord, BatchResult
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

    def test_batched_primaries_match_sequential_reference(self, exp: Explainer) -> None:
        from treecf.batch import _row_by_lever_blocking

        batch = exp.explain_batch(X, TARGET, n_per_example=3, diversity="lever-blocking", seed=5)
        reference = Explainer(_ir(), normalizers=np.ones(3))
        expected_records = []
        expected_essential = {}
        for i in range(len(X)):
            rows, ess = _row_by_lever_blocking(
                reference, X[i], TARGET, i, 3, "genetic", 10.0, 0.0, seed=5,
            )
            expected_records.extend(rows)
            expected_essential[i] = ess
        for got, want in zip(batch.records, expected_records, strict=True):
            assert (got.id, got.k, got.blocked_lever) == (want.id, want.k, want.blocked_lever)
            assert got.changes == want.changes
            assert got.distance == want.distance
        assert batch.essential_levers == expected_essential

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


class TestExactBatchOptIn:
    def test_no_flag_raises_with_estimate(self, exp: Explainer) -> None:
        X5 = np.zeros((5, 3))
        with pytest.raises(ValueError, match=r"5 rows x 1 plans x 10s") as excinfo:
            exp.explain_batch(X5, TARGET, backend="exact", seed=0)
        assert "hours" in str(excinfo.value)

    def test_no_flag_estimate_multiplies_plans_by_n_per_example(self, exp: Explainer) -> None:
        X3 = np.zeros((3, 3))
        with pytest.raises(ValueError, match=r"3 rows x 4 plans x 10s"):
            exp.explain_batch(
                X3, TARGET, backend="exact", seed=0,
                diversity="lever-blocking", n_per_example=4,
            )

    def test_no_flag_estimate_uses_coalition_count_as_plans(self, exp: Explainer) -> None:
        X2 = np.zeros((2, 3))
        with pytest.raises(ValueError, match=r"2 rows x 3 plans x 10s"):
            exp.explain_batch(
                X2, TARGET, backend="exact", seed=0, diversity="coalitions",
                coalitions={"c1": ["a"], "c2": ["b", "c"]}, include_full=True,
            )

    def test_flag_with_non_exact_backend_raises(self, exp: Explainer) -> None:
        with pytest.raises(ValueError, match="allow_exact_batch"):
            exp.explain_batch(X, TARGET, backend="genetic", allow_exact_batch=True)

    def test_coalitions_gated_by_flag(self, exp: Explainer) -> None:
        kwargs: dict[str, object] = {
            "diversity": "coalitions",
            "coalitions": {"c1": ["a"]},
            "backend": "exact",
            "seed": 0,
        }
        with pytest.raises(ValueError, match="allow_exact_batch"):
            exp.explain_batch(X[:1], TARGET, **kwargs)
        batch = exp.explain_batch(X[:1], TARGET, allow_exact_batch=True, **kwargs)
        assert len(batch) == 1

    def test_one_vectorized_warm_pass_and_no_genetic_fallback(
        self, exp: Explainer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A default (``diversity="seeds"``, ``n_per_example=1``) exact batch
        should run its warm pass through exactly one ``_solve_batch`` call and
        never fall back to the single-row ``_explain_genetic`` path."""
        calls = {"solve_batch": 0, "explain_genetic": 0}
        original_solve_batch = Explainer._solve_batch
        original_explain_genetic = Explainer._explain_genetic

        def spy_solve_batch(self: Explainer, *args: object, **kwargs: object) -> object:
            calls["solve_batch"] += 1
            return original_solve_batch(self, *args, **kwargs)  # type: ignore[arg-type]

        def spy_explain_genetic(self: Explainer, *args: object, **kwargs: object) -> object:
            calls["explain_genetic"] += 1
            return original_explain_genetic(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(Explainer, "_solve_batch", spy_solve_batch)
        monkeypatch.setattr(Explainer, "_explain_genetic", spy_explain_genetic)

        batch = exp.explain_batch(X, TARGET, backend="exact", seed=0, allow_exact_batch=True)
        assert len(batch) == 4
        assert calls["solve_batch"] == 1
        assert calls["explain_genetic"] == 0

    def test_infeasible_warm_row_runs_unwarmed_with_no_ga_fallback(
        self, exp: Explainer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A warm draw that comes back infeasible must not trigger a per-row
        genetic pass -- the exact search for that row simply runs unwarmed."""
        from treecf.backends.genetic import GeneticResult

        explain_genetic_calls = 0
        original_explain_genetic = Explainer._explain_genetic

        def spy_explain_genetic(self: Explainer, *args: object, **kwargs: object) -> object:
            nonlocal explain_genetic_calls
            explain_genetic_calls += 1
            return original_explain_genetic(self, *args, **kwargs)  # type: ignore[arg-type]

        def fake_solve_batch(
            self: Explainer, X: object, tasks: list[object], *args: object, **kwargs: object
        ) -> list[GeneticResult]:
            return [GeneticResult(x_cf=None, stats={}) for _ in tasks]

        monkeypatch.setattr(Explainer, "_explain_genetic", spy_explain_genetic)
        monkeypatch.setattr(Explainer, "_solve_batch", fake_solve_batch)

        X2 = np.zeros((2, 3))
        batch = exp.explain_batch(X2, TARGET, backend="exact", seed=0, allow_exact_batch=True)
        assert explain_genetic_calls == 0
        assert all(r.feasible for r in batch)  # the (generous default budget) exact
        # search still finds a counterfactual on its own, without the warm start

    def test_lever_blocking_smoke(self, exp: Explainer) -> None:
        batch = exp.explain_batch(
            X[:1], TARGET, n_per_example=2, diversity="lever-blocking",
            backend="exact", seed=0, allow_exact_batch=True,
        )
        records = [r for r in batch.for_id(0) if r.feasible]
        assert records
        assert records[0].blocked_lever is None


class TestRecordProof:
    """``BatchRecord.proof``/``solver_stats`` mirror the single-instance result
    that produced each record, so the aggregate degraded warning's pointer at
    "each result's own proof/solver_stats" is true as written."""

    def test_degraded_exact_batch_proofs_sum_to_the_warning_counts(
        self, exp: Explainer
    ) -> None:
        """The aggregate warning's per-kind solve counts must equal the counts
        recoverable from the records' own proof/solver_stats (one solve per
        record here: each row's first attempt returns the warm incumbent)."""
        node_budget = 1
        with pytest.warns(TreecfWarning) as record:
            batch = exp.explain_batch(
                X[:3], TARGET, backend="exact", seed=0,
                node_budget=node_budget, time_budget_s=5.0, allow_exact_batch=True,
            )
        assert len(record) == 1
        message = str(record[0].message)
        warned_counts = {kind: int(n) for kind, n in re.findall(r"(\w+): (\d+) solve", message)}

        record_counts: dict[str, int] = {}
        for r in batch:
            stats = r.solver_stats
            assert stats["completed"] is False  # every solve degraded here
            kind = "exhausted" if int(stats["nodes_expanded"]) >= node_budget else "withdrawn"
            record_counts[kind] = record_counts.get(kind, 0) + 1
        assert record_counts == warned_counts
        assert all(r.proof == "heuristic" for r in batch)  # exhausted with a warm row

    def test_genetic_rows_are_heuristic_with_empty_stats(self, exp: Explainer) -> None:
        batch = exp.explain_batch(X, TARGET, n_per_example=2, seed=0)
        assert all(r.proof == "heuristic" for r in batch if r.feasible)
        assert all(r.solver_stats == {} for r in batch)

    def test_certified_infeasible_row_carries_certified_proof(self, exp: Explainer) -> None:
        unreachable = Target.raw(op=">=", value=10.0)  # max raw score is 2.4
        batch = exp.explain_batch(
            X[:2], unreachable, backend="exact", seed=0, allow_exact_batch=True,
        )
        assert len(batch) == 2
        for r in batch:
            assert not r.feasible
            assert r.proof == "certified"
            assert r.solver_stats["completed"] is True

    def test_optimal_rows_mirror_proof_and_stats(self, exp: Explainer) -> None:
        batch = exp.explain_batch(
            X[:2], TARGET, backend="exact", seed=0, allow_exact_batch=True,
        )
        for r in batch:
            assert r.feasible
            assert r.proof == "optimal"
            assert r.solver_stats["completed"] is True

    def test_construction_without_the_new_fields_still_works(self) -> None:
        record = BatchRecord(
            id=0, k=0, feasible=False, x_cf=None, changes={},
            distance=None, n_changed=None, score_raw=None, score_prob=None,
        )
        assert record.proof == "heuristic"
        assert record.solver_stats == {}

    def test_proof_and_stats_round_trip_through_save_load(
        self, exp: Explainer, tmp_path: object
    ) -> None:
        unreachable = Target.raw(op=">=", value=10.0)
        batch = exp.explain_batch(
            X[:2], unreachable, backend="exact", seed=0, allow_exact_batch=True,
        )
        path = f"{tmp_path}/batch_proof.json"
        batch.save(path)
        loaded = BatchResult.load(path)
        for original, restored in zip(batch, loaded, strict=True):
            assert restored.proof == original.proof
            assert restored.solver_stats == original.solver_stats


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

    def test_region_round_trips_and_old_files_load_with_none(
        self, exp: Explainer, tmp_path: object
    ) -> None:
        from treecf import RecourseRegion

        batch = exp.explain_batch(X[:2], TARGET, n_per_example=1, seed=0, region=True)
        assert any(r.feasible and r.region is not None for r in batch)

        path = f"{tmp_path}/batch_region.json"
        batch.save(path)
        loaded = BatchResult.load(path)
        for original, restored in zip(batch, loaded, strict=True):
            if not original.feasible:
                assert restored.region is None
                continue
            assert isinstance(restored.region, RecourseRegion)
            np.testing.assert_array_equal(restored.region.lo, original.region.lo)
            np.testing.assert_array_equal(restored.region.hi, original.region.hi)
            assert restored.region.feature_intervals == original.region.feature_intervals
            assert restored.region.certified == original.region.certified

        # a file saved without region=True (or by an older version) has no
        # "region" key per record at all -- the loader must not choke on it.
        batch_no_region = exp.explain_batch(X[:2], TARGET, n_per_example=1, seed=0)
        old_path = f"{tmp_path}/batch_pre_region.json"
        batch_no_region.save(old_path)
        with open(old_path, encoding="utf-8") as fh:
            raw = json.load(fh)
        for record in raw["records"]:
            del record["region"]
        with open(old_path, "w", encoding="utf-8") as fh:
            json.dump(raw, fh)
        loaded_old = BatchResult.load(old_path)
        assert all(r.region is None for r in loaded_old)

    def test_to_frame_wide_columns(self, exp: Explainer) -> None:
        pd = pytest.importorskip("pandas")
        batch = exp.explain_batch(X[:2], TARGET, n_per_example=1, seed=0)
        frame = batch.to_frame()
        assert isinstance(frame, pd.DataFrame)
        for column in ("id", "k", "feasible", "distance", "proof", "cf_a", "cf_b", "cf_c"):
            assert column in frame.columns
        assert len(frame) == len(batch)


def test_bands_target_rejected(exp: Explainer) -> None:
    with pytest.raises(TreecfError, match="bands"):
        exp.explain_batch(X, Target.bands({"A": (0.1, 0.2)}, space="raw"))
