"""Coalitions mode: per-group counterfactuals (explain_coalitions + batch diversity)."""

from __future__ import annotations

import numpy as np
import pytest

from treecf import Counterfactual, Explainer, Freeze, Infeasible, Target, TreecfError
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
TARGET = Target.raw(op=">=", value=0.5)  # any single lever suffices
COALITIONS = {"first": ["a"], "rest": ["b", "c"]}


class TestValidation:
    def test_unknown_feature_raises_with_name(self, exp: Explainer) -> None:
        with pytest.raises(TreecfError, match="ghost"):
            exp.explain_coalitions(X0, TARGET, {"g": ["a", "ghost"]})

    def test_empty_mapping_and_empty_coalition_raise(self, exp: Explainer) -> None:
        with pytest.raises(TreecfError):
            exp.explain_coalitions(X0, TARGET, {})
        with pytest.raises(TreecfError):
            exp.explain_coalitions(X0, TARGET, {"g": []})

    def test_reserved_name_collides_with_include_full(self, exp: Explainer) -> None:
        with pytest.raises(TreecfError, match="reserved"):
            exp.explain_coalitions(
                X0, TARGET, {"(all levers)": ["a"]}, include_full=True
            )
        # without the baseline the name is just a name
        result = exp.explain_coalitions(X0, TARGET, {"(all levers)": ["a"]})
        assert set(result) == {"(all levers)"}

    def test_overlapping_coalitions_accepted(self, exp: Explainer) -> None:
        result = exp.explain_coalitions(
            X0, TARGET, {"g1": ["a", "b"], "g2": ["b", "c"]}, seed=0
        )
        assert set(result) == {"g1", "g2"}

    def test_bands_target_rejected(self, exp: Explainer) -> None:
        bands = Target.bands({"lo": (0.4, 0.7), "hi": (0.7, 2.0)}, space="raw")
        with pytest.raises(TreecfError, match="bands"):
            exp.explain_coalitions(X0, bands, COALITIONS)


class TestSingleRow:
    def test_plans_only_touch_their_coalition(self, exp: Explainer) -> None:
        result = exp.explain_coalitions(X0, TARGET, COALITIONS, seed=0)
        for name, outcome in result.items():
            assert isinstance(outcome, Counterfactual)
            assert set(outcome.changes) <= set(COALITIONS[name])

    def test_coalition_plan_equals_manual_freeze_complement(self, exp: Explainer) -> None:
        result = exp.explain_coalitions(X0, TARGET, COALITIONS, seed=3)
        clone = exp._with_extra_freezes(["b", "c"])  # complement of "first"
        manual = clone.explain(X0, TARGET, seed=3)
        assert isinstance(manual, Counterfactual)
        first = result["first"]
        assert isinstance(first, Counterfactual)
        assert np.array_equal(first.x_cf, manual.x_cf, equal_nan=True)
        assert first.distance == manual.distance

    def test_include_full_baseline_first_and_equal_to_plain_explain(
        self, exp: Explainer
    ) -> None:
        result = exp.explain_coalitions(
            X0, TARGET, COALITIONS, include_full=True, seed=1
        )
        assert next(iter(result)) == "(all levers)"
        plain = exp.explain(X0, TARGET, seed=1)
        baseline = result["(all levers)"]
        assert isinstance(baseline, Counterfactual) and isinstance(plain, Counterfactual)
        assert np.array_equal(baseline.x_cf, plain.x_cf, equal_nan=True)

    def test_allow_missing_outside_coalition_is_dropped_not_conflicting(self) -> None:
        from treecf import AllowMissing

        exp = Explainer(
            _ir(),
            normalizers=np.ones(3),
            constraints=[AllowMissing("c", delta_miss=1.0)],
        )
        # freezing the complement of {"a"} freezes "c"; AllowMissing("c") must
        # not blow up the clone, and "a"-only plans must still be found
        result = exp.explain_coalitions(X0, TARGET, {"first": ["a"]}, seed=0)
        outcome = result["first"]
        assert isinstance(outcome, Counterfactual)
        assert set(outcome.changes) == {"a"}

    def test_frozen_coalition_is_infeasible_alone(self) -> None:
        frozen_a = Explainer(_ir(), normalizers=np.ones(3), constraints=[Freeze("a")])
        result = frozen_a.explain_coalitions(X0, TARGET, COALITIONS, seed=0)
        assert isinstance(result["first"], Infeasible)  # its only lever is frozen
        assert isinstance(result["rest"], Counterfactual)


class TestBatch:
    X = np.zeros((3, 3))

    def test_one_record_per_coalition_per_row(self, exp: Explainer) -> None:
        batch = exp.explain_batch(
            self.X, TARGET, diversity="coalitions", coalitions=COALITIONS, seed=0
        )
        for row_id in range(3):
            records = batch.for_id(row_id)
            assert sorted(r.coalition for r in records) == ["first", "rest"]
            assert [r.k for r in records] == [0, 1]
            distances = [r.distance for r in records if r.feasible]
            assert distances == sorted(distances)

    def test_deterministic_and_python_backend_matches(self, exp: Explainer) -> None:
        kwargs = dict(diversity="coalitions", coalitions=COALITIONS, seed=5)
        b1 = exp.explain_batch(self.X, TARGET, **kwargs)
        b2 = exp.explain_batch(self.X, TARGET, **kwargs)
        for r1, r2 in zip(b1, b2, strict=True):
            assert (r1.coalition, r1.changes, r1.distance) == (
                r2.coalition, r2.changes, r2.distance
            )
        # the numpy reference walks the same clones row by row
        reference = Explainer(_ir(), normalizers=np.ones(3))
        b3 = reference.explain_batch(self.X, TARGET, backend="python", **kwargs)
        for r1, r3 in zip(b1, b3, strict=True):
            assert (r1.id, r1.coalition, r1.feasible) == (r3.id, r3.coalition, r3.feasible)
            assert set(r1.changes) == set(r3.changes)

    def test_n_per_example_has_no_effect(self, exp: Explainer) -> None:
        common = dict(diversity="coalitions", coalitions=COALITIONS, seed=0)
        b1 = exp.explain_batch(self.X, TARGET, n_per_example=1, **common)
        b5 = exp.explain_batch(self.X, TARGET, n_per_example=5, **common)
        assert len(b1) == len(b5)

    def test_infeasible_coalition_gets_named_record(self) -> None:
        frozen_a = Explainer(_ir(), normalizers=np.ones(3), constraints=[Freeze("a")])
        batch = frozen_a.explain_batch(
            self.X[:1], TARGET, diversity="coalitions", coalitions=COALITIONS,
            include_full=True, seed=0,
        )
        records = batch.for_id(0)
        by_name = {r.coalition: r for r in records}
        assert not by_name["first"].feasible
        assert by_name["rest"].feasible and by_name["(all levers)"].feasible
        assert len({r.k for r in records}) == len(records)  # (id, k) unique

    def test_mode_arguments_are_validated(self, exp: Explainer) -> None:
        with pytest.raises(TreecfError, match="requires"):
            exp.explain_batch(self.X, TARGET, diversity="coalitions")
        with pytest.raises(TreecfError, match="only valid"):
            exp.explain_batch(self.X, TARGET, coalitions=COALITIONS)
        with pytest.raises(TreecfError, match="only valid"):
            exp.explain_batch(self.X, TARGET, include_full=True)

    def test_round_trip_preserves_coalition(self, exp: Explainer, tmp_path: object) -> None:
        import pathlib

        batch = exp.explain_batch(
            self.X, TARGET, diversity="coalitions", coalitions=COALITIONS, seed=0
        )
        path = pathlib.Path(str(tmp_path)) / "batch.json"
        batch.save(path)
        from treecf import BatchResult

        loaded = BatchResult.load(path)
        for r1, r2 in zip(batch, loaded, strict=True):
            assert r1.coalition == r2.coalition

    def test_load_pre_coalition_json_defaults_to_none(
        self, exp: Explainer, tmp_path: object
    ) -> None:
        import json
        import pathlib

        batch = exp.explain_batch(self.X[:1], TARGET, seed=0)
        path = pathlib.Path(str(tmp_path)) / "old.json"
        batch.save(path)
        data = json.loads(path.read_text())
        for raw in data["records"]:
            raw.pop("coalition")  # simulate a file written before the field existed
        path.write_text(json.dumps(data))
        from treecf import BatchResult

        loaded = BatchResult.load(path)
        assert all(r.coalition is None for r in loaded)

    def test_to_frame_has_coalition_column(self, exp: Explainer) -> None:
        pytest.importorskip("pandas")
        batch = exp.explain_batch(
            self.X[:1], TARGET, diversity="coalitions", coalitions=COALITIONS, seed=0
        )
        frame = batch.to_frame()
        assert set(frame["coalition"]) == {"first", "rest"}
