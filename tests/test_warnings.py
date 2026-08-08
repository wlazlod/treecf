"""TreecfWarning: emitted when the factual violates constraints, and only then."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from treecf import Explainer, Range, Target, TreecfWarning, constraint


def _stump_dump() -> dict[str, object]:
    """Tiny two-tree binary ensemble as a LightGBM-format dump (no extras needed)."""

    def split(feat: int, thr: float, left: object, right: object) -> dict[str, object]:
        return {"split_feature": feat, "threshold": thr, "decision_type": "<=",
                "missing_type": "NaN", "default_left": True,
                "left_child": left, "right_child": right}

    return {"num_tree_per_iteration": 1, "objective": "binary", "max_feature_idx": 1,
            "feature_names": ["f0", "f1"],
            "tree_info": [
                {"tree_structure": split(0, 1.0, {"leaf_value": -0.5}, {"leaf_value": 0.5})},
                {"tree_structure": split(1, 0.0, {"leaf_value": -0.25}, {"leaf_value": 0.25})},
            ]}


TARGET = Target.probability(range=(0.0, 1.0))


class TestFactualViolationWarning:
    def test_linear_violation_warns_once(self) -> None:
        exp = Explainer(
            _stump_dump(), constraints=[constraint("f0 <= f1")], normalizers=np.ones(2)
        )
        with pytest.warns(TreecfWarning, match=r"factual violates 1 constraint\(s\)"):
            exp.explain(np.array([3.0, 0.25]), TARGET, seed=0)

    def test_range_violation_message(self) -> None:
        exp = Explainer(
            _stump_dump(), constraints=[Range("f0", 100.0, 1e9)], normalizers=np.ones(2)
        )
        with pytest.warns(TreecfWarning, match="changes made solely to satisfy them"):
            exp.explain(np.array([2.0, 0.0]), TARGET, seed=0)

    def test_feasible_factual_does_not_warn(self) -> None:
        exp = Explainer(
            _stump_dump(), constraints=[constraint("f0 <= f1")], normalizers=np.ones(2)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            exp.explain(np.array([0.0, 1.0]), TARGET, seed=0)

    def test_batch_aggregate_counts(self) -> None:
        exp = Explainer(
            _stump_dump(), constraints=[constraint("f0 <= f1")], normalizers=np.ones(2)
        )
        X = np.array([[3.0, 0.0], [0.0, 1.0], [5.0, 1.0]])
        with pytest.warns(TreecfWarning) as caught:
            exp.explain_batch(X, TARGET, seed=0)
        messages = [str(w.message) for w in caught if isinstance(w.message, TreecfWarning)]
        assert len(messages) == 1  # one aggregate warning, no per-row repeats
        assert "factual constraint violations in 2/3 rows" in messages[0]
        assert "1*f0 - 1*f1 <= 0: 2 rows" in messages[0]

    def test_batch_python_backend_warns_only_aggregate(self) -> None:
        exp = Explainer(
            _stump_dump(), constraints=[constraint("f0 <= f1")], normalizers=np.ones(2)
        )
        X = np.array([[3.0, 0.0], [0.0, 1.0]])
        with pytest.warns(TreecfWarning) as caught:
            exp.explain_batch(X, TARGET, backend="python", seed=0)
        messages = [str(w.message) for w in caught if isinstance(w.message, TreecfWarning)]
        assert len(messages) == 1
        assert messages[0].startswith("factual constraint violations in 1/2 rows")
