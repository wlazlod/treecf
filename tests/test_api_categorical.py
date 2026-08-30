"""API-level categorical behavior: flat change cost and unit normalizers."""

from __future__ import annotations

import numpy as np

from treecf import Counterfactual, Explainer, Target
from treecf.ir.model import CategoricalFeature, EnsembleIR, Link, Node, Tree


def _categorical_ir(categories: frozenset[int], cardinality: int) -> EnsembleIR:
    tree = Tree(
        nodes=(
            Node(0, 0, None, None, True, 1, 2, None, categories=categories),
            Node(1, None, None, None, None, None, None, 1.0),
            Node(2, None, None, None, None, None, None, 0.0),
        )
    )
    return EnsembleIR(
        trees=(tree,),
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=1,
        feature_names=("occupation",),
        meta={},
        categorical={0: CategoricalFeature(cardinality=cardinality)},
    )


class TestCategoricalDistance:
    """A category change contributes weights[j] * 1.0 / sigma[j] to distance."""

    def test_distance_is_flat_regardless_of_code_gap(self) -> None:
        exp = Explainer(_categorical_ir(frozenset({2, 3}), 5), normalizers=np.array([1.0]))
        result = exp.explain(np.array([0.0]), Target.raw(op=">=", value=0.5), seed=0)
        assert isinstance(result, Counterfactual)
        assert result.x_cf[0] in (2.0, 3.0)
        assert result.distance == 1.0
        assert result.n_changed == 1

    def test_background_fit_gives_unit_sigma_for_codes(self) -> None:
        background = np.array([[0.0], [0.0], [0.0], [1.0]])  # near-degenerate codes
        exp = Explainer(_categorical_ir(frozenset({1}), 3), background=background)
        assert exp.sigma[0] == 1.0


class TestAllowedCategoriesEndToEnd:
    def test_genetic_respects_the_allowed_set(self) -> None:
        from treecf import AllowedCategories

        exp = Explainer(
            _categorical_ir(frozenset({2, 3}), 5),
            normalizers=np.array([1.0]),
            constraints=[AllowedCategories("occupation", (0, 3))],  # 2 is banned
        )
        result = exp.explain(np.array([0.0]), Target.raw(op=">=", value=0.5), seed=0)
        assert isinstance(result, Counterfactual)
        assert result.x_cf[0] == 3.0

    def test_fingerprint_pins_the_allowed_set(self) -> None:
        from treecf import AllowedCategories, constraints_fingerprint

        base = Explainer(_categorical_ir(frozenset({2}), 5), normalizers=np.array([1.0]))
        restricted = Explainer(
            _categorical_ir(frozenset({2}), 5),
            normalizers=np.array([1.0]),
            constraints=[AllowedCategories("occupation", (0, 2))],
        )
        assert constraints_fingerprint(base) != constraints_fingerprint(restricted)

    def test_value_policy_on_categorical_raises(self) -> None:
        import pytest

        from treecf import ConstraintValidationError

        with pytest.raises(ConstraintValidationError, match="categorical feature"):
            Explainer(
                _categorical_ir(frozenset({2}), 5),
                normalizers=np.array([1.0]),
                value_policy={"occupation": "integer"},
            )
