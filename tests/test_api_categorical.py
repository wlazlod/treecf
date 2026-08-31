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


class TestCategoriesArgumentAndValidation:
    def test_categories_installs_names_on_a_parsed_ir(self) -> None:
        names = ("clerk", "manager", "nurse", "smith", "guard")
        exp = Explainer(
            _categorical_ir(frozenset({2, 3}), 5),
            normalizers=np.array([1.0]),
            categories={"occupation": names},
        )
        assert exp.ir.categorical[0].categories == names

    def test_categories_can_extend_the_cardinality(self) -> None:
        exp = Explainer(
            _categorical_ir(frozenset({2, 3}), 5),
            normalizers=np.array([1.0]),
            categories={"occupation": [f"c{i}" for i in range(8)]},
        )
        assert exp.ir.categorical[0].cardinality == 8

    def test_too_short_name_list_raises(self) -> None:
        import pytest

        from treecf import TreecfError

        with pytest.raises(TreecfError, match="5 codes"):
            Explainer(
                _categorical_ir(frozenset({2}), 5),
                normalizers=np.array([1.0]),
                categories={"occupation": ["a", "b"]},
            )

    def test_categories_on_a_numeric_feature_raises(self) -> None:
        import pytest

        from treecf import TreecfError
        from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree

        numeric = EnsembleIR(
            trees=(
                Tree(
                    nodes=(
                        Node(0, 0, 1.0, SplitOp.LT, True, 1, 2, None),
                        Node(1, None, None, None, None, None, None, -1.0),
                        Node(2, None, None, None, None, None, None, 1.0),
                    )
                ),
            ),
            base_score=0.0,
            link=Link.IDENTITY,
            n_features=1,
            feature_names=("amount",),
            meta={},
        )
        with pytest.raises(TreecfError, match=r"not a\s.*categorical feature"):
            Explainer(numeric, normalizers=np.array([1.0]), categories={"amount": ["a"]})

    def test_explain_rejects_invalid_codes(self) -> None:
        import pytest

        from treecf import TreecfError

        exp = Explainer(_categorical_ir(frozenset({2}), 5), normalizers=np.array([1.0]))
        with pytest.raises(TreecfError, match=r"occupation.*integral code"):
            exp.explain(np.array([2.5]), Target.raw(op=">=", value=0.5), seed=0)
        with pytest.raises(TreecfError, match="occupation"):
            exp.explain(np.array([7.0]), Target.raw(op=">=", value=0.5), seed=0)

    def test_background_and_batch_are_validated(self) -> None:
        import pytest

        from treecf import TreecfError

        with pytest.raises(TreecfError, match="background"):
            Explainer(_categorical_ir(frozenset({2}), 5), background=np.array([[9.0]]))
        exp = Explainer(_categorical_ir(frozenset({2}), 5), normalizers=np.array([1.0]))
        with pytest.raises(TreecfError, match="factual"):
            exp.explain_batch(
                np.array([[0.0], [1.5]]), Target.raw(op=">=", value=0.5)
            )


class TestExactBackendCategorical:
    def test_exact_proves_optimality_over_blocks(self) -> None:
        exp = Explainer(_categorical_ir(frozenset({2, 3}), 5), normalizers=np.array([1.0]))
        result = exp.explain(
            np.array([0.0]), Target.raw(op=">=", value=0.5), backend="exact", seed=0
        )
        assert isinstance(result, Counterfactual)
        assert result.proof == "optimal"
        assert result.x_cf[0] == 2.0  # the block's smallest member
        assert result.distance == 1.0

    def test_off_target_allowed_set_certifies_infeasibility(self) -> None:
        from treecf import AllowedCategories, Infeasible

        exp = Explainer(
            _categorical_ir(frozenset({2, 3}), 5),
            normalizers=np.array([1.0]),
            constraints=[AllowedCategories("occupation", (0, 1))],
        )
        result = exp.explain(
            np.array([0.0]), Target.raw(op=">=", value=0.5), backend="exact", seed=0
        )
        assert isinstance(result, Infeasible)
        assert result.proof == "certified"

    def test_empty_allowed_set_certifies_before_any_expansion(self) -> None:
        import pytest

        from treecf import AllowedCategories, Infeasible, TreecfWarning

        exp = Explainer(
            _categorical_ir(frozenset({2, 3}), 5),
            normalizers=np.array([1.0]),
            constraints=[AllowedCategories("occupation", ())],
        )
        with pytest.warns(TreecfWarning, match="factual's category not in allowed set"):
            result = exp.explain(
                np.array([0.0]), Target.raw(op=">=", value=0.5), backend="exact", seed=0
            )
        assert isinstance(result, Infeasible)
        assert result.proof == "certified"
        assert result.solver_stats["nodes_expanded"] == 0
