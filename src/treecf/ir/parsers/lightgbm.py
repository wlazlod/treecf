"""LightGBM parser: LE convention, double-precision thresholds, native categoricals.

Missing-value routing depends on each node's ``missing_type``:
- "NaN": NaN follows ``default_left``.
- "None": LightGBM substitutes 0.0 for NaN, so ``missing_left`` is resolved to
  the side that 0.0 takes (``0.0 <= threshold``; for a categorical split, the
  side category code 0 takes).
- "Zero" (``zero_as_missing``): unsupported — zeros and NaN collapse
  into one state that the IR cannot represent

Categorical splits (``decision_type "=="``) carry their member codes as a
``"||"``-joined list in ``threshold``; a member code routes left. A feature is
categorical iff its ``feature_infos`` entry lists bin ``values`` (this also
covers categorical features no tree split on), and display names come from the
model's stored ``pandas_categorical`` lists or the caller's ``categories``.

``boost_from_average`` folds the intercept into leaf values, so base_score = 0.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from treecf._errors import UnsupportedModelError
from treecf.ir.model import CategoricalFeature, EnsembleIR, Link, Node, SplitOp, Tree

# LightGBM zeroes inputs with |v| <= kZeroThreshold (1e-35f) before comparing, so its
# synthetic near-zero split thresholds need op/threshold rewrites to stay faithful.
_K_ZERO = float(np.float32(1e-35))

_OBJECTIVE_LINKS = {
    "binary": Link.SIGMOID,
    "regression": Link.IDENTITY,
    "regression_l1": Link.IDENTITY,
    "l2": Link.IDENTITY,
    "huber": Link.IDENTITY,
}


def parse_lightgbm(
    model: object, categories: Mapping[str, Sequence[str]] | None = None
) -> EnsembleIR:
    """Parse a ``lgb.Booster`` or sklearn-API wrapper via ``dump_model()``."""
    booster: Any = model.booster_ if hasattr(model, "booster_") else model
    dump: dict[str, Any] = booster.dump_model()
    return parse_lightgbm_dump(dump, categories)


def parse_lightgbm_dump(
    dump: dict[str, Any], categories: Mapping[str, Sequence[str]] | None = None
) -> EnsembleIR:
    """Parse the dict produced by ``Booster.dump_model()`` (or its JSON serialization)."""
    if int(dump.get("num_tree_per_iteration", 1)) > 1:
        raise UnsupportedModelError("multiclass LightGBM models are not supported")

    objective = str(dump.get("objective", "")).split(" ")[0]
    if objective not in _OBJECTIVE_LINKS:
        raise UnsupportedModelError(f"objective {objective!r} not supported")
    link = _OBJECTIVE_LINKS[objective]

    n_features = int(dump["max_feature_idx"]) + 1
    names = tuple(dump.get("feature_names") or (f"f{i}" for i in range(n_features)))

    trees = []
    max_code: dict[int, int] = {}  # per categorical feature, the largest member code seen
    for tree_info in dump["tree_info"]:
        nodes: list[Node] = []
        _walk(tree_info["tree_structure"], nodes, max_code)
        trees.append(Tree(nodes=tuple(nodes)))

    return EnsembleIR(
        trees=tuple(trees),
        base_score=0.0,  # boost_from_average folds the intercept into the first tree
        link=link,
        n_features=n_features,
        feature_names=names,
        meta={
            "source": "lightgbm",
            "objective": dump.get("objective"),
            "version": dump.get("version"),
        },
        categorical=_categorical_metadata(dump, names, max_code, categories),
    )


def _categorical_metadata(
    dump: dict[str, Any],
    names: tuple[str, ...],
    max_code: dict[int, int],
    categories: Mapping[str, Sequence[str]] | None,
) -> dict[int, CategoricalFeature]:
    """Kind, cardinality, and display names per categorical feature.

    A feature is categorical iff its ``feature_infos`` entry lists bin
    ``values`` (present even when no tree split on it); features only seen in
    ``"=="`` splits (a dump without ``feature_infos``) are covered by
    ``max_code``. Cardinality is the largest of: ``feature_infos.max_value+1``,
    the largest split member code + 1, and the length of the name list.
    """
    feature_infos = dump.get("feature_infos", {})
    categorical_indices: set[int] = set(max_code)
    cardinality: dict[int, int] = {j: c + 1 for j, c in max_code.items()}
    for j, name in enumerate(names):
        info = feature_infos.get(name, {})
        if info.get("values"):
            categorical_indices.add(j)
            top = max(int(v) for v in info["values"])
            cardinality[j] = max(cardinality.get(j, 0), top + 1)

    name_lists: dict[int, tuple[str, ...]] = {}
    pandas_categorical = dump.get("pandas_categorical")
    if pandas_categorical:
        for j, name_list in zip(sorted(categorical_indices), pandas_categorical, strict=False):
            name_lists[j] = tuple(str(n) for n in name_list)
    if categories:
        for feature_name, name_list in categories.items():
            if feature_name in names:
                name_lists[names.index(feature_name)] = tuple(name_list)

    result: dict[int, CategoricalFeature] = {}
    for j in sorted(categorical_indices):
        display = name_lists.get(j)
        k = max(cardinality.get(j, 1), len(display) if display else 0)
        if display is not None and len(display) < k:
            display = None  # names cover fewer codes than the model uses: drop them
        result[j] = CategoricalFeature(cardinality=k, categories=display)
    return result


def _walk(node: dict[str, Any], nodes: list[Node], max_code: dict[int, int]) -> int:
    """Preorder walk assigning node ids; returns this node's id."""
    node_id = len(nodes)
    if "leaf_value" in node and "split_feature" not in node:
        nodes.append(
            Node(node_id, None, None, None, None, None, None, float(node["leaf_value"]))
        )
        return node_id

    decision = node["decision_type"]
    if decision == "==":
        return _walk_categorical(node, nodes, max_code)
    if decision != "<=":
        raise UnsupportedModelError(
            f"decision_type {decision!r} not supported"
        )
    threshold = float(node["threshold"])
    op = SplitOp.LE
    if -1e-30 < threshold < 0.0:
        # boundary between negatives and the zero-collapse band: values equal to
        # -kZero are zeroed by LightGBM and go right -> strict comparison
        threshold, op = -_K_ZERO, SplitOp.LT
    elif 0.0 <= threshold < 1e-30:
        # zeros (and the whole collapse band) go left -> inclusive at +kZero
        threshold, op = _K_ZERO, SplitOp.LE
    missing_type = node.get("missing_type", "None")
    if missing_type == "NaN":
        missing_left = bool(node["default_left"])
    elif missing_type == "None":
        missing_left = threshold >= 0.0  # LightGBM substitutes 0.0 for NaN
    else:
        raise UnsupportedModelError(
            f"missing_type {missing_type!r} (zero_as_missing) not supported"
        )

    nodes.append(None)  # type: ignore[arg-type]  # placeholder until children are walked
    left_id = _walk(node["left_child"], nodes, max_code)
    right_id = _walk(node["right_child"], nodes, max_code)
    nodes[node_id] = Node(
        node_id=node_id,
        feature=int(node["split_feature"]),
        threshold=threshold,
        op=op,
        missing_left=missing_left,
        left=left_id,
        right=right_id,
        value=None,
    )
    return node_id


def _walk_categorical(node: dict[str, Any], nodes: list[Node], max_code: dict[int, int]) -> int:
    """A ``decision_type "=="`` node: member codes (``"||"``-joined) route left."""
    node_id = len(nodes)
    feature = int(node["split_feature"])
    members = frozenset(
        int(token) for token in str(node["threshold"]).split("||") if int(token) >= 0
    )
    if not members:
        raise UnsupportedModelError(f"categorical split at node {node_id} has no member codes")
    max_code[feature] = max(max_code.get(feature, 0), max(members))
    # LightGBM bins categorical NaN as the "other" category, which is never a
    # set member — NaN routes right regardless of missing_type/default_left
    missing_left = False

    nodes.append(None)  # type: ignore[arg-type]  # placeholder until children are walked
    left_id = _walk(node["left_child"], nodes, max_code)
    right_id = _walk(node["right_child"], nodes, max_code)
    nodes[node_id] = Node(
        node_id=node_id,
        feature=feature,
        threshold=None,
        op=None,
        missing_left=missing_left,
        left=left_id,
        right=right_id,
        value=None,
        categories=members,
    )
    return node_id
