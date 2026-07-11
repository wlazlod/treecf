"""LightGBM parser (spec §3.3): LE convention, double-precision thresholds.

Missing-value routing depends on each node's ``missing_type``:
- "NaN": NaN follows ``default_left``.
- "None": LightGBM substitutes 0.0 for NaN, so ``missing_left`` is resolved to
  the side that 0.0 takes (``0.0 <= threshold``).
- "Zero" (``zero_as_missing``): unsupported in v0.1 — zeros and NaN collapse
  into one state that the IR cannot represent per §3.2.

``boost_from_average`` folds the intercept into leaf values, so base_score = 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from treecf._errors import UnsupportedModelError
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree

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


def parse_lightgbm(model: object) -> EnsembleIR:
    """Parse a ``lgb.Booster`` or sklearn-API wrapper via ``dump_model()``."""
    booster: Any = model.booster_ if hasattr(model, "booster_") else model
    dump: dict[str, Any] = booster.dump_model()
    return parse_lightgbm_dump(dump)


def parse_lightgbm_dump(dump: dict[str, Any]) -> EnsembleIR:
    """Parse the dict produced by ``Booster.dump_model()`` (or its JSON serialization)."""
    if int(dump.get("num_tree_per_iteration", 1)) > 1:
        raise UnsupportedModelError("multiclass LightGBM models are not supported in v0.1")

    objective = str(dump.get("objective", "")).split(" ")[0]
    if objective not in _OBJECTIVE_LINKS:
        raise UnsupportedModelError(f"objective {objective!r} not supported in v0.1")
    link = _OBJECTIVE_LINKS[objective]

    n_features = int(dump["max_feature_idx"]) + 1
    names = tuple(dump.get("feature_names") or (f"f{i}" for i in range(n_features)))

    trees = []
    for tree_info in dump["tree_info"]:
        nodes: list[Node] = []
        _walk(tree_info["tree_structure"], nodes)
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
    )


def _walk(node: dict[str, Any], nodes: list[Node]) -> int:
    """Preorder walk assigning node ids; returns this node's id."""
    node_id = len(nodes)
    if "leaf_value" in node and "split_feature" not in node:
        nodes.append(
            Node(node_id, None, None, None, None, None, None, float(node["leaf_value"]))
        )
        return node_id

    decision = node["decision_type"]
    if decision != "<=":
        raise UnsupportedModelError(
            f"categorical split (decision_type {decision!r}) not supported in v0.1 (spec §1.2)"
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
            f"missing_type {missing_type!r} (zero_as_missing) not supported in v0.1"
        )

    nodes.append(None)  # type: ignore[arg-type]  # placeholder until children are walked
    left_id = _walk(node["left_child"], nodes)
    right_id = _walk(node["right_child"], nodes)
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
