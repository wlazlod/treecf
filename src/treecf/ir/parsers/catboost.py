"""CatBoost parser (spec §3.3): oblivious trees expanded to plain binary IR trees.

A depth-d oblivious tree stores d shared splits and 2^d leaf values; the leaf
index is the bit pattern of "x > border" decisions with splits[i] as bit i.
The expansion puts splits[d-1] at the root so leaf ranges stay contiguous, and
rewrites "x > border -> bit 1" as op LE (x <= border -> left/bit 0), §3.2.

Borders are float32-quantized (cast back through float32, as with XGBoost).
NaN routing: nan_value_treatment "AsFalse"/"AsIs" -> bit 0 (missing_left=True),
"AsTrue" -> bit 1. ``scale_and_bias`` folds scale into leaf values; bias is the
raw-space intercept.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from treecf._errors import UnsupportedModelError
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree

_LOSS_LINKS = {
    "Logloss": Link.SIGMOID,
    "CrossEntropy": Link.SIGMOID,
    "RMSE": Link.IDENTITY,
    "MAE": Link.IDENTITY,
    "Quantile": Link.IDENTITY,
}


def parse_catboost(model: object) -> EnsembleIR:
    """Parse a live CatBoost model via its JSON serialization."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.json"
        model.save_model(str(path), format="json")  # type: ignore[attr-defined]
        with open(path, encoding="utf-8") as fh:
            dump: dict[str, Any] = json.load(fh)
    return parse_catboost_dump(dump)


def parse_catboost_dump(dump: dict[str, Any]) -> EnsembleIR:
    scale, bias = dump["scale_and_bias"]
    if len(bias) != 1:
        raise UnsupportedModelError("multiclass CatBoost models are not supported in v0.1")

    info = dump.get("model_info", {})
    loss = (
        info.get("params", {}).get("loss_function", {}).get("type")
        or info.get("loss_function", {}).get("type")
        or ""
    )
    if loss not in _LOSS_LINKS:
        raise UnsupportedModelError(f"loss function {loss!r} not supported in v0.1")

    float_features = dump["features_info"]["float_features"]
    if "cat_features" in dump["features_info"] and dump["features_info"]["cat_features"]:
        raise UnsupportedModelError("categorical features are not supported in v0.1 (§1.2)")
    flat_of = {f["feature_index"]: f["flat_feature_index"] for f in float_features}
    missing_left_of = {
        f["feature_index"]: f.get("nan_value_treatment", "AsIs") != "AsTrue"
        for f in float_features
    }
    n_features = 1 + max((f["flat_feature_index"] for f in float_features), default=-1)

    trees = tuple(
        _expand_oblivious(tree, float(scale), flat_of, missing_left_of)
        for tree in dump["oblivious_trees"]
    )
    return EnsembleIR(
        trees=trees,
        base_score=float(bias[0]),
        link=_LOSS_LINKS[loss],
        n_features=n_features,
        feature_names=tuple(
            f["feature_id"] or f"f{f['flat_feature_index']}" for f in float_features
        ),
        meta={"source": "catboost", "loss_function": loss},
    )


def _expand_oblivious(
    tree: dict[str, Any],
    scale: float,
    flat_of: dict[int, int],
    missing_left_of: dict[int, bool],
) -> Tree:
    splits = tree["splits"]
    leaf_values = tree["leaf_values"]
    depth = len(splits)
    if len(leaf_values) != 2**depth:
        raise UnsupportedModelError("oblivious tree leaf count does not match its depth")

    nodes: list[Node] = []

    def build(bit: int, prefix: int) -> int:
        node_id = len(nodes)
        if bit < 0:
            value = scale * float(leaf_values[prefix])
            nodes.append(Node(node_id, None, None, None, None, None, None, value))
            return node_id
        split = splits[bit]
        if split.get("split_type") != "FloatFeature":
            raise UnsupportedModelError(
                f"split_type {split.get('split_type')!r} not supported in v0.1"
            )
        feature_index = int(split["float_feature_index"])
        nodes.append(None)  # type: ignore[arg-type]  # placeholder
        left = build(bit - 1, prefix)  # bit 0: x <= border
        right = build(bit - 1, prefix | (1 << bit))  # bit 1: x > border
        nodes[node_id] = Node(
            node_id=node_id,
            feature=int(flat_of[feature_index]),
            threshold=float(np.float32(split["border"])),
            op=SplitOp.LE,
            missing_left=missing_left_of[feature_index],
            left=left,
            right=right,
            value=None,
        )
        return node_id

    build(depth - 1, 0)
    return Tree(nodes=tuple(nodes))
