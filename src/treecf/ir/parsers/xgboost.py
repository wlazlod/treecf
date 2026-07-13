"""XGBoost parser: LT convention, explicit missing branch via default_left.

base_score semantics: the JSON model stores ``learner_model_param.base_score``
in *output* space — probability for ``binary:logistic``, target units for
``reg:squarederror``. The raw-space intercept is therefore ``logit(base_score)``
for the sigmoid link and ``base_score`` unchanged for the identity link.
Validated empirically by the conformance suite on every CI-pinned version.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from treecf._errors import UnsupportedModelError
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree

_OBJECTIVE_LINKS = {
    "binary:logistic": Link.SIGMOID,
    "reg:squarederror": Link.IDENTITY,
    "reg:linear": Link.IDENTITY,  # legacy alias of reg:squarederror
}


def parse_xgboost(model: object) -> EnsembleIR:
    """Parse a live ``Booster`` or sklearn-API wrapper via its JSON serialization."""
    booster: Any = model.get_booster() if hasattr(model, "get_booster") else model
    raw = bytes(booster.save_raw(raw_format="json"))
    dump: dict[str, Any] = json.loads(raw)
    return parse_xgboost_dump(dump)


def parse_xgboost_dump(dump: dict[str, Any]) -> EnsembleIR:
    """Parse the XGBoost JSON model format (``Booster.save_model('*.json')``)."""
    learner = dump["learner"]

    booster_kind = learner["gradient_booster"]["name"]
    if booster_kind != "gbtree":
        raise UnsupportedModelError(
            f"gradient booster {booster_kind!r} not supported (gbtree only)"
        )

    objective = learner["objective"]["name"]
    if objective not in _OBJECTIVE_LINKS:
        raise UnsupportedModelError(f"objective {objective!r} not supported in v0.1")
    link = _OBJECTIVE_LINKS[objective]

    model_param = learner["learner_model_param"]
    if int(model_param.get("num_class", "0")) > 1:
        raise UnsupportedModelError("multiclass models are not supported in v0.1")

    n_features = int(model_param["num_feature"])
    base_output = _parse_base_score(model_param["base_score"])
    if link is Link.SIGMOID:
        base_score = math.log(base_output / (1.0 - base_output))
    else:
        base_score = base_output

    names = learner.get("feature_names") or [f"f{i}" for i in range(n_features)]
    trees = tuple(_parse_tree(t) for t in learner["gradient_booster"]["model"]["trees"])

    return EnsembleIR(
        trees=trees,
        base_score=base_score,
        link=link,
        n_features=n_features,
        feature_names=tuple(names),
        meta={"source": "xgboost", "objective": objective, "version": dump.get("version")},
    )


def _parse_base_score(raw: str) -> float:
    """XGBoost <3 stores a scalar string ('5E-1'); 3.x a vector string ('[5.23E-1]')."""
    text = raw.strip()
    if text.startswith("["):
        values = [v for v in text.strip("[]").split(",") if v.strip()]
        if len(values) != 1:
            raise UnsupportedModelError(f"multi-output base_score {raw!r} not supported in v0.1")
        return float(values[0])
    return float(text)


def _parse_tree(tree: dict[str, Any]) -> Tree:
    left = tree["left_children"]
    right = tree["right_children"]
    split_indices = tree["split_indices"]
    split_conditions = tree["split_conditions"]
    default_left = tree["default_left"]
    split_type = tree.get("split_type")

    nodes: list[Node] = []
    for i in range(len(left)):
        # JSON stores float32 values as shortest decimals; parsing them as float64
        # yields numbers off the float32 grid, which flips routing for inputs equal
        # to a threshold. Cast through float32 to recover XGBoost's exact values.
        condition = float(np.float32(split_conditions[i]))
        if left[i] == -1:
            # Leaf: the JSON schema stores the leaf value in split_conditions.
            nodes.append(
                Node(
                    node_id=i,
                    feature=None,
                    threshold=None,
                    op=None,
                    missing_left=None,
                    left=None,
                    right=None,
                    value=condition,
                )
            )
            continue
        if split_type is not None and split_type[i] != 0:
            raise UnsupportedModelError(
                f"categorical split at node {i} not supported in v0.1"
            )
        nodes.append(
            Node(
                node_id=i,
                feature=int(split_indices[i]),
                threshold=condition,
                op=SplitOp.LT,
                missing_left=bool(default_left[i]),
                left=int(left[i]),
                right=int(right[i]),
                value=None,
            )
        )
    return Tree(nodes=tuple(nodes))
