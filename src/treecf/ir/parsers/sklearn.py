"""sklearn parsers (spec §3.3): RandomForest*, GradientBoosting*, HistGradientBoosting*.

Raw-score semantics per family (documented because they differ):
- RandomForestClassifier: raw score = averaged class-1 probability, link IDENTITY.
  Probability targets require SIGMOID, so use ``Target.raw`` for forests.
- GradientBoosting*: raw score = init prediction + lr * sum of tree outputs;
  SIGMOID link for the binary classifier.
- HistGradientBoosting*: baseline_prediction + sum of predictor outputs; NaN
  routing via ``missing_go_to_left``. Reads the private ``_predictors`` arrays —
  covered by the conformance matrix, raises on shape changes rather than guessing.

IsolationForest is parsed separately for plausibility (§9) with depth-based
leaf values; see ``parse_isolation_forest``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from treecf._errors import UnsupportedModelError
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree


def parse_sklearn(model: object) -> EnsembleIR:
    kind = type(model).__name__
    if kind in ("RandomForestClassifier", "RandomForestRegressor"):
        return _parse_random_forest(model)
    if kind in ("GradientBoostingClassifier", "GradientBoostingRegressor"):
        return _parse_gradient_boosting(model)
    if kind in ("HistGradientBoostingClassifier", "HistGradientBoostingRegressor"):
        return _parse_hist_gradient_boosting(model)
    raise UnsupportedModelError(f"sklearn model {kind} not supported in v0.1")


def _parse_random_forest(model: Any) -> EnsembleIR:
    classifier = type(model).__name__.endswith("Classifier")
    if classifier and model.n_classes_ > 2:
        raise UnsupportedModelError("multiclass forests are not supported in v0.1")
    n = len(model.estimators_)
    trees = tuple(
        _tree_from_arrays(est.tree_, scale=1.0 / n, classifier=classifier)
        for est in model.estimators_
    )
    return EnsembleIR(
        trees=trees,
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=int(model.n_features_in_),
        feature_names=_names(model),
        meta={"source": "sklearn", "estimator": type(model).__name__},
    )


def _parse_gradient_boosting(model: Any) -> EnsembleIR:
    classifier = type(model).__name__.endswith("Classifier")
    if classifier and model.n_classes_ > 2:
        raise UnsupportedModelError("multiclass gradient boosting is not supported in v0.1")
    base_score = float(
        model._raw_predict_init(np.zeros((1, model.n_features_in_), dtype=np.float64))[0, 0]
    )
    lr = float(model.learning_rate)
    trees = tuple(
        _tree_from_arrays(est.tree_, scale=lr, classifier=False)
        for est in model.estimators_[:, 0]
    )
    return EnsembleIR(
        trees=trees,
        base_score=base_score,
        link=Link.SIGMOID if classifier else Link.IDENTITY,
        n_features=int(model.n_features_in_),
        feature_names=_names(model),
        meta={"source": "sklearn", "estimator": type(model).__name__},
    )


def _parse_hist_gradient_boosting(model: Any) -> EnsembleIR:
    classifier = type(model).__name__.endswith("Classifier")
    if classifier and len(model.classes_) > 2:
        raise UnsupportedModelError("multiclass HistGradientBoosting is not supported in v0.1")
    baseline = float(np.ravel(model._baseline_prediction)[0])
    trees = []
    for predictors in model._predictors:
        if len(predictors) != 1:
            raise UnsupportedModelError("multi-output HistGradientBoosting is not supported")
        trees.append(_tree_from_hist_nodes(predictors[0].nodes))
    return EnsembleIR(
        trees=tuple(trees),
        base_score=baseline,
        link=Link.SIGMOID if classifier else Link.IDENTITY,
        n_features=int(model.n_features_in_),
        feature_names=_names(model),
        meta={"source": "sklearn", "estimator": type(model).__name__},
    )


def _tree_from_arrays(tree: Any, scale: float, classifier: bool) -> Tree:
    """Convert a fitted ``sklearn.tree._tree.Tree`` to IR nodes (LE convention)."""
    left = tree.children_left
    right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    value = tree.value
    missing = getattr(tree, "missing_go_to_left", None)

    nodes = []
    for i in range(tree.node_count):
        if left[i] == -1:
            if classifier:
                row = value[i, 0]
                leaf = float(row[1] / row.sum()) * scale  # class-1 fraction
            else:
                leaf = float(value[i, 0, 0]) * scale
            nodes.append(Node(i, None, None, None, None, None, None, leaf))
        else:
            missing_left = bool(missing[i]) if missing is not None else None
            nodes.append(
                Node(
                    node_id=i,
                    feature=int(feature[i]),
                    threshold=float(threshold[i]),
                    op=SplitOp.LE,
                    missing_left=missing_left,
                    left=int(left[i]),
                    right=int(right[i]),
                    value=None,
                )
            )
    return Tree(nodes=tuple(nodes))


def _tree_from_hist_nodes(nodes_array: Any) -> Tree:
    nodes = []
    for i, row in enumerate(nodes_array):
        if row["is_leaf"]:
            nodes.append(Node(i, None, None, None, None, None, None, float(row["value"])))
        else:
            nodes.append(
                Node(
                    node_id=i,
                    feature=int(row["feature_idx"]),
                    threshold=float(row["num_threshold"]),
                    op=SplitOp.LE,
                    missing_left=bool(row["missing_go_to_left"]),
                    left=int(row["left"]),
                    right=int(row["right"]),
                    value=None,
                )
            )
    return Tree(nodes=tuple(nodes))


def parse_isolation_forest(model: Any) -> EnsembleIR:
    """IsolationForest -> IR with depth-adjusted path lengths as leaf values (§9).

    Leaf value := depth(leaf) + c(n_samples(leaf)), so the ensemble raw score is
    ``sum_t h_t(x)`` and the anomaly score is ``2 ** (-mean_h / c(n))``.
    """
    if type(model).__name__ != "IsolationForest":
        raise UnsupportedModelError("expected an IsolationForest")
    trees = []
    for est in model.estimators_:
        tree = est.tree_
        depths = np.zeros(tree.node_count)
        stack = [(0, 0)]
        while stack:
            node, depth = stack.pop()
            depths[node] = depth
            if tree.children_left[node] != -1:
                stack.append((int(tree.children_left[node]), depth + 1))
                stack.append((int(tree.children_right[node]), depth + 1))
        nodes = []
        for i in range(tree.node_count):
            if tree.children_left[i] == -1:
                n_samples = float(tree.n_node_samples[i])
                nodes.append(
                    Node(i, None, None, None, None, None, None, depths[i] + _avg_path(n_samples))
                )
            else:
                nodes.append(
                    Node(
                        node_id=i,
                        feature=int(tree.feature[i]),
                        threshold=float(tree.threshold[i]),
                        op=SplitOp.LE,
                        missing_left=None,
                        left=int(tree.children_left[i]),
                        right=int(tree.children_right[i]),
                        value=None,
                    )
                )
        trees.append(Tree(nodes=tuple(nodes)))
    return EnsembleIR(
        trees=tuple(trees),
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=int(model.n_features_in_),
        feature_names=_names(model),
        meta={
            "source": "sklearn",
            "estimator": "IsolationForest",
            "max_samples": float(model.max_samples_),
        },
    )


def _avg_path(n: float) -> float:
    """Average path length c(n) of an unsuccessful BST search (Liu et al. 2008)."""
    if n <= 1.0:
        return 0.0
    if n == 2.0:
        return 1.0
    euler_gamma = 0.5772156649015329
    return 2.0 * (math.log(n - 1.0) + euler_gamma) - 2.0 * (n - 1.0) / n


def _names(model: Any) -> tuple[str, ...]:
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return tuple(str(n) for n in names)
    return tuple(f"f{i}" for i in range(int(model.n_features_in_)))
