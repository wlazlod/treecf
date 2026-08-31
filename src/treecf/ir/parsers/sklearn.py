"""sklearn parsers: RandomForest*, GradientBoosting*, HistGradientBoosting*.

Raw-score semantics per family (documented because they differ):
- RandomForestClassifier: raw score = averaged class-1 probability, link IDENTITY.
  Probability targets require SIGMOID, so use ``Target.raw`` for forests.
- GradientBoosting*: raw score = init prediction + lr * sum of tree outputs;
  SIGMOID link for the binary classifier.
- HistGradientBoosting*: baseline_prediction + sum of predictor outputs; NaN
  routing via ``missing_go_to_left``. Reads the private ``_predictors`` arrays —
  covered by the conformance matrix, raises on shape changes rather than guessing.

IsolationForest is parsed separately for plausibility with depth-based
leaf values; see ``parse_isolation_forest``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from treecf._errors import UnsupportedModelError
from treecf.ir.model import CategoricalFeature, EnsembleIR, Link, Node, SplitOp, Tree


def parse_sklearn(
    model: object, categories: Mapping[str, Sequence[str]] | None = None
) -> EnsembleIR:
    kind = type(model).__name__
    if kind in ("RandomForestClassifier", "RandomForestRegressor"):
        return _parse_random_forest(model)
    if kind in ("GradientBoostingClassifier", "GradientBoostingRegressor"):
        return _parse_gradient_boosting(model)
    if kind in ("HistGradientBoostingClassifier", "HistGradientBoostingRegressor"):
        return _parse_hist_gradient_boosting(model, categories)
    raise UnsupportedModelError(f"sklearn model {kind} not supported")


def _parse_random_forest(model: Any) -> EnsembleIR:
    classifier = type(model).__name__.endswith("Classifier")
    if classifier and model.n_classes_ > 2:
        raise UnsupportedModelError("multiclass forests are not supported")
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
        raise UnsupportedModelError("multiclass gradient boosting is not supported")
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


def _parse_hist_gradient_boosting(
    model: Any, categories: Mapping[str, Sequence[str]] | None = None
) -> EnsembleIR:
    classifier = type(model).__name__.endswith("Classifier")
    if classifier and len(model.classes_) > 2:
        raise UnsupportedModelError("multiclass HistGradientBoosting is not supported")
    baseline = float(np.ravel(model._baseline_prediction)[0])
    n_features = int(model.n_features_in_)
    names = _names(model)

    is_cat_attr = getattr(model, "is_categorical_", None)
    is_cat = (
        np.zeros(n_features, dtype=bool)
        if is_cat_attr is None
        else np.asarray(is_cat_attr, dtype=bool)
    )
    context = _hgb_categorical_context(model, is_cat, names, categories) if is_cat.any() else None

    trees = []
    for predictors in model._predictors:
        if len(predictors) != 1:
            raise UnsupportedModelError("multi-output HistGradientBoosting is not supported")
        trees.append(_tree_from_hist_nodes(predictors[0].nodes, predictors[0], context))
    return EnsembleIR(
        trees=tuple(trees),
        base_score=baseline,
        link=Link.SIGMOID if classifier else Link.IDENTITY,
        n_features=n_features,
        feature_names=names,
        meta={"source": "sklearn", "estimator": type(model).__name__},
        categorical={} if context is None else context.metadata,
    )


@dataclass(frozen=True)
class _HgbCategoricalContext:
    """Everything needed to lower HGB categorical nodes into user-code space.

    ``perm`` maps a predictor feature index to the original column (the fitted
    preprocessor reorders columns: encoded categoricals first, numericals
    after). ``position_codes[j]`` maps a bitset position to the user-facing
    code for original column ``j``; ``unknown_codes[j]`` are the codes inside
    ``[0, cardinality)`` the model never saw — its predictor routes them like
    missing values, so nodes with ``missing_go_to_left`` absorb them into
    their member set.
    """

    perm: tuple[int, ...]
    position_codes: dict[int, tuple[int, ...]]
    unknown_codes: dict[int, frozenset[int]]
    metadata: dict[int, CategoricalFeature]


def _hgb_categorical_context(
    model: Any,
    is_cat: npt.NDArray[np.bool_],
    names: tuple[str, ...],
    categories: Mapping[str, Sequence[str]] | None,
) -> _HgbCategoricalContext:
    cat_orig = [int(j) for j in np.flatnonzero(is_cat)]
    num_orig = [int(j) for j in np.flatnonzero(~is_cat)]
    preprocessor = getattr(model, "_preprocessor", None)
    if preprocessor is not None:
        # fitted preprocessing reorders columns to [categoricals..., numericals...]
        # and ordinal-encodes each categorical: bitset positions index the
        # encoder's sorted category values, and unknown values become NaN
        perm = tuple(cat_orig + num_orig)
        encoder = preprocessor.named_transformers_["encoder"]
        value_lists = [np.asarray(v) for v in encoder.categories_]
    else:
        # no preprocessor: bitset positions index the bin mapper's sorted
        # category values directly, in original column order
        perm = tuple(range(len(is_cat)))
        bin_thresholds = getattr(model, "_bin_mapper", None)
        if bin_thresholds is None:
            raise UnsupportedModelError(
                "HistGradientBoosting model exposes neither a fitted preprocessor "
                "nor a bin mapper; its categorical layout cannot be recovered"
            )
        value_lists = [np.asarray(bin_thresholds.bin_thresholds_[j]) for j in cat_orig]

    position_codes: dict[int, tuple[int, ...]] = {}
    unknown_codes: dict[int, frozenset[int]] = {}
    metadata: dict[int, CategoricalFeature] = {}
    for j, values in zip(cat_orig, value_lists, strict=True):
        # NaN appears as a trailing category when training data had missing
        # values; it encodes to NaN (the missing route), never to a position
        values = np.asarray(
            [v for v in values if not (isinstance(v, float) and math.isnan(v))]
        )
        display: tuple[str, ...] | None = None
        user_list = categories.get(names[j]) if categories else None
        if values.dtype.kind in "OUS":  # trained on string categories
            if user_list is None:
                raise UnsupportedModelError(
                    f"feature {names[j]!r} was trained on string categories; pass "
                    "categories= to the Explainer so codes can be assigned"
                )
            display = tuple(str(v) for v in user_list)
            lookup = {name: code for code, name in enumerate(display)}
            missing = [str(v) for v in values if str(v) not in lookup]
            if missing:
                raise UnsupportedModelError(
                    f"feature {names[j]!r}: categories= does not list trained "
                    f"category values {missing!r}"
                )
            codes = tuple(lookup[str(v)] for v in values)
        else:
            floats = values.astype(np.float64)
            if np.any(floats != np.floor(floats)) or np.any(floats < 0):
                raise UnsupportedModelError(
                    f"feature {names[j]!r}: categorical values must be "
                    "non-negative integer codes"
                )
            codes = tuple(int(v) for v in floats)
            if user_list is not None:
                display = tuple(str(v) for v in user_list)
        cardinality = max((max(codes) + 1) if codes else 1, len(display) if display else 0)
        if display is not None and len(display) < cardinality:
            display = None
        position_codes[j] = codes
        unknown_codes[j] = frozenset(range(cardinality)) - frozenset(codes)
        metadata[j] = CategoricalFeature(cardinality=cardinality, categories=display)
    return _HgbCategoricalContext(
        perm=perm,
        position_codes=position_codes,
        unknown_codes=unknown_codes,
        metadata=metadata,
    )


def _effective_le_threshold(t: float) -> float:
    """The float64 boundary of sklearn's float32 input cast, exactly.

    sklearn ``tree_``-based ensembles route ``float32(x) <= float64(t)`` —
    the input is cast to float32 (round-to-nearest-even) before the
    comparison. The IR evaluates in float64, so the stored threshold must be
    the largest float64 ``T`` with ``float32(T) <= t``; then ``x <= T``
    reproduces the native routing for *every* float64 ``x``, including points
    exactly on split boundaries — where a counterfactual search naturally
    lands. (Verified by a 138k-probe property sweep and the unquantized
    conformance tests.)

    Construction: let ``f`` be the largest float32 with ``f <= t`` and ``s``
    its float32 successor; every ``x`` below their float64 midpoint rounds to
    ``<= f``. The midpoint itself rounds half-to-even: it belongs to the left
    side exactly when it rounds back to ``f``.
    """
    f32 = np.float32(t)
    if float(f32) > t:
        f32 = np.nextafter(f32, np.float32(-np.inf))
    succ = np.nextafter(f32, np.float32(np.inf))
    mid = (float(f32) + float(succ)) / 2.0
    if float(np.float32(mid)) == float(f32):
        return mid
    return float(np.nextafter(mid, -np.inf))


def _tree_from_arrays(tree: Any, scale: float, classifier: bool) -> Tree:
    """Convert a fitted ``sklearn.tree._tree.Tree`` to IR nodes (LE convention).

    Thresholds are re-expressed on the float64 grid via
    :func:`_effective_le_threshold` so the IR's float64 routing matches
    sklearn's float32-cast routing bit-for-bit.
    """
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
                    threshold=_effective_le_threshold(float(threshold[i])),
                    op=SplitOp.LE,
                    missing_left=missing_left,
                    left=int(left[i]),
                    right=int(right[i]),
                    value=None,
                )
            )
    return Tree(nodes=tuple(nodes))


def _tree_from_hist_nodes(
    nodes_array: Any, predictor: Any, context: _HgbCategoricalContext | None
) -> Tree:
    field_names = nodes_array.dtype.names or ()
    has_categorical_fields = "is_categorical" in field_names
    nodes = []
    for i, row in enumerate(nodes_array):
        if row["is_leaf"]:
            nodes.append(Node(i, None, None, None, None, None, None, float(row["value"])))
            continue
        predictor_feature = int(row["feature_idx"])
        feature = context.perm[predictor_feature] if context is not None else predictor_feature
        missing_left = bool(row["missing_go_to_left"])
        if has_categorical_fields and bool(row["is_categorical"]):
            assert context is not None  # a categorical node implies categorical metadata
            words = predictor.raw_left_cat_bitsets[int(row["bitset_idx"])]
            positions = [
                w * 32 + b for w, word in enumerate(words) for b in range(32) if word >> b & 1
            ]
            codes = context.position_codes[feature]
            members = {codes[pos] for pos in positions if pos < len(codes)}
            if missing_left:
                # the predictor routes values it never saw like missing values;
                # missing goes left here, so the unseen codes join the set
                members |= context.unknown_codes[feature]
            nodes.append(
                Node(
                    node_id=i,
                    feature=feature,
                    threshold=None,
                    op=None,
                    missing_left=missing_left,
                    left=int(row["left"]),
                    right=int(row["right"]),
                    value=None,
                    categories=frozenset(members),
                )
            )
            continue
        nodes.append(
            Node(
                node_id=i,
                feature=feature,
                threshold=float(row["num_threshold"]),
                op=SplitOp.LE,
                missing_left=missing_left,
                left=int(row["left"]),
                right=int(row["right"]),
                value=None,
            )
        )
    return Tree(nodes=tuple(nodes))


def parse_isolation_forest(model: Any) -> EnsembleIR:
    """IsolationForest -> IR with depth-adjusted path lengths as leaf values.

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
                        threshold=_effective_le_threshold(float(tree.threshold[i])),
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
