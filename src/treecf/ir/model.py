"""IR data model.

Split operators are stored per node exactly as the source library defines them;
normalizing LT <-> LE via ``nextafter`` is forbidden.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

from treecf._errors import ParserError, TreecfError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


class SplitOp(Enum):
    """Comparison sending an instance to the left child."""

    LT = auto()  # x < threshold  -> left
    LE = auto()  # x <= threshold -> left


class Link(Enum):
    """Output link applied to the raw score."""

    IDENTITY = auto()
    SIGMOID = auto()


@dataclass(frozen=True)
class CategoricalFeature:
    """Per-feature categorical metadata: codes are ``0..cardinality-1``.

    ``categories`` optionally carries display names for the codes, in code
    order, with ``len(categories) == cardinality`` when present.
    """

    cardinality: int
    categories: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Node:
    """One tree node; ``feature is None`` marks a leaf.

    ``left``/``right`` are node ids, and parsers guarantee ``nodes[i].node_id == i``
    so children are addressed by index. A set-membership split carries
    ``categories`` (go left iff the integer code is in the set) with
    ``threshold``/``op`` both ``None``; a numeric split carries
    ``threshold``/``op`` with ``categories`` ``None``.
    """

    node_id: int
    feature: int | None
    threshold: float | None
    op: SplitOp | None
    missing_left: bool | None
    left: int | None
    right: int | None
    value: float | None
    categories: frozenset[int] | None = None


@dataclass(frozen=True)
class Tree:
    nodes: tuple[Node, ...]  # root = nodes[0]


@dataclass(frozen=True)
class EnsembleIR:
    """Raw score: ``S(x) = base_score + sum_t leaf_value_t(x)``; output = ``link(S(x))``."""

    trees: tuple[Tree, ...]
    base_score: float
    link: Link
    n_features: int
    feature_names: tuple[str, ...]
    meta: dict[str, object]
    categorical: dict[int, CategoricalFeature] = field(default_factory=dict)


def validate_ir(ir: EnsembleIR) -> None:
    """Check that an ensemble has the shape every consumer relies on.

    Feature names match the feature count; every tree is non-empty, its
    nodes carry their own index as ``node_id``, every child pointer stays
    inside the tree and the pointers form a tree (each node reached exactly
    once from the root); a split names a feature inside the model and is
    either numeric (a finite threshold with an operator) or set-membership
    (a non-empty code set within the feature's cardinality); a leaf carries a
    finite value; the base score is finite; categorical display names, when
    present, cover exactly the cardinality.

    Raises
    ------
    ParserError
        Naming the first violation found.
    """
    if ir.n_features < 0:
        raise ParserError(f"feature count {ir.n_features} is negative")
    if len(ir.feature_names) != ir.n_features:
        raise ParserError(
            f"{len(ir.feature_names)} feature names for {ir.n_features} features"
        )
    if not math.isfinite(ir.base_score):
        raise ParserError(f"base score {ir.base_score!r} is not finite")
    for j, info in ir.categorical.items():
        if not (0 <= j < ir.n_features):
            raise ParserError(f"categorical feature index {j} is outside the model")
        if info.cardinality < 1:
            raise ParserError(f"feature {j} has cardinality {info.cardinality}")
        if info.categories is not None and len(info.categories) != info.cardinality:
            raise ParserError(
                f"feature {j} names {len(info.categories)} categories for "
                f"cardinality {info.cardinality}"
            )
    for t, tree in enumerate(ir.trees):
        _validate_tree(ir, t, tree)


def _validate_tree(ir: EnsembleIR, t: int, tree: Tree) -> None:
    n = len(tree.nodes)
    if n == 0:
        raise ParserError(f"tree {t} has no nodes")
    for i, node in enumerate(tree.nodes):
        where = f"tree {t} node {i}"
        if node.node_id != i:
            raise ParserError(f"{where} carries node_id {node.node_id}")
        if node.feature is None:
            if node.value is None or not math.isfinite(node.value):
                raise ParserError(f"{where} is a leaf without a finite value")
            continue
        if not (0 <= node.feature < ir.n_features):
            raise ParserError(f"{where} splits on feature {node.feature}, outside the model")
        if node.left is None or node.right is None:
            raise ParserError(f"{where} is a split without two children")
        for child in (node.left, node.right):
            if not (0 <= child < n) or child == i:
                raise ParserError(f"{where} points at child {child}")
        if node.categories is not None:
            if node.threshold is not None or node.op is not None:
                raise ParserError(f"{where} mixes a threshold with a category set")
            if not node.categories:
                raise ParserError(f"{where} has an empty category set")
            info = ir.categorical.get(node.feature)
            if info is not None and any(
                not (0 <= code < info.cardinality) for code in node.categories
            ):
                raise ParserError(f"{where} names a category code outside the cardinality")
        else:
            if node.threshold is None or node.op is None:
                raise ParserError(f"{where} is a numeric split without a threshold")
            if not math.isfinite(node.threshold):
                raise ParserError(f"{where} has a non-finite threshold")
    # every node reached exactly once from the root: the pointers form a tree
    seen = [False] * n
    stack = [0]
    while stack:
        i = stack.pop()
        if seen[i]:
            raise ParserError(f"tree {t} reaches node {i} twice: not a tree")
        seen[i] = True
        node = tree.nodes[i]
        if node.feature is not None:
            assert node.left is not None and node.right is not None
            stack.append(node.right)
            stack.append(node.left)
    if not all(seen):
        unreached = seen.index(False)
        raise ParserError(f"tree {t} never reaches node {unreached}")


def apply_categories(
    ir: EnsembleIR, categories: Mapping[str, Sequence[str]]
) -> EnsembleIR:
    """A copy of ``ir`` with display names installed on its categorical features.

    Each name list must cover the feature's existing cardinality; a longer
    list extends the cardinality (declaring codes the training data never
    used). Naming a numeric or unknown feature is an error.
    """
    from dataclasses import replace

    updated = dict(ir.categorical)
    for feature_name, name_list in categories.items():
        if feature_name not in ir.feature_names:
            raise TreecfError(f"categories references unknown feature {feature_name!r}")
        j = ir.feature_names.index(feature_name)
        if j not in updated:
            raise TreecfError(
                f"categories[{feature_name!r}]: {feature_name!r} is not a "
                "categorical feature of this model"
            )
        display = tuple(str(v) for v in name_list)
        if len(display) < updated[j].cardinality:
            raise TreecfError(
                f"categories[{feature_name!r}] lists {len(display)} names but the "
                f"model uses {updated[j].cardinality} codes"
            )
        updated[j] = CategoricalFeature(cardinality=len(display), categories=display)
    return replace(ir, categorical=updated)


def code_goes_left(value: float, categories: frozenset[int]) -> bool:
    """Set-membership routing for a non-NaN value: left iff an integral member.

    A non-integral value is never a member (it cannot be a category code), and
    an unseen integral code outside the set routes right.
    """
    if not math.isfinite(value):
        return False
    code = int(value)
    return code == value and code in categories


def validate_feature_matrix(ir: EnsembleIR, X: npt.NDArray[np.float64], where: str) -> None:
    """Reject values that are not valid codes on the ir's categorical features.

    ``X`` is one row or a matrix; a categorical coordinate must be an integral
    code in ``[0, cardinality)`` or NaN.
    """
    if not ir.categorical:
        return
    import numpy as np

    rows = X.reshape(1, -1) if X.ndim == 1 else X
    for j, info in sorted(ir.categorical.items()):
        col = rows[:, j]
        bad = ~np.isnan(col) & (
            ~np.isfinite(col)
            | (col != np.floor(col))
            | (col < 0.0)
            | (col >= float(info.cardinality))
        )
        if bad.any():
            value = float(col[bad][0])
            name = ir.feature_names[j]
            raise TreecfError(
                f"feature {name!r} is categorical with {info.cardinality} "
                f"categories; {where} value {value!r} must be an integral code "
                f"in [0, {info.cardinality}) or NaN"
            )
