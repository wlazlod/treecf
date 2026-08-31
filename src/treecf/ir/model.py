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

from treecf._errors import TreecfError

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
