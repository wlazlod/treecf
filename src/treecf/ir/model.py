"""IR data model.

Split operators are stored per node exactly as the source library defines them;
normalizing LT <-> LE via ``nextafter`` is forbidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class SplitOp(Enum):
    """Comparison sending an instance to the left child."""

    LT = auto()  # x < threshold  -> left
    LE = auto()  # x <= threshold -> left


class Link(Enum):
    """Output link applied to the raw score."""

    IDENTITY = auto()
    SIGMOID = auto()


@dataclass(frozen=True)
class Node:
    """One tree node; ``feature is None`` marks a leaf.

    ``left``/``right`` are node ids, and parsers guarantee ``nodes[i].node_id == i``
    so children are addressed by index.
    """

    node_id: int
    feature: int | None
    threshold: float | None
    op: SplitOp | None
    missing_left: bool | None
    left: int | None
    right: int | None
    value: float | None


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
