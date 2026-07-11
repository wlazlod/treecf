"""Float-space IR evaluation: op semantics at thresholds, NaN routing, link (spec §3.1–§3.2)."""

import math

import numpy as np
import pytest

from treecf.ir.evaluate import apply_link, raw_score
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree

LEFT = -1.0
RIGHT = 1.0


def _leaf(node_id: int, value: float) -> Node:
    return Node(
        node_id=node_id,
        feature=None,
        threshold=None,
        op=None,
        missing_left=None,
        left=None,
        right=None,
        value=value,
    )


def _single_split_ir(
    op: SplitOp, threshold: float = 1.0, missing_left: bool | None = True
) -> EnsembleIR:
    """One tree: split on feature 0; left leaf = -1.0, right leaf = +1.0."""
    nodes = (
        Node(
            node_id=0,
            feature=0,
            threshold=threshold,
            op=op,
            missing_left=missing_left,
            left=1,
            right=2,
            value=None,
        ),
        _leaf(1, LEFT),
        _leaf(2, RIGHT),
    )
    return EnsembleIR(
        trees=(Tree(nodes=nodes),),
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=1,
        feature_names=("x0",),
        meta={},
    )


class TestSplitOpAtThreshold:
    def test_lt_at_threshold_goes_right(self) -> None:
        ir = _single_split_ir(SplitOp.LT)
        assert raw_score(ir, np.array([1.0])) == RIGHT

    def test_le_at_threshold_goes_left(self) -> None:
        ir = _single_split_ir(SplitOp.LE)
        assert raw_score(ir, np.array([1.0])) == LEFT

    @pytest.mark.parametrize("op", [SplitOp.LT, SplitOp.LE])
    def test_nextafter_below_goes_left(self, op: SplitOp) -> None:
        ir = _single_split_ir(op)
        x = np.array([np.nextafter(1.0, -np.inf)])
        assert raw_score(ir, x) == LEFT

    @pytest.mark.parametrize("op", [SplitOp.LT, SplitOp.LE])
    def test_nextafter_above_goes_right(self, op: SplitOp) -> None:
        ir = _single_split_ir(op)
        x = np.array([np.nextafter(1.0, np.inf)])
        assert raw_score(ir, x) == RIGHT


class TestNanRouting:
    def test_nan_routes_left_when_missing_left(self) -> None:
        ir = _single_split_ir(SplitOp.LT, missing_left=True)
        assert raw_score(ir, np.array([np.nan])) == LEFT

    def test_nan_routes_right_when_not_missing_left(self) -> None:
        ir = _single_split_ir(SplitOp.LT, missing_left=False)
        assert raw_score(ir, np.array([np.nan])) == RIGHT

    def test_nan_without_missing_routing_raises(self) -> None:
        ir = _single_split_ir(SplitOp.LT, missing_left=None)
        with pytest.raises(ValueError, match="missing"):
            raw_score(ir, np.array([np.nan]))


class TestScoreComposition:
    def test_base_score_and_tree_sum(self) -> None:
        one = _single_split_ir(SplitOp.LT)
        ir = EnsembleIR(
            trees=one.trees * 3,
            base_score=0.25,
            link=Link.IDENTITY,
            n_features=1,
            feature_names=("x0",),
            meta={},
        )
        assert raw_score(ir, np.array([0.0])) == pytest.approx(0.25 + 3 * LEFT)

    def test_apply_link_identity_and_sigmoid(self) -> None:
        assert apply_link(Link.IDENTITY, 0.3) == 0.3
        assert apply_link(Link.SIGMOID, 0.0) == 0.5
        assert apply_link(Link.SIGMOID, 4.0) == pytest.approx(1.0 / (1.0 + math.exp(-4.0)))
