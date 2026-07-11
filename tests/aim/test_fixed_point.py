"""Adversarial fixed-point case (spec §12.5): leaf values near 1/K resolution."""

from __future__ import annotations

import numpy as np
import pytest

from treecf import Counterfactual, Explainer, Target
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree

pytest.importorskip("ortools")


def _tiny_leaf_stump() -> EnsembleIR:
    """Split at 1.0; leaves 0 and 3e-7 — indistinguishable at the default K = 1e6."""
    nodes = (
        Node(0, 0, 1.0, SplitOp.LT, True, 1, 2, None),
        Node(1, None, None, None, None, None, None, 0.0),
        Node(2, None, None, None, None, None, None, 3e-7),
    )
    return EnsembleIR(
        trees=(Tree(nodes=nodes),),
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=1,
        feature_names=("a",),
        meta={},
    )


def test_k_retry_resolves_sub_resolution_targets() -> None:
    """The K*10 retry loop must recover when leaf differences vanish at scale K."""
    exp = Explainer(_tiny_leaf_stump(), normalizers=np.ones(1))
    res = exp.explain(np.array([0.0]), target=Target.raw(op=">=", value=2e-7))
    assert isinstance(res, Counterfactual)
    assert res.x_cf[0] >= 1.0  # forced into the right leaf
    assert res.score_raw >= 2e-7
