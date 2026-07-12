"""Flat-array (SoA) serialization of the IR — the cross-language boundary contract.

One format, three consumers: parity fixtures (JSON), Rust unit tests (serde),
and the PyO3 boundary (numpy arrays). Child indices are GLOBAL: per-tree node
ids are offset by the tree's start position; ``tree_roots[t]`` is tree t's root.

``missing_left`` is stored as u8 with None -> 0, matching the batch evaluator's
semantics (``raw_score_batch`` routes NaN right when a node defines no missing
direction) — the batch evaluator is the GA's reference.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree


def flatten_ir(ir: EnsembleIR) -> dict[str, Any]:
    n_nodes = sum(len(t.nodes) for t in ir.trees)
    feature = np.empty(n_nodes, dtype=np.int32)
    threshold = np.zeros(n_nodes, dtype=np.float64)
    is_lt = np.zeros(n_nodes, dtype=np.uint8)
    missing_left = np.zeros(n_nodes, dtype=np.uint8)
    left = np.zeros(n_nodes, dtype=np.uint32)
    right = np.zeros(n_nodes, dtype=np.uint32)
    value = np.zeros(n_nodes, dtype=np.float64)
    tree_roots = np.zeros(len(ir.trees), dtype=np.uint32)

    offset = 0
    for t, tree in enumerate(ir.trees):
        tree_roots[t] = offset
        for node in tree.nodes:
            i = offset + node.node_id
            if node.feature is None:
                feature[i] = -1
                value[i] = float(node.value)  # type: ignore[arg-type]
            else:
                feature[i] = node.feature
                threshold[i] = float(node.threshold)  # type: ignore[arg-type]
                is_lt[i] = 1 if node.op is SplitOp.LT else 0
                missing_left[i] = 1 if node.missing_left else 0
                left[i] = offset + int(node.left)  # type: ignore[arg-type]
                right[i] = offset + int(node.right)  # type: ignore[arg-type]
        offset += len(tree.nodes)

    return {
        "feature": feature,
        "threshold": threshold,
        "is_lt": is_lt,
        "missing_left": missing_left,
        "left": left,
        "right": right,
        "value": value,
        "tree_roots": tree_roots,
        "base_score": float(ir.base_score),
        "link": "sigmoid" if ir.link is Link.SIGMOID else "identity",
        "n_features": int(ir.n_features),
        "feature_names": list(ir.feature_names),
    }


def unflatten_ir(flat: dict[str, Any]) -> EnsembleIR:
    """Rebuild an EnsembleIR from flat arrays (fixtures need no ML libraries)."""
    feature = np.asarray(flat["feature"], dtype=np.int32)
    threshold = np.asarray(flat["threshold"], dtype=np.float64)
    is_lt = np.asarray(flat["is_lt"], dtype=np.uint8)
    missing_left = np.asarray(flat["missing_left"], dtype=np.uint8)
    left = np.asarray(flat["left"], dtype=np.uint32)
    right = np.asarray(flat["right"], dtype=np.uint32)
    value = np.asarray(flat["value"], dtype=np.float64)
    tree_roots = np.asarray(flat["tree_roots"], dtype=np.uint32)

    n_nodes = len(feature)
    boundaries = [*tree_roots.tolist(), n_nodes]
    trees = []
    for t in range(len(tree_roots)):
        start, end = boundaries[t], boundaries[t + 1]
        nodes = []
        for i in range(start, end):
            node_id = i - start
            if feature[i] < 0:
                nodes.append(
                    Node(node_id, None, None, None, None, None, None, float(value[i]))
                )
            else:
                nodes.append(
                    Node(
                        node_id=node_id,
                        feature=int(feature[i]),
                        threshold=float(threshold[i]),
                        op=SplitOp.LT if is_lt[i] else SplitOp.LE,
                        missing_left=bool(missing_left[i]),
                        left=int(left[i]) - start,
                        right=int(right[i]) - start,
                        value=None,
                    )
                )
        trees.append(Tree(nodes=tuple(nodes)))

    return EnsembleIR(
        trees=tuple(trees),
        base_score=float(flat["base_score"]),
        link=Link.SIGMOID if flat["link"] == "sigmoid" else Link.IDENTITY,
        n_features=int(flat["n_features"]),
        feature_names=tuple(flat["feature_names"]),
        meta={"source": "flat"},
    )
