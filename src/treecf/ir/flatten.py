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

from treecf.ir.evaluate import bitset_words
from treecf.ir.model import CategoricalFeature, EnsembleIR, Link, Node, SplitOp, Tree


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
    node_set = np.full(n_nodes, -1, dtype=np.int32)
    set_offsets = [0]
    set_words: list[int] = []

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
                missing_left[i] = 1 if node.missing_left else 0
                left[i] = offset + int(node.left)  # type: ignore[arg-type]
                right[i] = offset + int(node.right)  # type: ignore[arg-type]
                if node.categories is not None:
                    node_set[i] = len(set_offsets) - 1
                    set_words.extend(bitset_words(node.categories))
                    set_offsets.append(len(set_words))
                else:
                    threshold[i] = float(node.threshold)  # type: ignore[arg-type]
                    is_lt[i] = 1 if node.op is SplitOp.LT else 0
        offset += len(tree.nodes)

    flat = {
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
    if ir.categorical or set_words:
        flat["node_set"] = node_set
        flat["set_offsets"] = np.array(set_offsets, dtype=np.uint32)
        flat["set_words"] = np.array(set_words, dtype=np.uint64)
        flat["cat_idx"] = np.array(sorted(ir.categorical), dtype=np.uint32)
        flat["cat_card"] = np.array(
            [ir.categorical[j].cardinality for j in sorted(ir.categorical)], dtype=np.uint32
        )
    return flat


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

    node_set = np.asarray(flat.get("node_set", np.full(len(feature), -1)), dtype=np.int32)
    set_offsets = np.asarray(flat.get("set_offsets", [0]), dtype=np.uint32)
    set_words = np.asarray(flat.get("set_words", []), dtype=np.uint64)
    cat_idx = np.asarray(flat.get("cat_idx", []), dtype=np.uint32)
    cat_card = np.asarray(flat.get("cat_card", []), dtype=np.uint32)

    def set_members(sid: int) -> frozenset[int]:
        start, end = int(set_offsets[sid]), int(set_offsets[sid + 1])
        return frozenset(
            w * 64 + b
            for w in range(end - start)
            for b in range(64)
            if int(set_words[start + w]) >> b & 1
        )

    n_nodes = len(feature)
    boundaries = [*tree_roots.tolist(), n_nodes]
    trees = []
    for t in range(len(tree_roots)):
        start, end = boundaries[t], boundaries[t + 1]
        nodes = []
        for i in range(start, end):
            node_id = i - start
            if feature[i] < 0:
                nodes.append(Node(node_id, None, None, None, None, None, None, float(value[i])))
            elif node_set[i] >= 0:
                nodes.append(
                    Node(
                        node_id=node_id,
                        feature=int(feature[i]),
                        threshold=None,
                        op=None,
                        missing_left=bool(missing_left[i]),
                        left=int(left[i]) - start,
                        right=int(right[i]) - start,
                        value=None,
                        categories=set_members(int(node_set[i])),
                    )
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
        categorical={
            int(j): CategoricalFeature(cardinality=int(k))
            for j, k in zip(cat_idx.tolist(), cat_card.tolist(), strict=True)
        },
    )
