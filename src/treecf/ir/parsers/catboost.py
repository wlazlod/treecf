"""CatBoost parser: oblivious trees expanded to plain binary IR trees.

A depth-d oblivious tree stores d shared splits and 2^d leaf values; the leaf
index is the bit pattern of split decisions with splits[i] as bit i. The
expansion puts splits[d-1] at the root so leaf ranges stay contiguous. Float
splits rewrite "x > border -> bit 1" as op LE (x <= border -> left/bit 0).

Categorical splits are lowered to set-membership nodes at parse time:

- A one-hot split ("value equals this category's hash" -> bit 1) becomes a
  singleton set on the bit-1 side.
- A single-feature target-statistics split compares the category's quantized
  statistic against a border. The statistic is fully determined by the model's
  own table (category-hash bucket -> counts) plus the split's prior and scale,
  so each border lowers to the set of codes whose statistic falls on the
  bit-0 side; a category outside the table (never seen in training) takes the
  prior-only value, which routes every unseen code consistently.

Because categories are identified by a hash of their string form, ``categories``
(the caller's code -> name lists) is required whenever native categorical
features are present. The search core only ever sees set-membership nodes.

Borders are float32-quantized (cast back through float32, as with XGBoost).
NaN routing on float splits: nan_value_treatment "AsFalse"/"AsIs" -> bit 0
(missing_left=True), "AsTrue" -> bit 1. CatBoost rejects NaN categorical
inputs, so set-membership nodes route NaN right (never a member).
``scale_and_bias`` folds scale into leaf values; bias is the raw-space
intercept.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from treecf._errors import ParserError, UnsupportedModelError
from treecf.ir.model import CategoricalFeature, EnsembleIR, Link, Node, SplitOp, Tree
from treecf.ir.parsers._catboost_cat import calc_ctr_bucket, cat_feature_hash, signed32

_LOSS_LINKS = {
    "Logloss": Link.SIGMOID,
    "CrossEntropy": Link.SIGMOID,
    "RMSE": Link.IDENTITY,
    "MAE": Link.IDENTITY,
    "Quantile": Link.IDENTITY,
}

_RETRAIN_RECIPE = (
    "retrain with max_ctr_complexity=1 (single-feature statistics only), or "
    "one_hot_max_size >= the largest categorical cardinality"
)


def parse_catboost(
    model: object, categories: Mapping[str, Sequence[str]] | None = None
) -> EnsembleIR:
    """Parse a live CatBoost model via its JSON serialization."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.json"
        model.save_model(str(path), format="json")  # type: ignore[attr-defined]
        with open(path, encoding="utf-8") as fh:
            dump: dict[str, Any] = json.load(fh)
    return parse_catboost_dump(dump, categories)


def parse_catboost_dump(
    dump: dict[str, Any], categories: Mapping[str, Sequence[str]] | None = None
) -> EnsembleIR:
    scale, bias = dump["scale_and_bias"]
    if len(bias) != 1:
        raise UnsupportedModelError("multiclass CatBoost models are not supported")

    info = dump.get("model_info", {})
    loss = (
        info.get("params", {}).get("loss_function", {}).get("type")
        or info.get("loss_function", {}).get("type")
        or ""
    )
    if loss not in _LOSS_LINKS:
        raise UnsupportedModelError(f"loss function {loss!r} not supported")

    features_info = dump["features_info"]
    float_features = features_info.get("float_features") or []
    cat_features = features_info.get("categorical_features") or []
    flat_of = {f["feature_index"]: f["flat_feature_index"] for f in float_features}
    missing_left_of = {
        f["feature_index"]: f.get("nan_value_treatment", "AsIs") != "AsTrue"
        for f in float_features
    }

    context = _categorical_context(dump, cat_features, categories) if cat_features else None

    flat_indices = [f["flat_feature_index"] for f in float_features]
    flat_indices += [f["flat_feature_index"] for f in cat_features]
    n_features = 1 + max(flat_indices, default=-1)
    names: list[str] = [f"f{i}" for i in range(n_features)]
    for f in [*float_features, *cat_features]:
        if f.get("feature_id"):
            names[f["flat_feature_index"]] = f["feature_id"]

    trees = tuple(
        _expand_oblivious(tree, float(scale), flat_of, missing_left_of, context)
        for tree in dump["oblivious_trees"]
    )
    return EnsembleIR(
        trees=trees,
        base_score=float(bias[0]),
        link=_LOSS_LINKS[loss],
        n_features=n_features,
        feature_names=tuple(names),
        meta={"source": "catboost", "loss_function": loss},
        categorical={} if context is None else context.metadata,
    )


@dataclass(frozen=True)
class _CatContext:
    """Lowered categorical routing for one model.

    ``onehot_code``: per cat feature index, the JSON one-hot ``value`` (a
    signed category hash) -> the caller's code for it. ``ctr_members``: per
    ``split_index`` of a statistics split, the codes routing to the bit-0
    side. ``flat_of``: cat feature index -> flat column. ``metadata``: the
    per-column kind/cardinality/name record for the IR.
    """

    flat_of: dict[int, int]
    onehot_code: dict[int, dict[int, int]]
    ctr_members: dict[int, tuple[int, frozenset[int]]]
    metadata: dict[int, CategoricalFeature]


def _categorical_context(
    dump: dict[str, Any],
    cat_features: list[dict[str, Any]],
    categories: Mapping[str, Sequence[str]] | None,
) -> _CatContext:
    if categories is None:
        raise ParserError(
            "categories is a required Explainer argument for CatBoost models "
            "with native categorical features (categories are stored as hashes; "
            "the code -> name lists make them recoverable)"
        )

    flat_of: dict[int, int] = {}
    onehot_code: dict[int, dict[int, int]] = {}
    name_lists: dict[int, tuple[str, ...]] = {}
    metadata: dict[int, CategoricalFeature] = {}
    for f in cat_features:
        cat_idx = int(f["feature_index"])
        flat = int(f["flat_feature_index"])
        feature_name = f.get("feature_id") or f"f{flat}"
        if feature_name not in categories:
            raise ParserError(
                f"categories has no entry for categorical feature {feature_name!r}"
            )
        display = tuple(str(v) for v in categories[feature_name])
        signed_hashes = {
            signed32(cat_feature_hash(name)): code for code, name in enumerate(display)
        }
        stored = [int(v) for v in (f.get("values") or [])]
        uncovered = [v for v in stored if v not in signed_hashes]
        if uncovered:
            raise ParserError(
                f"categories[{feature_name!r}] does not name every category the "
                f"model was trained on ({len(uncovered)} stored hash(es) have no "
                "matching name)"
            )
        flat_of[cat_idx] = flat
        onehot_code[cat_idx] = signed_hashes
        name_lists[cat_idx] = display
        metadata[flat] = CategoricalFeature(cardinality=len(display), categories=display)

    ctr_members = _lower_ctr_splits(dump, flat_of, name_lists)
    return _CatContext(
        flat_of=flat_of,
        onehot_code=onehot_code,
        ctr_members=ctr_members,
        metadata=metadata,
    )


def _lower_ctr_splits(
    dump: dict[str, Any],
    flat_of: dict[int, int],
    name_lists: dict[int, tuple[str, ...]],
) -> dict[int, tuple[int, frozenset[int]]]:
    """Per statistics-split ``split_index``: (flat column, bit-0 member codes)."""
    entries = dump["features_info"].get("ctrs") or []
    if not entries:
        return {}

    combos: list[str] = []
    for entry in entries:
        elements = entry.get("elements", [])
        if len(elements) != 1 or any(
            e.get("combination_element") != "cat_feature_value" for e in elements
        ):
            combos.append(entry.get("identifier", "<unknown>"))
    if combos:
        raise ParserError(
            "CatBoost combination statistics span more than one feature and "
            f"cannot be lowered: {combos!r}; {_RETRAIN_RECIPE}"
        )

    used: dict[int, float] = {}
    for tree in dump["oblivious_trees"]:
        for s in tree["splits"] or []:
            if s["split_type"] == "OnlineCtr":
                used[int(s["split_index"])] = float(s["border"])
    if not used:
        return {}
    base = min(used)

    tables = _ctr_tables(dump)
    members: dict[int, tuple[int, frozenset[int]]] = {}
    flattened = [(entry, float(border)) for entry in entries for border in entry["borders"]]
    for offset, (entry, border) in enumerate(flattened):
        split_index = base + offset
        if int(entry.get("target_border_idx", 0)) != 0:
            raise UnsupportedModelError(
                "statistics over a non-primary target border are not supported"
            )
        ctr_type = entry.get("ctr_type")
        if ctr_type not in ("Borders", "Counter"):
            raise UnsupportedModelError(
                f"ctr_type {ctr_type!r} not supported (Borders and Counter only)"
            )
        cat_idx = int(entry["elements"][0]["cat_feature_index"])
        denominator, table = tables[entry["identifier"]]
        prior_num = float(entry["prior_numerator"])
        prior_denom = float(entry["prior_denomerator"])
        stat_scale = float(entry["scale"])
        shift = float(entry["shift"])
        codes = []
        for code, name in enumerate(name_lists[cat_idx]):
            bucket = calc_ctr_bucket(cat_feature_hash(name))
            counts = table.get(bucket)
            if ctr_type == "Borders":
                count_other, count_in_class = counts if counts is not None else (0, 0)
                total = count_other + count_in_class
                ratio = (count_in_class + prior_num) / (total + prior_denom)
            else:  # Counter: bucket -> occurrence count, shared denominator
                (count,) = counts if counts is not None else (0,)
                ratio = (count + prior_num) / (denominator + prior_denom)
            value = shift + stat_scale * ratio
            if not value > border:  # bit 0: the statistic does not clear the border
                codes.append(code)
        members[split_index] = (flat_of[cat_idx], frozenset(codes))
    misaligned = {
        s: border
        for s, border in used.items()
        if s not in members or border not in {b for _, b in flattened}
    }
    for s, border in used.items():
        offset = s - base
        if offset >= len(flattened) or flattened[offset][1] != border:
            misaligned[s] = border
    if misaligned:
        raise UnsupportedModelError(
            "statistics splits do not align with the model's declared tables "
            f"(indices {sorted(misaligned)}); the dump layout is not the one "
            "this parser understands"
        )
    return members


def _ctr_tables(dump: dict[str, Any]) -> dict[str, tuple[int, dict[int, tuple[int, ...]]]]:
    """Identifier -> (counter denominator, {category bucket: stored counts}).

    Class-conditional tables (stride 3) store (count outside the class, count
    in it) per bucket; occurrence tables (stride 2) store one count per bucket
    and share the table-level denominator.
    """
    tables: dict[str, tuple[int, dict[int, tuple[int, ...]]]] = {}
    for identifier, raw in (dump.get("ctr_data") or {}).items():
        stride = int(raw["hash_stride"])
        if stride not in (2, 3):
            raise UnsupportedModelError(
                f"statistics table stride {stride} not supported (expected 2 or 3)"
            )
        flat = raw["hash_map"]
        table: dict[int, tuple[int, ...]] = {}
        for i in range(0, len(flat), stride):
            bucket = int(flat[i])
            if bucket == (1 << 64) - 1:  # empty hash-map slot
                continue
            table[bucket] = tuple(int(v) for v in flat[i + 1 : i + stride])
        tables[identifier] = (int(raw.get("counter_denominator") or 0), table)
    return tables


def _expand_oblivious(
    tree: dict[str, Any],
    scale: float,
    flat_of: dict[int, int],
    missing_left_of: dict[int, bool],
    context: _CatContext | None,
) -> Tree:
    splits = tree["splits"] or []
    leaf_values = tree["leaf_values"]
    depth = len(splits)
    if len(leaf_values) != 2**depth:
        raise UnsupportedModelError("oblivious tree leaf count does not match its depth")

    nodes: list[Node] = []

    def build(bit: int, prefix: int) -> int:
        node_id = len(nodes)
        if bit < 0:
            value = scale * float(leaf_values[prefix])
            nodes.append(Node(node_id, None, None, None, None, None, None, value))
            return node_id
        split = splits[bit]
        split_type = split.get("split_type")
        nodes.append(None)  # type: ignore[arg-type]  # placeholder
        bit0 = build(bit - 1, prefix)
        bit1 = build(bit - 1, prefix | (1 << bit))
        if split_type == "FloatFeature":
            feature_index = int(split["float_feature_index"])
            nodes[node_id] = Node(
                node_id=node_id,
                feature=int(flat_of[feature_index]),
                threshold=float(np.float32(split["border"])),
                op=SplitOp.LE,
                missing_left=missing_left_of[feature_index],
                left=bit0,  # bit 0: x <= border
                right=bit1,  # bit 1: x > border
                value=None,
            )
        elif split_type == "OneHotFeature" and context is not None:
            cat_idx = int(split["cat_feature_index"])
            value = int(split["value"])
            code = context.onehot_code[cat_idx].get(value)
            if code is None:
                raise ParserError(
                    f"categories does not name the category a one-hot split tests "
                    f"(feature index {cat_idx}, stored hash {value})"
                )
            nodes[node_id] = Node(
                node_id=node_id,
                feature=context.flat_of[cat_idx],
                threshold=None,
                op=None,
                missing_left=False,  # CatBoost rejects NaN categorical inputs
                left=bit1,  # bit 1: the category equals the tested value
                right=bit0,
                value=None,
                categories=frozenset({code}),
            )
        elif split_type == "OnlineCtr" and context is not None:
            mapped = context.ctr_members.get(int(split["split_index"]))
            if mapped is None:
                raise UnsupportedModelError(
                    f"statistics split {split['split_index']} has no lowered table entry"
                )
            flat, member_codes = mapped
            nodes[node_id] = Node(
                node_id=node_id,
                feature=flat,
                threshold=None,
                op=None,
                missing_left=False,  # CatBoost rejects NaN categorical inputs
                left=bit0,  # bit 0: the category's statistic is at or below the border
                right=bit1,
                value=None,
                categories=member_codes,
            )
        else:
            raise UnsupportedModelError(f"split_type {split_type!r} not supported")
        return node_id

    build(depth - 1, 0)
    return Tree(nodes=tuple(nodes))
