"""Random model dumps in the three JSON formats the parsers read, plus mutations.

The builders produce structurally valid dumps of random shape from a seeded
``numpy`` generator (the house convention: hypothesis draws a seed, the case
is built from it). ``mutate`` then corrupts one such dump the way a broken
export or a hand edit would: a missing key, a wrong type, a truncated
parallel array, an out-of-range index, a non-finite number, a scalar where an
object belongs. A parser must answer either with a usable model or with
``ParserError`` — never with a bare ``KeyError``/``IndexError``/``TypeError``.
"""

from __future__ import annotations

import copy
import json
import math
from typing import Any

import numpy as np

FORMATS = ("xgboost", "lightgbm", "catboost")


def _shape(rng: np.random.Generator) -> tuple[int, int, int]:
    """(n_features, n_trees, depth) — small enough to score quickly."""
    return int(rng.integers(1, 6)), int(rng.integers(1, 5)), int(rng.integers(1, 4))


def xgboost_dump(rng: np.random.Generator) -> dict[str, Any]:
    n_features, n_trees, depth = _shape(rng)
    objective = str(rng.choice(["reg:squarederror", "binary:logistic"]))
    base = 0.5 if objective == "binary:logistic" else float(rng.normal())
    trees = [_xgboost_tree(rng, n_features, depth) for _ in range(n_trees)]
    return {
        "learner": {
            "gradient_booster": {"name": "gbtree", "model": {"trees": trees}},
            "objective": {"name": objective},
            "learner_model_param": {
                "num_feature": str(n_features),
                "num_class": "0",
                "base_score": f"[{base!r}]",
            },
            "feature_names": [f"f{i}" for i in range(n_features)],
        },
        "version": [3, 0, 0],
    }


def _xgboost_tree(rng: np.random.Generator, n_features: int, depth: int) -> dict[str, Any]:
    """One tree in the parallel-array layout, nodes numbered in preorder."""
    columns: dict[str, list[Any]] = {
        "left_children": [], "right_children": [], "split_indices": [],
        "split_conditions": [], "default_left": [],
    }

    def build(level: int) -> int:
        node_id = len(columns["left_children"])
        for column in columns.values():
            column.append(0)
        columns["left_children"][node_id] = -1
        columns["right_children"][node_id] = -1
        if level == 0 or rng.random() < 0.2:
            columns["split_conditions"][node_id] = float(np.float32(rng.normal()))
            return node_id
        columns["split_indices"][node_id] = int(rng.integers(0, n_features))
        columns["split_conditions"][node_id] = float(np.float32(rng.normal(scale=2.0)))
        columns["default_left"][node_id] = int(rng.random() < 0.5)
        columns["left_children"][node_id] = build(level - 1)
        columns["right_children"][node_id] = build(level - 1)
        return node_id

    build(depth)
    return columns


def lightgbm_dump(rng: np.random.Generator) -> dict[str, Any]:
    n_features, n_trees, depth = _shape(rng)
    objective = str(rng.choice(["regression", "binary"]))

    def node(level: int) -> dict[str, Any]:
        if level == 0 or rng.random() < 0.2:
            return {"leaf_value": float(rng.normal())}
        return {
            "split_feature": int(rng.integers(0, n_features)),
            "threshold": float(np.float32(rng.normal(scale=2.0))),
            "decision_type": "<=",
            "default_left": bool(rng.random() < 0.5),
            "missing_type": str(rng.choice(["None", "NaN"])),
            "left_child": node(level - 1),
            "right_child": node(level - 1),
        }

    return {
        "objective": objective,
        "max_feature_idx": n_features - 1,
        "num_tree_per_iteration": 1,
        "feature_names": [f"f{i}" for i in range(n_features)],
        "tree_info": [{"tree_structure": node(depth)} for _ in range(n_trees)],
        "version": "v4",
    }


def catboost_dump(rng: np.random.Generator) -> dict[str, Any]:
    n_features, n_trees, depth = _shape(rng)
    loss = str(rng.choice(["RMSE", "Logloss"]))
    trees = []
    for _ in range(n_trees):
        d = int(rng.integers(1, depth + 1))
        splits = [
            {
                "split_type": "FloatFeature",
                "float_feature_index": int(rng.integers(0, n_features)),
                "border": float(np.float32(rng.normal(scale=2.0))),
            }
            for _ in range(d)
        ]
        trees.append(
            {"splits": splits, "leaf_values": [float(v) for v in rng.normal(size=2**d)]}
        )
    return {
        "scale_and_bias": [1.0, [float(rng.normal())]],
        "model_info": {"params": {"loss_function": {"type": loss}}},
        "features_info": {
            "float_features": [
                {
                    "feature_index": i,
                    "flat_feature_index": i,
                    "feature_id": f"f{i}",
                    "nan_value_treatment": "AsIs",
                }
                for i in range(n_features)
            ]
        },
        "oblivious_trees": trees,
    }


BUILDERS = {"xgboost": xgboost_dump, "lightgbm": lightgbm_dump, "catboost": catboost_dump}


# --------------------------------------------------------------- mutation ---


def _paths(value: Any, prefix: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    """Every path to a container or scalar inside a JSON value."""
    out = [prefix] if prefix else []
    if isinstance(value, dict):
        for key, child in value.items():
            out.extend(_paths(child, (*prefix, key)))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            out.extend(_paths(child, (*prefix, i)))
    return out


def _get(value: Any, path: tuple[Any, ...]) -> Any:
    for key in path:
        value = value[key]
    return value


def _set(root: Any, path: tuple[Any, ...], new: Any) -> None:
    parent = _get(root, path[:-1])
    parent[path[-1]] = new


def _delete(root: Any, path: tuple[Any, ...]) -> None:
    parent = _get(root, path[:-1])
    del parent[path[-1]]


MUTATIONS = (
    "drop_key", "wrong_type", "truncate", "non_finite", "bad_index", "scalar_root",
    "huge_number", "negative_count",
)


def mutate(dump: dict[str, Any], rng: np.random.Generator, kind: str | None = None) -> Any:
    """One corruption applied to a deep copy of ``dump``; the kind is drawn
    when not given. May return a non-dict for ``scalar_root``."""
    dump = copy.deepcopy(dump)
    kind = kind or str(rng.choice(MUTATIONS))
    paths = _paths(dump)
    if kind == "scalar_root":
        options: list[Any] = [5, None, [1, 2, 3], 2.5]  # a string would be a path
        return options[int(rng.integers(len(options)))]
    if not paths:
        return dump
    if kind == "drop_key":
        keyed = [p for p in paths if isinstance(p[-1], str)]
        if keyed:
            _delete(dump, keyed[int(rng.integers(len(keyed)))])
        return dump
    if kind == "wrong_type":
        path = paths[int(rng.integers(len(paths)))]
        replacements: list[Any] = ["oops", None, [1], {"a": 1}, True]
        _set(dump, path, replacements[int(rng.integers(len(replacements)))])
        return dump
    if kind == "truncate":
        lists = [p for p in paths if isinstance(_get(dump, p), list) and _get(dump, p)]
        if lists:
            path = lists[int(rng.integers(len(lists)))]
            current = _get(dump, path)
            _set(dump, path, current[: int(rng.integers(len(current)))])
        return dump
    if kind in ("non_finite", "huge_number", "negative_count", "bad_index"):
        numeric = [
            p for p in paths
            if isinstance(_get(dump, p), int | float) and not isinstance(_get(dump, p), bool)
        ]
        if not numeric:
            return dump
        path = numeric[int(rng.integers(len(numeric)))]
        if kind == "non_finite":
            _set(dump, path, [math.nan, math.inf, -math.inf][int(rng.integers(3))])
        elif kind == "huge_number":
            _set(dump, path, [1e300, -1e300, 1e40][int(rng.integers(3))])
        elif kind == "negative_count":
            _set(dump, path, -int(rng.integers(1, 50)))
        else:
            _set(dump, path, int(rng.integers(50, 5000)))
        return dump
    raise ValueError(f"unknown mutation {kind!r}")


def as_json(dump: Any) -> Any:
    """Round-trip through JSON text (``allow_nan`` on, as a broken export
    might do), so the parser sees exactly what a file would carry."""
    return json.loads(json.dumps(dump, allow_nan=True))
