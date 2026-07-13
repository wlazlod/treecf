"""Dump-file entry point: parse library JSON dumps without the training framework."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from treecf._errors import UnsupportedModelError
from treecf.ir.model import EnsembleIR


def parse_dump(source: str | Path | dict[str, Any]) -> EnsembleIR:
    """Parse a model dump given as a dict, or a path to a JSON file."""
    if isinstance(source, str | Path):
        with open(source, encoding="utf-8") as fh:
            data: dict[str, Any] = json.load(fh)
    else:
        data = source

    if "learner" in data:  # XGBoost JSON model format
        from treecf.ir.parsers.xgboost import parse_xgboost_dump

        return parse_xgboost_dump(data)
    if "tree_info" in data:  # LightGBM dump_model() format
        from treecf.ir.parsers.lightgbm import parse_lightgbm_dump

        return parse_lightgbm_dump(data)
    if "oblivious_trees" in data:  # CatBoost JSON format
        from treecf.ir.parsers.catboost import parse_catboost_dump

        return parse_catboost_dump(data)
    raise UnsupportedModelError(
        "unrecognized dump format; expected an XGBoost or LightGBM JSON model (v0.1)"
    )
