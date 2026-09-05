"""Dump-file entry point: parse library JSON dumps without the training framework.

A dump is untrusted input: a file may be truncated, hand-edited, or written
by a version with a different layout. Whatever shape it has, the parsers
answer in one of two ways — a usable ``EnsembleIR`` (structurally checked
by ``validate_ir``) or ``ParserError`` naming what was wrong — never a bare
``KeyError``, ``IndexError`` or ``TypeError`` from inside a parser.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from treecf._errors import ParserError, UnsupportedModelError
from treecf.ir.model import EnsembleIR, validate_ir

# what a malformed dump makes a parser raise before it can say anything
# useful; each becomes a ParserError naming the format and the cause
_MALFORMED = (
    KeyError,
    IndexError,
    TypeError,
    ValueError,
    AttributeError,
    ZeroDivisionError,
    OverflowError,
    RecursionError,
)


def parse_dump(
    source: str | Path | dict[str, Any],
    categories: Mapping[str, Sequence[str]] | None = None,
) -> EnsembleIR:
    """Parse a model dump given as a dict, or a path to a JSON file.

    Raises
    ------
    ParserError
        If the file is not valid JSON, the dump is not an object, or a
        recognized format is structurally broken (a missing key, a wrong
        type, a truncated array, an index outside the model, a non-finite
        threshold or leaf value, a child pointer that does not form a tree).
    UnsupportedModelError
        If the format is not recognized, or the model uses something the IR
        cannot represent (a multiclass objective, an unsupported split).
    """
    if isinstance(source, str | Path):
        try:
            with open(source, encoding="utf-8") as fh:
                data = json.load(fh)
        except (ValueError, RecursionError, UnicodeDecodeError) as exc:
            raise ParserError(f"model dump is not readable JSON: {exc}") from exc
    else:
        data = source
    if not isinstance(data, dict):
        raise ParserError(
            f"a model dump must be a JSON object, got {type(data).__name__}"
        )

    parser: Callable[[dict[str, Any], Mapping[str, Sequence[str]] | None], EnsembleIR]
    if "learner" in data:  # XGBoost JSON model format
        from treecf.ir.parsers.xgboost import parse_xgboost_dump

        fmt, parser = "xgboost", parse_xgboost_dump
    elif "tree_info" in data:  # LightGBM dump_model() format
        from treecf.ir.parsers.lightgbm import parse_lightgbm_dump

        fmt, parser = "lightgbm", parse_lightgbm_dump
    elif "oblivious_trees" in data:  # CatBoost JSON format
        from treecf.ir.parsers.catboost import parse_catboost_dump

        fmt, parser = "catboost", parse_catboost_dump
    else:
        raise UnsupportedModelError(
            "unrecognized dump format; expected an XGBoost, LightGBM, or CatBoost JSON model"
        )
    try:
        # a float32 cast of an out-of-range threshold warns on overflow; the
        # resulting non-finite value is rejected by validate_ir below instead
        with np.errstate(all="ignore"):
            ir = parser(data, categories)
    except UnsupportedModelError:
        raise
    except _MALFORMED as exc:
        raise ParserError(
            f"malformed {fmt} dump: {type(exc).__name__}: {exc}"
        ) from exc
    validate_ir(ir)
    return ir
