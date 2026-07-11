"""Model parsers: native objects or JSON dumps in, ``EnsembleIR`` out (spec §3.3).

Dispatch never imports a model library; native objects are recognized by their
type's module and routed to the parser, which itself uses only dump payloads.
"""

from __future__ import annotations

from pathlib import Path

from treecf._errors import UnsupportedModelError
from treecf.ir.model import EnsembleIR
from treecf.ir.parsers.json_dump import parse_dump

__all__ = ["parse_dump", "parse_model"]


def parse_model(model: object) -> EnsembleIR:
    """Parse a native model object, a dump dict, or a path to a dump file."""
    if isinstance(model, str | Path | dict):
        return parse_dump(model)
    root_module = type(model).__module__.split(".")[0]
    if root_module == "xgboost":
        from treecf.ir.parsers.xgboost import parse_xgboost

        return parse_xgboost(model)
    raise UnsupportedModelError(
        f"cannot parse {type(model)!r}; supported in v0.1: xgboost models, JSON dumps"
    )
