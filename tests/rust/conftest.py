"""Rust cross-language tests: skipped when the dev extension is not built.

Build it with:
    VIRTUAL_ENV=$PWD/.venv uv run maturin develop --release -m rust/Cargo.toml
"""

from __future__ import annotations

import pytest

_treecf_core = pytest.importorskip("_treecf_core")
