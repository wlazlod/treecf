"""Rust cross-language tests: skipped when the extension is unavailable
(it is built by `uv sync` via the maturin build backend)."""

from __future__ import annotations

import pytest

_treecf_core = pytest.importorskip("treecf._treecf_core")
