"""The README's Python blocks run as written, in order, with warnings as errors."""

from __future__ import annotations

import pathlib
import re
import warnings

import pytest

pytestmark = pytest.mark.slow

_README = pathlib.Path(__file__).parent.parent / "README.md"
_CODE_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.S)


def test_readme_python_blocks_execute() -> None:
    blocks = _CODE_BLOCK_RE.findall(_README.read_text(encoding="utf-8"))
    assert blocks, "the README has no Python block to run"
    namespace: dict[str, object] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for block in blocks:
            exec(compile(block, str(_README), "exec"), namespace)
