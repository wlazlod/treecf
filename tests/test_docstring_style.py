"""The docstring convention is numpy-style; no Google-style section survives."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

GOOGLE_SECTION = re.compile(r"^\s*(Args|Arguments|Keyword Args):\s*$", re.MULTILINE)


def test_no_google_style_sections_under_src() -> None:
    out = subprocess.run(
        ["git", "ls-files", "src/"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    hits = []
    for rel in out.splitlines():
        if not rel.endswith(".py"):
            continue
        text = (REPO / rel).read_text(encoding="utf-8")
        for match in GOOGLE_SECTION.finditer(text):
            lineno = text.count("\n", 0, match.start()) + 1
            hits.append(f"{rel}:{lineno}")
    assert not hits, "Google-style docstring sections found (use numpy style):\n" + "\n".join(hits)
