"""Tracked text describes current behavior; release history belongs in CHANGELOG.md.

Version literals are allowed only where they are data: the changelog, the API
stability page's "Added in" lists, benchmark provenance (the exact library
versions a measurement was taken with), schema identifiers, and fixture
metadata recording what a golden file was captured from.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SCANNED_SUFFIXES = {".py", ".rs", ".md"}
SCANNED_ROOTS = ("src/", "rust/src/", "tests/", "docs/")

# Files where version strings are data, not narration.
ALLOWED_FILES = {
    "docs/api-stability.md",  # "Added in X" lists — a changelog-like page by convention
    "docs/changelog.md",  # renders CHANGELOG.md
    "docs/concepts/backends.md",  # benchmark provenance: versions a measurement used
    "tests/test_no_release_narration.py",
    "tests/test_version_everywhere.py",  # parses the version locations
}
ALLOWED_DIRS = (
    "docs/benchmarks/",  # generated from measured runs; carries version stamps
    "tests/fixtures/",  # fixture metadata records the release a fixture was captured from
)

# Lines where a version-shaped token is data even in an otherwise scanned file.
ALLOWED_LINE = re.compile(r"__version__|schema_version")

VERSION_LITERAL = re.compile(r"\b0\.\d+\.\d+\b")
RELEASE_SCOPING = re.compile(r"\bv0\.\d+\b")
WORKITEM_ID = re.compile(r"\b[TCSDVG]\d\b")
# matplotlib's colour-cycle strings ("C0".."C9") are code, not shorthand
_QUOTED_COLOR = re.compile(r"""["']C\d["']""")


def _tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", *SCANNED_ROOTS],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [Path(line) for line in out.splitlines() if Path(line).suffix in SCANNED_SUFFIXES]


def _violations(pattern: re.Pattern[str], files: list[Path]) -> list[str]:
    hits = []
    for rel in files:
        if str(rel) in ALLOWED_FILES or str(rel).startswith(ALLOWED_DIRS):
            continue
        text = (REPO / rel).read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if ALLOWED_LINE.search(line):
                continue
            if pattern.search(_QUOTED_COLOR.sub("", line)):
                hits.append(f"{rel}:{lineno}: {line.strip()}")
    return hits


def test_no_version_literals() -> None:
    files = _tracked_files()
    hits = _violations(VERSION_LITERAL, files) + _violations(RELEASE_SCOPING, files)
    assert not hits, "version literals belong in CHANGELOG.md, not here:\n" + "\n".join(hits)


def test_no_workitem_ids_in_python() -> None:
    files = [f for f in _tracked_files() if f.suffix == ".py"]
    hits = _violations(WORKITEM_ID, files)
    assert not hits, "work-item shorthand does not belong in tracked text:\n" + "\n".join(hits)
