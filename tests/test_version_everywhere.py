"""Every location the version string lives in must agree.

The release workflow runs this test before building any wheel, so a missed
location fails the release instead of shipping an inconsistency. The single
source of truth is ``treecf.__version__``; ``scripts/bump_version.py`` rewrites
every location in one operation.
"""

from __future__ import annotations

import importlib.metadata
import re
from datetime import date
from pathlib import Path

import pytest

import treecf

REPO = Path(__file__).resolve().parent.parent
VERSION = treecf.__version__


def _read(rel: str) -> str | None:
    path = REPO / rel
    if not path.is_file():
        return None  # installed-wheel context: repo files are absent
    return path.read_text(encoding="utf-8")


def test_version_matches_metadata() -> None:
    assert importlib.metadata.version("treecf") == VERSION


def test_pyproject_version() -> None:
    text = _read("pyproject.toml")
    if text is None:
        pytest.skip("pyproject.toml not present")
    match = re.search(r'^version = "([^"]+)"$', text, re.MULTILINE)
    assert match is not None and match.group(1) == VERSION


def test_cargo_toml_version() -> None:
    text = _read("rust/Cargo.toml")
    if text is None:
        pytest.skip("rust/Cargo.toml not present")
    match = re.search(r'^version = "([^"]+)"$', text, re.MULTILINE)
    assert match is not None and match.group(1) == VERSION


def test_cargo_lock_version() -> None:
    text = _read("rust/Cargo.lock")
    if text is None:
        pytest.skip("rust/Cargo.lock not present")
    match = re.search(r'name = "treecf-core"\nversion = "([^"]+)"', text)
    assert match is not None and match.group(1) == VERSION


def test_uv_lock_version() -> None:
    text = _read("uv.lock")
    if text is None:
        pytest.skip("uv.lock not present")
    match = re.search(r'name = "treecf"\nversion = "([^"]+)"', text)
    assert match is not None and match.group(1) == VERSION


def test_citation_version_and_date() -> None:
    text = _read("CITATION.cff")
    if text is None:
        pytest.skip("CITATION.cff not present")
    version = re.search(r"^version:\s*(\S+)\s*$", text, re.MULTILINE)
    assert version is not None and version.group(1) == VERSION
    released = re.search(r"^date-released:\s*(\S+)\s*$", text, re.MULTILINE)
    assert released is not None
    assert released.group(1) == _changelog_date()


def _changelog_date() -> str:
    text = _read("CHANGELOG.md")
    assert text is not None
    match = re.search(
        rf"^## \[{re.escape(VERSION)}\] - (\d{{4}}-\d{{2}}-\d{{2}})$", text, re.MULTILINE
    )
    assert match is not None, f"CHANGELOG.md has no dated heading for {VERSION}"
    date.fromisoformat(match.group(1))  # a real date, not a placeholder
    return match.group(1)


def test_changelog_heading() -> None:
    if _read("CHANGELOG.md") is None:
        pytest.skip("CHANGELOG.md not present")
    _changelog_date()


def test_api_stability_added_in() -> None:
    text = _read("docs/api-stability.md")
    if text is None:
        pytest.skip("docs/api-stability.md not present")
    assert re.search(rf"^## Added in {re.escape(VERSION)}$", text, re.MULTILINE), (
        f"docs/api-stability.md has no 'Added in {VERSION}' section"
    )
