"""Package-level invariants: version mirroring and lazy-import discipline."""

import re
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest

import treecf

OPTIONAL_MODULES = ("xgboost", "lightgbm", "catboost", "sklearn", "matplotlib")


def test_version_matches_metadata() -> None:
    assert treecf.__version__ == version("treecf")


def test_citation_version_matches() -> None:
    cff = Path(__file__).resolve().parents[1] / "CITATION.cff"
    if not cff.exists():
        pytest.skip("CITATION.cff not present (installed-wheel context)")
    match = re.search(r"^version:\s*(\S+)\s*$", cff.read_text(encoding="utf-8"), re.MULTILINE)
    assert match, "CITATION.cff has no version line"
    assert match.group(1) == treecf.__version__


def test_import_pulls_no_optional_dependencies() -> None:
    """`import treecf` must work with numpy alone: no optional module may load eagerly."""
    code = (
        "import sys; import treecf; "
        f"loaded = [m for m in {OPTIONAL_MODULES!r} if m in sys.modules]; "
        "assert not loaded, f'eagerly imported: {loaded}'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
