"""Package-level invariants: version mirroring and lazy-import discipline."""

import subprocess
import sys
from importlib.metadata import version

import treecf

OPTIONAL_MODULES = ("xgboost", "lightgbm", "catboost", "sklearn", "matplotlib")


def test_version_matches_metadata() -> None:
    assert treecf.__version__ == version("treecf")


def test_import_pulls_no_optional_dependencies() -> None:
    """`import treecf` must work with numpy alone: no optional module may load eagerly."""
    code = (
        "import sys; import treecf; "
        f"loaded = [m for m in {OPTIONAL_MODULES!r} if m in sys.modules]; "
        "assert not loaded, f'eagerly imported: {loaded}'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
