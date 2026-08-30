"""Set the release version in every location that carries it, in one operation.

Usage: ``uv run python scripts/bump_version.py NEW [--date YYYY-MM-DD] [--allow-dirty]``

Rewrites ``pyproject.toml``, ``src/treecf/__init__.py``, ``rust/Cargo.toml``,
``CITATION.cff`` (version and date-released), promotes ``## [Unreleased]`` in
``CHANGELOG.md`` to a dated heading, adds an "Added in NEW" section to
``docs/api-stability.md`` when that page exists, then refreshes both lockfiles.
Idempotent: rerunning with the same version changes nothing.
``tests/test_version_everywhere.py`` is the matching consistency check.
"""

from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _sub_once(path: Path, pattern: str, replacement: str, *, required: bool = True) -> bool:
    text = path.read_text(encoding="utf-8")
    new_text, n = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if n == 0:
        if required:
            raise SystemExit(f"{path}: pattern not found: {pattern!r}")
        return False
    if new_text != text:
        path.write_text(new_text, encoding="utf-8")
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="new version, e.g. 0.3.0")
    parser.add_argument("--date", default=datetime.date.today().isoformat())
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    if not re.fullmatch(r"\d+\.\d+\.\d+", args.version):
        raise SystemExit(f"not a X.Y.Z version: {args.version!r}")
    datetime.date.fromisoformat(args.date)

    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    if dirty and not args.allow_dirty:
        raise SystemExit("working tree is dirty; commit or stash first (or pass --allow-dirty)")

    version, day = args.version, args.date
    changed = []

    if _sub_once(REPO / "pyproject.toml", r'^version = "[^"]+"$', f'version = "{version}"'):
        changed.append("pyproject.toml")
    if _sub_once(
        REPO / "src/treecf/__init__.py",
        r'^__version__ = "[^"]+"$',
        f'__version__ = "{version}"',
    ):
        changed.append("src/treecf/__init__.py")
    if _sub_once(REPO / "rust/Cargo.toml", r'^version = "[^"]+"$', f'version = "{version}"'):
        changed.append("rust/Cargo.toml")
    citation = REPO / "CITATION.cff"
    if _sub_once(citation, r"^version: \S+$", f"version: {version}"):
        changed.append("CITATION.cff")
    if (
        _sub_once(citation, r"^date-released: \S+$", f"date-released: {day}")
        and "CITATION.cff" not in changed
    ):
        changed.append("CITATION.cff")

    changelog = REPO / "CHANGELOG.md"
    heading = f"## [{version}] - {day}"
    if heading not in changelog.read_text(encoding="utf-8"):
        if _sub_once(
            changelog,
            r"^## \[Unreleased\]$",
            f"## [Unreleased]\n\n{heading}",
            required=False,
        ):
            changed.append("CHANGELOG.md")
        else:
            raise SystemExit("CHANGELOG.md has neither an [Unreleased] section nor the heading")

    stability = REPO / "docs/api-stability.md"
    if stability.is_file():
        text = stability.read_text(encoding="utf-8")
        section = f"## Added in {version}"
        if section not in text:
            match = re.search(r"^## Added in \d+\.\d+\.\d+$", text, re.MULTILINE)
            if match is None:
                raise SystemExit("docs/api-stability.md has no 'Added in' section to anchor on")
            inserted = f"{section}\n\n- (nothing listed yet)\n\n"
            stability.write_text(
                text[: match.start()] + inserted + text[match.start() :],
                encoding="utf-8",
            )
            changed.append("docs/api-stability.md")

    if changed:
        subprocess.run(
            ["cargo", "update", "-p", "treecf-core", "--manifest-path", "rust/Cargo.toml"],
            cwd=REPO,
            check=True,
        )
        subprocess.run(["uv", "lock"], cwd=REPO, check=True)
        changed += ["rust/Cargo.lock", "uv.lock"]
        print(f"set {version} ({day}) in: " + ", ".join(changed))
    else:
        print(f"already at {version} ({day}); nothing to do")
    return 0


if __name__ == "__main__":
    sys.exit(main())
