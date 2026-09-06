"""Check that commit messages describe behavior, not internal planning.

Usage: ``python scripts/check_commit_messages.py BASE HEAD``

Reads the subject and body of every commit in ``BASE..HEAD`` (message text
only — never hashes or diffs) and fails when one mentions an internal
planning document, work-item shorthand, or decision-record language. The
history should read as a record of what the code does; see CONTRIBUTING.md.
"""

from __future__ import annotations

import re
import subprocess
import sys

# Case-sensitive on purpose: the single-letter-plus-digit shorthand is only a
# match for an uppercase letter, which is what keeps a hex commit hash or a
# unit like "3s" from tripping it.
DENY = (
    re.compile(r"_SPEC"),
    re.compile(r"SPEC\.md"),
    re.compile(r"per the spec", re.IGNORECASE),
    re.compile(r"work item", re.IGNORECASE),
    re.compile(r"as decided", re.IGNORECASE),
    re.compile(r"groom", re.IGNORECASE),
    re.compile(r"\b[A-Z][0-9]\b"),
)

RULE = (
    "Commit messages describe behavior; they must not name planning documents, "
    "work-item shorthand, or decisions taken elsewhere (see CONTRIBUTING.md, PR checklist)."
)


def violations(message: str) -> list[str]:
    """The deny-pattern matches in one commit message, as short excerpts."""
    found: list[str] = []
    for pattern in DENY:
        for match in pattern.finditer(message):
            start = max(match.start() - 20, 0)
            end = min(match.end() + 20, len(message))
            found.append(message[start:end].replace("\n", " ").strip())
    return found


def commit_messages(base: str, head: str) -> list[tuple[str, str]]:
    """``(short hash, full message)`` per commit in ``base..head``."""
    out = subprocess.run(
        ["git", "log", "--format=%h%x00%B%x01", f"{base}..{head}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    entries: list[tuple[str, str]] = []
    for chunk in out.split("\x01"):
        chunk = chunk.strip("\n")
        if not chunk:
            continue
        short, _, message = chunk.partition("\x00")
        entries.append((short.strip(), message))
    return entries


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    base, head = argv[1], argv[2]
    failed = False
    for short, message in commit_messages(base, head):
        for excerpt in violations(message):
            failed = True
            print(f"{short}: {excerpt!r}")
    if failed:
        print(RULE, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
