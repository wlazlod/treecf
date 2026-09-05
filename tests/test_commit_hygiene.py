"""The commit-message checker: what it refuses and what it lets through."""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "check_commit_messages.py"
_spec = importlib.util.spec_from_file_location("check_commit_messages", _SCRIPT)
assert _spec is not None and _spec.loader is not None
checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checker)


@pytest.mark.parametrize(
    "message",
    [
        "feat: opt-in branch-and-refine mode for the exact backend",
        "fix: parse a truncated dump as ParserError\n\nThe parser used to raise KeyError.",
        "ci: pin xgboost 3.1 with pandas below 3 (commit a1b2c3d)",
        "perf: 3x faster on the 200-tree case; the default colour cycle stays",
        "docs: describe the k1 tuning knob and the 2d layout",
    ],
)
def test_behavioral_messages_pass(message: str) -> None:
    assert checker.violations(message) == []


@pytest.mark.parametrize(
    "message",
    [
        "feat: implement RELEASE_SPEC section on regions",
        "feat: regions as described in SPEC.md",
        "feat: maximal regions per the spec",
        "chore: close work item for the report",
        "feat: trace plot, as decided in review",
        "chore: groomed backlog before the release",
        "feat: R4 additive keys policy",
        "feat: implement X1 and W2",
    ],
)
def test_planning_vocabulary_is_refused(message: str) -> None:
    assert checker.violations(message)


def test_repository_history_on_this_branch_passes() -> None:
    """Every commit between the default branch and HEAD, when the default
    branch is reachable; skipped in a shallow checkout."""
    import subprocess

    probe = subprocess.run(
        ["git", "merge-base", "main", "HEAD"], capture_output=True, text=True,
        cwd=_SCRIPT.parents[1],
    )
    if probe.returncode != 0:
        pytest.skip("no reachable main branch")
    base = probe.stdout.strip()
    for short, message in checker.commit_messages(base, "HEAD"):
        assert checker.violations(message) == [], f"{short}: {message!r}"
