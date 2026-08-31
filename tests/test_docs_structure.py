"""Structural tripwires for the docs site.

These tests pin the promises the docs make as a *site*, independent of any
one page's prose: no previously published URL disappears, the front page
names every capability, every public plot function is shown as a committed
figure with alt text, the workflow guides form a connected path, and the
changelog page resolves to the real changelog.
"""

from __future__ import annotations

import pathlib
import re

import pytest

pytestmark = pytest.mark.slow

REPO = pathlib.Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
MKDOCS = REPO / "mkdocs.yml"

# Every page path the previously released docs site served. Removing or
# renaming any of these breaks published links; moved content must leave a
# page (or a redirect) at the old path.
LEGACY_PAGES = [
    "index.md",
    "getting-started.md",
    "how-it-works.md",
    "concepts/models.md",
    "concepts/targets.md",
    "concepts/calibration.md",
    "concepts/constraints.md",
    "concepts/missing-values.md",
    "concepts/plausibility.md",
    "concepts/coalitions.md",
    "concepts/backends.md",
    "concepts/certification.md",
    "notebooks/01-quickstart.ipynb",
    "notebooks/02-credit-risk-tutorial.ipynb",
    "notebooks/03-no-solver-environments.ipynb",
    "api.md",
    "faq.md",
]

# The front page must name every capability, so a new headline feature that
# skips the front page fails here instead of silently staying invisible.
CAPABILITIES = [
    "constraint",
    "missing",
    "plausibility",
    "batch",
    "coalition",
    "exact",
    "certified infeasibility",
    "region",
    "certificate",
    "categorical",
    "probcal",
]

# Every public plot function must appear as a committed, alt-texted figure
# on at least one prose page.
PLOT_FUNCTIONS = [
    "plot_changes",
    "plot_counterfactuals",
    "plot_ladder",
    "plot_alternatives",
    "plot_tradeoff",
    "plot_recourse_map",
    "plot_waterfall",
    "plot_effort",
    "plot_region",
    "plot_batch_levers",
    "plot_batch_matrix",
    "plot_batch_summary",
    "plot_batch_deltas",
    "plot_recourse_burden",
]

WORKFLOW_PAGES = [
    "guide/models.md",
    "guide/targets.md",
    "guide/constraints.md",
    "guide/explain.md",
    "guide/certify.md",
    "guide/auditability.md",
    "guide/visualize.md",
]

_IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+\.png)\)")
_LINK_RE = re.compile(r"(?<!\!)\[[^\]]+\]\((?P<path>[^)#]+\.md)")


def _mkdocs_text() -> str:
    return MKDOCS.read_text(encoding="utf-8")


def _prose_pages() -> list[pathlib.Path]:
    return [p for p in DOCS.rglob("*.md") if p.name != "README.md"]


def test_no_legacy_url_disappears() -> None:
    nav = _mkdocs_text()
    missing = [page for page in LEGACY_PAGES if not (DOCS / page).is_file()]
    assert not missing, f"previously published docs pages missing from docs/: {missing}"
    unlisted = [page for page in LEGACY_PAGES if page not in nav]
    assert not unlisted, f"previously published pages no longer in mkdocs nav: {unlisted}"


def test_front_page_names_every_capability() -> None:
    text = (DOCS / "index.md").read_text(encoding="utf-8").lower()
    missing = [c for c in CAPABILITIES if c not in text]
    assert not missing, f"front page does not mention: {missing}"


def test_every_plot_function_has_a_committed_alt_texted_figure() -> None:
    embeds: dict[str, list[tuple[pathlib.Path, str]]] = {name: [] for name in PLOT_FUNCTIONS}
    for page in _prose_pages():
        for match in _IMAGE_RE.finditer(page.read_text(encoding="utf-8")):
            stem = pathlib.PurePosixPath(match.group("path")).stem
            base = stem.removesuffix("_schematic")
            if base in embeds:
                embeds[base].append((page, match.group("alt")))
                # The figure file itself must be committed next to the page.
                figure = (page.parent / match.group("path")).resolve()
                assert figure.is_file(), f"{page}: missing figure {match.group('path')}"
                assert match.group("alt").strip(), f"{page}: empty alt text for {stem}"
    missing = [name for name, hits in embeds.items() if not hits]
    assert not missing, f"plot functions without an embedded figure on any prose page: {missing}"


def test_workflow_pages_in_nav_and_cross_linked() -> None:
    nav = _mkdocs_text()
    for page in WORKFLOW_PAGES:
        assert page in nav, f"{page} not in mkdocs nav"
    guide = {pathlib.PurePosixPath(p).name for p in WORKFLOW_PAGES}
    for page in WORKFLOW_PAGES:
        text = (DOCS / page).read_text(encoding="utf-8")
        linked = {
            pathlib.PurePosixPath(m.group("path")).name for m in _LINK_RE.finditer(text)
        } & (guide - {pathlib.PurePosixPath(page).name})
        assert linked, f"{page} links no other workflow guide page"


def test_changelog_page_resolves() -> None:
    include = (DOCS / "changelog.md").read_text(encoding="utf-8")
    match = re.search(r'--8<--\s*"([^"]+)"', include)
    assert match is not None, "docs/changelog.md is not a snippet include"
    assert (REPO / match.group(1)).is_file(), f"included file missing: {match.group(1)}"
