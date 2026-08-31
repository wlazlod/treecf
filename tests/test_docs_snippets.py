"""Executable-snippet contract for the docs.

Every ```python fenced block in docs/**/*.md (except the changelog) is
executed, in order, in one namespace per page — pre-seeded with the fixed
vocabulary the snippet convention documents in docs/README.md. This turns
the fragment style from a rot risk into a tested contract: a block either
runs against that vocabulary, reuses a name an earlier block on the same
page defined, or is marked "# docs: no-run" as deliberate pseudo-code.
"""

from __future__ import annotations

import math
import pathlib
import re

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from treecf import Explainer, Target  # noqa: E402

pytestmark = pytest.mark.slow

_DOCS_DIR = pathlib.Path(__file__).parent.parent / "docs"
_EXCLUDED_PAGES = {"changelog.md", "README.md"}
_NO_RUN_MARKER = "# docs: no-run"
_CODE_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.S)
_REQUIRES_RE = re.compile(r"<!--\s*docs:\s*requires\s+([^>]*?)\s*-->")

_MODEL_PATH = pathlib.Path(__file__).parent / "fixtures" / "docs_model.json"
OCCUPATIONS = ("student", "clerk", "manager", "retired")


def docs_background(n: int = 400, seed: int = 7) -> np.ndarray:
    """The fixed docs data recipe (documented in docs/README.md)."""
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [
            rng.normal(loc=4200.0, scale=1600.0, size=n),  # income
            np.clip(rng.beta(2.0, 3.5, size=n), 0.0, 1.0),  # utilization
            np.floor(rng.exponential(scale=6.0, size=n)),  # dpd_12m
            np.floor(rng.uniform(3, 240, size=n)),  # tenure_months
            rng.integers(0, 4, size=n).astype(np.float64),  # occupation codes
        ]
    )


class StubCalibrator:
    """A calibrator satisfying the duck protocol without any probcal import:
    a fixed logit shift, monotone by construction."""

    is_monotone_ = True
    _shift = 0.3

    def predict_proba(self, scores):
        scores = np.asarray(scores, dtype=np.float64)
        logit = np.log(scores / (1.0 - scores))
        return 1.0 / (1.0 + np.exp(-(logit + self._shift)))

    def interval_inverse(self, lo, hi, *, space="probability", buffer_logit=0.0):
        def inv(p: float, side: int) -> float:
            if p <= 0.0:
                return -math.inf
            if p >= 1.0:
                return math.inf
            raw = math.log(p / (1.0 - p)) - self._shift
            return raw - side * buffer_logit

        del space  # the stub maps logit space to logit space
        return inv(lo, -1), inv(hi, 1)

    def fingerprint(self) -> str:
        return f"stub-logit-shift-{self._shift}"


def _make_vocabulary() -> dict:
    """The fixed vocabulary documented in docs/README.md."""
    X_bg = docs_background()
    exp = Explainer(
        str(_MODEL_PATH), background=X_bg, categories={"occupation": OCCUPATIONS}
    )
    target = Target.probability(range=(0.0, 0.05))
    x = X_bg[1]
    res = exp.explain(x, target=target, seed=0)
    batch = exp.explain_batch(X_bg[:20], target=target)
    return {
        "np": np,
        "X_bg": X_bg,
        "exp": exp,
        "target": target,
        "x": x,
        "res": res,
        "batch": batch,
        "cal": StubCalibrator(),
        "OCCUPATIONS": OCCUPATIONS,
    }


def _discover_pages() -> list[pathlib.Path]:
    return sorted(p for p in _DOCS_DIR.rglob("*.md") if p.name not in _EXCLUDED_PAGES)


def _required_packages(text: str) -> list[str]:
    names: list[str] = []
    for match in _REQUIRES_RE.findall(text):
        names.extend(name.strip() for name in match.split(",") if name.strip())
    return names


def _extract_blocks(page: pathlib.Path) -> list[str]:
    text = page.read_text(encoding="utf-8")
    blocks = []
    for raw in _CODE_BLOCK_RE.findall(text):
        lines = [line for line in raw.splitlines() if line.strip()]
        if not lines:
            continue
        if "--8<--" in raw:
            continue
        if lines[0].lstrip().startswith(">>>"):
            continue
        if _NO_RUN_MARKER in raw:
            continue
        blocks.append(raw)
    return blocks


_PAGES = _discover_pages()
_PAGE_IDS = [str(p.relative_to(_DOCS_DIR)) for p in _PAGES]


@pytest.mark.parametrize("page", _PAGES, ids=_PAGE_IDS)
def test_docs_page_snippets_execute(page: pathlib.Path, tmp_path, monkeypatch) -> None:
    blocks = _extract_blocks(page)
    if not blocks:
        pytest.skip("no runnable python blocks on this page")
    for package in _required_packages(page.read_text(encoding="utf-8")):
        pytest.importorskip(package)

    # Fresh cwd per page, so save(path=...) writes land in an isolated dir.
    monkeypatch.chdir(tmp_path)
    namespace = _make_vocabulary()
    try:
        for i, block in enumerate(blocks):
            try:
                exec(compile(block, str(page), "exec"), namespace)  # noqa: S102
            except Exception as exc:
                first_line = block.strip().splitlines()[0]
                raise AssertionError(
                    f"{page}: block {i} failed (first line: {first_line!r}): "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
    finally:
        plt.close("all")
