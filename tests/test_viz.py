"""Viz smoke tests (D15): figures render on Agg with the expected structure."""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from treecf import Counterfactual, Infeasible  # noqa: E402
from treecf.viz import plot_changes, plot_counterfactuals, plot_ladder  # noqa: E402


def _cf(changes: dict[str, tuple[float, float]], distance: float) -> Counterfactual:
    return Counterfactual(
        x_cf=np.zeros(3),
        changes=changes,
        distance=distance,
        n_changed=len(changes),
        score_raw=0.5,
        score_prob=None,
        proof="optimal",
    )


def test_plot_changes_renders_dumbbells() -> None:
    cf = _cf({"income": (1000.0, 2500.0), "dpd": (5.0, 0.0)}, distance=1.4)
    ax = plot_changes(cf)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert set(labels) == {"income", "dpd"}


def test_plot_changes_marks_nan_transitions() -> None:
    cf = _cf({"months": (7.0, float("nan"))}, distance=0.3)
    ax = plot_changes(cf)
    texts = [t.get_text() for t in ax.texts]
    assert any("NaN" in t for t in texts)


def test_plot_counterfactuals_matrix() -> None:
    results = [
        _cf({"a": (0.0, 1.0)}, 1.0),
        _cf({"b": (0.0, 2.0), "c": (1.0, 0.0)}, 2.0),
    ]
    ax = plot_counterfactuals(results)
    assert len(ax.get_xticklabels()) == 3  # union of changed features


def test_plot_ladder_costs_and_infeasible() -> None:
    ladder = {
        "C": _cf({"a": (0.0, 1.0)}, 0.5),
        "B": _cf({"a": (0.0, 2.0)}, 1.5),
        "A": Infeasible(reason="unreachable"),
    }
    ax = plot_ladder(ladder)
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["C", "B", "A"]
    texts = " ".join(t.get_text() for t in ax.texts)
    assert "infeasible" in texts.lower()
