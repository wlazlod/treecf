"""Counterfactual visualizations (D15). matplotlib lives behind the [viz] extra."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from treecf._errors import MissingExtraError
from treecf.api import Counterfactual, Infeasible


def plot_changes(cf: Counterfactual, ax: Any = None) -> Any:
    """Dumbbell chart of per-feature changes (from -> to); NaN transitions annotated."""
    plt = _import_pyplot()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.6 * max(2, len(cf.changes))))
    names = list(cf.changes)
    labeled = False
    for i, name in enumerate(names):
        source, target = cf.changes[name]
        if math.isnan(target) or math.isnan(source):
            anchor = source if math.isnan(target) else target
            ax.plot([anchor], [i], "o", color="tab:gray")
            ax.annotate(
                "-> NaN" if math.isnan(target) else "NaN ->",
                xy=(anchor, i),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                color="tab:red",
            )
            continue
        ax.plot([source, target], [i, i], "-", color="tab:gray", zorder=1)
        factual_label = None if labeled else "factual"
        cf_label = None if labeled else "counterfactual"
        ax.plot([source], [i], "o", color="tab:gray", label=factual_label)
        ax.plot([target], [i], "o", color="tab:blue", label=cf_label)
        labeled = True
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("feature value")
    ax.set_title(f"{cf.n_changed} change(s), distance {cf.distance:.3g} ({cf.proof})")
    if labeled:
        ax.legend(loc="best")
    return ax


def plot_counterfactuals(results: Sequence[Counterfactual], ax: Any = None) -> Any:
    """Changed-feature matrix comparing diverse counterfactuals (§8.3)."""
    plt = _import_pyplot()
    features = sorted({name for cf in results for name in cf.changes})
    if ax is None:
        _, ax = plt.subplots(figsize=(1.0 + 0.8 * len(features), 0.8 + 0.5 * len(results)))
    matrix = [[1.0 if f in cf.changes else 0.0 for f in features] for cf in results]
    ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(features)), features, rotation=45, ha="right")
    ax.set_yticks(
        range(len(results)),
        [f"#{i + 1} (J={cf.distance:.3g})" for i, cf in enumerate(results)],
    )
    ax.set_title("changed features per counterfactual")
    return ax


def plot_ladder(bands_result: Mapping[str, object], ax: Any = None) -> Any:
    """Cost of reaching each rating band (Target.bands): the price of every grade."""
    plt = _import_pyplot()
    if ax is None:
        _, ax = plt.subplots(figsize=(1.5 + 0.9 * len(bands_result), 4))
    names = list(bands_result)
    heights = []
    for name in names:
        outcome = bands_result[name]
        heights.append(outcome.distance if isinstance(outcome, Counterfactual) else 0.0)
    bars = ax.bar(names, heights, color="tab:blue")
    for bar, name in zip(bars, names, strict=True):
        outcome = bands_result[name]
        if isinstance(outcome, Infeasible):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.02,
                "infeasible",
                ha="center",
                va="bottom",
                rotation=90,
                color="tab:red",
            )
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel("distance J")
    ax.set_title("cost of reaching each band")
    return ax


def _import_pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise MissingExtraError(
            "visualization requires matplotlib: pip install treecf[viz]"
        ) from exc
    return plt
