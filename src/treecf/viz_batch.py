"""Batch-level counterfactual visualizations. matplotlib lives behind the [viz] extra.

Every function consumes a ``BatchResult``. ``k=0`` (the default) keeps each
row's best plan; ``k=None`` keeps every feasible plan, so shares are per plan,
not per row.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from treecf._errors import TreecfError
from treecf.batch import BatchRecord, BatchResult
from treecf.viz import _change_effort, _import_pyplot


def plot_batch_levers(
    batch: BatchResult,
    k: int | None = 0,
    normalize: bool = True,
    top_n: int = 20,
    show_essential: bool = True,
    ax: Any = None,
) -> Any:
    """Horizontal stacked bars: share of plans changing each feature, by direction.

    Increases, decreases, and NaN transitions stack per feature, ordered by how
    often the feature is used. For ``diversity="lever-blocking"`` results,
    features recorded as essential levers are annotated with their count.
    """
    plt = _import_pyplot()
    selected = _select_records(batch, k)
    increase: Counter[str] = Counter()
    decrease: Counter[str] = Counter()
    to_nan: Counter[str] = Counter()
    for record in selected:
        for name, (source, dest) in record.changes.items():
            if math.isnan(source) or math.isnan(dest):
                to_nan[name] += 1
            elif dest > source:
                increase[name] += 1
            else:
                decrease[name] += 1
    total = increase + decrease + to_nan
    order = sorted(total, key=lambda name: (-total[name], name))[:top_n]
    scale = 1.0 / len(selected) if normalize else 1.0

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.6 * max(2, len(order))))
    positions = range(len(order))
    left = [0.0] * len(order)
    parts = [("increase", increase, "tab:orange"), ("decrease", decrease, "tab:blue"),
             ("NaN", to_nan, "tab:gray")]
    for label, counter, color in parts:
        widths = [counter[name] * scale for name in order]
        if not any(widths):
            continue
        ax.barh(positions, widths, left=left, height=0.6, color=color, label=label)
        left = [acc + w for acc, w in zip(left, widths, strict=True)]

    essential: Counter[str] = Counter()
    if show_essential and batch.diversity == "lever-blocking":
        essential = Counter(
            lever for levers in batch.essential_levers.values() for lever in levers
        )
    for i, name in enumerate(order):
        if essential[name]:
            ax.annotate(
                f"essential ×{essential[name]}", xy=(left[i], i), xytext=(4, 0),
                textcoords="offset points", va="center", color="tab:red", fontsize=9,
            )
    ax.set_yticks(positions, order)
    ax.invert_yaxis()
    ax.set_xlabel("fraction of plans" if normalize else "plans")
    ax.set_title(f"levers used across {len(selected)} plan(s)")
    ax.legend(loc="best")
    return ax


def plot_batch_matrix(
    batch: BatchResult,
    explainer: Any = None,
    k: int | None = 0,
    sort_rows: bool = True,
    max_row_labels: int = 30,
    ax: Any = None,
) -> Any:
    """Plans × features heatmap: binary changes, or effort-shaded with an explainer.

    With ``explainer``, each cell shows the change's effort ``w·|Δ|/σ`` (NaN
    legs priced via ``AllowMissing``); without, cells mark changed features
    like ``plot_counterfactuals``. Rows sort by distance; columns by how often
    the feature is changed.
    """
    plt = _import_pyplot()
    import numpy as np

    if explainer is not None and tuple(explainer.ir.feature_names) != batch.feature_names:
        raise TreecfError("explainer and batch describe different feature spaces")
    selected = _select_records(batch, k)
    if sort_rows:
        selected.sort(key=lambda record: record.distance or 0.0)
    frequency = Counter(name for record in selected for name in record.changes)
    features = sorted(frequency, key=lambda name: (-frequency[name], name))

    matrix = np.zeros((len(selected), len(features)))
    for i, record in enumerate(selected):
        row_values = (
            {name: 1.0 for name in record.changes}
            if explainer is None
            else _change_effort(explainer, record.changes)
        )
        for jf, name in enumerate(features):
            matrix[i, jf] = row_values.get(name, 0.0)

    if ax is None:
        height = 0.8 + min(0.3 * max(2, len(selected)), 6.0)
        _, ax = plt.subplots(figsize=(1.0 + 0.8 * len(features), height))
    if explainer is None:
        vmax = 1.0
    else:
        # robust ceiling: one extreme change must not wash out the rest
        positive = matrix[matrix > 0]
        vmax = max(float(np.percentile(positive, 95)) if positive.size else 0.0, 1e-12)
    ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0.0, vmax=vmax)
    ax.set_xticks(range(len(features)), features, rotation=45, ha="right")
    if len(selected) <= max_row_labels:
        labels = [
            f"{r.id} (J={r.distance:.3g})" + (f" k={r.k}" if k is None else "")
            for r in selected
        ]
        ax.set_yticks(range(len(selected)), labels)
    else:
        ax.set_yticks([])
        ax.set_ylabel(f"{len(selected)} plans")
    ax.set_title(
        "effort per change (w·|Δ|/σ)" if explainer is not None else "changed features per plan"
    )
    return ax


def plot_batch_summary(batch: BatchResult, k: int | None = 0, axs: Any = None) -> Any:
    """Three-panel batch overview: plan cost, sparsity, and feasibility.

    Creates its own figure when ``axs`` is None and returns the array of three
    axes (unlike the single-axes functions, which return one ``ax``).
    """
    plt = _import_pyplot()
    ids_all = {record.id for record in batch.records}
    if not ids_all:
        raise TreecfError("empty batch")
    ids_ok = {record.id for record in batch.records if record.feasible}
    selected = [r for r in batch.records if r.feasible and (k is None or r.k == k)]

    own_figure = axs is None
    if own_figure:
        _, axs = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    distances = [record.distance for record in selected if record.distance is not None]
    if distances:
        axs[0].hist(distances, bins="auto", color="tab:blue")
    else:
        axs[0].text(
            0.5, 0.5, "no feasible plans", ha="center", va="center",
            transform=axs[0].transAxes, color="tab:red",
        )
    axs[0].set_xlabel("distance J")
    axs[0].set_title("plan cost")

    sparsity = Counter(record.n_changed for record in selected)
    counts = sorted((n, c) for n, c in sparsity.items() if n is not None)
    if counts:
        axs[1].bar([n for n, _ in counts], [c for _, c in counts], color="tab:blue")
        axs[1].set_xticks([n for n, _ in counts])
    axs[1].set_xlabel("features changed")
    axs[1].set_title("sparsity")

    axs[2].bar(
        ["feasible", "infeasible"],
        [len(ids_ok), len(ids_all) - len(ids_ok)],
        color=["tab:blue", "tab:red"],
    )
    axs[2].set_title(f"{len(ids_ok) / len(ids_all):.0%} of rows solvable")

    if own_figure:
        axs[0].figure.suptitle(
            f"batch summary — {len(ids_all)} rows, diversity={batch.diversity!r}"
        )
    return axs


def plot_batch_deltas(
    batch: BatchResult,
    explainer: Any = None,
    k: int | None = 0,
    top_n: int = 10,
    ax: Any = None,
) -> Any:
    """Strip plot of actual deltas (to − from) per feature, top-N most-changed.

    One jittered dot per plan, a median tick per feature; NaN transitions are
    counted in a per-feature annotation instead of plotted. With ``explainer``,
    deltas are divided by the per-feature normalizer sigma so features of
    different scales share one axis.
    """
    plt = _import_pyplot()
    import numpy as np

    if explainer is not None and tuple(explainer.ir.feature_names) != batch.feature_names:
        raise TreecfError("explainer and batch describe different feature spaces")
    selected = _select_records(batch, k)
    sigma = {name: 1.0 for name in batch.feature_names}
    if explainer is not None:
        sigma = dict(zip(batch.feature_names, (float(s) for s in explainer.sigma), strict=True))
    deltas: dict[str, list[float]] = {}
    nan_counts: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    for record in selected:
        for name, (source, dest) in record.changes.items():
            totals[name] += 1
            if math.isnan(source) or math.isnan(dest):
                nan_counts[name] += 1
            else:
                deltas.setdefault(name, []).append((dest - source) / sigma[name])
    order = sorted(totals, key=lambda name: (-totals[name], name))[:top_n]

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.6 * max(2, len(order))))
    rng = np.random.default_rng(0)  # fixed jitter: figures stay deterministic
    for i, name in enumerate(order):
        values = deltas.get(name, [])
        if values:
            jitter = rng.uniform(-0.15, 0.15, len(values))
            ax.plot(values, i + jitter, "o", color="tab:blue", alpha=0.6, markersize=4)
            ax.plot([float(np.median(values))], [i], "|", color="tab:orange", markersize=14)
        if nan_counts[name]:
            ax.annotate(
                f"→NaN ×{nan_counts[name]}", xy=(1.0, i),
                xycoords=("axes fraction", "data"), xytext=(-4, 0),
                textcoords="offset points", ha="right", va="center",
                color="tab:red", fontsize=9,
            )
    ax.axvline(0.0, color="0.6", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(order)), order)
    ax.invert_yaxis()
    ax.set_xlabel("delta (to − from)" if explainer is None else "standardized delta (Δ/σ)")
    ax.set_title(f"how far the levers move ({len(selected)} plan(s))")
    return ax


def _select_records(batch: BatchResult, k: int | None) -> list[BatchRecord]:
    """Feasible records; k=0 keeps each row's best plan, None keeps all plans."""
    selected = [r for r in batch.records if r.feasible and (k is None or r.k == k)]
    if not selected:
        raise TreecfError("no feasible plans to plot")
    return selected
