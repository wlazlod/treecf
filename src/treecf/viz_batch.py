"""Batch-level counterfactual visualizations. matplotlib lives behind the [viz] extra.

Every function consumes a ``BatchResult``. ``k=0`` (the default) keeps each
row's best plan; ``k=None`` keeps every feasible plan, so shares are per plan,
not per row.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
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

    Args:
        batch: The batch result to summarize.
        k: Which plan(s) to include per row — ``0`` (the default) keeps only
            each row's best plan; ``None`` keeps every feasible plan.
        normalize: When ``True`` (the default), bar widths are a fraction of
            the selected plans; when ``False``, raw plan counts.
        top_n: Maximum number of features to show, most-used first.
        show_essential: When ``True`` (the default) and
            ``batch.diversity == "lever-blocking"``, annotates each bar with
            how many rows recorded that feature as an essential lever
            (``batch.essential_levers``).
        ax: Existing axes to draw on; a new figure is created if omitted.

    Returns:
        The axes the chart was drawn on.

    Raises:
        MissingExtraError: If matplotlib is not installed.
        TreecfError: If ``batch`` has no plan matching ``k``.
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

    Args:
        batch: The batch result to visualize.
        explainer: When given, shades cells by change effort instead of a
            flat binary mark; must describe the same feature space as
            ``batch``.
        k: Which plan(s) to include per row — ``0`` (the default) keeps only
            each row's best plan; ``None`` keeps every feasible plan.
        sort_rows: When ``True`` (the default), rows are ordered by ascending
            distance.
        max_row_labels: Row id labels are drawn only when the selected plan
            count is at or below this limit; beyond it, the y-axis is left
            unlabeled with a plan-count caption instead.
        ax: Existing axes to draw on; a new figure is created if omitted.

    Returns:
        The axes the heatmap was drawn on.

    Raises:
        MissingExtraError: If matplotlib is not installed.
        TreecfError: If ``explainer`` is given and its feature space does not
            match ``batch.feature_names``, or if ``batch`` has no plan
            matching ``k``.
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
    axes (unlike the single-axes functions, which return one ``ax``). Panels:
    a histogram of ``distance`` over the selected plans, a bar chart of
    ``n_changed`` counts, and a feasible-vs-infeasible bar over every row
    (independent of ``k`` — every row counts once).

    Args:
        batch: The batch result to summarize.
        k: Which plan(s) feed the cost/sparsity panels — ``0`` (the default)
            keeps only each row's best plan; ``None`` keeps every feasible
            plan. Does not affect the feasibility panel.
        axs: Existing array of 3 axes to draw on; a new figure is created if
            omitted.

    Returns:
        The array of 3 axes (cost, sparsity, feasibility) the panels were
        drawn on.

    Raises:
        MissingExtraError: If matplotlib is not installed.
        TreecfError: If ``batch`` has no records at all.
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

    Args:
        batch: The batch result to visualize.
        explainer: When given, standardizes deltas by its per-feature
            ``sigma``; must describe the same feature space as ``batch``.
            Without it, raw deltas are plotted.
        k: Which plan(s) to include per row — ``0`` (the default) keeps only
            each row's best plan; ``None`` keeps every feasible plan.
        top_n: Maximum number of features to show, most-changed first.
        ax: Existing axes to draw on; a new figure is created if omitted.

    Returns:
        The axes the strip plot was drawn on.

    Raises:
        MissingExtraError: If matplotlib is not installed.
        TreecfError: If ``explainer`` is given and its feature space does not
            match ``batch.feature_names``, or if ``batch`` has no plan
            matching ``k``.
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


def recourse_burden_table(
    batch: BatchResult,
    groups: Sequence[object],
    *,
    group_order: Sequence[object] | None = None,
    min_group_size: int = 10,
) -> list[dict[str, object]]:
    """Recourse cost and availability by segment, one dict per group.

    ``groups`` assigns a segment label to every input row of the batch, in the
    order the rows were solved (one label per distinct id). A row's burden is
    its cheapest feasible plan's ``distance``; a row with no feasible plan has
    no burden and counts toward ``certified_no_share`` when every infeasibility
    marker it carries is certified, else ``unproven_no_share`` — an exhausted
    search is not a proven "no".

    Burden compares costs under one declared cost model and constraint set; a
    disparity between groups is a finding to investigate, not a fairness
    verdict — which metric matters is a choice this table does not make.

    Parameters
    ----------
    batch : BatchResult
        The batch whose rows are being segmented.
    groups : sequence
        One segment label per input row, aligned with the batch's row order.
    group_order : sequence, optional
        The groups to report, in order; defaults to the sorted labels.
    min_group_size : int
        Groups smaller than this are flagged ``small``.

    Returns
    -------
    list of dict
        Per group: ``group``, ``n``, ``recourse_share``,
        ``certified_no_share``, ``unproven_no_share``, ``median_burden``,
        ``mean_burden``, ``p90_burden``, ``small`` (NaN burdens where a group
        has no feasible row).

    Raises
    ------
    TreecfError
        If ``groups`` does not have one label per batch row.
    """
    import numpy as np

    rows = _rows_with_burdens(batch)
    if len(groups) != len(rows):
        raise TreecfError(
            f"groups has {len(groups)} labels but the batch has {len(rows)} rows"
        )
    by_group: dict[object, list[tuple[float | None, bool]]] = {}
    for label, (_row_id, burden, certified) in zip(groups, rows, strict=True):
        by_group.setdefault(label, []).append((burden, certified))
    order = list(group_order) if group_order is not None else sorted(by_group, key=str)

    table: list[dict[str, object]] = []
    for label in order:
        members = by_group.get(label, [])
        n = len(members)
        burdens = np.array([b for b, _ in members if b is not None], dtype=np.float64)
        certified_no = sum(1 for b, certified in members if b is None and certified)
        unproven_no = sum(1 for b, certified in members if b is None and not certified)
        table.append(
            {
                "group": label,
                "n": n,
                "recourse_share": len(burdens) / n if n else math.nan,
                "certified_no_share": certified_no / n if n else math.nan,
                "unproven_no_share": unproven_no / n if n else math.nan,
                "median_burden": float(np.median(burdens)) if len(burdens) else math.nan,
                "mean_burden": float(np.mean(burdens)) if len(burdens) else math.nan,
                "p90_burden": float(np.percentile(burdens, 90)) if len(burdens) else math.nan,
                "small": n < min_group_size,
            }
        )
    return table


def _rows_with_burdens(batch: BatchResult) -> list[tuple[object, float | None, bool]]:
    """Per distinct row id in first-appearance order: (id, cheapest feasible
    distance or None, every-infeasibility-marker-certified flag)."""
    order: list[object] = []
    per_row: dict[object, list[Any]] = {}
    for record in batch.records:
        if record.id not in per_row:
            order.append(record.id)
            per_row[record.id] = []
        per_row[record.id].append(record)
    rows: list[tuple[object, float | None, bool]] = []
    for row_id in order:
        records = per_row[row_id]
        feasible = [r.distance for r in records if r.feasible and r.distance is not None]
        markers = [r.proof for r in records if not r.feasible]
        if feasible:
            rows.append((row_id, min(feasible), False))
        else:
            certified = bool(markers) and all(proof == "certified" for proof in markers)
            rows.append((row_id, None, certified))
    return rows


def plot_recourse_burden(
    batch: BatchResult,
    groups: Sequence[object],
    *,
    axes: Any = None,
    group_order: Sequence[object] | None = None,
    min_group_size: int = 10,
    stat: str = "median",
) -> Any:
    """Who pays for recourse, and who has none: burden and availability by segment.

    Panel A draws one burden ECDF per group among the rows that have recourse
    (colour and linestyle both cycle, so groups stay tellable apart without
    colour); panel B stacks each group's availability into *has recourse*,
    *certified no recourse*, and *unproven no recourse* — the last hatched,
    because an exhausted search is not a proven "no" and the eye must not
    merge the two.

    Burden compares costs under one declared cost model and constraint set; a
    disparity between groups is a finding to investigate, not a fairness
    verdict — which metric matters is a choice this plot does not make.
    ``recourse_burden_table`` exposes the numbers behind the picture.

    Parameters
    ----------
    batch : BatchResult
        The batch whose rows are being segmented.
    groups : sequence
        One segment label per input row, aligned with the batch's row order.
    axes : array of two matplotlib axes, optional
        Target axes; a 1x2 figure is created when omitted.
    group_order : sequence, optional
        The groups to draw, in order; defaults to the sorted labels.
    min_group_size : int
        Groups smaller than this get ``" — small"`` appended in the legend.
    stat : {"median", "mean", "p90"}
        Which burden statistic the legend reports per group.

    Returns
    -------
    array of the two axes.

    Raises
    ------
    MissingExtraError
        If matplotlib is not installed.
    TreecfError
        If ``groups`` is misaligned or ``stat`` is unrecognized.
    """
    plt = _import_pyplot()
    import numpy as np

    stat_key = {"median": "median_burden", "mean": "mean_burden", "p90": "p90_burden"}
    if stat not in stat_key:
        raise TreecfError(f"unknown stat {stat!r}; use 'median', 'mean', or 'p90'")
    table = recourse_burden_table(
        batch, groups, group_order=group_order, min_group_size=min_group_size
    )
    rows = _rows_with_burdens(batch)
    by_group: dict[object, list[tuple[float | None, bool]]] = {}
    for label, (_row_id, burden, certified) in zip(groups, rows, strict=True):
        by_group.setdefault(label, []).append((burden, certified))

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11, 4))
    ecdf_ax, avail_ax = axes[0], axes[1]

    linestyles = ("-", "--", "-.", ":")
    for i, entry in enumerate(table):
        label = entry["group"]
        burdens = np.sort(
            np.array(
                [b for b, _ in by_group.get(label, []) if b is not None],
                dtype=np.float64,
            )
        )
        if len(burdens) == 0:
            continue
        y = np.arange(1, len(burdens) + 1) / len(burdens)
        suffix = " — small" if entry["small"] else ""
        value = entry[stat_key[stat]]
        ecdf_ax.step(
            burdens,
            y,
            where="post",
            color=f"C{i % 10}",
            linestyle=linestyles[i % len(linestyles)],
            label=f"{label} (n={entry['n']}, {stat} J={value:.3g}){suffix}",
        )
    ecdf_ax.set_xlabel("recourse cost J")
    ecdf_ax.set_ylabel("share of rows with recourse")
    ecdf_ax.set_ylim(0.0, 1.05)
    ecdf_ax.legend(fontsize=7, frameon=False)
    ecdf_ax.set_title("burden among rows with recourse")

    positions = np.arange(len(table), dtype=np.float64)
    for i, entry in enumerate(table):
        n = int(entry["n"])  # type: ignore[call-overload]
        members = by_group.get(entry["group"], [])
        has = sum(1 for b, _ in members if b is not None)
        certified_no = sum(1 for b, certified in members if b is None and certified)
        unproven_no = n - has - certified_no
        shares = [
            (has, "has recourse", "C0", None),
            (certified_no, "certified no recourse", "C3", None),
            (unproven_no, "unproven no recourse", "0.7", "///"),
        ]
        bottom = 0.0
        for count, seg_label, color, hatch in shares:
            share = count / n if n else 0.0
            avail_ax.bar(
                positions[i], share, bottom=bottom, width=0.7, color=color,
                hatch=hatch, edgecolor="white",
                label=seg_label if i == 0 else "_nolegend_",
            )
            if count:
                avail_ax.annotate(
                    str(count), xy=(positions[i], bottom + share / 2),
                    ha="center", va="center", fontsize=7, color="white" if hatch is None else "0.2",
                )
            bottom += share
        avail_ax.annotate(
            f"n={n}", xy=(positions[i], 1.02), ha="center", fontsize=7, color="0.35"
        )
    avail_ax.set_xticks(positions)
    avail_ax.set_xticklabels([str(entry["group"]) for entry in table])
    avail_ax.set_ylim(0.0, 1.12)
    avail_ax.set_ylabel("share of rows")
    avail_ax.legend(fontsize=7, frameon=False, loc="lower right")
    avail_ax.set_title("recourse availability")
    return axes


def _select_records(batch: BatchResult, k: int | None) -> list[BatchRecord]:
    """Feasible records; k=0 keeps each row's best plan, None keeps all plans."""
    selected = [r for r in batch.records if r.feasible and (k is None or r.k == k)]
    if not selected:
        raise TreecfError("no feasible plans to plot")
    return selected
