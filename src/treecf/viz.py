"""Counterfactual visualizations. matplotlib lives behind the [viz] extra."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from treecf._errors import MissingExtraError, TreecfError
from treecf.api import Counterfactual, Infeasible

__all__ = [
    "plot_alternatives",
    "plot_certification_trace",
    "plot_changes",
    "plot_counterfactuals",
    "plot_effort",
    "plot_ladder",
    "plot_recourse_map",
    "plot_recourse_menu",
    "plot_region",
    "plot_tradeoff",
    "plot_waterfall",
]


def plot_changes(cf: Counterfactual, ax: Any = None) -> Any:
    """Dumbbell chart of per-feature changes (from -> to); NaN transitions annotated.

    One row per changed feature, a gray dot at the factual value and a blue
    dot at the counterfactual value joined by a line; a feature that
    transitions to or from ``NaN`` is drawn as a single gray dot annotated
    ``"-> NaN"``/``"NaN ->"`` instead.

    Parameters
    ----------
    cf
        The counterfactual to plot.
    ax
        Existing axes to draw on; a new figure is created if omitted.

    Returns
    -------
    The axes the chart was drawn on.

    Raises
    ------
    MissingExtraError
        If matplotlib is not installed.
    """
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
    """Changed-feature matrix comparing diverse counterfactuals.

    One row per result, one column per feature changed by any of them; a
    filled cell marks that the row's plan changed that column's feature.
    Rows are labeled by rank and distance (``#1 (J=...)``, ...), in the order
    ``results`` is given.

    Parameters
    ----------
    results
        The counterfactuals to compare (e.g. the ``k`` alternatives
        for one row from ``diversity="seeds"``/``"lever-blocking"``).
    ax
        Existing axes to draw on; a new figure is created if omitted.

    Returns
    -------
    The axes the matrix was drawn on.

    Raises
    ------
    MissingExtraError
        If matplotlib is not installed.
    """
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
    """Cost of reaching each rating band (``Target.bands``): the price of every grade.

    One bar per band, named and ordered like ``bands_result``; a
    ``Counterfactual`` bar is its ``distance``, an ``Infeasible`` band is
    drawn at zero height and labeled ``"infeasible"``.

    Parameters
    ----------
    bands_result
        The dict returned by ``explain(x, target=Target.bands(...))``.
    ax
        Existing axes to draw on; a new figure is created if omitted.

    Returns
    -------
    The axes the chart was drawn on.

    Raises
    ------
    MissingExtraError
        If matplotlib is not installed.
    """
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


def plot_alternatives(results: Any, explainer: Any = None, ax: Any = None) -> Any:
    """Overlaid dumbbells: every alternative plan's changes for one instance.

    Accepts a sequence of ``Counterfactual`` objects or feasible
    ``BatchRecord`` entries, or a mapping of outcomes as returned by
    ``explain_coalitions`` (keys become legend labels; ``Infeasible`` values
    are skipped). Each plan keeps one color across all its changes — meant
    for a handful of alternatives for the same row (at most 10). With
    ``explainer``, changes are plotted as standardized deltas from the
    factual (Δ/σ), so features of different scales share one axis; without,
    raw values are shown with gray factual dots.

    Parameters
    ----------
    results
        The plans to overlay — a sequence, or a mapping keyed by
        plan name; see above for accepted element types.
    explainer
        When given, changes are standardized by its per-feature
        ``sigma``; when omitted, raw feature values are plotted instead.
    ax
        Existing axes to draw on; a new figure is created if omitted.

    Returns
    -------
    The axes the chart was drawn on.

    Raises
    ------
    MissingExtraError
        If matplotlib is not installed.
    TreecfError
        If ``results`` contains no feasible plans, or more than
        10.
    """
    plt = _import_pyplot()
    plans = _plans_with_labels(results)
    if not plans:
        raise TreecfError("no feasible plans to plot")
    if len(plans) > 10:
        raise TreecfError("plot_alternatives compares at most 10 plans")
    sigma: dict[str, float] = {}
    if explainer is not None:
        sigma = {
            name: float(s)
            for name, s in zip(explainer.ir.feature_names, explainer.sigma, strict=True)
        }
    frequency: dict[str, int] = {}
    for _, plan in plans:
        for name in plan.changes:
            frequency[name] = frequency.get(name, 0) + 1
    features = sorted(frequency, key=lambda name: (-frequency[name], name))
    slots = {name: i for i, name in enumerate(features)}

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.8 * max(2, len(features))))
    step = min(0.18, 0.7 / len(plans))
    for p, (plan_name, plan) in enumerate(plans):
        color = f"C{p}"
        offset = (p - (len(plans) - 1) / 2) * step
        base = plan_name if plan_name is not None else f"plan {p + 1}"
        label: str | None = f"{base} (J={plan.distance:.3g})"
        for name, (source, dest) in plan.changes.items():
            y = slots[name] + offset
            if math.isnan(dest) or math.isnan(source):
                anchor = source if math.isnan(dest) else dest
                if explainer is not None:
                    anchor = 0.0
                ax.plot([anchor], [y], "o", color=color, markersize=5, label=label)
                ax.annotate(
                    "-> NaN" if math.isnan(dest) else "NaN ->",
                    xy=(anchor, y), xytext=(6, 0), textcoords="offset points",
                    va="center", color="tab:red", fontsize=9,
                )
            else:
                if explainer is not None:
                    start, end = 0.0, (dest - source) / sigma[name]
                else:
                    start, end = source, dest
                ax.plot([start, end], [y, y], "-", color=color, alpha=0.5, zorder=1)
                ax.plot([start], [y], "o", color="tab:gray", markersize=4)
                ax.plot([end], [y], "o", color=color, markersize=5, label=label)
            label = None  # one legend entry per plan
    if explainer is not None:
        ax.axvline(0.0, color="0.6", linestyle="--", linewidth=1)
        ax.set_xlabel("standardized change from factual (Δ/σ)")
    else:
        ax.set_xlabel("feature value (gray = factual)")
    ax.set_yticks(range(len(features)), features)
    ax.invert_yaxis()
    ax.set_title(f"{len(plans)} alternative plan(s) for one instance")
    ax.legend(loc="best")
    return ax


def plot_tradeoff(results: Any, target: Any = None, ax: Any = None) -> Any:
    """Cost vs achieved score for alternative plans of one instance.

    One dot per plan: x = distance J, y = the achieved probability (sigmoid
    models) or raw score. ``target`` draws the interval bounds the plans had
    to reach. Accepts a sequence of ``Counterfactual`` objects or feasible
    ``BatchRecord`` entries, or a mapping as returned by
    ``explain_coalitions`` (keys label the dots; ``Infeasible`` skipped).

    Parameters
    ----------
    results
        The plans to plot; see above for accepted shapes.
    target
        When given, draws the target interval's finite bounds
        (mapped into the same probability/raw space as the plans) as
        horizontal reference lines.
    ax
        Existing axes to draw on; a new figure is created if omitted.

    Returns
    -------
    The axes the chart was drawn on.

    Raises
    ------
    MissingExtraError
        If matplotlib is not installed.
    TreecfError
        If ``results`` contains no feasible plans.
    """
    plt = _import_pyplot()
    plans = _plans_with_labels(results)
    if not plans:
        raise TreecfError("no feasible plans to plot")
    prob_space = all(plan.score_prob is not None for _, plan in plans)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    for p, (plan_name, plan) in enumerate(plans):
        score = plan.score_prob if prob_space else plan.score_raw
        ax.plot([plan.distance], [score], "o", color=f"C{p}", markersize=8)
        ax.annotate(
            plan_name if plan_name is not None else f"{p + 1}",
            xy=(plan.distance, score), xytext=(6, 4),
            textcoords="offset points", fontsize=9,
        )
    if target is not None:
        for bound in _target_bounds(target, prob_space):
            ax.axhline(bound, color="tab:red", linewidth=1)
    ax.set_xlabel("distance J (effort)")
    ax.set_ylabel("model probability" if prob_space else "raw score")
    ax.set_title("what each plan costs, and what it buys")
    return ax


def plot_recourse_map(
    explainer: Any,
    x: Any,
    results: Any,
    target: Any,
    *,
    ax: Any = None,
    space: str = "auto",
    annotate: bool = True,
    max_changes_per_label: int = 3,
    fmt: str = "{:.3g}",
    schematic: bool = False,
    region_labels: tuple[str, str] = ("Reject", "Accept"),
    show_factual_label: bool = True,
) -> Any:
    """Recourse diagram: what each plan costs and where it lands relative to the target.

    Plots one point per feasible plan in ``results`` at (model output, recourse
    cost ``J``), with the factual instance drawn as a red dot at cost 0 and an
    arrow from the factual to each plan. A green band marks the target
    interval on the model-output axis; the axis flips automatically so that
    "improving" always reads as a move toward the band. Model output is shown
    as a probability for sigmoid-link models (or when ``space="probability"``)
    and as the raw score otherwise (``space="raw"``; ``space="auto"`` picks
    based on the model's link function).

    In the default quantitative view, each plan is labeled with one line —
    its name (or ``"plan {i}"``, ascending by distance, when unnamed) and its
    cost — when ``annotate`` is set; the map's job here is the overview, not
    a change list. Infeasible entries in ``results`` are drawn as grey
    markers above the plans, labeled by name (``"infeasible"`` alone when
    unlabeled, with a ``(certified)`` suffix when the entry carries a
    certified proof) regardless of ``annotate``.

    ``schematic=True`` swaps the quantitative axes and target band for a
    slide-friendly rendering: a wavy decision-boundary line instead of a
    band, no ticks or axis labels, and "If ..." phrased plan labels (using
    each plan's changed features, largest-effort first, truncated to
    ``max_changes_per_label`` and formatted with ``fmt``). ``annotate`` also
    gates ``show_factual_label`` there — an anchored corner box on the
    factual's screen side listing the features any plan changed, at their
    original values (schematic mode only; the quantitative view never draws
    it). ``region_labels`` names the two sides of the boundary in
    ``schematic`` mode.

    Parameters
    ----------
    explainer
        Explainer wrapping the model; supplies the link function
        and the counterfactual distance weights used to order each
        plan's changes.
    x
        Factual feature vector.
    results
        Counterfactual outcomes for ``x`` — a single result, a
        sequence, or a mapping (as returned by ``explain_coalitions``).
        Feasible and infeasible entries are both accepted.
    target
        The target interval the plans were solved against; also
        drawn as the band (or boundary, in schematic mode).
    ax
        Existing axes to draw on; a new figure is created if omitted.
    space
        ``"probability"``, ``"raw"``, or ``"auto"`` (default) to pick
        the model-output axis space from the model's link function.
    annotate
        Draw a text label at each plan's point; in ``schematic``
        mode, also gates whether ``show_factual_label`` draws its block.
    max_changes_per_label
        Schematic mode only. Number of changed
        features shown per label before truncating to "(+k more)".
    fmt
        Schematic mode only. Format string for changed feature values
        in labels.
    schematic
        Render the slide-friendly boundary view instead of the
        quantitative axes.
    region_labels
        The (reject-side, accept-side) names drawn next to
        the boundary in schematic mode.
    show_factual_label
        Schematic mode only. Draw an anchored corner
        box, on the factual's screen side, listing the features any
        plan changed.

    Returns
    -------
    The axes the recourse map was drawn on.

    Raises
    ------
    MissingExtraError
        If matplotlib is not installed.
    TreecfError
        If ``results`` contains no plans at all, or more than
        10 feasible plans.
    """
    from treecf.ir.evaluate import apply_link, raw_score
    from treecf.ir.model import Link

    plt = _import_pyplot()
    plans, failures = _plans_and_failures(results)
    if not plans and not failures:
        raise TreecfError("no plans to plot")
    if len(plans) > 10:
        raise TreecfError("plot_recourse_map compares at most 10 plans")

    link = explainer.ir.link
    prob = space == "probability" or (space == "auto" and link is Link.SIGMOID)
    s_raw = raw_score(explainer.ir, x)
    x_fact = apply_link(Link.SIGMOID, s_raw) if prob else s_raw

    def _plan_x(plan: Any) -> float:
        if prob and plan.score_prob is not None:
            return float(plan.score_prob)
        if prob:
            return apply_link(Link.SIGMOID, plan.score_raw)
        return float(plan.score_raw)

    ordered = sorted(plans, key=lambda pair: pair[1].distance)
    plan_points = [(label, plan, _plan_x(plan), plan.distance) for label, plan in ordered]

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    lo, hi = _display_interval(target, link, space)
    finite_edges = [b for b in (lo, hi) if math.isfinite(b)]
    xs = [x_fact, *(px for _, _, px, _ in plan_points), *finite_edges]
    span = max(xs) - min(xs)
    pad = 0.08 * span

    if not schematic:
        lo_edge = lo if math.isfinite(lo) else min(xs)
        hi_edge = hi if math.isfinite(hi) else max(xs)
        ax.axvspan(lo_edge - pad, hi_edge + pad, color="tab:green", alpha=0.12)
        for edge in finite_edges:
            ax.axvline(edge, color="0.4", linestyle="--")

    if math.isfinite(hi) and hi < x_fact:
        ax.invert_xaxis()

    ax.plot([x_fact], [0.0], "o", color="tab:red", markersize=9, zorder=3)
    for i, (_label, _plan, px, py) in enumerate(plan_points):
        ax.plot([px], [py], "o", color="tab:green", markersize=8, zorder=3)
        sign = 1 if i % 2 else -1
        r = 0.0 if i == 0 else sign * 0.12 * math.ceil(i / 2)
        ax.annotate(
            "",
            xy=(px, py),
            xytext=(x_fact, 0.0),
            arrowprops={
                "arrowstyle": "->",
                "color": "0.15",
                "lw": 1.2,
                "connectionstyle": f"arc3,rad={r}",
            },
        )

    if not schematic:
        ax.set_xlabel("model output (probability)" if prob else "model output (raw score)")
        ax.set_ylabel("recourse cost J")
        ax.set_title(f"{len(plans)} recourse option(s)")

    label_bbox = {
        "boxstyle": "round,pad=0.25",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.75,
    }

    if annotate:
        for i, (label, plan, px, py) in enumerate(plan_points):
            if schematic:
                text = _format_plan(
                    label, plan, explainer, fmt, max_changes_per_label, schematic=True
                )
                fontsize = 8
            else:
                # Minimal mode: the map's job is the overview, not the change list —
                # one line naming the plan (or its ascending-distance ordinal) and its cost.
                plan_name = label if label is not None else f"plan {i + 1}"
                text = f"{plan_name} (J={plan.distance:.3g})"
                fontsize = 9
            dx, ha = _grow_inward(ax, px)
            ax.annotate(
                text,
                xy=(px, py),
                xytext=(dx, 4),
                textcoords="offset points",
                ha=ha,
                va="bottom",
                fontsize=fontsize,
                bbox=label_bbox,
                zorder=2,  # marker (zorder=3) stays on top of its own label box
            )

    if schematic and annotate and show_factual_label:
        touched: dict[str, float] = {}
        for _label, plan, _px, _py in plan_points:
            for name, (source, _dest) in plan.changes.items():
                touched.setdefault(name, source)
        if touched:
            lines = ["factual:"] + [
                f"{name} = NaN"
                if math.isnan(touched[name])
                else f"{name} = {fmt.format(touched[name])}"
                for name in sorted(touched)
            ]
            _, side_ha = _grow_inward(ax, x_fact)
            fx = 0.02 if side_ha == "left" else 0.98
            ax.text(
                fx,
                0.03,
                "\n".join(lines),
                transform=ax.transAxes,
                ha=side_ha,
                va="bottom",
                fontsize=8,
                bbox={**label_bbox, "facecolor": "0.96"},
                zorder=2,  # factual dot (zorder=3) stays on top of the box background
            )

    y_top = max((py for _, _, _, py in plan_points), default=0.0)
    step_y = 0.12 * (y_top or 1.0)
    fail_dx, fail_ha = _grow_inward(ax, x_fact)
    for i, (label, r) in enumerate(failures):
        y = y_top + (i + 1) * step_y
        ax.plot([x_fact], [y], "x", color="0.5", markersize=8, zorder=3)
        text = f"{label}: infeasible" if label is not None else "infeasible"
        if getattr(r, "proof", "") == "certified":
            text += " (certified)"
        ax.annotate(
            text,
            xy=(x_fact, y),
            xytext=(fail_dx, 0),
            textcoords="offset points",
            ha=fail_ha,
            fontsize=8,
            color="0.35",
            zorder=2,  # marker (zorder=3) stays on top of its own label
        )

    n_failures = len(failures)
    if schematic:
        # Extra top headroom keeps multi-line plan labels and the topmost infeasible
        # marker's label clear of the schematic region labels (y=0.95) and boundary
        # caption (y=0.86), which both sit near the top of the axes.
        top_ref = (y_top + n_failures * step_y) or 1.0
        ax.set_ylim(bottom=-0.05 * top_ref, top=1.5 * top_ref)
    else:
        # Minimal mode: inflate just enough to fit the infeasible stack above the
        # highest plan (or a touch of breathing room when there's no stack at all).
        top_ref = max(y_top + (n_failures + 1) * step_y, y_top * 1.1)
        ax.set_ylim(bottom=-0.05 * top_ref, top=top_ref)

    if schematic:
        _schematic_dressing(ax, finite_edges, span, region_labels)

    return ax


def _schematic_dressing(
    ax: Any,
    finite_edges: Sequence[float],
    x_span: float,
    region_labels: tuple[str, str],
) -> None:
    """Slide-style dressing: hide chrome, wavy boundary per edge, region labels.

    Runs after the caller's final ``set_ylim`` so the wave spans the visible
    y-range (``ax.get_ylim()``); ``x_span`` is the data-envelope x-range
    already computed by the caller, used for the wave amplitude.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    amplitude = 0.02 * x_span
    y_lo, y_hi = ax.get_ylim()
    t = [i / 199 for i in range(200)]
    y = [y_lo + ti * (y_hi - y_lo) for ti in t]
    caption_bbox = {
        "boxstyle": "round,pad=0.25",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.75,
    }
    for edge in finite_edges:
        wave_x = [edge + amplitude * math.sin(2 * math.pi * 1.5 * ti) for ti in t]
        ax.plot(wave_x, y, linestyle="--", color="tab:blue")
        dx, ha = _grow_inward(ax, edge)
        ax.annotate(
            "ML model decision boundary",
            xy=(edge, 0.86),
            xycoords=ax.get_xaxis_transform(),
            xytext=(dx, 0),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=8,
            bbox=caption_bbox,
        )

    # invert_xaxis() is only called when the band lies below the factual, which
    # flips the data->screen mapping so the accept side is screen-right in both
    # cases — the labels do not depend on ax.xaxis_inverted().
    ax.text(0.18, 0.95, region_labels[0], transform=ax.transAxes, fontsize=14, ha="center")
    ax.text(0.82, 0.95, region_labels[1], transform=ax.transAxes, fontsize=14, ha="center")


def _grow_inward(ax: Any, x_data: float) -> tuple[int, str]:
    """Offset/ha for a label anchored at ``x_data`` so its text grows toward plot center."""
    lo_v, hi_v = ax.get_xlim()  # reflects inversion
    frac = (x_data - lo_v) / (hi_v - lo_v) if hi_v != lo_v else 0.5
    return (8, "left") if frac < 0.5 else (-8, "right")


def _plans_with_labels(results: Any) -> list[tuple[str | None, Any]]:
    """Feasible plans paired with labels (mapping keys, `coalition` fields, or None)."""
    if isinstance(results, Mapping):
        return [(str(k), v) for k, v in results.items() if not isinstance(v, Infeasible)]
    return [
        (getattr(r, "coalition", None), r) for r in results if getattr(r, "feasible", True)
    ]


def _plans_and_failures(
    results: Any,
) -> tuple[list[tuple[str | None, Any]], list[tuple[str | None, Any]]]:
    """Split any results shape into labeled (feasible, infeasible) lists.

    Dispatches on shape: a ``Mapping`` keeps every value labeled by
    ``str(key)``; a ``Sequence`` (excluding ``str``/``bytes``) labels each
    entry by its ``coalition`` attribute; anything else is one bare result.
    A ``BatchResult`` itself is iterable but not a ``Sequence`` (no
    ``__getitem__``), so pass ``list(batch_result)`` (every row's records,
    ``coalition``-labeled when set) or a single row's records
    (``batch_result.for_id(row_id)``), not the ``BatchResult`` itself.
    """
    labeled: list[tuple[str | None, Any]]
    if isinstance(results, Mapping):
        labeled = [(str(k), v) for k, v in results.items()]
    elif isinstance(results, Sequence) and not isinstance(results, (str, bytes)):
        labeled = [(getattr(r, "coalition", None), r) for r in results]
    else:
        labeled = [(None, results)]

    plans: list[tuple[str | None, Any]] = []
    failures: list[tuple[str | None, Any]] = []
    for label, r in labeled:
        if isinstance(r, Infeasible) or getattr(r, "feasible", True) is False:
            failures.append((label, r))
        else:
            plans.append((label, r))
    return plans, failures


def _target_bounds(target: Any, prob_space: bool) -> list[float]:
    """Finite target-interval bounds in the plotted space (probability or raw)."""
    from treecf.ir.evaluate import apply_link
    from treecf.ir.model import Link

    lo, hi = float(target.lo), float(target.hi)
    if prob_space and target.space == "raw":
        lo, hi = apply_link(Link.SIGMOID, lo), apply_link(Link.SIGMOID, hi)
    elif not prob_space and target.space == "probability":
        return []  # probability targets only exist for sigmoid models
    return [b for b in (lo, hi) if math.isfinite(b)]


def _display_interval(target: Any, link: Any, space: str) -> tuple[float, float]:
    """Target's raw interval, mapped to display space; infinite endpoints pass through."""
    from treecf.ir.evaluate import apply_link
    from treecf.ir.model import Link

    resolved = ("probability" if link is Link.SIGMOID else "raw") if space == "auto" else space
    lo, hi = target.raw_interval(link)
    if resolved == "probability":
        lo = apply_link(Link.SIGMOID, lo) if math.isfinite(lo) else lo
        hi = apply_link(Link.SIGMOID, hi) if math.isfinite(hi) else hi
    return lo, hi


def plot_waterfall(explainer: Any, cf: Counterfactual, target: Any = None, ax: Any = None) -> Any:
    """SHAP-style waterfall: exact score deltas of the counterfactual's changes.

    Starts at the factual score, applies the changes one at a time (largest
    single effect first), each bar being the EXACT score delta from that change
    (recomputed through the IR — endpoints are exact; per-bar attribution is
    sequential and therefore order-dependent, like any sequential decomposition).
    Sigmoid-link models are plotted in probability space.

    Parameters
    ----------
    explainer
        Explainer wrapping the model; supplies the IR the score
        deltas are recomputed through and the link function.
    cf
        The counterfactual to decompose.
    target
        When given, draws the target interval's finite bounds (in the
        same display space) as vertical reference lines.
    ax
        Existing axes to draw on; a new figure is created if omitted.

    Returns
    -------
    The axes the waterfall was drawn on.

    Raises
    ------
    MissingExtraError
        If matplotlib is not installed.
    """
    import numpy as np

    from treecf.ir.evaluate import apply_link, raw_score
    from treecf.ir.model import Link

    plt = _import_pyplot()
    ir = explainer.ir
    index = {name: j for j, name in enumerate(ir.feature_names)}

    x = cf.x_cf.copy()
    for name, (source, _) in cf.changes.items():
        x[index[name]] = source

    def single_delta(name: str) -> float:
        probe = x.copy()
        probe[index[name]] = cf.changes[name][1]
        return raw_score(ir, probe) - raw_score(ir, x)

    order = sorted(cf.changes, key=lambda f: abs(single_delta(f)), reverse=True)

    sigmoid = ir.link is Link.SIGMOID
    to_display = (lambda s: apply_link(Link.SIGMOID, s)) if sigmoid else (lambda s: s)

    current = x.copy()
    scores = [to_display(raw_score(ir, current))]
    for name in order:
        current[index[name]] = cf.changes[name][1]
        scores.append(to_display(raw_score(ir, current)))

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.7 * max(2, len(order)) + 1))
    for i, _name in enumerate(order):
        before, after = scores[i], scores[i + 1]
        delta = after - before
        color = "tab:blue" if delta < 0 else "tab:orange"
        ax.barh(i, delta, left=before, color=color, height=0.6)
        ax.plot([after, after], [i, i + 1], color="0.6", linestyle=":", linewidth=1)
        ax.annotate(
            f"{delta:+.4g}",
            xy=(max(before, after), i),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
    ax.axvline(scores[0], color="0.4", linestyle="--", linewidth=1)
    ax.text(scores[0], -0.55, f"f(x) = {scores[0]:.4g}", ha="center", va="top", fontsize=9)
    ax.axvline(scores[-1], color="tab:green", linestyle="--", linewidth=1)
    ax.text(
        scores[-1], len(order) - 0.3, f"f(x') = {scores[-1]:.4g}",
        ha="center", va="bottom", fontsize=9, color="tab:green",
    )
    if target is not None:
        for bound in target.raw_interval(ir.link):
            if np.isfinite(bound):
                ax.axvline(to_display(bound), color="tab:red", linewidth=1)
    ax.set_yticks(range(len(order)), order)
    ax.invert_yaxis()  # largest effect on top, like SHAP
    ax.set_xlabel("model probability" if sigmoid else "raw score")
    ax.set_title("what moves the score (sequential, exact)")
    if sigmoid:
        low = min(0.0, min(scores))
        high = max(1.0, max(scores))
        ax.set_xlim(low - 0.02, min(high + 0.05, 1.05))
    return ax


def plot_effort(explainer: Any, cf: Counterfactual, ax: Any = None) -> Any:
    """Cost-space companion: how the distance J splits across the changes.

    One horizontal bar per changed feature, its length the feature's own
    contribution ``w * |delta| / sigma`` to ``cf.distance`` (a NaN transition
    priced via ``AllowMissing``'s ``delta_miss``/``delta_from_miss``),
    descending. Unlike ``plot_waterfall``'s exact score deltas, this
    decomposes the recourse *cost*, not the model score.

    Parameters
    ----------
    explainer
        Explainer wrapping the model; supplies the distance
        weights and normalizers each contribution is computed from.
    cf
        The counterfactual to decompose.
    ax
        Existing axes to draw on; a new figure is created if omitted.

    Returns
    -------
    The axes the chart was drawn on.

    Raises
    ------
    MissingExtraError
        If matplotlib is not installed.
    """
    plt = _import_pyplot()
    contributions = sorted(
        _change_effort(explainer, cf.changes).items(), key=lambda pair: pair[1], reverse=True
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.6 * max(2, len(contributions)) + 0.8))
    labels = [name for name, _ in contributions]
    efforts = [effort for _, effort in contributions]
    ax.barh(range(len(labels)), efforts, color="tab:blue", height=0.6)
    for i, effort in enumerate(efforts):
        ax.annotate(
            f"{effort:.3g}", xy=(effort, i), xytext=(4, 0),
            textcoords="offset points", va="center", fontsize=9,
        )
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("effort contribution (w·|Δ|/σ)")
    ax.set_title(f"where the effort goes — total J = {cf.distance:.3g}")
    return ax


def _change_effort(explainer: Any, changes: Mapping[str, tuple[float, float]]) -> dict[str, float]:
    """Per-change effort w_j*|delta|/sigma_j; NaN legs priced via compiled.allow_missing."""
    index = {name: j for j, name in enumerate(explainer.ir.feature_names)}
    allow = explainer.compiled.allow_missing
    efforts: dict[str, float] = {}
    for name, (source, dest) in changes.items():
        j = index[name]
        if math.isnan(dest):
            delta = allow[j][0]
        elif math.isnan(source):
            delta = allow[j][1]
        else:
            delta = abs(dest - source)
        efforts[name] = float(explainer.weights[j] * delta / explainer.sigma[j])
    return efforts


def _format_plan(
    name: str | None,
    plan: Any,
    explainer: Any,
    fmt: str = "{:.3g}",
    max_changes: int = 3,
    *,
    schematic: bool = False,
) -> str:
    """Arrow-label text for one plan: effort-ordered changes, one phrase per line.

    NaN legs read as ``drop {feature}`` / ``provide {feature} = value``;
    ``plan.region.describe()`` (when present and returning a ``Mapping``)
    supplies a phrase for changes it covers instead of ``feature = value`` —
    anything else ``describe()`` returns falls back to the plain phrasing, the
    same as no region at all. Truncated to ``max_changes``
    phrase lines with a trailing ``"(+k more)"`` line; ``schematic`` phrases
    each line as an "If ..." / "and ..." clause instead of a bare value.
    ``name``, when given, gets its own line in quantitative mode or prefixes
    the "If" line in schematic mode. ``(J=...)`` is appended to the last line.
    """
    changes = plan.changes
    efforts = _change_effort(explainer, changes)
    cat_info = _categorical_info(explainer)
    region = getattr(plan, "region", None)
    describe = getattr(region, "describe", None)
    region_phrases: Mapping[str, str] = {}
    if callable(describe):
        described = describe()
        if isinstance(described, Mapping):
            region_phrases = described
    ordered = sorted(changes, key=lambda f: (-efforts[f], f))

    parts = []
    for f in ordered[:max_changes]:
        source, dest = changes[f]
        if math.isnan(dest):
            parts.append(f"drop {f}")
        elif math.isnan(source):
            parts.append(f"provide {f} = {fmt.format(dest)}")
        elif f in region_phrases:
            parts.append(region_phrases[f])
        elif f in cat_info:
            source_label = _code_label(cat_info[f], source)
            dest_label = _code_label(cat_info[f], dest)
            parts.append(f"{f}: {source_label} → {dest_label}")
        else:
            parts.append(f"{f} = {fmt.format(dest)}")

    prefix = f"{name}: " if name is not None else ""
    if schematic and parts:
        lines = [f"{prefix}If {parts[0]}", *(f"and {p}" for p in parts[1:])]
    else:
        # No phrase lines to show (e.g. max_changes=0): fall back to the
        # quantitative layout, which degrades gracefully to just the name
        # line (if any) and the truncation/J line below.
        lines = list(parts)
        if name is not None:
            lines.insert(0, f"{name}:")

    remaining = len(ordered) - max_changes
    if remaining > 0:
        lines.append(f"(+{remaining} more)")

    if not lines:
        lines = [""]

    j_suffix = f"(J={plan.distance:.3g})"
    lines[-1] = f"{lines[-1]} {j_suffix}" if lines[-1] else j_suffix
    return "\n".join(lines)


_FillStyle = Literal["full", "left", "right", "bottom", "top", "none"]
_MENU_GLYPHS: dict[str, tuple[str, _FillStyle, str]] = {
    # kind -> (marker, fillstyle, legend label)
    "optimal": ("s", "full", "optimal"),
    "optimal_within_gap": ("s", "left", "optimal within gap"),
    "heuristic": ("s", "none", "heuristic"),
    "certified": ("x", "full", "certified infeasible"),
    "search_exhausted": (".", "full", "search exhausted"),
    "unresolved": ("$?$", "full", "unresolved"),
}


def _menu_glyph_kind(entry: Any) -> str:
    """Which glyph an entry gets: its proof, or ``"unresolved"`` for ``None``."""
    if entry is None:
        return "unresolved"
    if isinstance(entry, Counterfactual):
        return entry.proof if entry.proof in _MENU_GLYPHS else "heuristic"
    return "certified" if entry.proof == "certified" else "search_exhausted"


def plot_recourse_menu(
    menu: Any,
    *,
    ax: Any = None,
    order: str = "cost",
    max_rows: int = 25,
    annotate: bool = True,
    explainer: Any = None,
) -> Any:
    """Lever-set by feature matrix of a recourse menu.

    One row per menu entry — in the menu's own order (minimal frontier
    first) when ``order="cost"``, by set size then key when
    ``order="size"`` — followed by the unresolved sets, and one column per
    candidate lever. A filled cell marks a feature the plan changed, shaded
    by the size of the change: ``|Δ|/σ`` when ``explainer`` is given
    (categorical levers hatched), otherwise ``|Δ|`` relative to the largest
    change of that lever across the menu. The row label carries the plan
    cost, and a glyph before it the proof: filled square ``optimal``, half
    square ``optimal_within_gap``, open square ``heuristic``, cross
    certified infeasible, dot ``search_exhausted``, question mark
    unresolved. A ``DiverseSet`` built from a menu renders through it.

    Parameters
    ----------
    menu
        A ``RecourseMenu``, or a ``DiverseSet`` whose ``menu`` is set.
    ax
        Existing axes to draw on; a new figure is created if omitted.
    order
        ``"cost"`` (the menu's order) or ``"size"``.
    max_rows
        Rows drawn before the rest is cut; the title says how many of the
        total are shown.
    annotate
        Write each changed feature's new value in its cell.
    explainer
        The explainer the menu came from, for sigma-scaled shading and
        categorical hatching; optional.

    Returns
    -------
    The axes the matrix was drawn on.

    Raises
    ------
    MissingExtraError
        If matplotlib is not installed.
    TreecfError
        If a ``DiverseSet`` without a menu is given, or the menu has no
        levers.
    ValueError
        If ``order`` is unknown.
    """
    from matplotlib.patches import Rectangle

    from treecf._menu import DiverseSet

    plt = _import_pyplot()
    if isinstance(menu, DiverseSet):
        if menu.menu is None:
            raise TreecfError(
                "this DiverseSet carries no menu (coalition criterion); draw its plans "
                "with plot_alternatives instead"
            )
        menu = menu.menu
    if order not in ("cost", "size"):
        raise ValueError(f"order must be 'cost' or 'size', got {order!r}")
    levers = list(menu.levers)
    if not levers:
        raise TreecfError("the menu has no candidate levers to draw")

    rows: list[tuple[str, Any]] = [*menu.items(), *((key, None) for key in menu.unresolved)]
    if order == "size":
        rows.sort(key=lambda row: (len(row[0].split("+")), row[0]))
    total = len(rows)
    rows = rows[:max_rows]

    categorical = set()
    sigma: dict[str, float] = {}
    if explainer is not None:
        categorical = {explainer.ir.feature_names[j] for j in explainer.ir.categorical}
        sigma = dict(
            zip(explainer.ir.feature_names, [float(s) for s in explainer.sigma], strict=True)
        )

    def magnitude(name: str, before: float, after: float) -> float:
        if name in categorical:
            return 1.0
        if math.isnan(before) or math.isnan(after):
            return 1.0
        delta = abs(after - before)
        return delta / sigma[name] if sigma else delta

    scale: dict[str, float] = dict.fromkeys(levers, 0.0)
    for _key, entry in rows:
        if isinstance(entry, Counterfactual):
            for name, (before, after) in entry.changes.items():
                if name in scale:
                    scale[name] = max(scale[name], magnitude(name, before, after))
    if sigma:
        top = max(scale.values(), default=1.0) or 1.0
        scale = dict.fromkeys(levers, top)

    if ax is None:
        _, ax = plt.subplots(figsize=(0.6 * len(levers) + 3.0, 0.38 * len(rows) + 1.6))
    cmap = plt.get_cmap("Blues")
    labels: list[str] = []
    present: list[str] = []
    for i, (key, entry) in enumerate(rows):
        members = set(key.split("+"))
        changes = entry.changes if isinstance(entry, Counterfactual) else {}
        for j, name in enumerate(levers):
            if name in changes:
                before, after = changes[name]
                share = magnitude(name, before, after) / (scale[name] or 1.0)
                ax.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1.0, 1.0,
                    facecolor=cmap(0.35 + 0.6 * min(share, 1.0)),
                    edgecolor="white", hatch="//" if name in categorical else None,
                    label="_cell_filled",
                ))
                if annotate:
                    text = "NaN" if math.isnan(after) else f"{after:.3g}"
                    ax.text(j, i, text, ha="center", va="center", fontsize=7,
                            color="white" if share > 0.55 else "0.15")
            else:
                ax.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1.0, 1.0,
                    facecolor="0.97" if name in members else "white",
                    edgecolor="0.85", label="_cell_empty",
                ))
        kind = _menu_glyph_kind(entry)
        marker, fillstyle, _ = _MENU_GLYPHS[kind]
        ax.plot([-0.9], [i], marker=marker, fillstyle=fillstyle, color="0.2",
                markersize=7 if marker != "$?$" else 9, linestyle="none",
                label=f"_glyph_{kind}")
        if kind not in present:
            present.append(kind)
        if isinstance(entry, Counterfactual):
            labels.append(f"{key}  J={entry.distance:.3g}")
        else:
            labels.append(key)

    ax.set_xlim(-1.4, len(levers) - 0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks(range(len(levers)))
    ax.set_xticklabels(levers, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    title = f"recourse menu — {total} lever set(s)"
    if len(rows) < total:
        title += f" (showing {len(rows)} of {total})"
    ax.set_title(title)
    from matplotlib.lines import Line2D

    handles: list[Any] = [
        Line2D([], [], marker=_MENU_GLYPHS[kind][0], fillstyle=_MENU_GLYPHS[kind][1],
               color="0.2", linestyle="none", markersize=7, label=_MENU_GLYPHS[kind][2])
        for kind in present
    ]
    handles.append(Rectangle((0, 0), 1, 1, facecolor=cmap(0.7), edgecolor="white",
                             label="changed lever (shade: size of change)"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=7, frameon=False)
    return ax


def plot_region(
    explainer: Any,
    x: Any,
    result: Any,
    *,
    ax: Any = None,
    units: str = "sigma",
    order: str = "index",
    annotate: bool = True,
    fmt: str = "{:.3g}",
    max_features: int | None = None,
) -> Any:
    """Per-feature view of a certified recourse region: how far each value can
    move while staying certified, and what stopped it.

    Each widened numeric feature draws its certified interval as a thick bar
    (``units="sigma"``: one shared axis in sigma-units from the factual, so
    every factual sits at 0; ``units="raw"``: small multiples, one strip per
    feature on its own scale). The factual is a hollow circle, the
    counterfactual a filled marker, and the instance bounds faint whiskers. A
    finite bar end carries a cap saying what limited it: a bracket where the
    end coincides with an instance bound (a constraint stopped it), a plain
    tick where the model's own routing did; an infinite end runs to the axis
    edge with an open arrow. Categorical features draw one tile per category
    code — filled when certified, outlined at the factual's code, marked at
    the counterfactual's, hatched where a declared allowed set excludes the
    code; the tiles are nominal, so their positions carry no meaning. The
    legend records that the region is certified but not necessarily maximal.

    Parameters
    ----------
    explainer : Explainer
        The explainer that produced the result (bounds, normalizers, names).
    x : array-like
        The factual row the region is anchored at.
    result : Counterfactual or (RecourseRegion, array-like)
        A result carrying ``region``, or an explicit ``(region, x_cf)`` pair.
    ax : matplotlib axes, optional
        Target axes for ``units="sigma"``; ignored for ``units="raw"``.
    units : {"sigma", "raw"}
        Shared sigma-unit axis, or per-feature raw-value strips.
    order : {"index", "cost"}
        Row order: ascending feature index, or descending cost contribution.
    annotate : bool
        Annotate raw values at the bar ends.
    fmt : str
        Format string for annotations.
    max_features : int, optional
        Cap on rows; the rest are summarized as ``"(+k more)"``.

    Returns
    -------
    matplotlib axes, or an array of axes for ``units="raw"``.

    Raises
    ------
    MissingExtraError
        If matplotlib is not installed.
    TreecfError
        If the result carries no region, or an argument is unrecognized.
    """
    plt = _import_pyplot()
    import numpy as np

    from treecf._errors import TreecfError

    if isinstance(result, tuple):
        region, x_cf = result
    else:
        region = getattr(result, "region", None)
        x_cf = getattr(result, "x_cf", None)
    if region is None:
        raise TreecfError("result has no region; pass region=True to explain")
    if units not in ("sigma", "raw"):
        raise TreecfError(f"unknown units {units!r}; use 'sigma' or 'raw'")
    if order not in ("index", "cost"):
        raise TreecfError(f"unknown order {order!r}; use 'index' or 'cost'")

    x = np.asarray(x, dtype=np.float64)
    x_cf = np.asarray(x_cf, dtype=np.float64)
    names = tuple(explainer.ir.feature_names)
    index = {name: j for j, name in enumerate(names)}
    sigma = np.asarray(explainer.sigma, dtype=np.float64)
    weights = np.asarray(explainer.weights, dtype=np.float64)
    lo_b, hi_b, _frozen = explainer.compiled.instance_bounds(x)
    lo_b = np.where(np.isnan(lo_b), -math.inf, lo_b)
    hi_b = np.where(np.isnan(hi_b), math.inf, hi_b)

    rows: list[tuple[int, str, str]] = []  # (feature index, name, kind)
    for name in region.feature_intervals:
        rows.append((index[name], name, "numeric"))
    for name in getattr(region, "feature_categories", {}):
        rows.append((index[name], name, "categorical"))

    def cost_of(j: int) -> float:
        if math.isnan(x[j]) or math.isnan(x_cf[j]) or x[j] == x_cf[j]:
            return 0.0
        delta = 1.0 if j in explainer.ir.categorical else abs(x_cf[j] - x[j])
        return float(weights[j] * delta / sigma[j])

    if order == "index":
        rows.sort(key=lambda row: row[0])
    else:
        rows.sort(key=lambda row: (-cost_of(row[0]), row[0]))
    total_rows = len(rows)
    hidden = 0
    if max_features is not None and total_rows > max_features:
        hidden = total_rows - max_features
        rows = rows[:max_features]
    # the caveat line is only owed while some side is neither at a bound nor proved
    maximal_categories = getattr(region, "maximal_categories", {})
    show_caveat = any(
        _unproven_sides(region, name, float(lo_b[j]), float(hi_b[j]))
        if kind == "numeric"
        else not maximal_categories.get(name, False)
        for j, name, kind in rows
    )
    show_data = any(
        any(_data_limited_sides(region, name)) for _j, name, kind in rows if kind == "numeric"
    )

    if units == "raw":
        _, axes = plt.subplots(
            len(rows), 1, figsize=(7, 1.1 * max(2, len(rows))), squeeze=False
        )
        axes = axes[:, 0]
        for strip, (j, name, kind) in zip(axes, rows, strict=True):
            _region_row(
                strip, explainer, region, x, x_cf, j, name, kind,
                sigma_units=False, y=0.0, lo_b=lo_b, hi_b=hi_b,
                annotate=annotate, fmt=fmt,
            )
            strip.set_yticks([0.0])
            strip.set_yticklabels([name])
        _region_legend(axes[0], show_caveat, show_data)
        axes[0].set_title(f"certified recourse region — {total_rows} feature(s)")
        if hidden:
            axes[-1].annotate(
                f"(+{hidden} more)", xy=(0.99, 0.02), xycoords="axes fraction",
                ha="right", fontsize=8, color="0.4",
            )
        return axes

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.6 * max(2, len(rows))))
    for i, (j, name, kind) in enumerate(rows):
        _region_row(
            ax, explainer, region, x, x_cf, j, name, kind,
            sigma_units=True, y=float(len(rows) - 1 - i), lo_b=lo_b, hi_b=hi_b,
            annotate=annotate, fmt=fmt,
        )
    ax.set_yticks([float(len(rows) - 1 - i) for i in range(len(rows))])
    ax.set_yticklabels([name for _, name, _ in rows])
    ax.set_xlabel("distance from the factual (sigma units)")
    ax.set_title(f"certified recourse region — {total_rows} feature(s)")
    if hidden:
        ax.annotate(
            f"(+{hidden} more)", xy=(0.99, 0.02), xycoords="axes fraction",
            ha="right", fontsize=8, color="0.4",
        )
    _region_legend(ax, show_caveat, show_data)
    return ax


def plot_certification_trace(result: Any, *, ax: Any = None) -> Any:
    """How an exact search's proof formed: incumbent and lower bound over nodes.

    Reads ``solver_stats["trace"]`` — the samples the exact backend takes at
    every incumbent update and every power-of-two node count — and draws the
    incumbent cost (what has been found) and the lower bound (what can still
    be ruled out) against the nodes expanded on a log axis, with the gap
    between them shaded. The terminal marker names the outcome: ``optimal``,
    ``within gap``, ``certified infeasible``, or ``stopped early`` for a
    search that ran out of budget or withdrew its claim (the solve-time
    warning says which).

    Parameters
    ----------
    result
        A ``Counterfactual``, ``Infeasible``, or ``BatchRecord`` produced
        by ``backend="exact"``.
    ax
        Axes to draw on; a new figure is created when omitted.

    Returns
    -------
    The matplotlib ``Axes`` drawn on.

    Raises
    ------
    MissingExtraError
        If matplotlib is not installed.
    TreecfError
        If ``result`` carries no trace (a genetic or python-backend result).
    """
    plt = _import_pyplot()

    stats = getattr(result, "solver_stats", None)
    trace = stats.get("trace") if isinstance(stats, dict) else None
    if not trace:
        raise TreecfError(
            "result carries no certification trace; only backend='exact' records one"
        )
    assert isinstance(stats, dict)  # narrowed by the trace check above
    nodes = [max(int(n), 1) for n, _, _ in trace]  # a log axis cannot show node 0
    incumbents = [None if c is None else float(c) for _, c, _ in trace]
    bounds = [float(b) for _, _, b in trace]

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.2))
    known_bound = [(n, b) for n, b in zip(nodes, bounds, strict=True) if math.isfinite(b)]
    if known_bound:
        ax.plot(
            [n for n, _ in known_bound], [b for _, b in known_bound],
            drawstyle="steps-post", color="C3", linewidth=1.5, label="lower bound",
        )
    known_inc = [(n, c) for n, c in zip(nodes, incumbents, strict=True) if c is not None]
    if known_inc:
        ax.plot(
            [n for n, _ in known_inc], [c for _, c in known_inc],
            drawstyle="steps-post", color="C0", linewidth=1.5, label="incumbent",
        )
    both = [
        (n, c, b)
        for n, c, b in zip(nodes, incumbents, bounds, strict=True)
        if c is not None and math.isfinite(b)
    ]
    if both:
        ax.fill_between(
            [n for n, _, _ in both], [b for _, _, b in both], [c for _, c, _ in both],
            step="post", color="C0", alpha=0.15, linewidth=0, label="_gap",
        )
    ax.set_xscale("log")
    ax.set_xlabel("nodes expanded")
    ax.set_ylabel("cost")
    ax.set_title("certification trace")

    outcome = _trace_outcome(result, stats)
    last_n = nodes[-1]
    last_y = incumbents[-1] if incumbents[-1] is not None else (
        bounds[-1] if math.isfinite(bounds[-1]) else 0.0
    )
    ax.plot([last_n], [last_y], marker="o", color="0.2", markersize=5, zorder=5,
            label="_terminal")
    ax.annotate(
        outcome, xy=(last_n, last_y), xytext=(-6, 8), textcoords="offset points",
        ha="right", fontsize=8, color="0.2",
    )
    if known_bound or known_inc:
        ax.legend(fontsize=7, frameon=False, loc="best")
    return ax


def _trace_outcome(result: Any, stats: dict[str, Any]) -> str:
    """The claim a result makes, as the trace's terminal label."""
    proof = str(getattr(result, "proof", ""))
    feasible = getattr(result, "feasible", getattr(result, "x_cf", None) is not None)
    if proof == "optimal_within_gap":
        return "within gap"
    if proof == "optimal":
        return "optimal"
    if proof == "certified" or (not feasible and stats.get("completed") is True):
        return "certified infeasible"
    return "stopped early"


_CAP_MODEL = "stopped by the model"
_CAP_CONSTRAINT = "stopped by a constraint"
_CAP_PROVED = "stopped at a proved boundary"
_CAP_DATA = "stopped at the data range"
_CAP_CAVEAT = "certified, not necessarily maximal"


def _proved_sides(region: Any, name: str) -> tuple[bool, bool]:
    """What the maximal mode proved about a feature's two sides; nothing in
    fast mode."""
    flags = getattr(region, "maximal", {}).get(name, (False, False))
    return bool(flags[0]), bool(flags[1])


def _data_limited_sides(region: Any, name: str) -> tuple[bool, bool]:
    """Which sides of ``name`` stopped at the observed data range."""
    flags = getattr(region, "data_limited", {}).get(name, (False, False))
    return bool(flags[0]), bool(flags[1])


def _unproven_sides(region: Any, name: str, lo_b: float, hi_b: float) -> bool:
    """Whether some finite side of ``name`` is neither at its instance bound,
    nor at the data range, nor proved maximal — the case the legend's caveat
    line speaks to."""
    lo, hi = region.feature_intervals[name]
    proved_lo, proved_hi = _proved_sides(region, name)
    data_lo, data_hi = _data_limited_sides(region, name)
    for endpoint, bound, settled in (
        (lo, lo_b, proved_lo or data_lo), (hi, hi_b, proved_hi or data_hi)
    ):
        if math.isfinite(endpoint) and not _constraint_limited(endpoint, bound) and not settled:
            return True
    return False


def _constraint_limited(endpoint: float, bound: float) -> bool:
    """The endpoint sits at the instance bound, within the one-float32-ulp
    stepping region growth itself uses for open cell edges."""
    import numpy as np

    if not math.isfinite(bound):
        return False
    if endpoint == bound:
        return True
    with np.errstate(over="ignore"):
        b32 = np.float32(bound)
        ulp = abs(float(np.nextafter(b32, np.float32(math.inf))) - float(b32))
    return abs(endpoint - bound) <= max(ulp, 4.0 * math.ulp(abs(bound)))


def _region_row(
    ax: Any,
    explainer: Any,
    region: Any,
    x: Any,
    x_cf: Any,
    j: int,
    name: str,
    kind: str,
    *,
    sigma_units: bool,
    y: float,
    lo_b: Any,
    hi_b: Any,
    annotate: bool,
    fmt: str,
) -> None:
    import numpy as np

    if kind == "categorical":
        _categorical_tiles(ax, explainer, region, x, x_cf, j, name, y)
        return

    sigma_j = float(explainer.sigma[j]) if sigma_units else 1.0
    anchor = float(x[j]) if sigma_units else 0.0

    def to_axis(value: float) -> float:
        if not math.isfinite(value):
            return value
        return (value - anchor) / sigma_j if sigma_units else value

    lo, hi = region.feature_intervals[name]
    bound_lo, bound_hi = float(lo_b[j]), float(hi_b[j])

    # faint whiskers for the instance bounds (finite parts only)
    finite_lo = to_axis(bound_lo) if math.isfinite(bound_lo) else None
    finite_hi = to_axis(bound_hi) if math.isfinite(bound_hi) else None
    if finite_lo is not None or finite_hi is not None:
        span = [v for v in (finite_lo, to_axis(lo), to_axis(hi), finite_hi) if v is not None]
        span = [v for v in span if math.isfinite(v)]
        if span:
            ax.plot(
                [min(span), max(span)], [y, y],
                color="0.85", linewidth=1.0, zorder=1, solid_capstyle="butt",
            )

    seg_lo = to_axis(lo)
    seg_hi = to_axis(hi)
    finite_points = [v for v in (seg_lo, seg_hi, 0.0 if sigma_units else float(x[j]))
                     if math.isfinite(v)]
    draw_lo = seg_lo if math.isfinite(seg_lo) else min(finite_points, default=0.0) - 1.0
    draw_hi = seg_hi if math.isfinite(seg_hi) else max(finite_points, default=0.0) + 1.0
    ax.plot([draw_lo, draw_hi], [y, y], color="C0", linewidth=5, zorder=2,
            solid_capstyle="butt")

    for endpoint, drawn, side in ((lo, draw_lo, "lo"), (hi, draw_hi, "hi")):
        if not math.isfinite(endpoint):
            marker = "<" if side == "lo" else ">"
            ax.plot([drawn], [y], marker=marker, markerfacecolor="none",
                    markeredgecolor="C0", markersize=9, zorder=3,
                    label="_region_open_end")
            continue
        bound = float(lo_b[j]) if side == "lo" else float(hi_b[j])
        proved = _proved_sides(region, name)[0 if side == "lo" else 1]
        data_limited = _data_limited_sides(region, name)[0 if side == "lo" else 1]
        if data_limited:
            ax.plot([drawn], [y], marker="D", color="0.45", markersize=6,
                    zorder=4, label="_cap_data")
        elif _constraint_limited(endpoint, bound):
            ax.plot([drawn], [y], marker="$[$" if side == "lo" else "$]$",
                    color="C3", markersize=11, zorder=4, label="_cap_constraint")
        elif proved:
            ax.plot([drawn], [y], marker="s", color="C0", markersize=7,
                    zorder=4, label="_cap_proved")
        else:
            ax.plot([drawn], [y], marker="|", color="C0", markersize=11,
                    markeredgewidth=2.5, zorder=4, label="_cap_model")
        if annotate:
            ax.annotate(
                fmt.format(endpoint), xy=(drawn, y), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=7, color="0.35",
            )

    factual_pos = 0.0 if sigma_units else float(x[j])
    if math.isfinite(factual_pos):
        ax.plot([factual_pos], [y], marker="o", markerfacecolor="none",
                markeredgecolor="0.2", markersize=7, zorder=5)
    cf_pos = to_axis(float(x_cf[j]))
    if math.isfinite(cf_pos):
        ax.plot([cf_pos], [y], marker="o", color="C0", markersize=5, zorder=6)
    del np


def _categorical_tiles(
    ax: Any, explainer: Any, region: Any, x: Any, x_cf: Any, j: int, name: str, y: float
) -> None:
    """Nominal tiles, one per category code, in axes-fraction x at data y."""
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import blended_transform_factory

    info = explainer.ir.categorical[j]
    certified = set(region.feature_categories.get(name, ()))
    allowed = explainer.compiled.allowed_categories.get(j)
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    k = info.cardinality
    pad = 0.08
    width = (1.0 - 2 * pad) / k
    for code in range(k):
        left = pad + code * width
        excluded = allowed is not None and code not in allowed
        in_set = code in certified
        face = "C0" if in_set else "none"
        tile = Rectangle(
            (left + 0.06 * width, y - 0.28), 0.88 * width, 0.56,
            transform=trans,
            facecolor=face, alpha=0.55 if in_set else 1.0,
            edgecolor="0.2" if code == int(x[j]) else "0.6",
            linewidth=2.2 if code == int(x[j]) else 0.8,
            hatch="///" if excluded else None,
        )
        ax.add_patch(tile)
        if code == int(x_cf[j]):
            ax.plot([left + 0.5 * width], [y], marker="o", color="C0",
                    markersize=5, zorder=6, transform=trans)
        label = (
            str(info.categories[code])
            if info.categories is not None and code < len(info.categories)
            else str(code)
        )
        ax.annotate(
            label, xy=(left + 0.5 * width, y - 0.42), xycoords=trans,
            ha="center", fontsize=6, color="0.35",
        )


def _region_legend(ax: Any, show_caveat: bool = True, show_data: bool = False) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], marker="|", color="C0", markersize=10, markeredgewidth=2.5,
               linestyle="none", label=_CAP_MODEL),
        Line2D([], [], marker="$[$", color="C3", markersize=10, linestyle="none",
               label=_CAP_CONSTRAINT),
        Line2D([], [], marker="s", color="C0", markersize=7, linestyle="none",
               label=_CAP_PROVED),
    ]
    if show_data:
        handles.append(
            Line2D([], [], marker="D", color="0.45", markersize=6, linestyle="none",
                   label=_CAP_DATA)
        )
    if show_caveat:
        handles.append(Line2D([], [], linestyle="none", label=_CAP_CAVEAT))
    ax.legend(handles=handles, loc="best", fontsize=7, frameon=False)


def _categorical_info(explainer: Any) -> dict[str, Any]:
    """Per-feature categorical metadata keyed by name, when the explainer has any."""
    ir = getattr(explainer, "ir", None)
    categorical = getattr(ir, "categorical", None)
    if ir is None or not categorical:
        return {}
    return {ir.feature_names[j]: info for j, info in categorical.items()}


def _code_label(info: Any, value: float) -> str:
    """A category code rendered by its display name when the model carries one."""
    code = int(value)
    names = getattr(info, "categories", None)
    if names is not None and 0 <= code < len(names):
        return str(names[code])
    return str(code)


def _import_pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise MissingExtraError(
            "visualization requires matplotlib: pip install treecf[viz]"
        ) from exc
    return plt
