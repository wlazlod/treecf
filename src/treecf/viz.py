"""Counterfactual visualizations (D15). matplotlib lives behind the [viz] extra."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from treecf._errors import MissingExtraError, TreecfError
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


def _plans_with_labels(results: Any) -> list[tuple[str | None, Any]]:
    """Feasible plans paired with labels (mapping keys, `coalition` fields, or None)."""
    if isinstance(results, Mapping):
        return [(str(k), v) for k, v in results.items() if not isinstance(v, Infeasible)]
    return [
        (getattr(r, "coalition", None), r) for r in results if getattr(r, "feasible", True)
    ]


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


def plot_waterfall(explainer: Any, cf: Counterfactual, target: Any = None, ax: Any = None) -> Any:
    """SHAP-style waterfall: exact score deltas of the counterfactual's changes.

    Starts at the factual score, applies the changes one at a time (largest
    single effect first), each bar being the EXACT score delta from that change
    (recomputed through the IR — endpoints are exact; per-bar attribution is
    sequential and therefore order-dependent, like any sequential decomposition).
    Sigmoid-link models are plotted in probability space.
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
    """Cost-space companion: how the distance J splits across the changes."""
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


def _import_pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise MissingExtraError(
            "visualization requires matplotlib: pip install treecf[viz]"
        ) from exc
    return plt
