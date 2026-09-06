"""Regenerate every committed docs figure from the docs model.

One PNG per public plot function, written under ``docs/guide/img/``. Not run
at build time — see docs/README.md for when a regenerated figure should be
committed (only when the picture actually changed; PNGs are not
byte-reproducible across matplotlib versions or platforms).

Run from anywhere: ``uv run python docs/scripts/generate_figures.py``
"""

from __future__ import annotations

import pathlib
import shutil
import tempfile
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from treecf import Explainer, Range, Target, TreecfWarning
from treecf.audit import portfolio_report
from treecf.datasets import credit_demo
from treecf.viz import (
    plot_alternatives,
    plot_certification_trace,
    plot_changes,
    plot_counterfactuals,
    plot_effort,
    plot_ladder,
    plot_recourse_map,
    plot_recourse_menu,
    plot_region,
    plot_tradeoff,
    plot_waterfall,
)
from treecf.viz_batch import (
    plot_batch_deltas,
    plot_batch_levers,
    plot_batch_matrix,
    plot_batch_summary,
    plot_recourse_burden,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "guide" / "img"
SAMPLES = REPO / "docs" / "guide" / "samples"
def _save(name: str, fig: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")  # type: ignore[attr-defined]
    plt.close("all")
    print(f"wrote {path.relative_to(REPO)}")


def _fig_of(axes: object) -> object:
    ax = axes[0] if isinstance(axes, (list, tuple, np.ndarray)) else axes
    return ax.figure


def main() -> None:
    model, X_bg, _x = credit_demo()
    exp = Explainer(model, background=X_bg)
    target = Target.probability(range=(0.0, 0.05))
    x = X_bg[1]

    res = exp.explain(x, target=target, seed=0)
    _save("plot_changes", _fig_of(plot_changes(res)))
    _save("plot_waterfall", _fig_of(plot_waterfall(exp, res, target=target)))
    _save("plot_effort", _fig_of(plot_effort(exp, res)))

    second = exp.explain(x, target=target, seed=1)
    _save("plot_counterfactuals", _fig_of(plot_counterfactuals([res, second])))

    plans = exp.explain_coalitions(
        x,
        target=target,
        coalitions={
            "repayment": ["utilization", "dpd_12m"],
            "profile": ["income", "tenure_months", "occupation"],
        },
        include_full=True,
        seed=0,
    )
    _save("plot_alternatives", _fig_of(plot_alternatives(plans, explainer=exp)))
    _save("plot_tradeoff", _fig_of(plot_tradeoff(plans, target=target)))
    _save("plot_recourse_map", _fig_of(plot_recourse_map(exp, x, plans, target=target)))
    _save(
        "plot_recourse_map_schematic",
        _fig_of(plot_recourse_map(exp, x, plans, target=target, schematic=True)),
    )

    # the recourse menu, and the same menu on the recourse map: the two views
    # of one enumeration must agree
    menu = exp.recourse_menu(x, target=target, max_levers=2, backend="exact", seed=0)
    _save("plot_recourse_menu", _fig_of(plot_recourse_menu(menu, explainer=exp)))
    _save(
        "plot_recourse_menu_map",
        _fig_of(plot_recourse_map(exp, x, menu, target=target)),
    )

    ladder = exp.explain(
        x, target=Target.bands({"A": (0.0, 0.01), "B": (0.01, 0.05), "C": (0.05, 0.15)}), seed=0
    )
    _save("plot_ladder", _fig_of(plot_ladder(ladder)))

    # One Range constraint so the region shows both cap markers: bounds
    # stopped by the model and bounds stopped by a constraint.
    exp_rng = Explainer(
        model,
        background=X_bg,
        constraints=[Range("tenure_months", 0.0, 140.0)],
    )
    certified = exp_rng.explain(x, target=target, backend="exact", region=True, seed=0)
    _save("plot_region", _fig_of(plot_region(exp_rng, x, certified)))
    # the same region grown in the maximal mode: proved sides get the square cap
    maximal = exp_rng.explain(
        x, target=target, backend="exact", region=True, region_mode="maximal", seed=0
    )
    _save("plot_region_maximal", _fig_of(plot_region(exp_rng, x, maximal)))

    # a budget-limited exact solve leaves a visible gap between incumbent and bound
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TreecfWarning)
        cut = exp.explain(
            x, target=target, backend="exact", seed=0, node_budget=50_000, warm_start=False
        )
    _save("plot_certification_trace", _fig_of(plot_certification_trace(cut)))

    batch = exp.explain_batch(X_bg[:20], target=target, seed=0)
    _save("plot_batch_summary", _fig_of(plot_batch_summary(batch)))
    _save("plot_batch_levers", _fig_of(plot_batch_levers(batch)))
    _save("plot_batch_matrix", _fig_of(plot_batch_matrix(batch, explainer=exp)))
    _save("plot_batch_deltas", _fig_of(plot_batch_deltas(batch, explainer=exp)))

    groups = ["thin-file" if row[3] < 100 else "established" for row in X_bg[:20]]
    _save(
        "plot_recourse_burden",
        _fig_of(plot_recourse_burden(batch, groups, min_group_size=3)),
    )

    # the sample portfolio report: one self-contained HTML page under
    # docs/guide/samples/, and its dominant-levers chart as the page's figure
    SAMPLES.mkdir(parents=True, exist_ok=True)
    portfolio_report(
        batch, groups, explainer=exp, path=SAMPLES / "portfolio_report.html",
        title="Sample portfolio report", min_group_size=3,
    )
    with tempfile.TemporaryDirectory() as tmp:
        md_path = pathlib.Path(tmp) / "report.md"
        portfolio_report(
            batch, groups, explainer=exp, path=md_path, format="markdown", min_group_size=3
        )
        shutil.copy(
            md_path.parent / "report_figures" / "dominant_levers.png",
            OUT / "portfolio_report.png",
        )
        print(f"wrote {(OUT / 'portfolio_report.png').relative_to(REPO)}")
    print(f"wrote {(SAMPLES / 'portfolio_report.html').relative_to(REPO)}")


if __name__ == "__main__":
    main()
