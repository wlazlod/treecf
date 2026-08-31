"""Regenerate every committed docs figure from the docs model.

One PNG per public plot function, written under ``docs/guide/img/``. Not run
at build time — see docs/README.md for when a regenerated figure should be
committed (only when the picture actually changed; PNGs are not
byte-reproducible across matplotlib versions or platforms).

Run from anywhere: ``uv run python docs/scripts/generate_figures.py``
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from treecf import Explainer, Range, Target
from treecf.viz import (
    plot_alternatives,
    plot_changes,
    plot_counterfactuals,
    plot_effort,
    plot_ladder,
    plot_recourse_map,
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
MODEL = REPO / "tests" / "fixtures" / "docs_model.json"
OUT = REPO / "docs" / "guide" / "img"
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
    X_bg = docs_background()
    exp = Explainer(str(MODEL), background=X_bg, categories={"occupation": OCCUPATIONS})
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

    ladder = exp.explain(
        x, target=Target.bands({"A": (0.0, 0.01), "B": (0.01, 0.05), "C": (0.05, 0.15)}), seed=0
    )
    _save("plot_ladder", _fig_of(plot_ladder(ladder)))

    # One Range constraint so the region shows both cap markers: bounds
    # stopped by the model and bounds stopped by a constraint.
    exp_rng = Explainer(
        str(MODEL),
        background=X_bg,
        categories={"occupation": OCCUPATIONS},
        constraints=[Range("tenure_months", 0.0, 140.0)],
    )
    certified = exp_rng.explain(x, target=target, backend="exact", region=True, seed=0)
    _save("plot_region", _fig_of(plot_region(exp_rng, x, certified)))

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


if __name__ == "__main__":
    main()
