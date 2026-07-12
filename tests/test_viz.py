"""Viz smoke tests (D15): figures render on Agg with the expected structure."""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from treecf import Counterfactual, Infeasible, Target, TreecfError  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures() -> object:
    yield
    import matplotlib.pyplot as plt

    plt.close("all")  # >20 open figures raises under filterwarnings=error
from treecf.viz import (  # noqa: E402
    plot_alternatives,
    plot_changes,
    plot_counterfactuals,
    plot_ladder,
    plot_tradeoff,
)


def _cf(
    changes: dict[str, tuple[float, float]],
    distance: float,
    score_prob: float | None = None,
) -> Counterfactual:
    return Counterfactual(
        x_cf=np.zeros(3),
        changes=changes,
        distance=distance,
        n_changed=len(changes),
        score_raw=0.5,
        score_prob=score_prob,
        proof="heuristic",
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


def test_plot_alternatives_one_legend_entry_per_plan() -> None:
    plans = [
        _cf({"a": (0.0, 1.0), "b": (0.0, 2.0)}, 1.0),
        _cf({"a": (0.0, 3.0)}, 2.0),
    ]
    ax = plot_alternatives(plans)
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend_labels == ["plan 1 (J=1)", "plan 2 (J=2)"]
    yticklabels = [t.get_text() for t in ax.get_yticklabels()]
    assert yticklabels == ["a", "b"]  # 'a' used by both plans -> first


def test_plot_alternatives_skips_infeasible_records_and_marks_nan() -> None:
    from treecf.batch import BatchRecord

    feasible = BatchRecord(
        id=0, k=0, feasible=True, x_cf=np.zeros(3),
        changes={"c": (7.0, float("nan"))}, distance=0.4, n_changed=1,
        score_raw=0.1, score_prob=None,
    )
    infeasible = BatchRecord(
        id=0, k=0, feasible=False, x_cf=None, changes={},
        distance=None, n_changed=None, score_raw=None, score_prob=None,
    )
    ax = plot_alternatives([feasible, infeasible])
    assert "1 alternative plan(s)" in ax.get_title()
    assert any("NaN" in t.get_text() for t in ax.texts)


def test_plot_alternatives_empty_raises() -> None:
    with pytest.raises(TreecfError):
        plot_alternatives([])


def test_plot_alternatives_explainer_standardizes_by_sigma() -> None:
    from treecf import Explainer
    from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree

    stump = Tree(
        nodes=(
            Node(0, 0, 1.0, SplitOp.LT, True, 1, 2, None),
            Node(1, None, None, None, None, None, None, 0.0),
            Node(2, None, None, None, None, None, None, 1.0),
        )
    )
    ir = EnsembleIR(
        trees=(stump,), base_score=0.0, link=Link.IDENTITY,
        n_features=3, feature_names=("a", "b", "c"), meta={},
    )
    exp = Explainer(ir, normalizers=np.full(3, 4.0))
    ax = plot_alternatives([_cf({"a": (0.0, 2.0)}, 2.0)], explainer=exp)
    dots = [ln for ln in ax.lines if ln.get_marker() == "o"]
    assert {xs for ln in dots for xs in ln.get_xdata()} == {0.0, 0.5}  # 2.0 / sigma 4.0
    assert "σ" in ax.get_xlabel()


def test_plot_tradeoff_probability_space_with_target_lines() -> None:
    plans = [
        _cf({"a": (0.0, 1.0)}, 1.0, score_prob=0.25),
        _cf({"b": (0.0, 2.0)}, 2.0, score_prob=0.10),
    ]
    ax = plot_tradeoff(plans, target=Target.probability(range=(0.0, 0.30)))
    dots = [ln for ln in ax.lines if ln.get_marker() == "o"]
    assert len(dots) == 2
    assert ax.get_ylabel() == "model probability"
    line_ys = {ln.get_ydata()[0] for ln in ax.lines if ln.get_marker() != "o"}
    assert 0.30 in line_ys  # finite target bound drawn; -inf/0.0 lo also finite
    assert [t.get_text() for t in ax.texts] == ["1", "2"]


def test_plot_tradeoff_raw_space_without_prob() -> None:
    plans = [_cf({"a": (0.0, 1.0)}, 1.0)]
    ax = plot_tradeoff(plans, target=Target.raw(op=">=", value=0.4))
    assert ax.get_ylabel() == "raw score"
    line_ys = {ln.get_ydata()[0] for ln in ax.lines if ln.get_marker() != "o"}
    assert 0.4 in line_ys  # the infinite upper bound is skipped


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


def _waterfall_setup():
    from treecf import Explainer, Target
    from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree

    def stump(feature, threshold, right):
        return Tree(
            nodes=(
                Node(0, feature, threshold, SplitOp.LT, True, 1, 2, None),
                Node(1, None, None, None, None, None, None, 0.0),
                Node(2, None, None, None, None, None, None, right),
            )
        )

    ir = EnsembleIR(
        trees=(stump(0, 1.0, 1.0), stump(1, 1.0, 0.4)),
        base_score=-0.2,
        link=Link.IDENTITY,
        n_features=2,
        feature_names=("big", "small"),
        meta={},
    )
    exp = Explainer(ir, normalizers=np.ones(2))
    res = exp.explain(
        np.zeros(2), target=Target.raw(op=">=", value=1.1), seed=0
    )
    assert isinstance(res, Counterfactual) and res.n_changed == 2
    return exp, res


def test_plot_waterfall_bars_sum_to_score_move() -> None:
    from treecf.viz import plot_waterfall

    exp, res = _waterfall_setup()
    ax = plot_waterfall(exp, res)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == ["big", "small"]  # largest single effect first
    widths = [p.get_width() for p in ax.patches]
    assert sum(abs(w) for w in widths) == pytest.approx(abs(res.score_raw - (-0.2)))
    texts = " ".join(t.get_text() for t in ax.texts)
    assert "-0.2" in texts or "−0.2" in texts  # factual score annotated


def test_plot_waterfall_probability_space_for_sigmoid() -> None:
    from treecf import Explainer, Target
    from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree
    from treecf.viz import plot_waterfall

    nodes = (
        Node(0, 0, 1.0, SplitOp.LT, True, 1, 2, None),
        Node(1, None, None, None, None, None, None, -1.0),
        Node(2, None, None, None, None, None, None, 1.0),
    )
    ir = EnsembleIR(
        trees=(Tree(nodes=nodes),),
        base_score=0.0,
        link=Link.SIGMOID,
        n_features=1,
        feature_names=("x",),
        meta={},
    )
    exp = Explainer(ir, normalizers=np.ones(1))
    res = exp.explain(
        np.zeros(1), target=Target.probability(op=">=", value=0.6), seed=0
    )
    ax = plot_waterfall(exp, res, target=Target.probability(op=">=", value=0.6))
    assert ax.get_xlim()[0] >= -0.05 and ax.get_xlim()[1] <= 1.05  # probability axis
    assert len(ax.patches) == 1


def test_plot_effort_bars_sum_to_distance() -> None:
    from treecf.viz import plot_effort

    exp, res = _waterfall_setup()
    ax = plot_effort(exp, res)
    widths = [p.get_width() for p in ax.patches]
    assert sum(widths) == pytest.approx(res.distance)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert set(labels) == {"big", "small"}
