"""Viz smoke tests: figures render on Agg with the expected structure."""

from __future__ import annotations

import math

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


def test_plot_alternatives_mapping_uses_names_and_skips_infeasible() -> None:
    outcomes = {
        "debt": _cf({"a": (0.0, 1.0)}, 1.0),
        "income": Infeasible(reason="unreachable"),
        "behavior": _cf({"b": (0.0, 2.0)}, 2.0),
    }
    ax = plot_alternatives(outcomes)
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend_labels == ["debt (J=1)", "behavior (J=2)"]


def test_plot_tradeoff_mapping_labels_dots() -> None:
    outcomes = {
        "debt": _cf({"a": (0.0, 1.0)}, 1.0, score_prob=0.2),
        "income": Infeasible(reason="unreachable"),
    }
    ax = plot_tradeoff(outcomes)
    assert [t.get_text() for t in ax.texts] == ["debt"]


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


def _map_explainer(normalizers=None, constraints=()):
    """Small inline-IR Explainer over features a/b/c, default sigmoid link."""
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
        trees=(stump,),
        base_score=0.0,
        link=Link.SIGMOID,
        n_features=3,
        feature_names=("a", "b", "c"),
        meta={},
    )
    norm = np.ones(3) if normalizers is None else normalizers
    return Explainer(ir, normalizers=norm, constraints=constraints)


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


def test_display_interval_raw_space_explicit_no_link_mapping() -> None:
    from treecf.ir.model import Link
    from treecf.viz import _display_interval

    target = Target.raw(op=">=", value=0.5)
    lo, hi = _display_interval(target, Link.SIGMOID, "raw")
    assert lo == 0.5
    assert hi == math.inf


def test_display_interval_auto_identity_passthrough() -> None:
    from treecf.ir.model import Link
    from treecf.viz import _display_interval

    target = Target.raw(op=">=", value=0.5)
    lo, hi = _display_interval(target, Link.IDENTITY, "auto")
    assert lo == 0.5
    assert hi == math.inf


def test_display_interval_auto_sigmoid_maps_finite_endpoint_only() -> None:
    from treecf.ir.model import Link
    from treecf.viz import _display_interval

    target = Target.probability(range=(0.0, 0.05))
    lo, hi = _display_interval(target, Link.SIGMOID, "auto")
    assert lo == -math.inf
    assert hi == pytest.approx(0.05)


def test_plans_and_failures_bare_counterfactual() -> None:
    from treecf.viz import _plans_and_failures

    cf = _cf({"a": (0.0, 1.0)}, 1.0)
    plans, failures = _plans_and_failures(cf)
    assert plans == [(None, cf)]
    assert failures == []


def test_plans_and_failures_bare_infeasible() -> None:
    from treecf.viz import _plans_and_failures

    inf = Infeasible(reason="unreachable")
    plans, failures = _plans_and_failures(inf)
    assert plans == []
    assert failures == [(None, inf)]


def test_plans_and_failures_mapping_keeps_all_labeled_by_key_in_order() -> None:
    from treecf.viz import _plans_and_failures

    debt_cf = _cf({"a": (0.0, 1.0)}, 1.0)
    income_inf = Infeasible(reason="unreachable")
    behavior_cf = _cf({"b": (0.0, 2.0)}, 2.0)
    outcomes = {"debt": debt_cf, "income": income_inf, "behavior": behavior_cf}
    plans, failures = _plans_and_failures(outcomes)
    assert plans == [("debt", debt_cf), ("behavior", behavior_cf)]
    assert failures == [("income", income_inf)]


def test_plans_and_failures_sequence_of_batch_records() -> None:
    from treecf.batch import BatchRecord
    from treecf.viz import _plans_and_failures

    feasible = BatchRecord(
        id=0, k=0, feasible=True, x_cf=np.zeros(3),
        changes={"a": (0.0, 1.0)}, distance=1.0, n_changed=1,
        score_raw=0.1, score_prob=None, coalition="grp1",
    )
    infeasible = BatchRecord(
        id=0, k=0, feasible=False, x_cf=None, changes={},
        distance=None, n_changed=None, score_raw=None, score_prob=None,
        coalition="grp2",
    )
    plans, failures = _plans_and_failures([feasible, infeasible])
    assert plans == [("grp1", feasible)]
    assert failures == [("grp2", infeasible)]


def test_format_plan_orders_changes_by_descending_effort() -> None:
    from treecf.viz import _format_plan

    exp = _map_explainer(normalizers=np.array([2.0, 1.0, 4.0]))
    plan = _cf({"a": (0.0, 2.0), "b": (0.0, 3.0), "c": (0.0, 4.0)}, distance=5.0)
    text = _format_plan(None, plan, exp)
    assert text == "b = 3, a = 2, c = 4 (J=5)"


def test_format_plan_nan_legs_use_drop_and_provide_words() -> None:
    from treecf import AllowMissing
    from treecf.viz import _format_plan

    exp = _map_explainer(
        constraints=[
            AllowMissing("a", delta_miss=0.3),
            AllowMissing("b", delta_miss=0.3, delta_from_miss=0.5),
        ]
    )
    plan = _cf(
        {"a": (1.0, float("nan")), "b": (float("nan"), 2.0), "c": (0.0, 1.0)}, distance=1.8
    )
    text = _format_plan(None, plan, exp)
    assert text == "c = 1, provide b = 2, drop a (J=1.8)"


def test_format_plan_truncates_and_dresses_with_more_count() -> None:
    from treecf.viz import _format_plan

    exp = _map_explainer()
    plan = _cf({"a": (0.0, 1.0), "b": (0.0, 2.0), "c": (0.0, 3.0)}, distance=6.0)
    text = _format_plan(None, plan, exp, max_changes=2)
    assert text == "c = 3, b = 2 (+1 more) (J=6)"


def test_format_plan_uses_region_phrase_when_available() -> None:
    from types import SimpleNamespace

    from treecf.viz import _format_plan

    exp = _map_explainer()

    class _RegionStub:
        def describe(self) -> dict[str, str]:
            return {"a": "keep a within the safe band"}

    plan = SimpleNamespace(
        changes={"a": (0.0, 1.0)}, distance=1.0, region=_RegionStub()
    )
    text = _format_plan("bandA", plan, exp)
    assert text == "bandA: keep a within the safe band (J=1)"


def test_format_plan_schematic_joins_with_if_and_and() -> None:
    from treecf.viz import _format_plan

    exp = _map_explainer()
    plan = _cf({"a": (0.0, 1.0), "b": (0.0, 2.0)}, distance=3.0)
    text = _format_plan("bandA", plan, exp, schematic=True)
    assert text == "bandA: If b = 2 and a = 1 (J=3)"


def test_plot_recourse_map_smoke_single_cf() -> None:
    from treecf.viz import plot_recourse_map

    exp = _map_explainer()
    plan = _cf({"a": (0.0, 2.0)}, distance=1.0, score_prob=0.7)
    ax = plot_recourse_map(exp, np.zeros(3), [plan], Target.probability(op=">=", value=0.6))
    assert ax.get_title() == "1 recourse option(s)"


def test_plot_recourse_map_returns_given_ax() -> None:
    import matplotlib.pyplot as plt

    from treecf.viz import plot_recourse_map

    exp = _map_explainer()
    plan = _cf({"a": (0.0, 2.0)}, distance=1.0, score_prob=0.7)
    _, ax0 = plt.subplots()
    ax = plot_recourse_map(
        exp, np.zeros(3), [plan], Target.probability(op=">=", value=0.6), ax=ax0
    )
    assert ax is ax0


def test_plot_recourse_map_marker_and_arrow_counts() -> None:
    from treecf.viz import plot_recourse_map

    exp = _map_explainer()
    plans = [
        _cf({"a": (0.0, 2.0)}, distance=1.0, score_prob=0.6),
        _cf({"b": (0.0, 3.0)}, distance=2.0, score_prob=0.7),
    ]
    ax = plot_recourse_map(
        exp, np.zeros(3), plans, Target.probability(op=">=", value=0.6), annotate=False
    )
    dots = [ln for ln in ax.lines if ln.get_marker() == "o"]
    assert len(dots) == 3  # factual + 2 plans
    arrows = [t for t in ax.texts if t.get_text() == "" and t.arrow_patch is not None]
    assert len(arrows) == 2


def test_plot_recourse_map_draws_target_band() -> None:
    from treecf.viz import plot_recourse_map

    exp = _map_explainer()
    plan = _cf({"a": (0.0, 2.0)}, distance=1.0, score_prob=0.7)
    ax = plot_recourse_map(exp, np.zeros(3), [plan], Target.probability(op=">=", value=0.6))
    assert len(ax.patches) == 1  # the axvspan band


def test_plot_recourse_map_inverts_when_band_is_below_high_factual() -> None:
    from treecf.viz import plot_recourse_map

    exp = _map_explainer()
    plan = _cf({"a": (2.0, 0.0)}, distance=1.0, score_prob=0.2)
    ax = plot_recourse_map(
        exp, np.array([2.0, 0.0, 0.0]), [plan], Target.probability(op="<=", value=0.3)
    )
    assert ax.xaxis_inverted()


def test_plot_recourse_map_no_inversion_when_factual_inside_band() -> None:
    from treecf.viz import plot_recourse_map

    exp = _map_explainer()
    plan = _cf({"b": (0.0, 1.0)}, distance=1.0, score_prob=0.6)
    ax = plot_recourse_map(
        exp, np.zeros(3), [plan], Target.probability(range=(0.3, 0.7))
    )
    assert not ax.xaxis_inverted()


def test_plot_recourse_map_too_many_plans_raises() -> None:
    from treecf.viz import plot_recourse_map

    exp = _map_explainer()
    plans = [_cf({"a": (0.0, float(i))}, distance=float(i + 1)) for i in range(11)]
    with pytest.raises(TreecfError, match="at most 10 plans"):
        plot_recourse_map(exp, np.zeros(3), plans, Target.probability(op=">=", value=0.5))


def test_plot_recourse_map_empty_raises_no_plans_message() -> None:
    from treecf.viz import plot_recourse_map

    exp = _map_explainer()
    with pytest.raises(TreecfError, match="no plans to plot"):
        plot_recourse_map(exp, np.zeros(3), [], Target.probability(op=">=", value=0.5))


def test_plot_recourse_map_identity_link_never_touches_score_prob() -> None:
    from types import SimpleNamespace

    from treecf import Explainer
    from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree
    from treecf.viz import plot_recourse_map

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
    exp = Explainer(ir, normalizers=np.ones(3))
    plan = SimpleNamespace(changes={"a": (0.0, 2.0)}, distance=1.0, score_raw=0.8)
    ax = plot_recourse_map(exp, np.zeros(3), [plan], Target.raw(op=">=", value=0.5))
    assert ax.get_xlabel() == "model output (raw score)"


def test_plot_recourse_map_annotate_false_and_no_factual_label_has_no_text() -> None:
    from treecf.viz import plot_recourse_map

    exp = _map_explainer()
    plans = [
        _cf({"a": (0.0, 2.0)}, distance=1.0, score_prob=0.6),
        _cf({"b": (0.0, 3.0)}, distance=2.0, score_prob=0.7),
    ]
    ax = plot_recourse_map(
        exp,
        np.zeros(3),
        plans,
        Target.probability(op=">=", value=0.6),
        annotate=False,
        show_factual_label=False,
    )
    labels = [t.get_text() for t in ax.texts if t.get_text()]
    assert labels == []


def test_plot_recourse_map_annotate_false_alone_suppresses_factual_block() -> None:
    from treecf.viz import plot_recourse_map

    exp = _map_explainer()
    plan = _cf({"a": (1.0, 2.0)}, distance=1.0, score_prob=0.65)
    ax = plot_recourse_map(
        exp,
        np.array([1.0, 0.0, 0.0]),
        [plan],
        Target.probability(op=">=", value=0.6),
        annotate=False,  # show_factual_label left at its default True
    )
    assert not any("a = 1" in t.get_text() for t in ax.texts)


def test_plot_recourse_map_annotate_false_still_labels_infeasible_entries() -> None:
    from treecf.viz import plot_recourse_map

    exp = _map_explainer()
    results = {
        "debt": _cf({"a": (0.0, 1.0)}, distance=1.0, score_prob=0.65),
        "income": Infeasible(reason="unreachable"),
    }
    ax = plot_recourse_map(
        exp, np.zeros(3), results, Target.probability(op=">=", value=0.6), annotate=False
    )
    texts = [t.get_text() for t in ax.texts]
    assert "income: infeasible" in texts


def test_plot_recourse_map_infeasible_markers_and_labels() -> None:
    from treecf.viz import plot_recourse_map

    exp = _map_explainer()
    results = {
        "debt": _cf({"a": (0.0, 1.0)}, distance=1.0, score_prob=0.65),
        "behavior": _cf({"b": (0.0, 2.0)}, distance=2.0, score_prob=0.7),
        "income": Infeasible(reason="unreachable"),
    }
    ax = plot_recourse_map(exp, np.zeros(3), results, Target.probability(op=">=", value=0.6))
    greens = [ln for ln in ax.lines if ln.get_marker() == "o" and ln.get_color() == "tab:green"]
    greys = [ln for ln in ax.lines if ln.get_marker() == "x"]
    assert len(greens) == 2
    assert len(greys) == 1
    texts = [t.get_text() for t in ax.texts]
    assert "income: infeasible" in texts
    assert not any("certified" in t for t in texts)


def test_plot_recourse_map_all_infeasible_draws_only_markers() -> None:
    from treecf.viz import plot_recourse_map

    exp = _map_explainer()
    failures = {
        "income": Infeasible(reason="unreachable"),
        "debt": Infeasible(reason="constraint violated"),
    }
    ax = plot_recourse_map(exp, np.zeros(3), failures, Target.probability(op=">=", value=0.6))
    greens = [ln for ln in ax.lines if ln.get_marker() == "o" and ln.get_color() == "tab:green"]
    greys = [ln for ln in ax.lines if ln.get_marker() == "x"]
    assert len(greens) == 0
    assert len(greys) == 2


def test_plot_recourse_map_infeasible_certified_stub_appends_word() -> None:
    from types import SimpleNamespace

    from treecf.viz import plot_recourse_map

    exp = _map_explainer()
    plan = _cf({"a": (0.0, 1.0)}, distance=1.0, score_prob=0.65)
    infeasible = SimpleNamespace(feasible=False, proof="certified")
    ax = plot_recourse_map(
        exp, np.zeros(3), [plan, infeasible], Target.probability(op=">=", value=0.6)
    )
    texts = [t.get_text() for t in ax.texts]
    assert "infeasible (certified)" in texts


def test_plot_recourse_map_labels_use_drop_and_provide_words() -> None:
    from treecf import AllowMissing
    from treecf.viz import plot_recourse_map

    exp = _map_explainer(constraints=[AllowMissing("a", delta_miss=0.3)])
    plan = _cf({"a": (1.0, float("nan"))}, distance=0.5, score_prob=0.65)
    ax = plot_recourse_map(
        exp, np.array([1.0, 0.0, 0.0]), [plan], Target.probability(op=">=", value=0.6)
    )
    texts = " ".join(t.get_text() for t in ax.texts)
    assert "drop a" in texts


def test_plot_recourse_map_max_changes_per_label_truncates() -> None:
    from treecf.viz import plot_recourse_map

    exp = _map_explainer()
    plan = _cf(
        {"a": (0.0, 1.0), "b": (0.0, 2.0), "c": (0.0, 3.0)}, distance=6.0, score_prob=0.65
    )
    ax = plot_recourse_map(
        exp,
        np.zeros(3),
        [plan],
        Target.probability(op=">=", value=0.6),
        max_changes_per_label=1,
    )
    texts = " ".join(t.get_text() for t in ax.texts)
    assert "(+2 more)" in texts


def test_plot_recourse_map_factual_label_present_and_absent() -> None:
    from treecf.viz import plot_recourse_map

    exp = _map_explainer()
    plan = _cf({"a": (1.0, 2.0)}, distance=1.0, score_prob=0.65)
    ax_on = plot_recourse_map(
        exp, np.array([1.0, 0.0, 0.0]), [plan], Target.probability(op=">=", value=0.6)
    )
    assert any("a = 1" in t.get_text() for t in ax_on.texts)

    ax_off = plot_recourse_map(
        exp,
        np.array([1.0, 0.0, 0.0]),
        [plan],
        Target.probability(op=">=", value=0.6),
        show_factual_label=False,
    )
    assert not any("a = 1" in t.get_text() for t in ax_off.texts)


def test_plot_recourse_map_region_phrase_end_to_end() -> None:
    from types import SimpleNamespace

    from treecf.viz import plot_recourse_map

    exp = _map_explainer()

    class _RegionStub:
        def describe(self) -> dict[str, str]:
            return {"a": "keep a within [0, 5]"}

    plan = SimpleNamespace(
        changes={"a": (0.0, 1.0)}, distance=1.0, score_prob=0.65, region=_RegionStub()
    )
    ax = plot_recourse_map(exp, np.zeros(3), [plan], Target.probability(op=">=", value=0.6))
    texts = " ".join(t.get_text() for t in ax.texts)
    assert "keep a within [0, 5]" in texts


def test_plot_recourse_map_schematic_hides_ticks_spines_and_band() -> None:
    from treecf.viz import plot_recourse_map

    exp = _map_explainer()
    plan = _cf({"a": (0.0, 2.0)}, distance=1.0, score_prob=0.7)
    ax = plot_recourse_map(
        exp, np.zeros(3), [plan], Target.probability(op=">=", value=0.6), schematic=True
    )
    assert len(ax.get_xticks()) == 0
    assert len(ax.get_yticks()) == 0
    assert not any(spine.get_visible() for spine in ax.spines.values())
    assert len(ax.patches) == 0  # axvspan band dropped
    assert ax.get_title() == ""
    assert ax.get_xlabel() == ""
    assert ax.get_ylabel() == ""


def test_plot_recourse_map_schematic_wavy_boundary_has_200_samples_and_spans_ylim() -> None:
    from treecf.viz import plot_recourse_map

    exp = _map_explainer()
    plan = _cf({"a": (0.0, 2.0)}, distance=1.0, score_prob=0.7)
    ax = plot_recourse_map(
        exp, np.zeros(3), [plan], Target.probability(op=">=", value=0.6), schematic=True
    )
    waves = [
        ln for ln in ax.lines if ln.get_linestyle() == "--" and ln.get_color() == "tab:blue"
    ]
    assert len(waves) == 1  # one finite target edge
    xdata, ydata = waves[0].get_data()
    assert len(xdata) == 200
    ylo, yhi = ax.get_ylim()
    assert ydata[0] == pytest.approx(ylo)
    assert ydata[-1] == pytest.approx(yhi)


def test_plot_recourse_map_schematic_region_labels_and_boundary_annotation() -> None:
    from treecf.viz import plot_recourse_map

    exp = _map_explainer()
    plan = _cf({"a": (0.0, 2.0)}, distance=1.0, score_prob=0.7)
    ax = plot_recourse_map(
        exp,
        np.zeros(3),
        [plan],
        Target.probability(op=">=", value=0.6),
        schematic=True,
        annotate=False,  # region labels/boundary draw regardless of annotate
    )
    texts = [t.get_text() for t in ax.texts]
    assert "Reject" in texts
    assert "Accept" in texts
    assert "ML model decision boundary" in texts


def test_plot_recourse_map_schematic_custom_region_labels() -> None:
    from treecf.viz import plot_recourse_map

    exp = _map_explainer()
    plan = _cf({"a": (0.0, 2.0)}, distance=1.0, score_prob=0.7)
    ax = plot_recourse_map(
        exp,
        np.zeros(3),
        [plan],
        Target.probability(op=">=", value=0.6),
        schematic=True,
        region_labels=("No", "Yes"),
    )
    texts = [t.get_text() for t in ax.texts]
    assert "No" in texts
    assert "Yes" in texts
    assert "Reject" not in texts
    assert "Accept" not in texts


def test_plot_recourse_map_schematic_plan_labels_start_with_if() -> None:
    from treecf.viz import plot_recourse_map

    exp = _map_explainer()
    plan = _cf({"a": (0.0, 2.0)}, distance=1.0, score_prob=0.7)
    ax = plot_recourse_map(
        exp, np.zeros(3), [plan], Target.probability(op=">=", value=0.6), schematic=True
    )
    texts = [t.get_text() for t in ax.texts]
    assert any(t.startswith("If ") for t in texts)


def _axes_frac_x(ax, data_x):
    """Data-space x, transformed to axes-fraction x (0=left edge, 1=right edge)."""
    display = ax.transData.transform((data_x, 0.0))
    return float(ax.transAxes.inverted().transform(display)[0])


def test_plot_recourse_map_schematic_accept_label_matches_accept_screen_side() -> None:
    """Accept/Reject label sides track real geometry, not the inversion flag directly.

    ``invert_xaxis()`` fires exactly when the band lies below the factual, which
    flips the data->screen mapping so the accept side is screen-right in both
    orientations — checked here against the actual plotted dot positions rather
    than hardcoded fx values, for both an increase-direction and a
    decrease-direction target.
    """
    from treecf.viz import plot_recourse_map

    exp = _map_explainer()
    cases = [
        (
            Target.probability(op=">=", value=0.6),
            _cf({"a": (0.0, 2.0)}, distance=1.0, score_prob=0.7),
            np.zeros(3),
            False,
        ),
        (
            Target.probability(op="<=", value=0.3),
            _cf({"a": (2.0, 0.0)}, distance=1.0, score_prob=0.2),
            np.array([2.0, 0.0, 0.0]),
            True,
        ),
    ]
    for target, plan, x_factual, expect_inverted in cases:
        ax = plot_recourse_map(exp, x_factual, [plan], target, schematic=True)
        assert bool(ax.xaxis_inverted()) == expect_inverted

        accept_fx = next(t for t in ax.texts if t.get_text() == "Accept").get_position()[0]
        reject_fx = next(t for t in ax.texts if t.get_text() == "Reject").get_position()[0]
        assert accept_fx > reject_fx

        plan_dot = next(
            ln for ln in ax.lines if ln.get_marker() == "o" and ln.get_color() == "tab:green"
        )
        factual_dot = next(
            ln for ln in ax.lines if ln.get_marker() == "o" and ln.get_color() == "tab:red"
        )
        plan_fx = _axes_frac_x(ax, plan_dot.get_xdata()[0])
        factual_fx = _axes_frac_x(ax, factual_dot.get_xdata()[0])

        # Accept sits on the same screen side as the (accepted) plan's dot;
        # Reject sits on the same screen side as the factual's dot.
        assert (accept_fx > 0.5) == (plan_fx > 0.5)
        assert (reject_fx > 0.5) == (factual_fx > 0.5)
