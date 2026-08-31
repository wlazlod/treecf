"""Structural checks for the batch-level plots (Agg backend, no image comparison)."""

from __future__ import annotations

import math

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from treecf import Explainer, TreecfError  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures() -> object:
    yield
    import matplotlib.pyplot as plt

    plt.close("all")  # >20 open figures raises under filterwarnings=error
from treecf.batch import BatchRecord, BatchResult  # noqa: E402
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree  # noqa: E402
from treecf.viz_batch import (  # noqa: E402
    plot_batch_deltas,
    plot_batch_levers,
    plot_batch_matrix,
    plot_batch_summary,
)


def _rec(
    rid: object,
    k: int,
    changes: dict[str, tuple[float, float]],
    distance: float,
    *,
    feasible: bool = True,
    blocked: str | None = None,
) -> BatchRecord:
    return BatchRecord(
        id=rid, k=k, feasible=feasible,
        x_cf=np.zeros(3) if feasible else None,
        changes=changes, distance=distance if feasible else None,
        n_changed=len(changes) if feasible else None,
        score_raw=1.0 if feasible else None, score_prob=None,
        blocked_lever=blocked,
    )


def _infeasible(rid: object) -> BatchRecord:
    return BatchRecord(
        id=rid, k=0, feasible=False, x_cf=None, changes={},
        distance=None, n_changed=None, score_raw=None, score_prob=None,
    )


def _batch(
    records: list[BatchRecord],
    diversity: str = "seeds",
    essential: dict[object, list[str]] | None = None,
) -> BatchResult:
    return BatchResult(
        feature_names=("a", "b", "c"), diversity=diversity,
        records=tuple(records), essential_levers=essential or {},
    )


THREE_PLANS = [
    _rec(0, 0, {"a": (0.0, 2.0)}, 2.0),
    _rec(1, 0, {"a": (0.0, 1.0), "b": (0.0, -1.0)}, 1.5),
    _rec(2, 0, {"b": (0.0, 0.5)}, 0.5),
]


def _stump(feature: int) -> Tree:
    return Tree(
        nodes=(
            Node(0, feature, 1.0, SplitOp.LT, True, 1, 2, None),
            Node(1, None, None, None, None, None, None, 0.0),
            Node(2, None, None, None, None, None, None, 1.0),
        )
    )


def _ir() -> EnsembleIR:
    return EnsembleIR(
        trees=(_stump(0), _stump(1), _stump(2)),
        base_score=0.0, link=Link.IDENTITY, n_features=3,
        feature_names=("a", "b", "c"), meta={},
    )


class TestLevers:
    def test_frequency_order_and_normalized_width(self) -> None:
        ax = plot_batch_levers(_batch(THREE_PLANS))
        assert [t.get_text() for t in ax.get_yticklabels()] == ["a", "b"]
        widths = [p.get_width() for p in ax.patches]
        assert max(widths) == pytest.approx(2 / 3)

    def test_nan_direction_gets_own_segment(self) -> None:
        batch = _batch([_rec(0, 0, {"c": (7.0, math.nan)}, 1.0)])
        ax = plot_batch_levers(batch)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ["NaN"]

    def test_essential_levers_annotated(self) -> None:
        batch = _batch(THREE_PLANS, diversity="lever-blocking", essential={0: ["a"]})
        ax = plot_batch_levers(batch)
        assert any("essential" in t.get_text() for t in ax.texts)

    def test_all_infeasible_raises(self) -> None:
        with pytest.raises(TreecfError):
            plot_batch_levers(_batch([_infeasible(0)]))


class TestMatrix:
    def test_binary_matrix_shape_and_values(self) -> None:
        ax = plot_batch_matrix(_batch(THREE_PLANS))
        data = np.asarray(ax.images[0].get_array())
        assert data.shape == (3, 2)
        assert set(np.unique(data)) <= {0.0, 1.0}

    def test_effort_shading_uses_change_magnitude(self) -> None:
        exp = Explainer(_ir(), normalizers=np.ones(3))
        ax = plot_batch_matrix(_batch(THREE_PLANS), explainer=exp)
        data = np.asarray(ax.images[0].get_array())
        assert float(data.max()) == pytest.approx(2.0)  # |0 -> 2| on feature a

    def test_row_labels_suppressed_beyond_cap(self) -> None:
        ax = plot_batch_matrix(_batch(THREE_PLANS), max_row_labels=1)
        assert not ax.get_yticklabels()
        assert "3" in ax.get_ylabel()

    def test_rows_sorted_by_distance(self) -> None:
        ax = plot_batch_matrix(_batch(THREE_PLANS))
        assert "J=0.5" in ax.get_yticklabels()[0].get_text()

    def test_feature_space_mismatch_raises(self) -> None:
        exp = Explainer(_ir(), normalizers=np.ones(3))
        batch = BatchResult(
            feature_names=("x", "y"), diversity="seeds",
            records=(_rec(0, 0, {"x": (0.0, 1.0)}, 1.0),), essential_levers={},
        )
        with pytest.raises(TreecfError):
            plot_batch_matrix(batch, explainer=exp)


class TestSummary:
    def test_three_panels_and_feasibility_counts(self) -> None:
        axs = plot_batch_summary(_batch([*THREE_PLANS, _infeasible(3)]))
        assert len(axs) == 3
        assert axs[0].patches  # histogram drawn
        heights = [p.get_height() for p in axs[2].patches]
        assert heights == [3, 1]
        assert "75%" in axs[2].get_title()

    def test_all_infeasible_renders_without_raising(self) -> None:
        axs = plot_batch_summary(_batch([_infeasible(0), _infeasible(1)]))
        assert any("no feasible plans" in t.get_text() for t in axs[0].texts)
        assert "0%" in axs[2].get_title()


class TestDeltas:
    def test_top_n_limits_rows(self) -> None:
        ax = plot_batch_deltas(_batch(THREE_PLANS), top_n=1)
        assert [t.get_text() for t in ax.get_yticklabels()] == ["a"]

    def test_dots_match_numeric_delta_count(self) -> None:
        ax = plot_batch_deltas(_batch(THREE_PLANS))
        dots = [ln for ln in ax.lines if ln.get_marker() == "o"]
        assert sum(len(ln.get_xdata()) for ln in dots) == 4  # a: 2 deltas, b: 2 deltas

    def test_nan_transition_annotated(self) -> None:
        batch = _batch([_rec(0, 0, {"c": (7.0, math.nan)}, 1.0)])
        ax = plot_batch_deltas(batch)
        assert any("NaN" in t.get_text() for t in ax.texts)

    def test_explainer_standardizes_by_sigma(self) -> None:
        exp = Explainer(_ir(), normalizers=np.full(3, 4.0))
        batch = _batch([_rec(0, 0, {"a": (0.0, 2.0)}, 2.0)])
        ax = plot_batch_deltas(batch, explainer=exp)
        dots = [ln for ln in ax.lines if ln.get_marker() == "o"]
        assert list(dots[0].get_xdata()) == [pytest.approx(0.5)]  # 2.0 / sigma 4.0
        assert "σ" in ax.get_xlabel()


class TestRecourseBurden:
    """Segment-level burden and availability, numbers first."""

    @staticmethod
    def _burden_batch() -> BatchResult:
        from dataclasses import replace as dc_replace

        records = (
            # row 0 (group A): two plans; the cheaper one carries the burden
            _rec(0, 0, {"a": (0.0, 2.0)}, 2.0),
            _rec(0, 1, {"b": (0.0, 5.0)}, 5.0),
            # row 1 (group A): single plan
            _rec(1, 0, {"a": (0.0, 4.0)}, 4.0),
            # row 2 (group B): certified no recourse
            dc_replace(_infeasible(2), proof="certified"),
            # row 3 (group B): search exhausted — not a proven no
            dc_replace(_infeasible(3), proof="search_exhausted"),
            # row 4 (group B): feasible
            _rec(4, 0, {"a": (0.0, 8.0)}, 8.0),
        )
        return _batch(list(records))

    GROUPS = ("A", "A", "B", "B", "B")

    def test_table_values_on_a_hand_built_batch(self) -> None:
        from treecf.viz_batch import recourse_burden_table

        table = recourse_burden_table(self._burden_batch(), self.GROUPS, min_group_size=2)
        assert [entry["group"] for entry in table] == ["A", "B"]
        a, b = table
        assert a["n"] == 2 and a["recourse_share"] == 1.0
        assert a["median_burden"] == 3.0  # cheapest plans: 2.0 and 4.0
        assert a["certified_no_share"] == 0.0 and a["unproven_no_share"] == 0.0
        assert b["n"] == 3
        assert b["recourse_share"] == pytest.approx(1 / 3)
        assert b["certified_no_share"] == pytest.approx(1 / 3)
        assert b["unproven_no_share"] == pytest.approx(1 / 3)
        assert b["median_burden"] == 8.0
        assert a["small"] is False and b["small"] is False

    def test_group_order_is_deterministic_and_overridable(self) -> None:
        from treecf.viz_batch import recourse_burden_table

        default = recourse_burden_table(self._burden_batch(), self.GROUPS)
        assert [entry["group"] for entry in default] == ["A", "B"]
        flipped = recourse_burden_table(
            self._burden_batch(), self.GROUPS, group_order=("B", "A")
        )
        assert [entry["group"] for entry in flipped] == ["B", "A"]

    def test_mismatched_groups_length_raises(self) -> None:
        from treecf import TreecfError
        from treecf.viz_batch import recourse_burden_table

        with pytest.raises(TreecfError, match="5 rows"):
            recourse_burden_table(self._burden_batch(), ("A", "B"))

    def test_ecdf_line_count_matches_groups_with_feasible_rows(self) -> None:
        from treecf.viz_batch import plot_recourse_burden

        axes = plot_recourse_burden(self._burden_batch(), self.GROUPS, min_group_size=2)
        assert len(axes[0].lines) == 2  # A and B both have >= 1 feasible row

    def test_zero_feasible_group_has_a_bar_but_no_line(self) -> None:
        from treecf.viz_batch import plot_recourse_burden

        from dataclasses import replace as dc_replace

        records = [
            _rec(0, 0, {"a": (0.0, 2.0)}, 2.0),
            dc_replace(_infeasible(1), proof="certified"),
        ]
        batch = _batch(records)
        axes = plot_recourse_burden(batch, ("A", "B"), min_group_size=1)
        assert len(axes[0].lines) == 1  # only A draws an ECDF
        assert len(axes[1].get_xticklabels()) == 2  # both groups get a bar

    def test_three_stacked_segments_with_hatched_unproven(self) -> None:
        from treecf.viz_batch import plot_recourse_burden

        axes = plot_recourse_burden(self._burden_batch(), self.GROUPS, min_group_size=2)
        bars = axes[1].patches
        hatched = [bar for bar in bars if bar.get_hatch()]
        assert hatched  # the search-exhausted share is visually distinct
        labels = [t.get_text() for t in axes[1].get_legend().get_texts()]
        assert labels == ["has recourse", "certified no recourse", "unproven no recourse"]

    def test_small_group_flag_lands_in_the_legend(self) -> None:
        from treecf.viz_batch import plot_recourse_burden

        axes = plot_recourse_burden(self._burden_batch(), self.GROUPS, min_group_size=10)
        legend_texts = [t.get_text() for t in axes[0].get_legend().get_texts()]
        assert all("— small" in text for text in legend_texts)

    def test_agg_smoke_on_both_entry_points(self) -> None:
        from treecf.viz_batch import plot_recourse_burden, recourse_burden_table

        table = recourse_burden_table(self._burden_batch(), self.GROUPS)
        assert len(table) == 2
        axes = plot_recourse_burden(self._burden_batch(), self.GROUPS, stat="p90")
        assert len(axes) == 2
