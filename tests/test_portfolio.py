"""The portfolio report: a strict-JSON artifact, optionally rendered self-contained."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from treecf import Explainer, TreecfError
from treecf.audit import portfolio_report
from treecf.batch import BatchRecord, BatchResult
from treecf.ir.model import CategoricalFeature, EnsembleIR, Link, Node, SplitOp, Tree


def _leaf(i: int, v: float) -> Node:
    return Node(i, None, None, None, None, None, None, v)


def _stump(feature: int, threshold: float, right_value: float) -> Tree:
    return Tree(
        nodes=(
            Node(0, feature, threshold, SplitOp.LT, True, 1, 2, None),
            _leaf(1, 0.0),
            _leaf(2, right_value),
        )
    )


def _ir() -> EnsembleIR:
    return EnsembleIR(
        trees=(_stump(0, 1.0, 1.0), _stump(1, 1.0, 0.8), _stump(2, 1.0, 0.6)),
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=3,
        feature_names=("a", "b", "occ"),
        meta={},
        categorical={2: CategoricalFeature(cardinality=3, categories=("x", "y", "z"))},
    )


def _rec(
    rid: object,
    k: int,
    changes: dict[str, tuple[float, float]],
    distance: float | None,
    *,
    feasible: bool = True,
    proof: str = "heuristic",
) -> BatchRecord:
    return BatchRecord(
        id=rid, k=k, feasible=feasible,
        x_cf=np.zeros(3) if feasible else None,
        changes=changes if feasible else {},
        distance=distance if feasible else None,
        n_changed=len(changes) if feasible else None,
        score_raw=1.0 if feasible else None, score_prob=None,
        proof=proof,
    )


def _batch() -> BatchResult:
    records = (
        _rec(0, 0, {"a": (0.0, 2.0), "occ": (0.0, 1.0)}, 2.0, proof="optimal"),
        _rec(0, 1, {"b": (0.0, 5.0)}, 5.0),
        _rec(1, 0, {"a": (math.nan, 4.0)}, 4.0),
        _rec(2, 0, {}, None, feasible=False, proof="certified"),
        _rec(3, 0, {}, None, feasible=False, proof="search_exhausted"),
        _rec(4, 0, {"b": (3.0, math.nan), "occ": (0.0, 2.0)}, 8.0),
    )
    return BatchResult(feature_names=("a", "b", "occ"), diversity="seeds", records=records)


GROUPS = ("A", "A", "B", "B", "B")


def _dumps(report: dict[str, object]) -> str:
    return json.dumps(report, allow_nan=False, sort_keys=True)


class TestContent:
    def test_strict_json_round_trip_with_one_group_by_default(self) -> None:
        report = portfolio_report(_batch(), min_group_size=2)
        restored = json.loads(_dumps(report))
        assert restored["portfolio_schema_version"] == 1
        assert restored["population"] == {
            "rows": 5, "records": 6, "with_recourse": 3,
            "certified_no_recourse": 1, "unproven_no_recourse": 1,
        }
        assert restored["proof_mix"] == {
            "optimal": 1, "heuristic": 3, "certified": 1, "search_exhausted": 1,
        }
        assert [row["group"] for row in restored["recourse_burden_table"]] == ["all"]
        assert "disparity" not in restored

    def test_levers_and_missing_transitions_per_group(self) -> None:
        report = portfolio_report(_batch(), GROUPS, min_group_size=2, explainer=None)
        levers = report["dominant_levers"]
        assert isinstance(levers, dict)
        a_levers = {entry["feature"]: entry for entry in levers["A"]}
        # cheapest plans in A: row 0 -> {a, occ}, row 1 -> {a}
        assert a_levers["a"]["count"] == 2 and a_levers["a"]["share"] == 1.0
        assert a_levers["a"]["median_abs_delta"] == pytest.approx(2.0)  # |2-0|, NaN->4 excluded
        assert a_levers["occ"]["count"] == 1
        missing = report["missing_transitions"]
        assert missing == {"a": {"provide": 1, "drop": 0}, "b": {"provide": 0, "drop": 1}}

    def test_explainer_adds_fingerprints_sigma_scaling_and_category_names(self) -> None:
        exp = Explainer(_ir(), normalizers=np.array([2.0, 1.0, 1.0]))
        report = portfolio_report(_batch(), GROUPS, explainer=exp, min_group_size=2)
        fingerprints = report["fingerprints"]
        assert isinstance(fingerprints, dict)
        assert len(fingerprints["ir"]) == 64 and len(fingerprints["constraints"]) == 64
        a = {e["feature"]: e for e in report["dominant_levers"]["A"]}
        assert a["a"]["median_abs_delta_sigma"] == pytest.approx(1.0)
        assert a["occ"]["kind"] == "categorical"
        assert a["occ"]["top_targets"] == [["y", 1]]

    def test_small_groups_are_flagged(self) -> None:
        report = portfolio_report(_batch(), GROUPS, min_group_size=10)
        table = report["recourse_burden_table"]
        assert all(row["small"] for row in table)

    def test_group_count_mismatch_is_an_error(self) -> None:
        with pytest.raises(TreecfError, match="rows"):
            portfolio_report(_batch(), ("A", "B"))


class TestDisparity:
    def test_absent_by_default_present_with_framing_when_enabled(self) -> None:
        plain = portfolio_report(_batch(), GROUPS, min_group_size=2)
        assert "disparity" not in plain
        framed = portfolio_report(
            _batch(), GROUPS, min_group_size=2, disparity=True, reference_group="A"
        )
        block = framed["disparity"]
        assert isinstance(block, dict)
        assert block["reference_group"] == "A"
        assert "which comparison matters is a modeling choice" in block["framing"]
        ratios = {row["group"]: row for row in block["groups"]}
        assert ratios["A"]["median_burden_ratio"] == 1.0
        # B's median burden 8.0 against A's 3.0; B's no-recourse share 2/3 against A's 0
        assert ratios["B"]["median_burden_ratio"] == pytest.approx(8.0 / 3.0)
        assert ratios["B"]["no_recourse_share_ratio"] is None
        json.loads(_dumps(framed))

    def test_reference_group_required_iff_disparity(self) -> None:
        with pytest.raises(ValueError, match="reference_group"):
            portfolio_report(_batch(), GROUPS, disparity=True)
        with pytest.raises(ValueError, match="reference_group"):
            portfolio_report(_batch(), GROUPS, reference_group="A")
        with pytest.raises(ValueError, match="reference_group"):
            portfolio_report(_batch(), GROUPS, disparity=True, reference_group="Z")


class TestRendering:
    def test_json_path_writes_strict_json_without_matplotlib(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import treecf.viz as viz

        def no_plots() -> None:
            raise AssertionError("matplotlib must not be touched for format='json'")

        monkeypatch.setattr(viz, "_import_pyplot", no_plots)
        path = f"{tmp_path}/report.json"
        report = portfolio_report(_batch(), GROUPS, path=path, format="json", min_group_size=2)
        with open(path, encoding="utf-8") as fh:
            assert json.load(fh) == json.loads(_dumps(report))

    def test_html_is_self_contained(self, tmp_path: object) -> None:
        pytest.importorskip("matplotlib")
        path = f"{tmp_path}/report.html"
        portfolio_report(
            _batch(), GROUPS, path=path, format="html", title="Q3 <review>", min_group_size=2,
            disparity=True, reference_group="A",
        )
        with open(path, encoding="utf-8") as fh:
            html = fh.read()
        assert "data:image/png;base64," in html
        assert "http" not in html
        assert "<script" not in html
        assert "Q3 &lt;review&gt;" in html
        assert "modeling choice" in html

    def test_markdown_writes_figures_beside_the_file(self, tmp_path: object) -> None:
        pytest.importorskip("matplotlib")
        import pathlib

        path = pathlib.Path(str(tmp_path)) / "report.md"
        portfolio_report(_batch(), GROUPS, path=str(path), format="markdown", min_group_size=2)
        text = path.read_text(encoding="utf-8")
        assert "data:image" not in text
        figures = sorted((path.parent / "report_figures").glob("*.png"))
        assert figures
        assert all(f"report_figures/{f.name}" in text for f in figures)

    def test_markdown_requires_a_path_and_unknown_formats_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="path"):
            portfolio_report(_batch(), GROUPS, format="markdown")
        with pytest.raises(ValueError, match="format"):
            portfolio_report(_batch(), GROUPS, format="pdf")

    def test_report_is_deterministic_apart_from_the_timestamp(self) -> None:
        first = portfolio_report(_batch(), GROUPS, min_group_size=2)
        second = portfolio_report(_batch(), GROUPS, min_group_size=2)
        first.pop("created_utc")
        second.pop("created_utc")
        assert first == second
