"""Audit certificates: fingerprints, strict-JSON round trips, and validation.

A certificate is a reproducibility record plus a fresh verification — the
tests here pin the three properties that make it useful to a validator: it
serializes under ``json.dumps(..., allow_nan=False)`` even for NaN/inf-bearing
plans, its fingerprints are stable and sensitive, and ``check_certificate``
catches a tampered plan, a swapped model, and a changed constraint set.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest

from treecf import (
    AllowMissing,
    Counterfactual,
    Explainer,
    Freeze,
    Grid,
    Infeasible,
    Target,
    TreecfError,
    TreecfWarning,
    constraints_fingerprint,
    ir_fingerprint,
)
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree

TARGET = Target.raw(op=">=", value=0.5)  # any single lever suffices


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


def _ir(c_value: float = 0.6) -> EnsembleIR:
    """Three independent levers worth 1.0 / 0.8 / ``c_value`` on a/b/c."""
    return EnsembleIR(
        trees=(_stump(0, 1.0, 1.0), _stump(1, 1.0, 0.8), _stump(2, 1.0, c_value)),
        base_score=0.0,
        link=Link.IDENTITY,
        n_features=3,
        feature_names=("a", "b", "c"),
        meta={},
    )


@pytest.fixture()
def exp() -> Explainer:
    return Explainer(_ir(), normalizers=np.ones(3))


def _dumps(cert: dict[str, object]) -> str:
    return json.dumps(cert, allow_nan=False, sort_keys=True)


class TestStrictJsonRoundTrip:
    def test_nan_factual_and_nan_to_value_change(self) -> None:
        exp = Explainer(
            _ir(), normalizers=np.ones(3),
            constraints=[AllowMissing("a", delta_miss=0.05)],
        )
        x = np.array([math.nan, 0.0, 0.0])
        result = exp.explain(x, Target.raw(op=">=", value=0.9), seed=0)  # only lever a reaches
        assert isinstance(result, Counterfactual)
        assert math.isnan(result.changes["a"][0])  # a NaN -> value change
        cert = exp.certificate(x, result, Target.raw(op=">=", value=0.9))
        restored = json.loads(_dumps(cert))
        assert restored["factual"]["x"][0] == "NaN"
        assert restored["plan"]["changes"]["a"][0] == "NaN"
        assert restored["verification"]["in_target_interval"] is True
        assert restored["verification"]["constraints_ok"] is True

    def test_infinite_region_endpoints(self, exp: Explainer) -> None:
        x = np.zeros(3)
        result = exp.explain(x, TARGET, seed=0, region=True)
        assert isinstance(result, Counterfactual)
        assert result.region is not None
        assert any(
            math.isinf(lo) or math.isinf(hi)
            for lo, hi in result.region.feature_intervals.values()
        )
        cert = exp.certificate(x, result, TARGET)
        restored = json.loads(_dumps(cert))
        intervals = restored["plan"]["region_feature_intervals"]
        assert any("Infinity" in pair or "-Infinity" in pair for pair in intervals.values())
        points = restored["verification"]["region_points"]
        assert points and all(p["ok"] for p in points)

    def test_certified_infeasible(self, exp: Explainer) -> None:
        x = np.zeros(3)
        unreachable = Target.raw(op=">=", value=10.0)  # max raw score is 2.4
        result = exp.explain(x, unreachable, backend="exact", seed=0)
        assert isinstance(result, Infeasible)
        assert result.proof == "certified"
        cert = exp.certificate(x, result, unreachable)
        restored = json.loads(_dumps(cert))
        assert restored["infeasible"]["proof"] == "certified"
        assert restored["solve"]["backend"] == "exact"
        assert restored["target"]["raw_interval"][1] == "Infinity"
        assert restored["verification"]["factual_in_target_interval"] is False

    def test_declared_solve_parameters_are_recorded(self, exp: Explainer) -> None:
        x = np.zeros(3)
        result = exp.explain(x, TARGET, backend="exact", seed=7, time_budget_s=5.0)
        cert = exp.certificate(
            x, result, TARGET,
            seed=7, node_budget=2_000_000, gap=0.0, time_budget_s=5.0, warm_start=True,
        )
        declared = json.loads(_dumps(cert))["solve"]["declared"]
        assert declared == {
            "seed": 7, "node_budget": 2_000_000, "gap": 0.0,
            "time_budget_s": 5.0, "warm_start": True,
        }


class TestFingerprints:
    def test_same_ir_fingerprints_equal(self) -> None:
        assert ir_fingerprint(_ir()) == ir_fingerprint(_ir())

    def test_one_ulp_leaf_perturbation_changes_the_fingerprint(self) -> None:
        perturbed = float(np.nextafter(0.6, 1.0))
        assert ir_fingerprint(_ir()) != ir_fingerprint(_ir(c_value=perturbed))

    def test_constraint_set_changes_the_fingerprint(self, exp: Explainer) -> None:
        frozen = Explainer(_ir(), normalizers=np.ones(3), constraints=[Freeze("b")])
        assert constraints_fingerprint(exp) != constraints_fingerprint(frozen)

    def test_constraints_fingerprint_is_deterministic(self, exp: Explainer) -> None:
        assert constraints_fingerprint(exp) == constraints_fingerprint(
            Explainer(_ir(), normalizers=np.ones(3))
        )


class TestCheckCertificate:
    def test_clean_pass(self, exp: Explainer) -> None:
        x = np.zeros(3)
        result = exp.explain(x, TARGET, seed=0)
        cert = json.loads(_dumps(exp.certificate(x, result, TARGET)))
        report = exp.check_certificate(cert)
        assert report == {
            "model_match": True,
            "constraints_match": True,
            "verification_ok": True,
            "mismatches": [],
        }

    def test_tampered_x_cf_fails_verification(self, exp: Explainer) -> None:
        x = np.zeros(3)
        result = exp.explain(x, TARGET, seed=0)
        cert = json.loads(_dumps(exp.certificate(x, result, TARGET)))
        cert["plan"]["x_cf"] = [0.2, 0.0, 0.0]  # score 0.0, outside the target
        report = exp.check_certificate(cert)
        assert report["verification_ok"] is False
        assert any("in_target_interval" in m for m in report["mismatches"])
        assert report["model_match"] is True  # only the plan was tampered with

    def test_swapped_model_is_caught(self, exp: Explainer) -> None:
        x = np.zeros(3)
        result = exp.explain(x, TARGET, seed=0)
        cert = exp.certificate(x, result, TARGET)
        other = Explainer(_ir(c_value=0.7), normalizers=np.ones(3))
        report = other.check_certificate(cert)
        assert report["model_match"] is False
        assert any("model fingerprint" in m for m in report["mismatches"])

    def test_changed_constraint_set_is_caught(self, exp: Explainer) -> None:
        x = np.zeros(3)
        result = exp.explain(x, TARGET, seed=0)
        cert = exp.certificate(x, result, TARGET)
        stricter = Explainer(_ir(), normalizers=np.ones(3), constraints=[Freeze("a")])
        report = stricter.check_certificate(cert)
        assert report["constraints_match"] is False
        assert report["model_match"] is True  # same ensemble, different constraints
        assert any("constraints fingerprint" in m for m in report["mismatches"])


class TestReproducibility:
    def test_callable_value_policy_marks_not_reproducible(self) -> None:
        exp = Explainer(
            _ir(), normalizers=np.ones(3),
            value_policy={"a": lambda v: float(round(v))},
        )
        x = np.zeros(3)
        result = exp.explain(x, TARGET, seed=0)
        cert = exp.certificate(x, result, TARGET)
        assert cert["reproducible"] is False
        assert "value_policy" in str(cert["reproducible_reason"])

    def test_grid_value_policy_stays_reproducible(self) -> None:
        exp = Explainer(
            _ir(), normalizers=np.ones(3), value_policy={"a": Grid(step=0.5)}
        )
        x = np.zeros(3)
        result = exp.explain(x, TARGET, seed=0)
        cert = exp.certificate(x, result, TARGET)
        assert cert["reproducible"] is True
        assert "reproducible_reason" not in cert


class TestVerificationFailurePath:
    def test_corrupt_result_still_issues_a_certificate_and_warns(
        self, exp: Explainer
    ) -> None:
        x = np.zeros(3)
        result = exp.explain(x, TARGET, seed=0)
        assert isinstance(result, Counterfactual)
        corrupted = replace(result, x_cf=np.array([0.2, 0.0, 0.0]))  # score 0.0 < 0.5
        with pytest.warns(TreecfWarning, match="in_target_interval"):
            cert = exp.certificate(x, corrupted, TARGET)
        assert cert["verification"]["in_target_interval"] is False  # type: ignore[index]
        json.loads(_dumps(cert))  # still strict-JSON serializable


class TestBandsAndErrors:
    def test_band_result_records_the_band(self, exp: Explainer) -> None:
        ladder = Target.bands({"lo": (0.5, 0.7), "hi": (1.3, 1.5)}, space="raw")
        results = exp.explain(np.zeros(3), ladder, seed=0)
        assert isinstance(results, dict)
        band_result = results["hi"]
        assert isinstance(band_result, Counterfactual)
        cert = exp.certificate(np.zeros(3), band_result, ladder, band="hi")
        target_block = cert["target"]
        assert target_block == {  # type: ignore[comparison-overlap]
            "space": "raw", "lo": 1.3, "hi": 1.5,
            "raw_interval": [1.3, 1.5], "band": "hi",
        }

    def test_bands_target_without_band_raises(self, exp: Explainer) -> None:
        ladder = Target.bands({"lo": (0.5, 0.7)}, space="raw")
        result = Infeasible(reason="test")
        with pytest.raises(TreecfError, match="band="):
            exp.certificate(np.zeros(3), result, ladder)

    def test_band_on_plain_target_raises(self, exp: Explainer) -> None:
        result = Infeasible(reason="test")
        with pytest.raises(TreecfError, match="band="):
            exp.certificate(np.zeros(3), result, TARGET, band="lo")


class TestCategoricalFingerprints:
    """Set-split encodings extend the fingerprint; numeric encodings are frozen."""

    RECORDED_NUMERIC = "4e004fa506fd23b3a655b562cb0627a01786fcbc1e4c4c3b1771627b6e72d778"

    def test_numeric_encoding_is_frozen(self) -> None:
        from treecf.ir.flatten import unflatten_ir

        with open("tests/fixtures/exact/01-basic-lt-le.json", encoding="utf-8") as fh:
            payload = json.load(fh)
        ir = unflatten_ir(payload["ensemble"])
        assert ir_fingerprint(ir) == self.RECORDED_NUMERIC

    def test_set_membership_changes_the_fingerprint(self) -> None:
        from tests.conftest import make_random_mixed_ir

        a = make_random_mixed_ir(np.random.default_rng(0), categorical={1: 4})
        b = make_random_mixed_ir(np.random.default_rng(0), categorical={1: 4})
        assert ir_fingerprint(a) == ir_fingerprint(b)
        c = make_random_mixed_ir(np.random.default_rng(1), categorical={1: 4})
        assert ir_fingerprint(a) != ir_fingerprint(c)

    def test_cardinality_changes_the_fingerprint(self) -> None:
        from dataclasses import replace as dc_replace

        from tests.conftest import make_random_mixed_ir
        from treecf.ir.model import CategoricalFeature

        a = make_random_mixed_ir(np.random.default_rng(0), categorical={1: 4})
        wider = dc_replace(a, categorical={1: CategoricalFeature(cardinality=9)})
        assert ir_fingerprint(a) != ir_fingerprint(wider)
