"""Calibrated-target provenance in certificates and batch records."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from treecf import Explainer, Target
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree


def _logit(p: float) -> float:
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


class StubCalibrator:
    """Affine-logit calibrator with the full optional duck surface."""

    is_monotone_ = True

    def __init__(self, a: float = 1.0, b: float = 0.0, tag: str = "stub-1") -> None:
        self.a, self.b, self.tag = a, b, tag
        self.inverse_calls = 0

    def fingerprint(self) -> str:
        return f"fp-{self.tag}-{self.a}-{self.b}"

    def predict_proba(self, p: object) -> np.ndarray:
        arr = np.asarray(p, dtype=np.float64)
        z = self.a * np.log(arr / (1.0 - arr)) + self.b
        return 1.0 / (1.0 + np.exp(-z))

    def interval_inverse(
        self, lo: float, hi: float, *, space: str = "probability", buffer_logit: float = 0.0
    ) -> tuple[float, float]:
        assert space == "logit"
        self.inverse_calls += 1
        lo_z = -math.inf if lo <= 0.0 else (_logit(lo) + buffer_logit - self.b) / self.a
        hi_z = math.inf if hi >= 1.0 else (_logit(hi) - buffer_logit - self.b) / self.a
        return lo_z, hi_z


class BareCalibrator(StubCalibrator):
    """Protocol-minimal: no fingerprint, no predict_proba."""

    fingerprint = None  # type: ignore[assignment]
    predict_proba = None  # type: ignore[assignment]


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


def _sig_ir() -> EnsembleIR:
    """Sigmoid-link, three independent levers worth 1.0 / 0.8 / 0.6."""
    return EnsembleIR(
        trees=(_stump(0, 1.0, 1.0), _stump(1, 1.0, 0.8), _stump(2, 1.0, 0.6)),
        base_score=-1.0,
        link=Link.SIGMOID,
        n_features=3,
        feature_names=("a", "b", "c"),
        meta={},
    )


@pytest.fixture()
def exp() -> Explainer:
    return Explainer(_sig_ir(), normalizers=np.ones(3))


X0 = np.zeros(3)  # raw score -1.0; raising levers pushes the score up


def _feasible(exp: Explainer, cal: StubCalibrator, **kw) -> tuple:
    target = Target.calibrated(cal, op=">=", value=0.5, **kw)
    res = exp.explain(X0, target, seed=0)
    assert hasattr(res, "x_cf"), res
    return res, target


class TestCertificateProvenance:
    def test_calibrator_block_is_structured(self, exp: Explainer) -> None:
        cal = StubCalibrator(a=1.0, b=0.3)
        res, target = _feasible(exp, cal, buffer_logit=0.1)
        cert = exp.certificate(X0, res, target)
        block = cert["target"]["calibrator"]
        assert block == {
            "embedded": False,
            "fingerprint": cal.fingerprint(),
            "type": "StubCalibrator",
            "buffer_logit": 0.1,
        }
        json.dumps(cert)  # stays strict-JSON serializable

    def test_missing_fingerprint_yields_null(self, exp: Explainer) -> None:
        cal = BareCalibrator(a=1.0, b=0.3)
        res, target = _feasible(exp, cal)
        cert = exp.certificate(X0, res, target)
        assert cert["target"]["calibrator"]["fingerprint"] is None

    def test_band_certificate_carries_the_block_once(self, exp: Explainer) -> None:
        cal = StubCalibrator(a=1.0, b=0.0)
        target = Target.bands(
            {"good": (0.0, 0.4), "bad": (0.4, 1.0)}, space="calibrated", calibrator=cal
        )
        out = exp.explain(X0, target, seed=0)
        band = next(name for name, r in out.items() if hasattr(r, "x_cf"))
        cert = exp.certificate(X0, out[band], target, band=band)
        assert cert["target"]["band"] == band
        assert cert["target"]["calibrator"]["type"] == "StubCalibrator"

    def test_check_without_calibrator_matches_0_2_3_shape(self, exp: Explainer) -> None:
        cal = StubCalibrator()
        res, target = _feasible(exp, cal)
        cert = exp.certificate(X0, res, target)
        report = exp.check_certificate(cert)
        assert set(report) == {
            "model_match",
            "constraints_match",
            "verification_ok",
            "mismatches",
        }
        assert report["verification_ok"] and not report["mismatches"]

    def test_check_with_matching_calibrator_is_clean(self, exp: Explainer) -> None:
        cal = StubCalibrator(a=0.9, b=0.2)
        res, target = _feasible(exp, cal, buffer_logit=0.05)
        cert = exp.certificate(X0, res, target)
        report = exp.check_certificate(cert, calibrator=cal)
        assert report["calibrator_match"] is True
        assert not report["mismatches"]

    def test_check_flags_a_different_calibrator(self, exp: Explainer) -> None:
        cal = StubCalibrator(a=0.9, b=0.2)
        res, target = _feasible(exp, cal)
        cert = exp.certificate(X0, res, target)
        other = StubCalibrator(a=0.9, b=0.2, tag="stub-2")  # same math, other identity
        report = exp.check_certificate(cert, calibrator=other)
        assert report["calibrator_match"] is False
        assert any("fingerprint" in m for m in report["mismatches"])

    def test_check_flags_a_perturbed_stored_interval(self, exp: Explainer) -> None:
        cal = StubCalibrator(a=0.9, b=0.2)
        res, target = _feasible(exp, cal)
        cert = exp.certificate(X0, res, target)
        cert["target"]["raw_interval"][0] = cert["target"]["raw_interval"][0] + 0.01 \
            if math.isfinite(cert["target"]["raw_interval"][0]) else -3.0
        cert["target"]["raw_interval"][1] = 5.0
        report = exp.check_certificate(cert, calibrator=cal)
        assert report["calibrator_match"] is False
        assert any("re-invert" in m or "interval" in m for m in report["mismatches"])

    def test_check_with_fingerprintless_calibrator_notes_and_reinverts(
        self, exp: Explainer
    ) -> None:
        cal = BareCalibrator(a=1.0, b=0.3)
        res, target = _feasible(exp, cal)
        cert = exp.certificate(X0, res, target)
        report = exp.check_certificate(cert, calibrator=cal)
        # fingerprint unavailable on both sides -> noted, but the interval
        # re-inversion still passes, so calibrator_match reflects only real evidence
        assert any("fingerprint" in m for m in report["mismatches"])


class TestBatchProvenance:
    def test_records_carry_the_calibrator_fingerprint(self, exp: Explainer, tmp_path) -> None:
        cal = StubCalibrator(a=1.0, b=0.2)
        target = Target.calibrated(cal, op=">=", value=0.4)
        X = np.zeros((3, 3))
        batch = exp.explain_batch(X, target, seed=0)
        assert len(batch.records) == 3
        for rec in batch.records:
            assert rec.calibrator_fingerprint == cal.fingerprint()
        path = tmp_path / "batch.json"
        batch.save(path)
        loaded = type(batch).load(path)
        assert all(r.calibrator_fingerprint == cal.fingerprint() for r in loaded.records)

    def test_non_calibrated_target_leaves_field_none(self, exp: Explainer) -> None:
        batch = exp.explain_batch(np.zeros((2, 3)), Target.raw(op=">=", value=0.5), seed=0)
        assert all(r.calibrator_fingerprint is None for r in batch.records)

    def test_pre_0_2_4_batch_json_loads_defaulted(self, exp: Explainer, tmp_path) -> None:
        from treecf.batch import BatchResult

        cal = StubCalibrator()
        batch = exp.explain_batch(np.zeros((2, 3)), Target.calibrated(cal, op=">=", value=0.4))
        path = tmp_path / "old.json"
        batch.save(path)
        data = json.loads(path.read_text())
        for raw in data["records"]:  # simulate a 0.2.x writer
            raw.pop("calibrator_fingerprint", None)
            raw.pop("score_calibrated", None)
        path.write_text(json.dumps(data))
        loaded = BatchResult.load(path)
        assert all(r.calibrator_fingerprint is None for r in loaded.records)
        assert all(r.score_calibrated is None for r in loaded.records)


class TestCalibratedReadout:
    def test_counterfactual_readout_matches_external_recompute(self, exp: Explainer) -> None:
        cal = StubCalibrator(a=0.8, b=0.4)
        res, _target = _feasible(exp, cal)
        assert res.score_calibrated is not None
        expected = float(cal.predict_proba(np.array([_sigmoid(res.score_raw)]))[0])
        assert res.score_calibrated == pytest.approx(expected, abs=1e-12)
        assert res.score_calibrated >= 0.5 - 1e-12  # inside the closed target

    def test_none_for_raw_targets_and_bare_calibrators(self, exp: Explainer) -> None:
        raw_res = exp.explain(X0, Target.raw(op=">=", value=0.5), seed=0)
        assert raw_res.score_calibrated is None
        bare_res, _ = _feasible(exp, BareCalibrator(a=1.0, b=0.3))
        assert bare_res.score_calibrated is None

    def test_band_results_carry_the_readout(self, exp: Explainer) -> None:
        cal = StubCalibrator(a=1.0, b=0.0)
        target = Target.bands(
            {"good": (0.0, 0.4), "bad": (0.4, 1.0)}, space="calibrated", calibrator=cal
        )
        out = exp.explain(X0, target, seed=0)
        for res in out.values():
            if hasattr(res, "x_cf"):
                assert res.score_calibrated is not None

    def test_batch_records_carry_the_readout(self, exp: Explainer) -> None:
        cal = StubCalibrator(a=1.0, b=0.2)
        batch = exp.explain_batch(np.zeros((2, 3)), Target.calibrated(cal, op=">=", value=0.4))
        for rec in batch.records:
            if rec.feasible:
                expected = float(cal.predict_proba(np.array([_sigmoid(rec.score_raw)]))[0])
                assert rec.score_calibrated == pytest.approx(expected, abs=1e-12)

    def test_certificate_factual_block_carries_the_readout(self, exp: Explainer) -> None:
        cal = StubCalibrator(a=1.0, b=0.2)
        res, target = _feasible(exp, cal)
        cert = exp.certificate(X0, res, target)
        # X0 has raw score -1.0 (base_score, no levers raised)
        expected = float(cal.predict_proba(np.array([_sigmoid(-1.0)]))[0])
        assert cert["factual"]["score_calibrated"] == pytest.approx(expected, abs=1e-12)

    def test_certificate_factual_readout_none_for_raw_target(self, exp: Explainer) -> None:
        target = Target.raw(op=">=", value=0.5)
        res = exp.explain(X0, target, seed=0)
        cert = exp.certificate(X0, res, target)
        assert cert["factual"].get("score_calibrated") is None


class TestInversionCaching:
    def test_explain_batch_inverts_exactly_once(self, exp: Explainer) -> None:
        cal = StubCalibrator(a=1.0, b=0.2)
        target = Target.calibrated(cal, op=">=", value=0.4)
        exp.explain_batch(np.zeros((4, 3)), target, seed=0)
        assert cal.inverse_calls == 1

    def test_band_ladder_inverts_once_per_band(self, exp: Explainer) -> None:
        cal = StubCalibrator(a=1.0, b=0.0)
        target = Target.bands(
            {"good": (0.0, 0.4), "bad": (0.4, 1.0)}, space="calibrated", calibrator=cal
        )
        exp.explain(X0, target, seed=0)
        assert cal.inverse_calls == 2  # one per band
