"""Plateau-aware exactness for calibrated-space targets.

A step calibrator maps every raw score inside a pooled block to one shared
level, so a calibrated target whose bound sits exactly on a plateau level is
the sharpest test of the interval contract: the closed target must admit the
plateau (``<=``/``>=`` at the level itself) and exclude it one float above.
Ground truth is direct enumeration in calibrated space — every reachable
lever combination's raw score mapped through the calibrator and checked for
closed membership — and the brute-force oracle run on the raw interval the
calibrator's ``interval_inverse`` produced must agree with it, as must both
search backends.

The stub follows the probcal step contract exactly: left block edges from
``inf{s : g(s) >= lo}``, attained right bounds one float below the next
block's edge, ``+-inf`` logit extension when a bound covers the extreme
plateau, closed intervals throughout. Under ``importorskip("probcal")`` the
same cases run against a real ``IsotonicCalibrator`` fitted so its pooled
blocks reproduce the stub's geometry (identical raw intervals, identical
decisions) and a ``CenteredIsotonicCalibrator`` (interpolating, so only the
decision-vs-enumeration invariant applies).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from treecf import Explainer, Target
from treecf.constraints import compile_constraints
from treecf.ir.evaluate import raw_score
from treecf.ir.model import EnsembleIR, Link, Node, SplitOp, Tree

from .brute_force import solve_brute_force

UP = float(np.nextafter(0.5, 1.0))  # one float above the middle plateau


def _logit(p: float) -> float:
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


class _StepCalibrator:
    """Right-continuous step map with the probcal generalized-inverse contract."""

    is_monotone_ = True

    def __init__(self, block_first_s: tuple[float, ...], levels: tuple[float, ...]) -> None:
        self.first = np.asarray(block_first_s, dtype=np.float64)
        self.levels = np.asarray(levels, dtype=np.float64)

    def predict_proba(self, s: object) -> np.ndarray:
        arr = np.asarray(s, dtype=np.float64)
        idx = np.searchsorted(self.first, arr, side="right") - 1
        return self.levels[np.clip(idx, 0, len(self.levels) - 1)]

    def _inverse_left(self, t: float) -> float:
        j = int(np.searchsorted(self.levels, t, side="left"))
        return float(self.first[j])

    def _inverse_right(self, t: float) -> float:
        j = int(np.searchsorted(self.levels, t, side="right")) - 1
        if j >= len(self.levels) - 1:
            return 1.0
        return float(np.nextafter(self.first[j + 1], 0.0))

    def interval_inverse(
        self, lo: float, hi: float, *, space: str = "probability", buffer_logit: float = 0.0
    ) -> tuple[float, float]:
        assert space == "logit" and buffer_logit == 0.0  # this suite never buffers
        gmin, gmax = float(self.levels[0]), float(self.levels[-1])
        if lo > gmax or hi < gmin:
            raise ValueError(f"[{lo}, {hi}] outside output range [{gmin}, {gmax}]")
        raw_lo = 0.0 if lo <= gmin else self._inverse_left(lo)
        raw_hi = 1.0 if hi >= gmax else self._inverse_right(hi)
        return _logit(raw_lo), _logit(raw_hi)


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
    """Three independent levers worth 1.0 / 0.8 / 0.6 over base -1.0.

    Reachable raw scores (subset sums): -1.0, -0.4, -0.2, 0.0, 0.4, 0.6,
    0.8, 1.4 — sigmoids 0.269..0.802, so with plateau edges at s=0.5 and
    s=0.7 the middle plateau's left edge is *attained exactly* by the
    raw-score-0.0 row (sigmoid(0.0) == 0.5).
    """
    return EnsembleIR(
        trees=(_stump(0, 1.0, 1.0), _stump(1, 1.0, 0.8), _stump(2, 1.0, 0.6)),
        base_score=-1.0,
        link=Link.SIGMOID,
        n_features=3,
        feature_names=("a", "b", "c"),
        meta={},
    )


# plateau levels 0.2 / 0.5 / 0.8 with edges at s = 0.5 and s = 0.7
STUB = _StepCalibrator(block_first_s=(0.30, 0.50, 0.70), levels=(0.2, 0.5, 0.8))
IR = _sig_ir()
X_LOW = np.zeros(3)  # raw -1.0, calibrated 0.2
X_HIGH = np.full(3, 2.0)  # raw 1.4, calibrated 0.8

# (name, factual, target interval in calibrated space)
CASES = (
    ("ge-at-plateau", X_LOW, (0.5, 1.0)),
    ("ge-above-plateau", X_LOW, (UP, 1.0)),
    ("le-at-plateau", X_HIGH, (0.0, 0.5)),
    ("range-endpoint-on-plateau", X_HIGH, (0.5, 0.75)),
    ("range-between-plateaus", X_LOW, (UP, 0.75)),  # no level inside: infeasible
)


def _enumerated_feasible(cal, lo: float, hi: float) -> bool:
    """Ground truth: closed membership in calibrated space over all lever rows."""
    for bits in itertools.product((0.0, 2.0), repeat=3):
        p = float(cal.predict_proba(np.array([_sigmoid(raw_score(IR, np.asarray(bits)))]))[0])
        if lo <= p <= hi:
            return True
    return False


def _target_for(cal, lo: float, hi: float) -> Target:
    if hi >= 1.0:
        return Target.calibrated(cal, op=">=", value=lo)
    if lo <= 0.0:
        return Target.calibrated(cal, op="<=", value=hi)
    return Target.calibrated(cal, range=(lo, hi))


def _run_case(cal, x: np.ndarray, lo: float, hi: float) -> None:
    truth = _enumerated_feasible(cal, lo, hi)
    target = _target_for(cal, lo, hi)
    interval = target.raw_interval(Link.SIGMOID)

    compiled = compile_constraints([], IR.feature_names)
    oracle = solve_brute_force(IR, x, interval, compiled, np.ones(3), np.ones(3))
    assert oracle.feasible == truth

    exp = Explainer(IR, normalizers=np.ones(3))
    for backend in ("exact", "genetic"):
        res = exp.explain(x, target, seed=0, backend=backend)
        feasible = hasattr(res, "x_cf")
        if backend == "exact":
            assert feasible == truth
        else:
            assert not feasible or truth  # heuristic may miss, never invent
        if feasible:
            p = float(cal.predict_proba(np.array([_sigmoid(res.score_raw)]))[0])
            assert lo <= p <= hi  # closed membership, plateau values included
            assert res.score_calibrated == pytest.approx(p, abs=1e-12)


class TestStubPlateaus:
    @pytest.mark.parametrize("name,x,bounds", CASES, ids=[c[0] for c in CASES])
    def test_backends_match_calibrated_space_enumeration(self, name, x, bounds) -> None:
        _run_case(STUB, x, *bounds)

    def test_plateau_membership_is_closed(self) -> None:
        # >= at the plateau level admits the plateau; one float above excludes it
        assert _enumerated_feasible(STUB, 0.5, 1.0)
        exp = Explainer(IR, normalizers=np.ones(3))
        at = exp.explain(X_LOW, Target.calibrated(STUB, op=">=", value=0.5), backend="exact")
        assert at.score_calibrated == 0.5  # the plateau itself, attained exactly
        above = exp.explain(X_LOW, Target.calibrated(STUB, op=">=", value=UP), backend="exact")
        assert above.score_calibrated == 0.8  # forced up to the next plateau

    def test_bands_with_edge_on_plateau(self) -> None:
        target = Target.bands(
            {"low": (0.2, 0.5), "mid": (0.5, 0.75), "high": (0.75, 1.0)},
            space="calibrated",
            calibrator=STUB,
        )
        exp = Explainer(IR, normalizers=np.ones(3))
        out = exp.explain(X_HIGH, target, seed=0, backend="exact")
        for (name, lo, hi) in target.bands_spec:
            res = out[name]
            assert hasattr(res, "x_cf") == _enumerated_feasible(STUB, lo, hi), name
            if hasattr(res, "x_cf"):
                p = float(STUB.predict_proba(np.array([_sigmoid(res.score_raw)]))[0])
                assert lo <= p <= hi


class TestProbcalPlateaus:
    """Same geometry through real probcal calibrators."""

    @staticmethod
    def _fit_isotonic():
        probcal = pytest.importorskip("probcal")
        # PAVA pools each decreasing run into one block: levels 0.2 / 0.5 / 0.8
        # with block_first_s = (0.30, 0.50, 0.70) — the stub's exact geometry.
        s = np.array([0.30, 0.32, 0.34, 0.36, 0.38, 0.50, 0.52, 0.70, 0.72, 0.74, 0.76, 0.78])
        y = np.array([1.0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0])
        return probcal.IsotonicCalibrator().fit(s, y)

    def test_isotonic_reproduces_the_stub_geometry(self) -> None:
        iso = self._fit_isotonic()
        np.testing.assert_allclose(iso.block_mean_, [0.2, 0.5, 0.8])
        np.testing.assert_allclose(iso.block_first_s_, [0.30, 0.50, 0.70])

    @pytest.mark.parametrize("name,x,bounds", CASES, ids=[c[0] for c in CASES])
    def test_isotonic_matches_the_stub_decisions(self, name, x, bounds) -> None:
        iso = self._fit_isotonic()
        lo, hi = bounds
        stub_interval = _target_for(STUB, lo, hi).raw_interval(Link.SIGMOID)
        iso_interval = _target_for(iso, lo, hi).raw_interval(Link.SIGMOID)
        assert iso_interval == pytest.approx(stub_interval, rel=1e-12)
        _run_case(iso, x, lo, hi)

    @pytest.mark.parametrize(
        "name,x,bounds",
        [c for c in CASES if c[0] != "range-between-plateaus"],
        ids=[c[0] for c in CASES if c[0] != "range-between-plateaus"],
    )
    def test_centered_isotonic_backends_match_enumeration(self, name, x, bounds) -> None:
        # CIR interpolates (no flat plateaus away from the ends), so only the
        # decision-vs-enumeration invariant applies — through CIR's own map.
        probcal = pytest.importorskip("probcal")
        s = np.array([0.30, 0.32, 0.34, 0.36, 0.38, 0.50, 0.52, 0.70, 0.72, 0.74, 0.76, 0.78])
        y = np.array([1.0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0])
        cir = probcal.CenteredIsotonicCalibrator().fit(s, y)
        _run_case(cir, x, *bounds)
