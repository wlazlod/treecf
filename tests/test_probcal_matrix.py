"""probcal calibrator matrix on real boosted models.

Every cell fits a real probcal calibrator on a model's scores, asks the exact
backend for a counterfactual against a calibrated target, and re-verifies the
result *through the model and calibrator themselves* — never trusting
treecf's own report. Cells that return a plan must satisfy the closed
calibrated target; buffered cells must additionally clear the unbuffered
bound by the buffer's logit margin; targets below a step calibrator's output
floor must fail loudly (``TargetError``), never silently clamp.

Skipped without probcal + scikit-learn; the LightGBM half also needs
lightgbm. Sized to stay fast: 30 trees of depth 2 keep the exact backend's
search small while still exercising real fitted maps.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

probcal = pytest.importorskip("probcal")
pytest.importorskip("sklearn")

from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402

from treecf import Explainer, Target, TargetError  # noqa: E402

RNG_SEED = 7
N = 3000
BUFFER = 0.2


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _dataset() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RNG_SEED)
    X = np.column_stack(
        [
            rng.normal(size=N),
            rng.normal(size=N),
            rng.uniform(0, 1, N),
            rng.normal(size=N),
            rng.integers(0, 3, N).astype(float),
        ]
    )
    z = 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.9 * X[:, 2] + 0.4 * X[:, 3] - 3.1
    y = (rng.random(N) < 1.0 / (1.0 + np.exp(-z))).astype(float)
    return X, y


@pytest.fixture(scope="module")
def gbc_setup():
    X, y = _dataset()
    assert 0.05 < y.mean() < 0.12  # ~8% event rate
    model = GradientBoostingClassifier(n_estimators=30, max_depth=2, random_state=0).fit(X, y)
    scores = model.predict_proba(X)[:, 1]
    return model, X, y, scores


@pytest.fixture(scope="module")
def lgbm_setup():
    lgb = pytest.importorskip("lightgbm")
    X, y = _dataset()
    model = lgb.LGBMClassifier(n_estimators=30, num_leaves=4, random_state=0, verbose=-1).fit(
        X, y.astype(int)
    )
    scores = model.predict_proba(X)[:, 1]
    return model, X, y, scores


def _calibrators(scores: np.ndarray, y: np.ndarray, model, X) -> dict[str, object]:
    beta = probcal.BetaCalibrator().fit(scores, y)
    off = probcal.LogitOffset(delta=0.3).fit(beta.predict_proba(scores))
    return {
        "platt": probcal.PlattCalibrator().fit(scores, y),
        "temperature": probcal.TemperatureCalibrator().fit(scores, y),
        "beta-abm": beta,
        "isotonic": probcal.IsotonicCalibrator().fit(scores, y),
        "centered-isotonic": probcal.CenteredIsotonicCalibrator().fit(scores, y),
        "chain-beta-offset": probcal.Chain([beta, off]),
        "calibrated-model-chain": probcal.CalibratedModel(
            model, probcal.BetaCalibrator(), flow="prefit"
        )
        .fit(X, y)
        .chain_,
    }


@pytest.fixture(scope="module")
def gbc_cals(gbc_setup):
    model, X, y, scores = gbc_setup
    return _calibrators(scores, y, model, X)


@pytest.fixture(scope="module")
def lgbm_cals(lgbm_setup):
    model, X, y, scores = lgbm_setup
    return _calibrators(scores, y, model, X)


CAL_KEYS = (
    "platt",
    "temperature",
    "beta-abm",
    "isotonic",
    "centered-isotonic",
    "chain-beta-offset",
    "calibrated-model-chain",
)
OPS = ("<=", ">=", "range", "bands")
BUFFERS = (0.0, BUFFER)


def _cell_target(cal, op: str, q30: float, q70: float, buffer_logit: float) -> Target:
    if op == "<=":
        return Target.calibrated(cal, op="<=", value=q30, buffer_logit=buffer_logit)
    if op == ">=":
        return Target.calibrated(cal, op=">=", value=q70, buffer_logit=buffer_logit)
    if op == "range":
        return Target.calibrated(cal, range=(q30, q70), buffer_logit=buffer_logit)
    return Target.bands(
        {"low": (0.0, q30), "mid": (q30, q70), "high": (q70, 1.0)},
        space="calibrated",
        calibrator=cal,
        buffer_logit=buffer_logit,
    )


def _closed_bounds(op: str, q30: float, q70: float) -> tuple[float, float]:
    if op == "<=":
        return 0.0, q30
    if op == ">=":
        return q70, 1.0
    return q30, q70


def _recompute(cal, model, x_cf: np.ndarray) -> float:
    s = model.predict_proba(np.asarray(x_cf, dtype=np.float64).reshape(1, -1))[:, 1]
    return float(np.asarray(cal.predict_proba(s)).reshape(-1)[0])


def _run_matrix_cell(setup, cals, cal_key: str, op: str, buffer_logit: float) -> None:
    model, X, _y, scores = setup
    cal = cals[cal_key]
    p_cal = np.asarray(cal.predict_proba(scores), dtype=np.float64)
    q30, q70 = float(np.quantile(p_cal, 0.30)), float(np.quantile(p_cal, 0.70))
    if not q30 < q70:  # step maps can collapse quantiles; nothing to test then
        pytest.skip(f"{cal_key}: degenerate calibrated quantiles")

    # factual: worst row for downward ops, best-behaved low row for upward
    x0 = X[int(np.argmax(p_cal))] if op in ("<=", "range", "bands") else X[int(np.argmin(p_cal))]
    target = _cell_target(cal, op, q30, q70, buffer_logit)
    exp = Explainer(model=model, background=X[:500])

    try:
        out = exp.explain(x0, target, seed=0, backend="exact")
    except TargetError:
        assert buffer_logit > 0.0  # only the buffer may empty an attainable target
        return

    results = out.values() if isinstance(out, dict) else [out]
    checked = 0
    for res in results:
        if not hasattr(res, "x_cf"):
            continue
        checked += 1
        p_cf = _recompute(cal, model, res.x_cf)
        assert res.score_calibrated == pytest.approx(p_cf, abs=1e-9)
        if isinstance(out, dict):  # bands: verify against each band's own bounds
            name = next(k for k, v in out.items() if v is res)
            lo, hi = next((b, c) for (n, b, c) in target.bands_spec if n == name)
        else:
            lo, hi = _closed_bounds(op, q30, q70)
        assert lo <= p_cf + 1e-12 and p_cf - 1e-12 <= hi
        if buffer_logit > 0.0 and op in ("<=", ">="):
            # buffered plan clears the unbuffered bound by the buffer's margin
            bound = q30 if op == "<=" else q70
            margin = (_logit(bound) - _logit(p_cf)) * (1 if op == "<=" else -1)
            assert margin >= buffer_logit - 1e-9
    if not isinstance(out, dict) and buffer_logit == 0.0:
        assert checked == 1, f"{cal_key}/{op}: unbuffered single target must be feasible"


@pytest.mark.parametrize("buffer_logit", BUFFERS, ids=["buf0", "buf0.2"])
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("cal_key", CAL_KEYS)
def test_gbc_matrix(gbc_setup, gbc_cals, cal_key: str, op: str, buffer_logit: float) -> None:
    _run_matrix_cell(gbc_setup, gbc_cals, cal_key, op, buffer_logit)


@pytest.mark.parametrize("buffer_logit", BUFFERS, ids=["buf0", "buf0.2"])
@pytest.mark.parametrize("op", ("<=", ">="))
@pytest.mark.parametrize("cal_key", ("platt", "isotonic", "chain-beta-offset"))
def test_lgbm_matrix(lgbm_setup, lgbm_cals, cal_key: str, op: str, buffer_logit: float) -> None:
    _run_matrix_cell(lgbm_setup, lgbm_cals, cal_key, op, buffer_logit)


def test_out_of_range_target_fails_loudly(gbc_setup) -> None:
    # A target entirely outside the step calibrator's output range must raise
    # (probcal's no-silent-clamp doctrine, surfaced as treecf's TargetError),
    # never come back as a quietly clamped plan.
    model, X, _y, scores = gbc_setup
    # PAVA pools each decreasing run: blocks with levels exactly 0.2 / 0.5 / 0.8,
    # so anything below 0.2 (or above 0.8) is outside the output range.
    s = np.array([0.30, 0.32, 0.34, 0.36, 0.38, 0.50, 0.52, 0.70, 0.72, 0.74, 0.76, 0.78])
    iso = probcal.IsotonicCalibrator().fit(s, np.array([1.0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0]))
    assert float(np.min(iso.block_mean_)) == pytest.approx(0.2)
    target = Target.calibrated(iso, op="<=", value=0.1)
    exp = Explainer(model=model, background=X[:500])
    with pytest.raises(TargetError, match="could not invert"):
        exp.explain(X[int(np.argmax(scores))], target, seed=0)


def test_probability_target_is_not_the_calibrated_target(gbc_setup) -> None:
    # The Target.probability trap: it bounds the *model's own* probability;
    # a fitted non-identity calibrator inverts to a different raw interval.
    from treecf.ir.model import Link

    _model, _X, y, scores = gbc_setup
    platt = probcal.PlattCalibrator().fit(scores, y)
    t = 0.05
    prob_interval = Target.probability(op="<=", value=t).raw_interval(Link.SIGMOID)
    cal_interval = Target.calibrated(platt, op="<=", value=t).raw_interval(Link.SIGMOID)
    assert prob_interval != pytest.approx(cal_interval)


@pytest.fixture(scope="module")
def lgbm_categorical_setup():
    lgb = pytest.importorskip("lightgbm")
    X, y = _dataset()
    rng = np.random.default_rng(7)
    X = X.copy()
    codes = rng.integers(0, 5, size=len(X)).astype(np.float64)
    X[:, 2] = codes
    y = (y.astype(int) | ((codes == 4) & (rng.random(len(X)) < 0.6))).astype(int)
    model = lgb.LGBMClassifier(
        n_estimators=30, num_leaves=4, random_state=0, verbose=-1, min_data_per_group=1,
    ).fit(X, y, categorical_feature=[2])
    scores = model.predict_proba(X)[:, 1]
    return model, X, y, scores


@pytest.fixture(scope="module")
def lgbm_categorical_cals(lgbm_categorical_setup):
    model, X, y, scores = lgbm_categorical_setup
    return _calibrators(scores, y, model, X)


@pytest.mark.parametrize("buffer_logit", BUFFERS, ids=["buf0", "buf0.2"])
@pytest.mark.parametrize("op", ("<=", ">="))
@pytest.mark.parametrize("cal_key", ("platt", "isotonic"))
def test_lgbm_categorical_matrix(
    lgbm_categorical_setup, lgbm_categorical_cals, cal_key: str, op: str, buffer_logit: float
) -> None:
    """Calibrated targets solve exactly over a model with a native categorical lever."""
    _run_matrix_cell(lgbm_categorical_setup, lgbm_categorical_cals, cal_key, op, buffer_logit)
