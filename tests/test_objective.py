"""Normalizer chain MAD -> IQR -> range -> 1."""

import numpy as np
import pytest

from treecf.objective import DegenerateFeatureWarning, fit_normalizers


def test_mad_used_when_positive() -> None:
    col = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    sigma = fit_normalizers(col.reshape(-1, 1))
    # median = 3, |x - 3| = [2, 1, 0, 1, 97] -> MAD = 1
    assert sigma[0] == 1.0


def test_iqr_fallback_when_mad_zero() -> None:
    # 60% zeros: median 0, MAD 0; IQR positive.
    col = np.array([0.0] * 60 + list(np.linspace(1, 10, 40)))
    sigma = fit_normalizers(col.reshape(-1, 1))
    expected_iqr = float(np.percentile(col, 75) - np.percentile(col, 25))
    assert expected_iqr > 0
    assert sigma[0] == pytest.approx(expected_iqr)


def test_range_fallback_when_iqr_zero() -> None:
    # 90% zeros: MAD 0 and IQR 0, but range 5.
    col = np.array([0.0] * 90 + [5.0] * 10)
    sigma = fit_normalizers(col.reshape(-1, 1))
    assert sigma[0] == 5.0


def test_constant_column_warns_and_returns_one() -> None:
    col = np.full(50, 7.0)
    with pytest.warns(DegenerateFeatureWarning, match="feature 0"):
        sigma = fit_normalizers(col.reshape(-1, 1))
    assert sigma[0] == 1.0


def test_nan_values_are_ignored() -> None:
    col = np.array([1.0, 2.0, 3.0, 4.0, 100.0, np.nan, np.nan])
    sigma = fit_normalizers(col.reshape(-1, 1))
    assert sigma[0] == 1.0  # same as the NaN-free case


def test_all_nan_column_warns_and_returns_one() -> None:
    col = np.full(10, np.nan)
    with pytest.warns(DegenerateFeatureWarning):
        sigma = fit_normalizers(col.reshape(-1, 1))
    assert sigma[0] == 1.0


def test_categorical_features_get_unit_normalizers() -> None:
    # column 1 is a constant code column: without the categorical marker it
    # would warn as degenerate; with it, sigma is 1.0 outright and no warning
    # escapes (the suite runs with warnings-as-errors)
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(size=200), np.full(200, 3.0)])
    sigma = fit_normalizers(X, categorical={1})
    assert sigma[1] == 1.0
    assert sigma[0] > 0


def test_cost_of_row_prices_a_category_change_flat() -> None:
    from treecf.backends._exact_domains import _cost_of_row

    x = np.array([0.0, 10.0])
    row = np.array([4.0, 10.0])  # code 0 -> 4: four codes away, still one unit
    flat = _cost_of_row(x, row, np.ones(2), np.ones(2), 0.25, {}, categorical=frozenset({0}))
    assert flat == 0.25 + 1.0
    numeric = _cost_of_row(x, row, np.ones(2), np.ones(2), 0.25, {})
    assert numeric == 0.25 + 4.0
