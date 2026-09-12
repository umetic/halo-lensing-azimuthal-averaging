from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose
import pytest

from azlens.statistics import (
    finite_values,
    least_squares_fit,
    ordinary_percentiles,
    r2_score,
    summarize_signed_values,
)


def test_finite_values_apply_selection_before_finite_filtering():
    values = np.array([1.0, np.nan, -2.0, np.inf, 4.0])
    mask = np.array([True, True, False, True, True])
    out = finite_values(values, mask)
    assert_allclose(out, [1.0, 4.0], rtol=0, atol=0)


def test_summarize_signed_values_keeps_sign_and_counts_nonfinite():
    values = np.array([-3.0, -1.0, np.nan, 2.0, 5.0])
    mask = np.array([True, True, True, False, True])
    stats = summarize_signed_values(values, mask)
    assert stats.n_total == 5
    assert stats.n_selected == 4
    assert stats.n_finite == 3
    assert_allclose(stats.mean, np.mean([-3.0, -1.0, 5.0]), rtol=0, atol=0)
    assert stats.p50 == -1.0
    assert stats.finite_fraction_selected == 0.75


def test_summarize_empty_selected_sample_reports_nan_values():
    stats = summarize_signed_values(np.array([1.0, 2.0]), np.array([False, False]))
    assert stats.n_total == 2
    assert stats.n_selected == 0
    assert stats.n_finite == 0
    assert np.isnan(stats.p50)
    assert np.isnan(stats.finite_fraction_selected)


def test_ordinary_percentiles_matches_numpy_default_linear_rule():
    values = np.array([-2.0, 0.0, 3.0, 9.0])
    probs = [16.0, 50.0, 84.0, 97.5]
    assert_allclose(ordinary_percentiles(values, probs), np.percentile(values, probs), rtol=0, atol=0)


def test_ordinary_percentiles_rejects_nonfinite_input():
    with pytest.raises(ValueError):
        ordinary_percentiles([1.0, np.nan])


def test_r2_score_matches_manual_definition():
    y = np.array([1.0, 2.0, 4.0, np.nan])
    p = np.array([1.1, 1.9, 3.5, 0.0])
    expected = 1.0 - np.sum((y[:3]-p[:3])**2)/np.sum((y[:3]-np.mean(y[:3]))**2)
    assert_allclose(r2_score(y, p), expected, rtol=0, atol=1e-15)


def test_r2_score_returns_nan_for_constant_target():
    assert np.isnan(r2_score([2.0, 2.0, 2.0], [2.0, 2.1, 1.9]))


def test_least_squares_fit_recovers_known_coefficients():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    X = np.column_stack((np.ones_like(x), x))
    y = 2.5 - 0.75*x
    result = least_squares_fit(X, y)
    assert_allclose(result.coefficients, [2.5, -0.75], rtol=0, atol=1e-14)
    assert result.rank == 2
    assert_allclose(result.prediction, y, rtol=0, atol=1e-14)


def test_least_squares_fit_applies_selection_and_finite_masks():
    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, np.nan], [1.0, 4.0]])
    y = np.array([1.0, 3.0, 5.0, 9.0, 9.0])
    mask = np.array([True, True, True, True, False])
    result = least_squares_fit(X, y, mask)
    assert_allclose(result.coefficients, [1.0, 2.0], rtol=0, atol=1e-14)


def test_mask_shape_mismatches_are_rejected():
    with pytest.raises(ValueError):
        finite_values([1.0, 2.0], [True])
    with pytest.raises(ValueError):
        summarize_signed_values([1.0, 2.0], [True])
    with pytest.raises(ValueError):
        r2_score([1.0, 2.0], [1.0, 2.0], [True])
