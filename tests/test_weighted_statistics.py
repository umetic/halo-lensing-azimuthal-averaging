from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose
import pytest

from azlens.weighted_statistics import (
    effective_sample_size,
    summarize_weighted_values,
    weighted_mean,
    weighted_midpoint_quantile,
    weighted_r2_score,
)


def test_weighted_mean_and_effective_sample_size_equal_weights():
    values = np.array([1.0, 2.0, 4.0, 8.0])
    weights = np.ones_like(values)
    assert_allclose(weighted_mean(values, weights), np.mean(values), rtol=0, atol=0)
    assert_allclose(effective_sample_size(weights), 4.0, rtol=0, atol=0)


def test_midpoint_quantile_equal_weights_gives_interpolated_median():
    values = np.array([0.0, 10.0, 20.0, 30.0])
    weights = np.ones_like(values)
    # Midpoint coordinates are 0.125, 0.375, 0.625, 0.875, so q=0.5
    # interpolates halfway between the two central values.
    assert_allclose(weighted_midpoint_quantile(values, weights, [0.5]), [15.0], rtol=0, atol=0)


def test_midpoint_quantile_unequal_weights_and_endpoints():
    values = np.array([0.0, 10.0, 20.0])
    weights = np.array([1.0, 8.0, 1.0])
    result = weighted_midpoint_quantile(values, weights, [0.0, 0.5, 1.0])
    assert_allclose(result[0], 0.0, rtol=0, atol=0)
    assert_allclose(result[1], 10.0, rtol=0, atol=0)
    assert_allclose(result[2], 20.0, rtol=0, atol=0)


def test_summary_reports_counts_and_ignores_nan_values():
    values = np.array([1.0, np.nan, 3.0, 5.0, 7.0])
    weights = np.ones_like(values)
    selected = np.array([True, True, False, True, True])
    summary = summarize_weighted_values(values, weights, selected)
    assert summary.n_total == 5
    assert summary.n_selected == 4
    assert summary.n_used == 3
    assert_allclose(summary.mean, (1.0 + 5.0 + 7.0)/3.0, rtol=0, atol=0)


def test_negative_weights_are_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        weighted_mean([1.0, 2.0], [1.0, -1.0])


def test_weighted_r2_perfect_and_nonperfect():
    y = np.array([1.0, 2.0, 4.0, 8.0])
    w = np.array([1.0, 2.0, 3.0, 4.0])
    assert_allclose(weighted_r2_score(y, y, w), 1.0, rtol=0, atol=0)
    assert weighted_r2_score(y, y + 1.0, w) < 1.0
