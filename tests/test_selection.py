from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from azlens.selection import (
    key_radius_indices,
    key_radius_values,
    lambda_minus_minimum,
    local_safety_mask,
    outer_contiguous_mask,
    publication_radial_grid,
    represented_fraction,
    summarize_selected_values,
)

TARGETS = json.loads(Path("reference/population_fields/population_field_targets.json").read_text())


def test_publication_radial_grid_and_key_radii():
    target = TARGETS["publication_radial_grid"]
    radii = publication_radial_grid()
    assert radii.size == target["n_radial"]
    assert_allclose(radii[0], target["first"], rtol=0, atol=1e-15)
    assert_allclose(radii[-1], target["last"], rtol=0, atol=1e-15)
    assert_array_equal(key_radius_indices(radii), target["key_indices"])
    assert_allclose(key_radius_values(radii), target["key_values"], rtol=0, atol=1e-15)


def test_lambda_minus_minimum_uses_total_shear_amplitude():
    k = np.array([[0.1, 0.2], [0.3, 0.1]])
    gp = np.array([[0.2, 0.0], [0.0, 0.4]])
    gx = np.array([[0.0, 0.3], [0.4, 0.0]])
    expected = np.min(1.0 - k - np.hypot(gp, gx), axis=1)
    assert_allclose(lambda_minus_minimum(k, gp, gx), expected, rtol=0, atol=0)


def test_outer_contiguous_mask_from_large_radius_inward():
    local = np.array([
        [True, True, False, True],
        [True, False, True, True],
        [False, True, True, True],
    ])
    expected = np.array([
        [False, False, False, True],
        [False, False, True, True],
        [False, True, True, True],
    ])
    assert_array_equal(outer_contiguous_mask(local, radius_axis=1), expected)


def test_represented_fraction_and_summary():
    safe = np.array([[[True, False], [True, True], [False, True]]])
    assert_allclose(represented_fraction(safe, halo_axis=1), [[2/3, 2/3]], rtol=0, atol=0)
    values = np.array([0.0, 1.0, np.nan, 3.0])
    mask = np.array([True, True, True, False])
    stats = summarize_selected_values(values, mask)
    assert stats.n_selected == 3
    assert stats.n_finite == 2
    assert_allclose(stats.mean, 0.5, rtol=0, atol=0)
    assert_allclose(stats.p50, 0.5, rtol=0, atol=0)


def test_local_safety_mask_threshold():
    assert_array_equal(local_safety_mask([0.099, 0.1, 0.101]), [False, True, True])
