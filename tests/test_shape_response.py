from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from azlens.population_fields import PopulationFieldProducts
from azlens.shape_response import (
    CENTERED_SHAPE_RADIUS,
    SHAPE_RESPONSE_COLUMNS,
    centered_shape_input_from_product,
    centered_shape_response_rows,
    centered_shape_response_rows_from_products,
    compact_shape_response_metrics,
    leading_shape_coefficient_Aepsilon,
    nfw_f_derivative,
    nfw_j4_shape,
    quantile_bins_epsilon2,
    read_shape_response_csv,
    retained_counts_by_grid,
    write_shape_response_csv,
)
from azlens.nfw import nfw_f

TARGETS = json.loads(Path("reference/shape_response/centered_shape_targets.json").read_text())


def test_nfw_f_derivative_matches_finite_difference_away_from_singularity():
    x = np.array([0.05, 0.3, 0.8, 1.2, 3.0, 12.0])
    h = 2e-6*np.maximum(1.0, x)
    fd = (nfw_f(x+h) - nfw_f(x-h))/(2.0*h)
    assert_allclose(nfw_f_derivative(x), fd, rtol=3e-7, atol=3e-8)


def test_nfw_f_derivative_is_regular_near_unity():
    x = np.array([1.0, np.nextafter(1.0, 0.0), np.nextafter(1.0, 2.0)])
    values = nfw_f_derivative(x)
    assert np.all(np.isfinite(values))
    assert_allclose(values[0], -0.4, rtol=0, atol=0)


def test_nfw_j4_and_leading_shape_coefficient_are_finite():
    x = np.array([0.7, 1.0, 1.8])
    kappa_s = np.array([0.05, 0.08, 0.04])
    j4 = nfw_j4_shape(x)
    coeff = leading_shape_coefficient_Aepsilon(x, kappa_s)
    assert np.all(np.isfinite(j4))
    assert np.all(np.isfinite(coeff))
    assert coeff.shape == x.shape


def test_quantile_bins_assign_internal_edge_ties_to_upper_bin():
    eps = np.sqrt(np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    result = quantile_bins_epsilon2(eps, np.ones(5, dtype=bool), n_bins=2)
    assert_allclose(result.edges, [0.0, 2.0, 4.0], rtol=0, atol=1e-15)
    assert_array_equal(result.bin_index, [0, 0, 1, 1, 1])


def _synthetic_shape_input():
    from azlens.shape_response import ShapeResponseInput
    eps = np.sqrt(np.array([0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64]))
    return ShapeResponseInput(
        grid_index=7,
        mass_msun_h=1e15,
        z_l=0.3,
        source_z=1.0,
        sigma_d=0.0,
        radius_over_r200c=CENTERED_SHAPE_RADIUS,
        epsilon_perp=eps,
        c200c=np.array([3.0, 3.2, 3.5, 3.8, 4.0, 4.3, 4.6, 5.0]),
        projected_scale_factor=np.array([0.9, 1.0, 1.1, 0.95, 1.05, 1.2, 0.85, 1.15]),
        kappa_s_perp_infinity=np.array([0.06, 0.07, 0.08, 0.09, 0.075, 0.065, 0.055, 0.095]),
        source_weight=0.8,
        safe_outer_infinity=np.ones(8, dtype=bool),
        delta_g_plus_fractional=np.array([0.001, 0.002, 0.004, 0.006, 0.009, 0.012, 0.015, 0.020]),
        delta_g_plus_second_order=np.array([0.0011, 0.0021, 0.0042, 0.0064, 0.0092, 0.0125, 0.0155, 0.0204]),
    )


def test_shape_response_allows_nonfinite_unselected_residuals():
    data = _synthetic_shape_input()
    from azlens.shape_response import ShapeResponseInput
    residual = data.delta_g_plus_fractional.copy()
    residual[-1] = np.nan
    safe = data.safe_outer_infinity.copy()
    safe[-1] = False
    new_data = ShapeResponseInput(
        grid_index=data.grid_index,
        mass_msun_h=data.mass_msun_h,
        z_l=data.z_l,
        source_z=data.source_z,
        sigma_d=data.sigma_d,
        radius_over_r200c=data.radius_over_r200c,
        epsilon_perp=data.epsilon_perp,
        c200c=data.c200c,
        projected_scale_factor=data.projected_scale_factor,
        kappa_s_perp_infinity=data.kappa_s_perp_infinity,
        source_weight=data.source_weight,
        safe_outer_infinity=safe,
        delta_g_plus_fractional=residual,
        delta_g_plus_second_order=data.delta_g_plus_second_order,
    )
    rows = centered_shape_response_rows(new_data, n_bins=2)
    assert sum(row["n_bin"] for row in rows) == 7


def test_centered_shape_response_rows_use_median_of_halo_level_products():
    data = _synthetic_shape_input()
    rows = centered_shape_response_rows(data, n_bins=2)
    assert len(rows) == 2
    coeff = leading_shape_coefficient_Aepsilon(data.x_nfw, data.kappa_s_perp_source)
    eps2 = data.epsilon_perp**2
    for row in rows:
        mask = quantile_bins_epsilon2(data.epsilon_perp, data.safe_outer_infinity, n_bins=2).bin_index == row["bin_index"]
        expected = 100.0*np.median(coeff[mask]*eps2[mask])
        product_of_medians = 100.0*np.median(coeff[mask])*np.median(eps2[mask])
        assert_allclose(row["delta_g_leading_prediction_median_percent"], expected, rtol=0, atol=1e-14)
        assert abs(expected - product_of_medians) > 1e-7


def _synthetic_product(grid_index: int = 0, safe_count: int = 5) -> PopulationFieldProducts:
    n = 6
    radii = np.array([0.2091279105182546, CENTERED_SHAPE_RADIUS, 0.5113850166642591, 1.0])
    sigma = (0.0, 0.05)
    safe = np.zeros((2, n, 4), dtype=bool)
    safe[:, :, :] = True
    safe[0, safe_count:, 1] = False
    delta = np.full((3, 2, n, 4), np.nan)
    delta_second = np.full_like(delta, np.nan)
    delta[0, 0, :, 1] = np.linspace(0.001, 0.006, n)
    delta_second[0, 0, :, 1] = np.linspace(0.0011, 0.0061, n)
    key = {
        "R_over_r200c": radii,
        "sigma_d": np.array(sigma),
        "source_weights": np.array([0.8, 0.9, 1.0]),
        "lambda_min_infinity": np.ones((2, n, 4)),
        "safe_local": safe.copy(),
        "safe_outer": safe.copy(),
        "epsilon_perp": np.sqrt(np.linspace(0.01, 0.36, n)),
        "q_perp": np.ones(n)*0.8,
        "b_los": np.ones(n),
        "projected_scale_factor": np.ones(n),
        "c200c": np.linspace(3.0, 4.2, n),
        "kappa_s_perp_infinity": np.linspace(0.06, 0.08, n),
        "rayleigh_unit": np.ones(n),
        "phi_offset": np.zeros(n),
        "d_over_r200c": np.zeros((2, n)),
        "delta_g_plus_fractional": delta,
        "delta_g_plus_second_order": delta_second,
        "Delta_g_plus": delta,
        "Delta_g_plus_second_order": delta_second,
        "g_cross_mean_reduced": np.zeros_like(delta),
        "g_plus_mf": np.ones_like(delta),
        "delta_g_shape_only": delta,
        "delta_g_offset_only": np.zeros_like(delta),
        "delta_g_additive": delta,
        "delta_g_interaction": np.zeros_like(delta),
    }
    return PopulationFieldProducts(
        grid_index=grid_index,
        mass_msun_h=[3e14, 1e15, 2e15, 3e14, 1e15, 2e15, 1.0][grid_index if grid_index < 6 else 6],
        z_l=0.2 if grid_index < 3 else 0.5,
        radial_grid=radii,
        sigma_d=sigma,
        source_redshifts=(1.0, 2.0),
        source_weights={"z1": 0.8, "z2": 0.9, "infinity": 1.0},
        key_indices=np.arange(4),
        lambda_min_infinity=np.ones((2, n, 4)),
        safe_local=safe.copy(),
        safe_outer=safe.copy(),
        represented_fraction_outer=np.mean(safe, axis=1),
        radial_summary_rows=tuple(),
        key_samples=key,
    )


def test_centered_shape_input_from_product_uses_sigma_zero_z1_and_diagnostic_radius():
    product = _synthetic_product(grid_index=0, safe_count=4)
    data = centered_shape_input_from_product(product)
    assert data.grid_index == 0
    assert data.retained_count == 4
    assert_allclose(data.source_weight, 0.8, rtol=0, atol=0)
    assert_allclose(data.radius_over_r200c, CENTERED_SHAPE_RADIUS, rtol=0, atol=0)
    assert_allclose(data.delta_g_plus_fractional, product.key_samples["delta_g_plus_fractional"][0, 0, :, 1])


def test_centered_shape_response_does_not_apply_common_domain_cut():
    product = _synthetic_product(grid_index=0, safe_count=4)  # below a 95% represented fraction.
    rows = centered_shape_response_rows_from_products([product], n_bins=2)
    assert len(rows) == 2
    assert sum(row["n_bin"] for row in rows) == 4


def test_shape_response_csv_roundtrip(tmp_path):
    rows = centered_shape_response_rows(_synthetic_shape_input(), n_bins=2)
    path = tmp_path/"shape_response.csv"
    write_shape_response_csv(rows, path)
    loaded = read_shape_response_csv(path)
    assert tuple(loaded[0].keys()) == SHAPE_RESPONSE_COLUMNS
    assert int(loaded[0]["grid_index"]) == 7
    assert_allclose(float(loaded[0]["R_over_R200c"]), CENTERED_SHAPE_RADIUS, rtol=0, atol=0)


def test_shape_response_metrics_and_retained_counts():
    products = [_synthetic_product(0, 6), _synthetic_product(1, 5)]
    rows = centered_shape_response_rows_from_products(products, n_bins=2)
    counts = retained_counts_by_grid(rows)
    assert counts == {0: 6, 1: 5}
    metrics = compact_shape_response_metrics(rows)
    assert metrics["n_rows"] == 4
    assert metrics["grid_counts"] == {"0": 6, "1": 5}


def test_reference_figure3_targets_are_complete_and_ordered():
    rows = TARGETS["figure3_reference_rows"]
    assert len(rows) == 60
    assert {row["grid_index"] for row in rows} == set(range(6))
    assert sorted({row["bin_index"] for row in rows}) == list(range(10))
    counts = {}
    for row in rows:
        counts[row["grid_index"]] = counts.get(row["grid_index"], 0) + int(row["n_bin"])
    assert [counts[i] for i in range(6)] == TARGETS["retained_counts_by_grid"]
    first = rows[0]
    last = rows[-1]
    assert_allclose(first["delta_g_leading_prediction_median_percent"], 0.01594999954704009, rtol=0, atol=2e-15)
    assert_allclose(last["delta_g_median_percent"], 2.508952346609483, rtol=0, atol=2e-13)
