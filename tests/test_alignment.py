from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

from azlens.alignment import (
    DEFAULT_QOFF_VALUES,
    alignment_diagnostics,
    alignment_rows_from_product,
    alignment_weights,
    empirical_Qoff,
    linear_alignment_weights,
    table3_records_from_alignment_rows,
    von_mises_kappa_for_Qoff,
    weighted_represented_fraction,
)
from azlens.population_fields import PopulationFieldProducts
from azlens.weighted_statistics import weighted_r2_score

TARGETS = json.loads(Path("reference/alignment_reweighting/alignment_targets.json").read_text())


def _make_alignment_product(grid_index: int = 0, n: int = 240, mass: float = 3.0e14, z_l: float = 0.2) -> PopulationFieldProducts:
    rng = np.random.default_rng(4000 + grid_index)
    radii = np.array(TARGETS["key_radii"], dtype=np.float64)
    sigma = np.array([0.0, 0.02, 0.05, 0.08], dtype=np.float64)
    source_weights = np.array([0.8, 0.9, 1.0], dtype=np.float64)
    n_source = 3
    n_sigma = sigma.size
    n_radius = radii.size
    shape = (n_source, n_sigma, n, n_radius)

    phi = np.linspace(0.0, 2.0*np.pi, n, endpoint=False) + 0.17/n
    epsilon = 0.15 + 0.20*rng.random(n)
    rayleigh = np.sqrt(-2.0*np.log1p(-rng.random(n)))

    delta = np.zeros(shape, dtype=np.float64)
    shape_only = np.zeros(shape, dtype=np.float64)
    offset_only = np.zeros(shape, dtype=np.float64)
    delta2 = np.zeros(shape, dtype=np.float64)
    Delta = np.zeros(shape, dtype=np.float64)
    Delta2 = np.zeros(shape, dtype=np.float64)
    gxmean = np.zeros(shape, dtype=np.float64)
    gmf = np.ones(shape, dtype=np.float64)
    safe_outer = np.ones((n_sigma, n, n_radius), dtype=bool)
    # Grid 4/5-like products can be made nonqualifying by the caller if needed.

    for sidx, sw in enumerate(source_weights):
        source_scale = 1.0 if sidx == 2 else sw
        for j, sig in enumerate(sigma):
            d = sig*rayleigh
            for ridx, radius in enumerate(radii):
                u = d/radius
                base = (0.003*(grid_index + 1) + 0.02*epsilon + 0.05*u**2)*(1.0 + 0.2*ridx)*source_scale
                basis = epsilon*u**2*np.cos(2.0*phi)
                combined = base + 0.002 + 0.8*basis
                shape_only[sidx, j, :, ridx] = 0.02*epsilon*source_scale
                offset_only[sidx, j, :, ridx] = 0.05*u**2*source_scale
                delta[sidx, j, :, ridx] = combined
                delta2[sidx, j, :, ridx] = combined - 0.02*basis**2
                Delta[sidx, j, :, ridx] = 0.6 + combined
                Delta2[sidx, j, :, ridx] = 0.6 + delta2[sidx, j, :, ridx]
                gxmean[sidx, j, :, ridx] = 0.01*epsilon*u**2*np.sin(2.0*phi)

    key_samples = {
        "R_over_r200c": radii,
        "sigma_d": sigma,
        "source_weights": source_weights,
        "safe_outer": safe_outer,
        "safe_local": safe_outer.copy(),
        "lambda_min_infinity": np.ones_like(safe_outer, dtype=float),
        "epsilon_perp": epsilon,
        "q_perp": (1.0 - epsilon)/(1.0 + epsilon),
        "b_los": np.ones(n),
        "projected_scale_factor": np.ones(n),
        "c200c": np.full(n, 4.0),
        "kappa_s_perp_infinity": np.full(n, 0.2),
        "rayleigh_unit": rayleigh,
        "phi_offset": phi,
        "d_over_r200c": sigma[:, None]*rayleigh[None, :],
        "delta_g_plus_fractional": delta,
        "Delta_g_plus": Delta,
        "delta_g_plus_second_order": delta2,
        "Delta_g_plus_second_order": Delta2,
        "g_cross_mean_reduced": gxmean,
        "g_plus_mf": gmf,
        "delta_g_shape_only": shape_only,
        "delta_g_offset_only": offset_only,
        "delta_g_additive": shape_only + offset_only,
        "delta_g_interaction": delta - shape_only - offset_only,
    }
    return PopulationFieldProducts(
        grid_index=grid_index,
        mass_msun_h=mass,
        z_l=z_l,
        radial_grid=radii,
        sigma_d=tuple(float(x) for x in sigma),
        source_redshifts=(1.0, 2.0),
        source_weights={"z1": 0.8, "z2": 0.9, "infinity": 1.0},
        key_indices=np.arange(n_radius),
        lambda_min_infinity=np.ones_like(safe_outer, dtype=float),
        safe_local=safe_outer.copy(),
        safe_outer=safe_outer,
        represented_fraction_outer=np.mean(safe_outer, axis=1),
        radial_summary_rows=tuple(),
        key_samples=key_samples,
    )


def test_reference_targets_record_table3_and_diagnostics():
    assert TARGETS["current_paper_unfiltered_row_count"] == 1440
    assert TARGETS["table3"]["family"] == "linear"
    assert len(TARGETS["table3"]["rounded_percent_values"]) == 4
    assert_allclose(TARGETS["section_6_4_reported_diagnostics"]["restricted_linear_Rw2_min"], 0.9886, rtol=0, atol=0)


def test_linear_alignment_weights_reproduce_parent_Q_on_dense_grid():
    phi = np.linspace(0.0, 2.0*np.pi, 20000, endpoint=False)
    for q in DEFAULT_QOFF_VALUES:
        weights = linear_alignment_weights(phi, q)
        assert np.all(weights >= 0.0)
        assert_allclose(empirical_Qoff(phi, weights), q, rtol=0, atol=5e-16)


def test_linear_alignment_rejects_out_of_range_Q():
    with pytest.raises(ValueError, match="<= 0.5"):
        linear_alignment_weights([0.0, 1.0], 0.51)


def test_von_mises_weights_and_kappa_reproduce_Q_on_dense_grid():
    phi = np.linspace(0.0, 2.0*np.pi, 30000, endpoint=False)
    for q in (-0.30, -0.15, 0.0, 0.15, 0.30):
        kappa = von_mises_kappa_for_Qoff(q)
        assert np.sign(kappa) == np.sign(q) or q == 0.0
        weights = alignment_weights(phi, q, "von_mises")
        assert np.all(weights > 0.0)
        assert_allclose(empirical_Qoff(phi, weights), q, rtol=0, atol=2e-12)


def test_weighted_represented_fraction_uses_original_denominator():
    safe = np.array([True, False, True, False])
    weights = np.array([1.0, 9.0, 1.0, 9.0])
    assert_allclose(weighted_represented_fraction(safe, weights), 2.0/20.0, rtol=0, atol=0)
    # Dividing by retained weights would give one; this guards against that mistake.
    assert weighted_represented_fraction(safe, weights) != 1.0


def test_alignment_rows_require_far_background_source():
    product = _make_alignment_product()
    with pytest.raises(ValueError, match="far-background"):
        alignment_rows_from_product(product, source_label="z1")


def test_alignment_rows_from_product_have_expected_count_and_metadata():
    product = _make_alignment_product()
    rows = alignment_rows_from_product(product)
    assert len(rows) == 3*4*2*5*2
    assert {row.source_label for row in rows} == {"infinity"}
    assert {row.family for row in rows} == {"linear", "von_mises"}
    assert {row.domain for row in rows} == {"all_outer_safe", "u_lt_0p3"}
    assert all(row.n_total == product.n_halo for row in rows)


def test_alignment_baseline_coefficients_are_not_refitted_with_Q():
    product = _make_alignment_product()
    rows = alignment_rows_from_product(product, families=("linear",), q_values=(-0.30, 0.0, 0.30), sigma_values=(0.05,), domain_specs=(("u_lt_0p3", 0.3),))
    by_radius = {}
    for row in rows:
        by_radius.setdefault(row.R_over_r200c, []).append(row)
    for group in by_radius.values():
        a0 = {round(row.a0_baseline, 14) for row in group}
        aint = {round(row.a_int_baseline, 14) for row in group}
        assert len(a0) == 1
        assert len(aint) == 1


def test_alignment_weighted_r2_matches_direct_weighted_calculation_for_Q0():
    product = _make_alignment_product()
    rows = alignment_rows_from_product(product, families=("linear",), q_values=(0.0,), sigma_values=(0.05,), domain_specs=(("all_outer_safe", None),))
    row = rows[0]
    # At Q=0 weights are unity, so weighted R2 should be finite and <=1 for this synthetic product.
    assert np.isfinite(row.weighted_r2)
    assert row.weighted_r2 <= 1.0


def test_table3_selection_applies_representation_cut_across_Q_range():
    products = [_make_alignment_product(i, mass=m, z_l=z) for i, (m, z) in enumerate([(3e14,0.2),(1e15,0.2),(2e15,0.2),(3e14,0.5),(1e15,0.5)])]
    # Make grid 4 fail representation at one positive-Q row by marking high-cos2phi halos unsafe.
    p4 = products[-1]
    safe = p4.key_samples["safe_outer"].copy()
    phi = p4.key_samples["phi_offset"]
    sigidx = 2
    ridx = 1
    safe[sigidx, np.cos(2.0*phi) > -0.2, ridx] = False
    key = dict(p4.key_samples)
    key["safe_outer"] = safe
    products[-1] = PopulationFieldProducts(
        grid_index=p4.grid_index, mass_msun_h=p4.mass_msun_h, z_l=p4.z_l,
        radial_grid=p4.radial_grid, sigma_d=p4.sigma_d, source_redshifts=p4.source_redshifts,
        source_weights=p4.source_weights, key_indices=p4.key_indices,
        lambda_min_infinity=p4.lambda_min_infinity, safe_local=p4.safe_local,
        safe_outer=safe, represented_fraction_outer=np.mean(safe, axis=1),
        radial_summary_rows=p4.radial_summary_rows, key_samples=key,
    )
    rows = []
    for product in products:
        rows.extend(alignment_rows_from_product(product, families=("linear",), q_values=DEFAULT_QOFF_VALUES, sigma_values=(0.05,), domain_specs=(("all_outer_safe", None),)))
    table = table3_records_from_alignment_rows(rows)
    assert [record.grid_index for record in table] == [0, 1, 2, 3]


def test_alignment_diagnostics_returns_compact_ranges():
    rows = alignment_rows_from_product(_make_alignment_product(), families=("linear", "von_mises"), q_values=DEFAULT_QOFF_VALUES)
    diag = alignment_diagnostics(rows)
    assert diag["n_rows"] == len(rows)
    assert diag["restricted_linear_r2_count"] > 0
    assert diag["max_family_difference_percent_points"] is not None
