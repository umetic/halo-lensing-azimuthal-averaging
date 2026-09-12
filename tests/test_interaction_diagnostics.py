from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

from azlens.interaction_diagnostics import (
    conditioning_r2_diagnostic,
    conditioning_rows_from_product,
    fit_hybrid_interaction,
    hybrid_fit_rows_from_product,
    interaction_basis,
    restricted_hybrid_r2_summary,
)
from azlens.population_fields import PopulationFieldProducts
from azlens.statistics import r2_score

TARGETS = json.loads(Path("reference/interaction_diagnostics/interaction_diagnostics_targets.json").read_text())


def _synthetic_vectors():
    epsilon = np.array([0.12, 0.21, 0.33, 0.18, 0.27, 0.39], dtype=float)
    u = np.array([0.08, 0.16, 0.24, 0.31, 0.37, 0.44], dtype=float)
    phi = np.deg2rad([0.0, 20.0, 45.0, 70.0, 110.0, 150.0])
    shape = np.array([0.011, 0.009, 0.006, 0.003, -0.002, -0.004], dtype=float)
    offset = np.array([0.001, 0.002, 0.003, 0.006, 0.009, 0.010], dtype=float)
    basis = interaction_basis(epsilon, u, phi)
    return epsilon, u, phi, shape, offset, basis


def _make_synthetic_product() -> PopulationFieldProducts:
    n = 6
    radii = np.array(TARGETS["key_radii"], dtype=np.float64)
    sigma = np.array([0.0, 0.02, 0.05, 0.08], dtype=np.float64)
    n_source = 3
    n_sigma = sigma.size
    n_radius = radii.size
    shape = (n_source, n_sigma, n, n_radius)

    delta = np.zeros(shape, dtype=np.float64)
    shape_only = np.zeros(shape, dtype=np.float64)
    offset_only = np.zeros(shape, dtype=np.float64)
    delta2 = np.zeros(shape, dtype=np.float64)
    Delta = np.zeros(shape, dtype=np.float64)
    Delta2 = np.zeros(shape, dtype=np.float64)

    epsilon = np.array([0.12, 0.21, 0.33, 0.18, 0.27, 0.39], dtype=float)
    phi = np.deg2rad([0.0, 20.0, 45.0, 70.0, 110.0, 150.0])
    rayleigh = np.array([0.30, 0.50, 0.70, 1.00, 2.00, 3.00], dtype=float)
    safe_outer = np.ones((n_sigma, n, n_radius), dtype=bool)
    safe_outer[3, 5, 0] = False

    for sidx in range(n_source):
        for j, sig in enumerate(sigma):
            d = sig*rayleigh
            for ridx, radius in enumerate(radii):
                u = d/radius
                additive = 0.02*epsilon + 0.03*u**2
                offset_component = 0.03*u**2
                shape_component = 0.02*epsilon
                basis = epsilon*u**2*np.cos(2.0*phi)
                combined = additive + 0.004 + 1.7*basis
                shape_only[sidx, j, :, ridx] = shape_component
                offset_only[sidx, j, :, ridx] = offset_component
                delta[sidx, j, :, ridx] = combined
                delta2[sidx, j, :, ridx] = combined - 0.1*basis**2
                Delta[sidx, j, :, ridx] = 0.5 + 2.0*combined
                Delta2[sidx, j, :, ridx] = 0.5 + 2.0*delta2[sidx, j, :, ridx]

    key_samples = {
        "R_over_r200c": radii,
        "sigma_d": sigma,
        "source_weights": np.array([0.8, 0.9, 1.0]),
        "safe_outer": safe_outer,
        "safe_local": np.ones_like(safe_outer),
        "lambda_min_infinity": np.ones((n_sigma, n, n_radius)),
        "epsilon_perp": epsilon,
        "phi_offset": phi,
        "rayleigh_unit": rayleigh,
        "d_over_r200c": sigma[:, None]*rayleigh[None, :],
        "delta_g_plus_fractional": delta,
        "delta_g_shape_only": shape_only,
        "delta_g_offset_only": offset_only,
        "delta_g_additive": shape_only + offset_only,
        "delta_g_interaction": delta - shape_only - offset_only,
        "delta_g_plus_second_order": delta2,
        "Delta_g_plus": Delta,
        "Delta_g_plus_second_order": Delta2,
    }
    return PopulationFieldProducts(
        grid_index=0,
        mass_msun_h=3.0e14,
        z_l=0.2,
        radial_grid=radii,
        sigma_d=tuple(float(x) for x in sigma),
        source_redshifts=(1.0, 2.0),
        source_weights={"z1": 0.8, "z2": 0.9, "infinity": 1.0},
        key_indices=np.arange(n_radius),
        lambda_min_infinity=np.ones((n_sigma, n, n_radius)),
        safe_local=np.ones_like(safe_outer),
        safe_outer=safe_outer,
        represented_fraction_outer=np.mean(safe_outer, axis=1),
        radial_summary_rows=tuple(),
        key_samples=key_samples,
    )


def test_reference_targets_record_section_6_values():
    target = TARGETS["section_6_1_restricted"]
    assert target["n_fits"] == 72
    assert_allclose(target["r2_min"], 0.9943526126179945, rtol=0, atol=0)
    assert_allclose(target["r2_median"], 0.9988671021584998, rtol=0, atol=0)
    cond = TARGETS["section_6_2_conditioning"]
    assert cond[0]["n_used"] == 9594
    assert_allclose(cond[0]["r2_fractional"], 0.9008233931123506, rtol=0, atol=0)


def test_interaction_basis_matches_definition():
    epsilon, u, phi, *_ = _synthetic_vectors()
    assert_allclose(interaction_basis(epsilon, u, phi), epsilon*u**2*np.cos(2.0*phi), rtol=0, atol=0)


def test_fit_hybrid_interaction_recovers_intercept_and_interaction_coefficient():
    epsilon, u, phi, shape, offset, basis = _synthetic_vectors()
    a0 = -0.003
    aint = 2.25
    y = shape + offset + a0 + aint*basis
    fit = fit_hybrid_interaction(y, shape, offset, epsilon, u, phi, np.ones_like(y, dtype=bool))
    assert fit.n_total == y.size
    assert fit.n_selected == y.size
    assert fit.n_used == y.size
    assert_allclose(fit.a0, a0, rtol=0, atol=2e-15)
    assert_allclose(fit.a_int, aint, rtol=0, atol=2e-14)
    assert_allclose(fit.r2, 1.0, rtol=0, atol=2e-14)
    assert_allclose(fit.prediction, y, rtol=0, atol=2e-15)


def test_fit_hybrid_interaction_u_cut_applies_without_changing_arrays():
    epsilon, u, phi, shape, offset, basis = _synthetic_vectors()
    y = shape + offset + 0.1 + 3.0*basis
    y[-1] += 99.0  # outside u<0.3 and should not affect the restricted fit
    fit = fit_hybrid_interaction(y, shape, offset, epsilon, u, phi, u_max=0.3)
    assert fit.n_selected == int(np.count_nonzero(u < 0.3))
    assert fit.n_used == fit.n_selected
    assert_allclose(fit.a0, 0.1, rtol=0, atol=2e-15)
    assert_allclose(fit.a_int, 3.0, rtol=0, atol=2e-14)


def test_fit_hybrid_interaction_keeps_additive_coefficient_fixed():
    epsilon, u, phi, shape, offset, basis = _synthetic_vectors()
    additive = shape + offset
    y = 2.0*additive + 0.04 + 1.5*basis
    fit = fit_hybrid_interaction(y, shape, offset, epsilon, u, phi)
    # If an additive amplitude were fitted freely this synthetic construction
    # would be reproduced exactly. With the coefficient fixed to unity it is not.
    assert fit.r2 < 0.9999
    assert not np.allclose(fit.prediction, y, rtol=0, atol=1e-12)


def test_conditioning_diagnostic_is_no_fit_comparison():
    x = np.linspace(-1.0, 1.0, 8)
    pred = x.copy()
    y = 2.0*x + 0.5
    diag = conditioning_r2_diagnostic(y, pred, y, pred)
    assert diag.n_used == x.size
    assert diag.r2_fractional < 1.0
    assert diag.r2_difference < 1.0
    assert_allclose(diag.r2_fractional, r2_score(y, pred), rtol=0, atol=0)


def test_conditioning_diagnostic_uses_joint_finite_mask():
    y = np.array([0.0, 1.0, np.nan, 3.0, 4.0])
    p = np.array([0.0, 1.1, 2.0, 3.1, 4.1])
    d = np.array([2.0, 4.0, 6.0, np.inf, 10.0])
    dp = np.array([2.0, 4.1, 6.1, 8.1, 10.1])
    diag = conditioning_r2_diagnostic(y, p, d, dp, [True, True, True, True, False])
    assert diag.n_selected == 4
    assert diag.n_used == 2
    assert np.array_equal(np.where(diag.selected_finite_mask)[0], np.array([0, 1]))


def test_hybrid_rows_from_product_iterate_nonzero_offsets_and_domains():
    product = _make_synthetic_product()
    rows = hybrid_fit_rows_from_product(product)
    assert len(rows) == 3*4*2
    restricted = [row for row in rows if row.domain == "u_lt_0p3"]
    assert len(restricted) == 12
    assert all(row.source_label == "z1" for row in rows)
    assert {row.sigma_d for row in rows} == {0.02, 0.05, 0.08}
    summary = restricted_hybrid_r2_summary(rows)
    assert summary["n_fits"] == 12
    assert summary["r2_min"] > 0.99


def test_hybrid_rows_from_product_do_not_apply_common_domain_or_representation_cut():
    product = _make_synthetic_product()
    rows = hybrid_fit_rows_from_product(product, sigma_values=(0.08,), domain_specs=(("u_lt_0p3", 0.3),))
    # The first key radius has one unsafe halo, but it is still analyzed with
    # the selected retained subset rather than being rejected by a 95% rule.
    first = rows[0]
    assert_allclose(first.R_over_r200c, TARGETS["key_radii"][0], rtol=0, atol=0)
    assert first.n_selected < first.n_total
    assert first.n_used == first.n_selected


def test_conditioning_rows_from_product_select_requested_radii_and_sigma():
    product = _make_synthetic_product()
    requested = [TARGETS["key_radii"][0], TARGETS["key_radii"][2]]
    rows = conditioning_rows_from_product(product, sigma_d=0.08, radii=requested)
    assert len(rows) == 2
    assert_allclose([r.R_over_r200c for r in rows], requested, rtol=0, atol=0)
    assert rows[0].n_selected == 5
    assert rows[1].n_selected == 6
    assert rows[0].r2_fractional <= 1.0
    assert rows[0].r2_difference <= 1.0


def test_product_source_label_validation():
    product = _make_synthetic_product()
    with pytest.raises(ValueError):
        hybrid_fit_rows_from_product(product, source_label="z3")
