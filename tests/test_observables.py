"""Nonlinear ring observable tests."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from azlens.cosmology import PUBLICATION_COSMOLOGY
from azlens.nfw import nfw_normalization
from azlens.observables import angular_covariance, evaluate_ring_observables, scale_lensing_fields
from azlens.ring_fields import CircularNFWFieldEvaluator, RingFields, evaluate_ring_fields, midpoint_angles


def sample_fields():
    phi = midpoint_angles(64)
    kappa = 0.12 + 0.02*np.cos(2*phi) + 0.01*np.sin(phi)
    gp = 0.08 + 0.015*np.cos(2*phi) - 0.006*np.sin(3*phi)
    gx = 0.01 + 0.012*np.sin(2*phi) + 0.004*np.cos(phi)
    return kappa[None, :], gp[None, :], gx[None, :]


def test_scale_lensing_fields_applies_linear_weight_only_to_fields():
    k, gp, gx = sample_fields()
    out = scale_lensing_fields(k, gp, gx, source_weight=0.5)
    assert_allclose(out[0], 0.5*k)
    assert_allclose(out[1], 0.5*gp)
    assert_allclose(out[2], 0.5*gx)
    with pytest.raises(ValueError):
        scale_lensing_fields(k, gp, gx, source_weight=0.0)


def test_angular_covariance_uses_ddof_zero():
    a = np.array([[1.0, 2.0, 5.0, 6.0]])
    b = np.array([[0.0, 1.0, 1.0, 2.0]])
    expected = np.mean((a-a.mean(axis=1)[:, None])*(b-b.mean(axis=1)[:, None]), axis=1)
    assert_allclose(angular_covariance(a, b), expected, rtol=0, atol=0)
    assert not np.allclose(angular_covariance(a, b), np.cov(a[0], b[0], ddof=1)[0, 1])


def test_inverse_magnification_identity_with_nonzero_cross_mean():
    k, gp, gx = sample_fields()
    summary = evaluate_ring_observables(k, gp, gx, source_weight=1.0, count_slopes=(0.3,))
    cov = summary.covariances
    identity = cov.C_kk - cov.C_plus_plus - cov.C_cross_cross
    assert_allclose(summary.magnification.Y_difference, identity, rtol=3e-15, atol=3e-15)
    D = 1.0 - summary.kappa_bar
    y_mf_with_cross = D**2 - summary.gamma_plus_bar**2 - summary.gamma_cross_bar**2
    y_mf_without_cross = D**2 - summary.gamma_plus_bar**2
    assert_allclose(summary.magnification.Y_mf, y_mf_with_cross)
    assert np.max(np.abs(y_mf_with_cross-y_mf_without_cross)) > 0


def test_finite_source_scaling_before_reduced_shear_is_not_posthoc_scaling():
    k, gp, gx = sample_fields()
    far = evaluate_ring_observables(k, gp, gx, source_weight=1.0)
    finite = evaluate_ring_observables(k, gp, gx, source_weight=0.5)
    assert np.max(np.abs(finite.reduced_shear.g_plus_mean - 0.5*far.reduced_shear.g_plus_mean)) > 1e-3


def test_reduced_shear_denominator_guard_affects_fractional_not_difference():
    phi = midpoint_angles(32)
    kappa = np.full((1, phi.size), 0.1)
    gp = 0.02*np.cos(phi)[None, :]
    gx = np.zeros_like(gp)
    summary = evaluate_ring_observables(kappa, gp, gx, denominator_tol=1e-14)
    assert_allclose(summary.reduced_shear.g_plus_mf, 0.0, atol=5e-18)
    assert np.isnan(summary.reduced_shear.g_plus_fractional_residual[0])
    assert np.isfinite(summary.reduced_shear.g_plus_difference[0])


def test_reduced_shear_second_order_formula_is_used():
    k, gp, gx = sample_fields()
    summary = evaluate_ring_observables(k, gp, gx)
    D = 1.0 - summary.kappa_bar
    expected = summary.covariances.C_k_plus/D**2 + summary.gamma_plus_bar*summary.covariances.C_kk/D**3
    assert_allclose(summary.reduced_shear.g_plus_second_order_difference, expected, rtol=0, atol=0)
    assert_allclose(
        summary.reduced_shear.g_plus_second_order_fractional,
        expected/summary.reduced_shear.g_plus_mf,
        rtol=0,
        atol=0,
    )


def test_magnification_bias_q1_recovers_inverse_magnification_residual():
    k, gp, gx = sample_fields()
    summary = evaluate_ring_observables(k, gp, gx, count_slopes=(0.0, 0.3, 1.4))
    assert_allclose(
        summary.magnification.magnification_bias_fractional_residual[0.0],
        summary.magnification.Y_fractional_residual,
        rtol=3e-15,
        atol=3e-15,
    )
    assert_allclose(
        summary.magnification.magnification_bias_second_order_fractional[0.0],
        summary.magnification.Y_second_order_fractional,
        rtol=0,
        atol=0,
    )


def test_ringfields_input_and_array_input_are_equivalent():
    k, gp, gx = sample_fields()
    fields = RingFields(kappa=k, gamma_plus=gp, gamma_cross=gx)
    a = evaluate_ring_observables(fields, source_weight=0.7)
    b = evaluate_ring_observables(k, gp, gx, source_weight=0.7)
    assert_allclose(a.reduced_shear.g_plus_mean, b.reduced_shear.g_plus_mean)
    assert_allclose(a.magnification.mu_fractional_residual, b.magnification.mu_fractional_residual)
    with pytest.raises(ValueError):
        evaluate_ring_observables(fields, gp, gx)


def test_lambda_min_uses_scaled_fields():
    k, gp, gx = sample_fields()
    summary = evaluate_ring_observables(k, gp, gx, source_weight=0.4)
    expected = np.min(1.0-0.4*k-np.hypot(0.4*gp, 0.4*gx), axis=1)
    assert_allclose(summary.lambda_min, expected)


def test_appendix_d_displaced_spherical_anchor_at_zs1():
    cosmo = PUBLICATION_COSMOLOGY
    z_l, z_s = 0.3, 1.0
    c = 3.843
    norm = nfw_normalization(1.0e15, c, z_l, cosmology=cosmo)
    fields = evaluate_ring_fields(
        CircularNFWFieldEvaluator(),
        concentration=c,
        projected_scale_factor=1.0,
        kappa_s_projected=float(norm.kappa_s_infinity),
        radius_over_r200c=0.3,
        d_over_r200c=0.10,
        phi_offset=0.0,
        n_phi=8192,
    )
    summary = evaluate_ring_observables(fields, source_weight=cosmo.source_weight(z_l, z_s))
    assert_allclose(100.0*summary.reduced_shear.g_plus_fractional_residual[0], 0.6377685010562573, rtol=3e-5, atol=3e-6)
    assert abs(summary.reduced_shear.g_cross_mean[0]) < 1e-15


def test_invalid_observable_inputs():
    k, gp, gx = sample_fields()
    with pytest.raises(ValueError):
        evaluate_ring_observables(k[:, :-1], gp, gx)
    with pytest.raises(ValueError):
        evaluate_ring_observables(k, gp, gx, count_slopes=(np.nan,))
    bad = k.copy(); bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        evaluate_ring_observables(bad, gp, gx)
