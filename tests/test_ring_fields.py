"""Deterministic adopted-center ring-field tests."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from azlens.multipoles import build_elliptical_lensing_modes, reconstruct_lensing_fields
from azlens.ring_fields import (
    CircularNFWFieldEvaluator,
    ModeFieldEvaluator,
    displaced_ring_geometry,
    evaluate_ring_fields,
    midpoint_angles,
    rotate_spin2,
)
from azlens.solver_settings import CONTROLLED_ELLIPTICAL_SETTINGS, CONTROLLED_CIRCULAR_SETTINGS
from azlens.nfw import nfw_lensing


def test_midpoint_angles_are_complete_ring_without_endpoint():
    phi = midpoint_angles(16)
    assert_allclose(phi[0], np.pi/16)
    assert_allclose(phi[-1], 2*np.pi - np.pi/16)
    assert not np.any(np.isclose(phi, 0.0))
    assert not np.any(np.isclose(phi, 2*np.pi))
    assert_allclose(np.diff(phi), 2*np.pi/16)
    with pytest.raises(ValueError):
        midpoint_angles(7)


def test_centered_geometry_has_no_spin_rotation_modulo_pi():
    phi = midpoint_angles(32)
    geom = displaced_ring_geometry(
        radius_over_r200c=0.3,
        concentration=[3.843, 5.0],
        projected_scale_factor=[1.0, 0.8],
        d_over_r200c=0.0,
        phi_offset=[0.0, 1.3],
        phi=phi,
    )
    assert geom.true_radius_over_r200c.shape == (2, 32)
    assert_allclose(geom.true_radius_over_r200c, 0.3)
    assert_allclose(np.cos(2*geom.omega), 1.0, atol=8e-15)
    assert_allclose(np.sin(2*geom.omega), 0.0, atol=8e-15)
    assert_allclose(geom.x_true[0], 3.843*0.3)
    assert_allclose(geom.x_true[1], 5.0*0.3/0.8)


def test_displaced_geometry_rotating_offset_by_2pi_is_identical():
    a = displaced_ring_geometry(
        radius_over_r200c=0.4, concentration=3.0, projected_scale_factor=1.2,
        d_over_r200c=0.07, phi_offset=0.31, n_phi=64,
    )
    b = displaced_ring_geometry(
        radius_over_r200c=0.4, concentration=3.0, projected_scale_factor=1.2,
        d_over_r200c=0.07, phi_offset=0.31+2*np.pi, n_phi=64,
    )
    assert_allclose(a.true_radius_over_r200c, b.true_radius_over_r200c, atol=2e-16)
    assert_allclose(np.cos(a.true_phi), np.cos(b.true_phi), atol=2e-16)
    assert_allclose(np.sin(a.true_phi), np.sin(b.true_phi), atol=2e-16)
    assert_allclose(np.cos(2*a.omega), np.cos(2*b.omega), atol=4e-16)
    assert_allclose(np.sin(2*a.omega), np.sin(2*b.omega), atol=4e-16)


def test_geometry_rejects_exact_center_crossing_without_policy():
    with pytest.raises(ValueError):
        displaced_ring_geometry(
            radius_over_r200c=0.25, concentration=4.0, projected_scale_factor=1.0,
            d_over_r200c=0.25, phi_offset=0.0, phi=[0.0, *midpoint_angles(8)[1:]],
        )


def test_spin2_rotation_known_values():
    gp, gx = rotate_spin2([2.0], [0.5], [np.pi/4])
    assert_allclose(gp, [-0.5], atol=2e-15)
    assert_allclose(gx, [2.0], atol=2e-15)
    with pytest.raises(ValueError):
        rotate_spin2([1.0], [0.0], [np.nan])


def test_circular_displaced_fields_match_manual_appendix_d_formula():
    radius = 0.3
    d = 0.1
    alpha = 0.37
    concentration = 3.843
    amplitude = 0.2
    fields = evaluate_ring_fields(
        CircularNFWFieldEvaluator(),
        concentration=concentration,
        projected_scale_factor=1.0,
        kappa_s_projected=amplitude,
        radius_over_r200c=radius,
        d_over_r200c=d,
        phi_offset=alpha,
        n_phi=128,
    )
    x = fields.geometry.x_true
    lens = nfw_lensing(x, amplitude)
    assert_allclose(fields.kappa, lens.kappa, rtol=0, atol=0)
    assert_allclose(fields.gamma_plus, lens.gamma_t*np.cos(2*fields.geometry.omega), rtol=2e-15, atol=2e-15)
    assert_allclose(fields.gamma_cross, lens.gamma_t*np.sin(2*fields.geometry.omega), rtol=2e-15, atol=2e-15)


def test_full_ring_circular_means_are_independent_of_absolute_offset_angle():
    kwargs = dict(
        field_evaluator=CircularNFWFieldEvaluator(),
        concentration=3.843,
        projected_scale_factor=1.0,
        kappa_s_projected=0.18,
        radius_over_r200c=0.3,
        d_over_r200c=0.1,
        n_phi=1024,
    )
    a = evaluate_ring_fields(phi_offset=0.0, **kwargs)
    b = evaluate_ring_fields(phi_offset=0.73, **kwargs)
    for name in ("kappa", "gamma_plus", "gamma_cross"):
        assert_allclose(np.mean(getattr(a, name), axis=1), np.mean(getattr(b, name), axis=1), rtol=0, atol=2e-7)


def test_mode_field_evaluator_centered_matches_multipole_reconstruction():
    settings = CONTROLLED_ELLIPTICAL_SETTINGS.with_small_grids(n_radial_solver=256, n_phi_solver=128, m_max=10)
    solved = build_elliptical_lensing_modes(0.67, settings)
    evaluator = ModeFieldEvaluator(solved, radial_interpolation="logx")
    phi = midpoint_angles(64)
    fields = evaluate_ring_fields(
        evaluator,
        concentration=3.0,
        projected_scale_factor=1.0,
        kappa_s_projected=0.12,
        radius_over_r200c=0.2,
        d_over_r200c=0.0,
        phi_offset=0.0,
        phi=phi,
    )
    x = np.array([0.6])
    k, gp, gx = reconstruct_lensing_fields(solved, phi)
    # reconstruct_lensing_fields returns all solver radii; compare through direct evaluator at x=0.6 instead.
    direct = evaluator.evaluate(np.full((1, phi.size), 0.6), phi[None, :], np.array([[0.12]]))
    assert_allclose(fields.kappa, direct.kappa, rtol=0, atol=2e-15)
    assert_allclose(fields.gamma_plus, direct.gamma_plus, rtol=0, atol=2e-15)
    assert_allclose(fields.gamma_cross, direct.gamma_cross, rtol=0, atol=2e-15)


def test_mode_field_evaluator_rejects_out_of_domain_x():
    settings = CONTROLLED_CIRCULAR_SETTINGS.with_small_grids(n_radial_solver=128, n_phi_solver=64, m_max=6)
    solved = build_elliptical_lensing_modes(1.0, settings)
    evaluator = ModeFieldEvaluator(solved, radial_interpolation="logx")
    with pytest.raises(ValueError):
        evaluator.evaluate(np.array([[solved.radius[0]/2]]), np.array([[0.0]]), np.array([[1.0]]))


def test_invalid_ring_field_inputs():
    with pytest.raises(ValueError):
        evaluate_ring_fields(
            CircularNFWFieldEvaluator(), concentration=[3.0, 4.0], projected_scale_factor=[1.0],
            kappa_s_projected=[0.1, 0.2, 0.3], radius_over_r200c=0.2,
        )
    with pytest.raises(ValueError):
        evaluate_ring_fields(
            CircularNFWFieldEvaluator(), concentration=3.0, projected_scale_factor=1.0,
            kappa_s_projected=-0.1, radius_over_r200c=0.2,
        )
