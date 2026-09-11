"""Fourier--Green multipole reconstruction tests."""
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import cumulative_simpson

from azlens.multipoles import (
    FourierModes,
    angular_fourier_coefficients,
    reconstruct_from_fourier,
    nonmonopole_mode_power,
    area_preserving_elliptical_radius_factor,
    elliptical_nfw_convergence_on_grid,
    solve_lensing_from_convergence_modes,
    reconstruct_lensing_fields,
    interpolate_lensing_modes,
    build_elliptical_lensing_modes,
    build_stored_elliptical_mode_grid,
)
from azlens.nfw import nfw_lensing
from azlens.solver_settings import SolverSettings, CONTROLLED_CIRCULAR_SETTINGS

REF = json.loads((Path(__file__).resolve().parents[1]/"reference/multipoles/elliptical_modes_reference.json").read_text())


def compact_settings(case):
    return SolverSettings(
        name="compact",
        q_fixed=case["q"],
        x_min=min(case["target_radius"]),
        x_max=max(case["target_radius"]),
        n_x=len(case["target_radius"]),
        x_solver_min=case["x_solver_min"],
        x_solver_max=case["x_solver_max"],
        n_radial_solver=case["n_radial_solver"],
        n_phi_solver=case["n_phi_solver"],
        m_max=case["m_max"],
        radial_interpolation=case["radial_interpolation"],
    )


def test_fourier_coefficients_and_power():
    n = 128
    phi = 2*np.pi*np.arange(n)/n
    values = 2.5 + 1.25*np.cos(3*phi) - 0.75*np.sin(2*phi)
    modes = angular_fourier_coefficients(values, 6)
    assert_allclose(modes.cosine[0], 2.5, rtol=0, atol=3e-16)
    assert_allclose(modes.cosine[3], 1.25, rtol=0, atol=8e-16)
    assert_allclose(modes.sine[2], -0.75, rtol=0, atol=8e-16)
    residual = np.delete(modes.cosine.copy(), [0, 3])
    assert_allclose(residual, 0, rtol=0, atol=8e-16)
    residual_sin = np.delete(modes.sine.copy(), [2])
    assert_allclose(residual_sin, 0, rtol=0, atol=8e-16)
    assert_allclose(nonmonopole_mode_power(modes).sum(), np.var(values), rtol=5e-15)


def test_fourier_roundtrip_for_batched_inputs():
    n = 96
    phi = 2*np.pi*np.arange(n)/n
    values = np.stack([
        np.cos(phi) + 0.2*np.sin(4*phi),
        -0.4 + 0.7*np.cos(2*phi) - 0.1*np.sin(5*phi),
    ])
    modes = angular_fourier_coefficients(values, 8)
    assert reconstruct_from_fourier(modes, phi).shape == values.shape
    assert_allclose(reconstruct_from_fourier(modes, phi), values, rtol=0, atol=3e-15)


def test_nyquist_mode_is_rejected():
    with pytest.raises(ValueError):
        angular_fourier_coefficients(np.ones(16), 8)
    with pytest.raises(ValueError):
        angular_fourier_coefficients(np.ones(16), -1)


@pytest.mark.parametrize("q", [1.0, 0.67, 0.3])
def test_area_preserving_radius_factor(q):
    phi = np.linspace(0, 2*np.pi, 1001, endpoint=False)
    factor = area_preserving_elliptical_radius_factor(q, phi)
    assert np.all(factor > 0)
    assert_allclose(factor[0], np.sqrt(q), rtol=3e-15)
    assert_allclose(factor[250], 1/np.sqrt(q), rtol=2e-4)
    if q == 1.0:
        assert_allclose(factor, 1.0, rtol=0, atol=3e-16)


def test_elliptical_convergence_spherical_limit():
    x = np.geomspace(0.01, 20, 9)
    phi = np.linspace(0, 2*np.pi, 32, endpoint=False)
    kappa = elliptical_nfw_convergence_on_grid(x, phi, 1.0, kappa_s=0.25)
    expected = np.broadcast_to(nfw_lensing(x, 0.25).kappa[:, None], kappa.shape)
    assert_allclose(kappa, expected, rtol=2e-15)


def test_solve_lensing_m0_matches_its_integral_definition():
    r = np.geomspace(1e-4, 10.0, 401)
    k0 = nfw_lensing(r, 1.0).kappa
    modes = solve_lensing_from_convergence_modes(r, k0[:, None], np.zeros((r.size, 1)))
    interior = cumulative_simpson(r*k0, x=r, initial=0.0)
    assert_allclose(modes.gamma_plus_cos[:, 0], 2*interior/r**2 - k0, rtol=0, atol=0)
    for arr in (modes.kappa_sin, modes.potential_cos, modes.potential_sin, modes.gamma_plus_sin, modes.gamma_cross_cos, modes.gamma_cross_sin):
        assert_allclose(arr, 0, rtol=0, atol=0)


def test_build_q1_modes_have_spherical_symmetry():
    settings = CONTROLLED_CIRCULAR_SETTINGS.with_small_grids(n_radial_solver=384, n_phi_solver=64, m_max=6)
    modes = build_elliptical_lensing_modes(1.0, settings)
    assert_allclose(modes.kappa_cos[:, 1:], 0, rtol=0, atol=3e-14)
    assert_allclose(modes.kappa_sin, 0, rtol=0, atol=3e-14)
    assert_allclose(modes.gamma_plus_sin, 0, rtol=0, atol=3e-13)
    assert_allclose(modes.gamma_cross_cos, 0, rtol=0, atol=3e-13)
    assert_allclose(modes.gamma_cross_sin, 0, rtol=0, atol=3e-13)


@pytest.mark.parametrize("case", REF["cases"])
def test_compact_historical_mode_references(case):
    settings = compact_settings(case)
    solved = build_elliptical_lensing_modes(case["q"], settings)
    interp = interpolate_lensing_modes(solved, case["target_radius"], radial_interpolation=case["radial_interpolation"])
    for name in ("kappa_cos", "kappa_sin", "gamma_plus_cos", "gamma_plus_sin", "gamma_cross_cos", "gamma_cross_sin"):
        assert_allclose(getattr(interp, name), np.array(case[name]), rtol=4e-10, atol=2e-12)
    phi = np.array(case["fields"]["phi"])
    kappa, gp, gx = reconstruct_lensing_fields(interp, phi)
    assert_allclose(kappa, np.array(case["fields"]["kappa"]), rtol=4e-10, atol=2e-12)
    assert_allclose(gp, np.array(case["fields"]["gamma_plus"]), rtol=4e-10, atol=2e-12)
    assert_allclose(gx, np.array(case["fields"]["gamma_cross"]), rtol=4e-10, atol=2e-12)


def test_interpolation_identity_on_solved_grid():
    settings = CONTROLLED_CIRCULAR_SETTINGS.with_small_grids(n_radial_solver=96, n_phi_solver=64, m_max=6)
    modes = build_elliptical_lensing_modes(1.0, settings)
    subset = modes.radius[[5, 20, 70]]
    out = interpolate_lensing_modes(modes, subset, radial_interpolation="logx")
    assert_allclose(out.kappa_cos, modes.kappa_cos[[5, 20, 70]], rtol=0, atol=2e-15)
    assert_allclose(out.gamma_plus_cos, modes.gamma_plus_cos[[5, 20, 70]], rtol=0, atol=2e-15)
    with pytest.raises(ValueError):
        interpolate_lensing_modes(modes, [modes.radius[0]/2], radial_interpolation="logx")


def test_stored_grid_builder_small_fixed_q():
    settings = SolverSettings(
        name="small_fixed",
        q_fixed=0.67,
        x_min=0.03,
        x_max=3.0,
        n_x=5,
        x_solver_min=1e-4,
        x_solver_max=30.0,
        n_radial_solver=128,
        n_phi_solver=64,
        m_max=6,
        radial_interpolation="x",
    )
    grid = build_stored_elliptical_mode_grid(settings)
    assert grid.q_grid.shape == (1,)
    assert grid.kappa_cos.shape == (1, 5, 7)
    assert grid.radial_interpolation == "x"
    assert_allclose(grid.radius, settings.stored_x_grid(), rtol=0, atol=0)
    assert_allclose(grid.kappa_sin, 0, rtol=0, atol=2e-13)
    assert np.any(np.abs(grid.kappa_cos[0, :, 2]) > 1e-3)


def test_invalid_multipole_inputs():
    with pytest.raises(ValueError):
        area_preserving_elliptical_radius_factor(0.0, [0.0])
    with pytest.raises(ValueError):
        elliptical_nfw_convergence_on_grid([1.0, 0.5], [0.0, 1.0], 0.8)
    with pytest.raises(ValueError):
        solve_lensing_from_convergence_modes([1.0, 0.5], [[1.0], [1.0]], [[0.0], [0.0]])
    with pytest.raises(ValueError):
        build_elliptical_lensing_modes(0.7, CONTROLLED_CIRCULAR_SETTINGS)
