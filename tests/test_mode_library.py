from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

from azlens.mode_library import ModeLibrary
from azlens.multipoles import build_elliptical_lensing_modes, interpolate_lensing_modes, reconstruct_lensing_fields
from azlens.ring_fields import evaluate_ring_fields, midpoint_angles
from azlens.solver_settings import SolverSettings
from azlens.nfw import nfw_lensing


def small_library_settings() -> SolverSettings:
    return SolverSettings(
        name="test_population_small",
        q_min=0.5,
        q_max=1.0,
        n_q=4,
        x_min=1.0e-3,
        x_max=3.0,
        n_x=18,
        x_solver_min=1.0e-4,
        x_solver_max=20.0,
        n_radial_solver=80,
        n_phi_solver=64,
        m_max=6,
        radial_interpolation="x",
    )


def test_mode_library_build_shapes_and_settings():
    library = ModeLibrary.build(small_library_settings())
    assert library.q_grid.size == 4
    assert library.x_grid.size == 18
    assert library.m_max == 6
    assert library.radial_interpolation == "x"
    assert library.grid.kappa_cos.shape == (4, 18, 7)


def test_mode_library_reproduces_grid_node_modes_at_q_node():
    settings = small_library_settings()
    library = ModeLibrary.build(settings)
    q = library.q_grid[1]
    solved = build_elliptical_lensing_modes(float(q), settings, kappa_s=1.0)
    interp = interpolate_lensing_modes(solved, library.x_grid, radial_interpolation=settings.radial_interpolation)
    assert_allclose(library.grid.kappa_cos[1], interp.kappa_cos, rtol=0, atol=0)
    assert_allclose(library.grid.gamma_cross_sin[1], interp.gamma_cross_sin, rtol=0, atol=0)


def test_mode_library_q_one_matches_circular_nfw_to_solver_accuracy():
    library = ModeLibrary.build(small_library_settings())
    evaluator = library.evaluator_for_q([1.0])
    phi = midpoint_angles(96)[None, :]
    x = np.full_like(phi, 0.8)
    fields = evaluator.evaluate(x, phi, kappa_s=np.ones_like(x))
    lens = nfw_lensing(0.8, 1.0)
    assert_allclose(np.mean(fields.kappa), lens.kappa, rtol=0, atol=5e-2)
    assert_allclose(np.mean(fields.gamma_plus), lens.gamma_t, rtol=0, atol=5e-2)
    assert abs(np.mean(fields.gamma_cross)) < 1e-13


def test_mode_library_evaluator_rejects_out_of_domain_x():
    library = ModeLibrary.build(small_library_settings())
    evaluator = library.evaluator_for_q([0.75])
    phi = midpoint_angles(16)[None, :]
    with pytest.raises(ValueError, match="outside"):
        evaluator.evaluate(np.full_like(phi, library.x_grid[-1]*1.1), phi, kappa_s=1.0)


def test_mode_library_save_load_roundtrip(tmp_path: Path):
    library = ModeLibrary.build(small_library_settings())
    path = tmp_path/"library.npz"
    library.save_npz(path, extra_metadata={"purpose": "test"})
    loaded = ModeLibrary.load_npz(path)
    assert_allclose(loaded.q_grid, library.q_grid, rtol=0, atol=0)
    assert_allclose(loaded.x_grid, library.x_grid, rtol=0, atol=0)
    assert_allclose(loaded.grid.gamma_plus_cos, library.grid.gamma_plus_cos, rtol=0, atol=0)


def test_mode_library_can_feed_adopted_ring_evaluator():
    library = ModeLibrary.build(small_library_settings())
    evaluator = library.evaluator_for_q([0.67, 1.0])
    fields = evaluate_ring_fields(
        evaluator,
        concentration=[3.8, 4.0],
        projected_scale_factor=[1.0, 1.0],
        kappa_s_projected=[0.1, 0.1],
        radius_over_r200c=0.2,
        d_over_r200c=[0.0, 0.02],
        phi_offset=[0.0, 0.5],
        n_phi=32,
    )
    assert fields.kappa.shape == (2, 32)
    assert np.all(np.isfinite(fields.gamma_plus))
