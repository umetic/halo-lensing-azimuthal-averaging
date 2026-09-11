"""Named Fourier--Green solver settings and validation."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from azlens.solver_settings import (
    SolverSettings,
    POPULATION_MODE_SETTINGS,
    CONTROLLED_ELLIPTICAL_SETTINGS,
    CONTROLLED_CIRCULAR_SETTINGS,
)


def test_population_contract_values():
    s = POPULATION_MODE_SETTINGS
    assert s.name == "population"
    assert (s.q_min, s.q_max, s.n_q) == (0.18, 1.0, 83)
    assert (s.x_min, s.x_max, s.n_x) == (1e-6, 31.0, 289)
    assert (s.x_solver_min, s.x_solver_max, s.n_radial_solver) == (1e-8, 1e3, 1800)
    assert (s.n_phi_solver, s.m_max, s.radial_interpolation) == (256, 20, "x")
    q = s.q_grid()
    assert_allclose([q[0], q[-1], q.size], [0.18, 1.0, 83], rtol=0, atol=0)
    assert np.all(np.diff(q) > 0)


def test_controlled_contract_values():
    e = CONTROLLED_ELLIPTICAL_SETTINGS
    c = CONTROLLED_CIRCULAR_SETTINGS
    assert e.q_fixed == 0.67
    assert (e.n_radial_solver, e.n_phi_solver, e.m_max, e.radial_interpolation) == (5000, 512, 30, "logx")
    assert c.q_fixed == 1.0
    assert (c.n_radial_solver, c.n_phi_solver, c.m_max, c.radial_interpolation) == (4000, 128, 8, "logx")
    assert_allclose(e.q_grid(), [0.67], rtol=0, atol=0)
    assert_allclose(c.q_grid(), [1.0], rtol=0, atol=0)


@pytest.mark.parametrize("settings", [POPULATION_MODE_SETTINGS, CONTROLLED_ELLIPTICAL_SETTINGS, CONTROLLED_CIRCULAR_SETTINGS])
def test_grids_and_angles(settings):
    x = settings.stored_x_grid()
    xs = settings.solver_x_grid()
    phi = settings.solver_phi_grid()
    assert_allclose([x[0], x[-1]], [settings.x_min, settings.x_max], rtol=5e-15)
    assert_allclose([xs[0], xs[-1]], [settings.x_solver_min, settings.x_solver_max], rtol=5e-15)
    assert np.all(np.diff(x) > 0) and np.all(np.diff(xs) > 0)
    assert phi[0] == 0.0
    assert_allclose(phi[1] - phi[0], 2*np.pi/settings.n_phi_solver, rtol=3e-16)
    assert phi[-1] < 2*np.pi
    assert settings.m_max < settings.n_phi_solver // 2


def test_small_grid_copy_preserves_conventions():
    s = POPULATION_MODE_SETTINGS.with_small_grids(n_x=9, n_radial_solver=17, n_phi_solver=64, m_max=7)
    assert s.name == POPULATION_MODE_SETTINGS.name
    assert s.radial_interpolation == "x"
    assert s.q_min == 0.18 and s.q_max == 1.0 and s.n_q == 83
    assert s.stored_x_grid().size == 9
    assert s.solver_x_grid().size == 17


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(name="", q_fixed=1.0),
        dict(name="bad", q_fixed=0.0),
        dict(name="bad", q_min=0.2, q_max=1.1, n_q=3),
        dict(name="bad", q_min=0.2, q_max=1.0, n_q=1),
        dict(name="bad", q_fixed=1.0, q_min=0.2),
        dict(name="bad", q_fixed=1.0, x_min=2.0, x_max=1.0),
        dict(name="bad", q_fixed=1.0, m_max=32, n_phi_solver=64),
        dict(name="bad", q_fixed=1.0, radial_interpolation="radius"),
    ],
)
def test_invalid_settings(kwargs):
    base = dict(x_min=1e-6, x_max=31.0, n_x=10, x_solver_min=1e-6, x_solver_max=1e3, n_radial_solver=32, n_phi_solver=64, m_max=8, radial_interpolation="x")
    base.update(kwargs)
    with pytest.raises(ValueError):
        SolverSettings(**base)
