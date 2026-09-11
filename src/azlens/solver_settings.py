"""Named numerical settings for the Fourier--Green lensing solver.

These settings make explicit the two numerical profiles identified in the
public numerical specification: the population mode library and the controlled
single-halo benchmarks share the same equations but not all discretization and
interpolation choices.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

RadialInterpolation = Literal["x", "logx"]


@dataclass(frozen=True)
class SolverSettings:
    """Immutable discretization settings for Fourier--Green reconstruction.

    Parameters refer to the dimensionless projected NFW radius ``x=R/r_s`` of
    the circular projected reference profile. ``q_grid`` is the grid of
    projected axis ratios for a reusable library; controlled calculations can
    instead use a fixed ``q_fixed`` value.
    """

    name: str
    x_min: float
    x_max: float
    n_x: int
    x_solver_min: float
    x_solver_max: float
    n_radial_solver: int
    n_phi_solver: int
    m_max: int
    radial_interpolation: RadialInterpolation
    q_min: float | None = None
    q_max: float | None = None
    n_q: int | None = None
    q_fixed: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        for field in ("x_min", "x_max", "x_solver_min", "x_solver_max"):
            value = float(getattr(self, field))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive")
        if not (self.x_min < self.x_max and self.x_solver_min < self.x_solver_max):
            raise ValueError("x ranges must be strictly increasing")
        if self.n_x < 2 or self.n_radial_solver < 8 or self.n_phi_solver < 8:
            raise ValueError("grid sizes are too small")
        if self.m_max < 0 or self.m_max >= self.n_phi_solver // 2:
            raise ValueError("m_max must be non-negative and below the Nyquist mode")
        if self.radial_interpolation not in ("x", "logx"):
            raise ValueError("radial_interpolation must be 'x' or 'logx'")
        grid_specified = self.q_fixed is None
        if grid_specified:
            if self.q_min is None or self.q_max is None or self.n_q is None:
                raise ValueError("either q_fixed or q_min/q_max/n_q must be supplied")
            if self.n_q < 2:
                raise ValueError("n_q must be at least two for a q grid")
            if not (0.0 < self.q_min <= self.q_max <= 1.0):
                raise ValueError("q grid must satisfy 0 < q_min <= q_max <= 1")
        else:
            if self.q_min is not None or self.q_max is not None or self.n_q is not None:
                raise ValueError("q grid fields must be omitted when q_fixed is supplied")
            if not (0.0 < float(self.q_fixed) <= 1.0):
                raise ValueError("q_fixed must satisfy 0 < q_fixed <= 1")

    def stored_x_grid(self) -> NDArray[np.float64]:
        """Geometric stored-evaluation x grid."""
        return np.geomspace(self.x_min, self.x_max, int(self.n_x), dtype=np.float64)

    def solver_x_grid(self) -> NDArray[np.float64]:
        """Geometric radial grid on which Green integrals are solved."""
        return np.geomspace(self.x_solver_min, self.x_solver_max, int(self.n_radial_solver), dtype=np.float64)

    def solver_phi_grid(self) -> NDArray[np.float64]:
        """Endpoint-excluding angular grid used for Fourier coefficients."""
        return 2.0*np.pi*np.arange(int(self.n_phi_solver), dtype=np.float64)/float(self.n_phi_solver)

    def q_grid(self) -> NDArray[np.float64]:
        """Projected-axis-ratio grid or a one-node fixed controlled value."""
        if self.q_fixed is not None:
            return np.array([float(self.q_fixed)], dtype=np.float64)
        return np.linspace(float(self.q_min), float(self.q_max), int(self.n_q), dtype=np.float64)

    def with_small_grids(
        self,
        *,
        n_x: int | None = None,
        n_radial_solver: int | None = None,
        n_phi_solver: int | None = None,
        m_max: int | None = None,
    ) -> "SolverSettings":
        """Return a test copy with reduced resolution and the same conventions."""
        return replace(
            self,
            n_x=self.n_x if n_x is None else n_x,
            n_radial_solver=self.n_radial_solver if n_radial_solver is None else n_radial_solver,
            n_phi_solver=self.n_phi_solver if n_phi_solver is None else n_phi_solver,
            m_max=self.m_max if m_max is None else m_max,
        )


POPULATION_MODE_SETTINGS = SolverSettings(
    name="population",
    q_min=0.18,
    q_max=1.0,
    n_q=83,
    x_min=1.0e-6,
    x_max=31.0,
    n_x=289,
    x_solver_min=1.0e-8,
    x_solver_max=1.0e3,
    n_radial_solver=1800,
    n_phi_solver=256,
    m_max=20,
    radial_interpolation="x",
)

CONTROLLED_ELLIPTICAL_SETTINGS = SolverSettings(
    name="controlled_elliptical",
    q_fixed=0.67,
    x_min=1.0e-6,
    x_max=31.0,
    n_x=289,
    x_solver_min=1.0e-6,
    x_solver_max=1.0e3,
    n_radial_solver=5000,
    n_phi_solver=512,
    m_max=30,
    radial_interpolation="logx",
)

CONTROLLED_CIRCULAR_SETTINGS = SolverSettings(
    name="controlled_circular",
    q_fixed=1.0,
    x_min=1.0e-6,
    x_max=31.0,
    n_x=289,
    x_solver_min=1.0e-6,
    x_solver_max=1.0e3,
    n_radial_solver=4000,
    n_phi_solver=128,
    m_max=8,
    radial_interpolation="logx",
)
