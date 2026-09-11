"""Fourier--Green multipole reconstruction for area-preserving elliptical NFW fields.

This module is deterministic and non-stochastic. It implements the common
Fourier--Green equations and the explicitly named numerical profiles in
``solver_settings.py``. Miscentering, nonlinear observables, population
sampling, and plotting are deliberately outside this increment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import cumulative_simpson

from .nfw import nfw_lensing
from .solver_settings import SolverSettings

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FourierModes:
    """Real Fourier coefficients for ``f=a0+sum_m(a_m cos mphi+b_m sin mphi)``."""

    cosine: FloatArray
    sine: FloatArray

    @property
    def m_max(self) -> int:
        return int(self.cosine.shape[-1] - 1)


@dataclass(frozen=True)
class LensingModes:
    """Radial arrays of convergence, potential, and shear coefficients."""

    radius: FloatArray
    kappa_cos: FloatArray
    kappa_sin: FloatArray
    potential_cos: FloatArray
    potential_sin: FloatArray
    gamma_plus_cos: FloatArray
    gamma_plus_sin: FloatArray
    gamma_cross_cos: FloatArray
    gamma_cross_sin: FloatArray

    def __post_init__(self) -> None:
        r = np.asarray(self.radius, dtype=np.float64)
        if r.ndim != 1 or r.size < 2 or np.any(~np.isfinite(r)) or np.any(r <= 0.0) or np.any(np.diff(r) <= 0.0):
            raise ValueError("radius must be a positive, strictly increasing one-dimensional grid")
        expected = (r.size, self.kappa_cos.shape[1])
        for name in (
            "kappa_cos", "kappa_sin", "potential_cos", "potential_sin",
            "gamma_plus_cos", "gamma_plus_sin", "gamma_cross_cos", "gamma_cross_sin",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != expected or value.ndim != 2 or np.any(~np.isfinite(value)):
                raise ValueError(f"{name} must have finite shape {expected}")

    @property
    def m_max(self) -> int:
        return int(self.kappa_cos.shape[1] - 1)


@dataclass(frozen=True)
class EllipticalModeGrid:
    """Mode coefficients for one or more projected axis ratios.

    Coefficient arrays have shape ``(n_q, n_x, m_max+1)`` and unit ``kappa_s``.
    """

    q_grid: FloatArray
    radius: FloatArray
    kappa_cos: FloatArray
    kappa_sin: FloatArray
    gamma_plus_cos: FloatArray
    gamma_plus_sin: FloatArray
    gamma_cross_cos: FloatArray
    gamma_cross_sin: FloatArray
    settings_name: str
    radial_interpolation: str

    def __post_init__(self) -> None:
        q = np.asarray(self.q_grid, dtype=np.float64)
        r = np.asarray(self.radius, dtype=np.float64)
        if q.ndim != 1 or q.size < 1 or np.any(~np.isfinite(q)) or np.any(q <= 0.0) or np.any(q > 1.0):
            raise ValueError("q_grid must be finite and in (0,1]")
        if q.size > 1 and np.any(np.diff(q) <= 0.0):
            raise ValueError("q_grid must be strictly increasing")
        if r.ndim != 1 or r.size < 2 or np.any(~np.isfinite(r)) or np.any(r <= 0.0) or np.any(np.diff(r) <= 0.0):
            raise ValueError("radius must be finite, positive and increasing")
        expected = (q.size, r.size, self.kappa_cos.shape[2])
        for name in (
            "kappa_cos", "kappa_sin", "gamma_plus_cos", "gamma_plus_sin",
            "gamma_cross_cos", "gamma_cross_sin",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.ndim != 3 or value.shape != expected or np.any(~np.isfinite(value)):
                raise ValueError(f"{name} must have finite shape {expected}")


def _as_1d_positive(values: ArrayLike, name: str) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0 or np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"{name} must contain positive finite values")
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return arr


def _as_phi(values: ArrayLike) -> FloatArray:
    phi = np.asarray(values, dtype=np.float64).reshape(-1)
    if phi.size < 2 or np.any(~np.isfinite(phi)):
        raise ValueError("phi must contain at least two finite angles")
    return phi


def angular_fourier_coefficients(values: ArrayLike, m_max: int) -> FourierModes:
    """Return real Fourier coefficients on an endpoint-excluding uniform grid.

    The Nyquist mode is deliberately disallowed because its real-FFT coefficient
    has a different normalization from ordinary positive modes. All production
    and controlled settings use ``m_max`` safely below Nyquist.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim < 1:
        raise ValueError("values must have an angular axis")
    n_phi = int(arr.shape[-1])
    if m_max < 0 or m_max >= n_phi // 2:
        raise ValueError("m_max must be non-negative and below the Nyquist mode")
    coeff = np.fft.rfft(arr, axis=-1)/float(n_phi)
    cosine = np.zeros(arr.shape[:-1] + (int(m_max)+1,), dtype=np.float64)
    sine = np.zeros_like(cosine)
    cosine[..., 0] = coeff[..., 0].real
    if m_max >= 1:
        positive = coeff[..., 1:int(m_max)+1]
        cosine[..., 1:] = 2.0*positive.real
        sine[..., 1:] = -2.0*positive.imag
    return FourierModes(cosine=cosine, sine=sine)


def reconstruct_from_fourier(modes: FourierModes, phi: ArrayLike) -> FloatArray:
    """Reconstruct scalar fields from real Fourier coefficients."""
    angles = _as_phi(phi)
    m = np.arange(modes.cosine.shape[-1], dtype=np.float64)
    cos = np.cos(np.outer(m, angles))
    sin = np.sin(np.outer(m, angles))
    return np.asarray(modes.cosine @ cos + modes.sine @ sin)


def nonmonopole_mode_power(modes: FourierModes) -> FloatArray:
    """Return ``P_m=(a_m^2+b_m^2)/2`` for m>=1, preserving leading dimensions."""
    if modes.cosine.shape != modes.sine.shape:
        raise ValueError("cosine and sine arrays must have the same shape")
    return 0.5*(modes.cosine[..., 1:]**2 + modes.sine[..., 1:]**2)


def area_preserving_elliptical_radius_factor(q_perp: float, phi: ArrayLike) -> FloatArray:
    """Return zeta/R for an area-preserving ellipse with major axis at phi=0."""
    q = float(q_perp)
    if not np.isfinite(q) or not (0.0 < q <= 1.0):
        raise ValueError("q_perp must satisfy 0 < q_perp <= 1")
    angle = np.asarray(phi, dtype=np.float64)
    return np.sqrt(q*np.cos(angle)**2 + np.sin(angle)**2/q)


def elliptical_nfw_convergence_on_grid(
    radius: ArrayLike,
    phi: ArrayLike,
    q_perp: float,
    *,
    kappa_s: float = 1.0,
) -> FloatArray:
    """Convergence of an area-preserving elliptical NFW reference field.

    ``radius`` is the circular-reference x-coordinate. The returned array has
    shape ``(n_radius, n_phi)``. The projected major axis is ``phi=0``.
    """
    r = _as_1d_positive(radius, "radius")
    angle = _as_phi(phi)
    if not np.isfinite(kappa_s) or kappa_s < 0.0:
        raise ValueError("kappa_s must be finite and non-negative")
    x_ell = r[:, None]*area_preserving_elliptical_radius_factor(q_perp, angle)[None, :]
    return np.asarray(nfw_lensing(x_ell, kappa_s).kappa)


def _reverse_cumulative_integral(y: FloatArray, x: FloatArray) -> FloatArray:
    return cumulative_simpson(y[::-1], x=-x[::-1], initial=0.0)[::-1]


def _potential_mode_from_kappa(radius: FloatArray, kappa_mode: FloatArray, m: int) -> tuple[FloatArray, FloatArray]:
    if m < 1:
        raise ValueError("m must be at least one")
    r = radius
    left = cumulative_simpson(r**(m+1)*kappa_mode, x=r, initial=0.0)
    right = _reverse_cumulative_integral(r**(1-m)*kappa_mode, r)
    potential = -(r**(-m)*left + r**m*right)/float(m)
    derivative = r**(-m-1)*left - r**(m-1)*right
    return potential, derivative


def solve_lensing_from_convergence_modes(radius: ArrayLike, kappa_cos: ArrayLike, kappa_sin: ArrayLike) -> LensingModes:
    """Derive potential and shear modes from convergence modes.

    Input arrays must have shape ``(n_radius, m_max+1)``. The m=0 sine
    coefficient is ignored but must be present for shape consistency.
    """
    r = _as_1d_positive(radius, "radius")
    kc = np.asarray(kappa_cos, dtype=np.float64)
    ks = np.asarray(kappa_sin, dtype=np.float64)
    if kc.shape != ks.shape or kc.ndim != 2 or kc.shape[0] != r.size or kc.shape[1] < 1:
        raise ValueError("mode arrays must have shape (n_radius, m_max+1)")
    if np.any(~np.isfinite(kc)) or np.any(~np.isfinite(ks)):
        raise ValueError("mode arrays must be finite")

    pc = np.zeros_like(kc)
    ps = np.zeros_like(ks)
    gp_c = np.zeros_like(kc)
    gp_s = np.zeros_like(ks)
    gx_c = np.zeros_like(kc)
    gx_s = np.zeros_like(ks)

    interior = cumulative_simpson(r*kc[:, 0], x=r, initial=0.0)
    gp_c[:, 0] = 2.0*interior/r**2 - kc[:, 0]

    for m in range(1, kc.shape[1]):
        ac, ac_prime = _potential_mode_from_kappa(r, kc[:, m], m)
        bs, bs_prime = _potential_mode_from_kappa(r, ks[:, m], m)
        pc[:, m] = ac
        ps[:, m] = bs
        gp_c[:, m] = -kc[:, m] + ac_prime/r - (m*m)*ac/r**2
        gp_s[:, m] = -ks[:, m] + bs_prime/r - (m*m)*bs/r**2
        op_c = m*(ac_prime/r - ac/r**2)
        op_s = m*(bs_prime/r - bs/r**2)
        gx_c[:, m] = -op_s
        gx_s[:, m] = op_c

    return LensingModes(r, kc, ks, pc, ps, gp_c, gp_s, gx_c, gx_s)


def reconstruct_lensing_fields(modes: LensingModes, phi: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Reconstruct kappa, gamma_plus, and gamma_cross on supplied angles."""
    angles = _as_phi(phi)
    m = np.arange(modes.m_max+1, dtype=np.float64)
    cos = np.cos(np.outer(m, angles))
    sin = np.sin(np.outer(m, angles))
    return (
        np.asarray(modes.kappa_cos @ cos + modes.kappa_sin @ sin),
        np.asarray(modes.gamma_plus_cos @ cos + modes.gamma_plus_sin @ sin),
        np.asarray(modes.gamma_cross_cos @ cos + modes.gamma_cross_sin @ sin),
    )


def interpolate_lensing_modes(
    modes: LensingModes,
    target_radius: ArrayLike,
    *,
    radial_interpolation: str,
) -> LensingModes:
    """Linearly interpolate mode coefficients in x or log(x)."""
    targets = _as_1d_positive(target_radius, "target_radius")
    if targets[0] < modes.radius[0] or targets[-1] > modes.radius[-1]:
        raise ValueError("target radii lie outside the solved radius grid")
    if radial_interpolation == "x":
        grid = modes.radius
        points = targets
    elif radial_interpolation == "logx":
        grid = np.log(modes.radius)
        points = np.log(targets)
    else:
        raise ValueError("radial_interpolation must be 'x' or 'logx'")

    def interp(array: FloatArray) -> FloatArray:
        out = np.empty((targets.size, array.shape[1]), dtype=np.float64)
        for j in range(array.shape[1]):
            out[:, j] = np.interp(points, grid, array[:, j])
        return out

    return LensingModes(
        radius=targets,
        kappa_cos=interp(modes.kappa_cos),
        kappa_sin=interp(modes.kappa_sin),
        potential_cos=interp(modes.potential_cos),
        potential_sin=interp(modes.potential_sin),
        gamma_plus_cos=interp(modes.gamma_plus_cos),
        gamma_plus_sin=interp(modes.gamma_plus_sin),
        gamma_cross_cos=interp(modes.gamma_cross_cos),
        gamma_cross_sin=interp(modes.gamma_cross_sin),
    )


def build_elliptical_lensing_modes(q_perp: float, settings: SolverSettings, *, kappa_s: float = 1.0) -> LensingModes:
    """Solve unit-amplitude modes for one projected axis ratio using settings."""
    if settings.q_fixed is not None and not np.isclose(q_perp, settings.q_fixed, rtol=0.0, atol=1e-14):
        raise ValueError("q_perp does not match the fixed-q solver setting")
    x = settings.solver_x_grid()
    phi = settings.solver_phi_grid()
    kappa = elliptical_nfw_convergence_on_grid(x, phi, q_perp, kappa_s=kappa_s)
    fourier = angular_fourier_coefficients(kappa, settings.m_max)
    return solve_lensing_from_convergence_modes(x, fourier.cosine, fourier.sine)


def build_stored_elliptical_mode_grid(settings: SolverSettings, *, kappa_s: float = 1.0) -> EllipticalModeGrid:
    """Build interpolated unit-amplitude coefficients on the settings' stored grid."""
    q_nodes = settings.q_grid()
    stored_x = settings.stored_x_grid()
    arrays: dict[str, list[FloatArray]] = {name: [] for name in (
        "kappa_cos", "kappa_sin", "gamma_plus_cos", "gamma_plus_sin", "gamma_cross_cos", "gamma_cross_sin"
    )}
    for q in q_nodes:
        solved = build_elliptical_lensing_modes(float(q), settings, kappa_s=kappa_s)
        interpolated = interpolate_lensing_modes(solved, stored_x, radial_interpolation=settings.radial_interpolation)
        for name in arrays:
            arrays[name].append(getattr(interpolated, name))
    return EllipticalModeGrid(
        q_grid=q_nodes,
        radius=stored_x,
        settings_name=settings.name,
        radial_interpolation=settings.radial_interpolation,
        **{name: np.stack(values, axis=0) for name, values in arrays.items()},
    )
