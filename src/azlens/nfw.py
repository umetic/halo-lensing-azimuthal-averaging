"""Untruncated spherical NFW normalization and stable projected functions.

Public Numerical Specification v0.1 §3. Input x is R/r_s and kappa_s is
rho_s*r_s/Sigma_crit (NOT twice this value). All profile outputs are float64
arrays, including zero-dimensional arrays for scalar inputs.

Analytic series avoid cancellation at small x and near x=1. Unlike the legacy
helpers, every valid argument belongs to exactly one branch. No radius floor
or extrapolation is applied: x=0 is outside this profile API.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .cosmology import PUBLICATION_COSMOLOGY, PublicationCosmology

SMALL_X_MAX = 1.0e-3
NEAR_ONE_HALF_WIDTH = 1.0e-3

# f(1+t), through t^6: the audited mode-construction coefficients.
_F_NEAR_ONE = (1/3, -2/5, 13/35, -20/63, 61/231, -94/429, 1181/6435)
# Integrate G'(x)=x*f(x), with G(1)=1-log(2), through t^7.
_G_NEAR_ONE = (1-np.log(2.0), 1/3, -1/30, -1/105, 17/1260, -37/3465, 15/2002, -229/45045)


def _positive(value: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64)
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"{name} must be finite and strictly positive")
    return arr


def _horner(x: NDArray[np.float64], coefficients: tuple) -> NDArray[np.float64]:
    out = np.zeros_like(x) + coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        out = out*x + coefficient
    return out


def _projected_shapes(x: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return f, G, h, where h=4G/x^2-2f is evaluated stably at small x."""
    arr = _positive(x, "x")
    f, g, h = (np.empty_like(arr) for _ in range(3))
    small = arr < SMALL_X_MAX
    near = (~small) & (np.abs(arr - 1.0) <= NEAR_ONE_HALF_WIDTH)
    below = (~small) & (~near) & (arr < 1.0)
    above = (~small) & (~near) & (~below)

    if np.any(small):
        xx = arr[small]
        y = xx*xx
        ell = np.log(2.0) - np.log(xx)
        # Series in y=x^2 through y^4. G itself starts at x^2.
        f[small] = _horner(y, (
            ell-1, 3*ell/2-5/4, 15*ell/8-47/32,
            35*ell/16-319/192, 315*ell/128-1879/1024,
        ))
        g[small] = y*_horner(y, (
            ell/2-1/4, 3*ell/8-7/32, 5*ell/16-37/192,
            35*ell/128-533/3072, 63*ell/256-1627/10240,
        ))
        # Avoid subtracting two logarithmically large profiles to obtain h->1.
        h[small] = _horner(y, (
            np.ones_like(ell), 13/8-3*ell/2, 13/6-5*ell/2,
            673/256-105*ell/32, 971/320-63*ell/16,
        ))

    if np.any(near):
        t = arr[near] - 1.0
        f[near] = _horner(t, _F_NEAR_ONE)
        g[near] = _horner(t, _G_NEAR_ONE)
        h[near] = 4.0*g[near]/arr[near]**2 - 2.0*f[near]

    if np.any(below):
        xx = arr[below]
        root = np.sqrt((1.0-xx)*(1.0+xx))
        u = np.sqrt((1.0-xx)/(1.0+xx))
        t = 2.0*np.arctanh(u)/root
        f[below] = (t-1.0)/((1.0-xx)*(1.0+xx))
        # Algebraically equal to log(x/2)+T, without small-x cancellation.
        y = xx*xx
        ell = np.log(2.0) - np.log(xx)
        g[below] = (ell*y/(1.0+root) + np.log1p(-y/(2.0*(1.0+root))))/root
        h[below] = 4.0*g[below]/y - 2.0*f[below]

    if np.any(above):
        xx = arr[above]
        root = np.sqrt((xx-1.0)*(xx+1.0))
        u = np.sqrt((xx-1.0)/(xx+1.0))
        t = 2.0*np.arctan(u)/root
        f[above] = (1.0-t)/((xx-1.0)*(xx+1.0))
        g[above] = np.log(xx/2.0) + t
        h[above] = 4.0*g[above]/xx**2 - 2.0*f[above]
    return f, g, h


def nfw_f(x: ArrayLike) -> NDArray[np.float64]:
    """Surface-density shape f, defined by Sigma=2*rho_s*r_s*f(R/r_s)."""
    return _projected_shapes(x)[0]


def nfw_g(x: ArrayLike) -> NDArray[np.float64]:
    """Enclosed projected shape G, with G'(x)=x*f(x) and G(0)=0."""
    return _projected_shapes(x)[1]


def nfw_shear_shape(x: ArrayLike) -> NDArray[np.float64]:
    """h=4G/x^2-2f, so gamma_t=kappa_s*h; h tends to 1 as x tends to zero."""
    return _projected_shapes(x)[2]


@dataclass(frozen=True)
class SphericalLensing:
    """Dimensionless spherical fields at one source plane; positive gamma_t convention."""

    kappa: NDArray[np.float64]
    gamma_t: NDArray[np.float64]
    mean_kappa: NDArray[np.float64]


def nfw_lensing(x: ArrayLike, kappa_s: ArrayLike) -> SphericalLensing:
    """Spherical convergence/shear/interior mean convergence; inputs broadcast.

    The caller supplies the source-plane normalization. This routine neither
    makes a reduced-shear approximation nor imposes a subcriticality selection.
    """
    radii, amplitude = np.broadcast_arrays(_positive(x, "x"), np.asarray(kappa_s, dtype=np.float64))
    if np.any(~np.isfinite(amplitude)) or np.any(amplitude < 0.0):
        raise ValueError("kappa_s must be finite and non-negative")
    f, _, h = _projected_shapes(radii)
    kappa = np.asarray(2.0*amplitude*f)
    gamma_t = np.asarray(amplitude*h)
    return SphericalLensing(kappa, gamma_t, np.asarray(kappa+gamma_t))


def nfw_mass_shape(c: ArrayLike) -> NDArray[np.float64]:
    """log(1+c)-c/(1+c), with a cancellation-free small-c series."""
    arr = _positive(c, "c")
    out = np.empty_like(arr)
    small = arr < 1.0e-3
    out[small] = arr[small]**2 * _horner(arr[small], tuple((-1)**k*(k-1)/k for k in range(2, 10)))
    out[~small] = np.log1p(arr[~small]) - arr[~small]/(1.0+arr[~small])
    return out


@dataclass(frozen=True)
class NFWNormalization:
    """Broadcast halo normalizations for one lens redshift.

    Masses and radii are in Msun/h and physical Mpc/h. rho_s is in
    (Msun/h)/(Mpc/h)^3. kappa_s_infinity uses physical beta_infinity.
    The concentration is supplied; no concentration relation is inferred.
    """

    r200c_mpc_h: NDArray[np.float64]
    r_s_mpc_h: NDArray[np.float64]
    rho_s_hunits: NDArray[np.float64]
    kappa_s_infinity: NDArray[np.float64]


def nfw_normalization(
    mass_msun_h: ArrayLike,
    concentration: ArrayLike,
    z_l: float,
    cosmology: PublicationCosmology = PUBLICATION_COSMOLOGY,
) -> NFWNormalization:
    """Normalize the spherical-equivalent NFW reference profile, with z_l>0."""
    mass, c = np.broadcast_arrays(_positive(mass_msun_h, "mass_msun_h"), _positive(concentration, "concentration"))
    sigma_crit_inf = cosmology.sigma_crit_infinity_hunits(z_l)
    r200 = cosmology.r200c_mpc_h(mass, z_l)
    rs = np.asarray(r200/c)
    rho_s = np.asarray(mass/(4.0*np.pi*rs**3*nfw_mass_shape(c)))
    ks_inf = np.asarray(rho_s*rs/sigma_crit_inf)
    return NFWNormalization(r200, rs, rho_s, ks_inf)


def nfw_density(radius_mpc_h: ArrayLike, normalization: NFWNormalization) -> NDArray[np.float64]:
    """Untruncated three-dimensional spherical density, in h-scaled units."""
    x = _positive(radius_mpc_h, "radius_mpc_h")/normalization.r_s_mpc_h
    return np.asarray(normalization.rho_s_hunits/(x*(1.0+x)**2))


def nfw_surface_density(radius_mpc_h: ArrayLike, normalization: NFWNormalization) -> NDArray[np.float64]:
    """Untruncated spherical surface density, in (Msun/h)/(Mpc/h)^2."""
    x = _positive(radius_mpc_h, "radius_mpc_h")/normalization.r_s_mpc_h
    return np.asarray(2.0*normalization.rho_s_hunits*normalization.r_s_mpc_h*nfw_f(x))
