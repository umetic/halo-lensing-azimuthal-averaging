"""Publication-specific physical normalization for halo lensing.

Distances: Mpc/h. Halo density: (Msun/h)/(Mpc/h)^3. Surface density:
(Msun/h)/(Mpc/h)^2. Halo radii are physical, not comoving.

The late-time density background and radiation-inclusive distance background
are deliberately distinct, following PUBLIC_NUMERICAL_SPECIFICATION v0.1 §3.
This module does not replace the separate Colossus population backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad

C_LIGHT_KMS = 299792.458
G_MPC_KMS2_MSUN = 4.30091727003628e-9
RHO_CRIT_0_HUNITS = 2.77536627e11


def _scalar(value: float, name: str, *, positive: bool = False) -> float:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 0 or not np.isfinite(arr):
        raise ValueError(f"{name} must be a finite scalar")
    result = float(arr)
    if result < 0.0 or (positive and result == 0.0):
        condition = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {condition}")
    return result


def _redshift_array(z: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(z, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("redshift must be finite and non-negative")
    return values


@dataclass(frozen=True)
class PublicationCosmology:
    """The explicit normalization used by the publication.

    Scalar distances are cached. Density and halo-radius methods broadcast
    arrays. The default parameter set is the supported publication setting;
    other parameter choices do not constitute a validated cosmology package.
    """

    h: float = 0.6766
    omega_m: float = 0.3111
    omega_b: float = 0.0490
    sigma8: float = 0.8102
    n_s: float = 0.9665
    tcmb: float = 2.7255
    n_eff_rel: float = 3.046

    def __post_init__(self) -> None:
        for name in ("h", "omega_m", "sigma8", "n_s", "tcmb"):
            object.__setattr__(self, name, _scalar(getattr(self, name), name, positive=True))
        for name in ("omega_b", "n_eff_rel"):
            object.__setattr__(self, name, _scalar(getattr(self, name), name))
        if not self.omega_b <= self.omega_m < 1.0:
            raise ValueError("require 0 <= omega_b <= omega_m < 1")
        if not 0.0 < self.omega_r < 1.0 - self.omega_m:
            raise ValueError("distance background requires positive radiation and Lambda densities")

    @property
    def omega_gamma(self) -> float:
        return 2.472e-5 / self.h**2 * (self.tcmb / 2.7255)**4

    @property
    def omega_r(self) -> float:
        return self.omega_gamma * (1.0 + 0.22710731766 * self.n_eff_rel)

    def e2_late(self, z: ArrayLike) -> NDArray[np.float64]:
        """H(z)^2/H0^2 used ONLY for the halo critical-density convention."""
        redshift = _redshift_array(z)
        return np.asarray(self.omega_m * (1.0 + redshift)**3 + (1.0 - self.omega_m))

    def e2_distance(self, z: ArrayLike) -> NDArray[np.float64]:
        """Radiation-inclusive H(z)^2/H0^2 used for distances."""
        redshift = _redshift_array(z)
        return np.asarray(
            self.omega_r * (1.0 + redshift)**4
            + self.omega_m * (1.0 + redshift)**3
            + (1.0 - self.omega_m - self.omega_r)
        )

    def rho_crit_hunits(self, z: ArrayLike) -> NDArray[np.float64]:
        """Critical density in (Msun/h)/(Mpc/h)^3, using the late-time background."""
        return np.asarray(RHO_CRIT_0_HUNITS * self.e2_late(z))

    def r200c_mpc_h(self, mass_msun_h: ArrayLike, z: ArrayLike) -> NDArray[np.float64]:
        """Physical spherical-equivalent r200c in Mpc/h; inputs broadcast."""
        mass, redshift = np.broadcast_arrays(
            np.asarray(mass_msun_h, dtype=np.float64), _redshift_array(z)
        )
        if np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
            raise ValueError("mass_msun_h must be finite and positive")
        return np.asarray(
            (3.0 * mass / (4.0 * np.pi * 200.0 * self.rho_crit_hunits(redshift)))**(1.0 / 3.0)
        )

    @lru_cache(maxsize=128)
    def _distance_from_scale_factor(self, a0: float) -> float:
        orad, om = self.omega_r, self.omega_m
        ol = 1.0 - om - orad
        value, _ = quad(
            lambda a: 1.0 / np.sqrt(orad + om * a + ol * a**4),
            a0, 1.0, epsabs=2e-11, epsrel=2e-11, limit=300,
        )
        return float(C_LIGHT_KMS / 100.0 * value)

    def comoving_distance_mpc_h(self, z: float) -> float:
        """Comoving distance for a finite scalar z >= 0; not an infinite-z proxy."""
        redshift = _scalar(z, "z")
        return self._distance_from_scale_factor(1.0 / (1.0 + redshift))

    def comoving_distance_infinity_mpc_h(self) -> float:
        """True asymptotic distance: integrate from scale factor zero."""
        return self._distance_from_scale_factor(0.0)

    def angular_diameter_distance_mpc_h(self, z: float) -> float:
        redshift = _scalar(z, "z")
        return self.comoving_distance_mpc_h(redshift) / (1.0 + redshift)

    def beta(self, z_l: float, z_s: float) -> float:
        """D_ls/D_s in the flat distance background; require z_s > z_l >= 0."""
        lens, source = _scalar(z_l, "z_l"), _scalar(z_s, "z_s")
        if source <= lens:
            raise ValueError("require z_s > z_l >= 0")
        return 1.0 - self.comoving_distance_mpc_h(lens) / self.comoving_distance_mpc_h(source)

    def beta_infinity(self, z_l: float) -> float:
        lens = _scalar(z_l, "z_l")
        return 1.0 - self.comoving_distance_mpc_h(lens) / self.comoving_distance_infinity_mpc_h()

    def source_weight(self, z_l: float, z_s: float) -> float:
        """w=beta/beta_infinity, for scaling fields before nonlinear mapping."""
        return self.beta(z_l, z_s) / self.beta_infinity(z_l)

    def sigma_crit_infinity_hunits(self, z_l: float) -> float:
        """Critical surface density in (Msun/h)/(Mpc/h)^2; require z_l > 0."""
        lens = _scalar(z_l, "z_l", positive=True)
        distance = self.angular_diameter_distance_mpc_h(lens)
        beta = self.beta_infinity(lens)
        if distance <= 0.0 or beta <= 0.0:
            raise ValueError("degenerate lens-distance or far-background efficiency")
        return C_LIGHT_KMS**2 / (4.0 * np.pi * G_MPC_KMS2_MSUN) / (distance * beta)

    def sigma_crit_hunits(self, z_l: float, z_s: float) -> float:
        """Finite-source critical surface density, in the same h-scaled units."""
        return self.sigma_crit_infinity_hunits(z_l) / self.source_weight(z_l, z_s)


PUBLICATION_COSMOLOGY = PublicationCosmology()
