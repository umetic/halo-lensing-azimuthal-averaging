"""Deterministic adopted-center ring-field evaluation.

This layer consumes already constructed true-center field evaluators and returns
local lensing fields on complete midpoint-sampled rings. It contains geometry
and spin-2 rotation only: no nonlinear observable maps, Monte Carlo sampling,
alignment reweighting, plotting, or population statistics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .multipoles import LensingModes
from .nfw import nfw_lensing

FloatArray = NDArray[np.float64]


def _positive_scalar(value: float, name: str) -> float:
    scalar = float(np.asarray(value, dtype=np.float64))
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return scalar


def _angle_array(phi: ArrayLike, name: str = "phi") -> FloatArray:
    values = np.asarray(phi, dtype=np.float64).reshape(-1)
    if values.size < 8 or np.any(~np.isfinite(values)):
        raise ValueError(f"{name} must contain at least eight finite angles")
    return values


def _halo_arrays(*arrays: ArrayLike) -> tuple[FloatArray, ...]:
    broadcast = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64).reshape(-1) for a in arrays))
    if len(broadcast) == 0 or broadcast[0].size < 1:
        raise ValueError("at least one halo must be supplied")
    return tuple(np.asarray(a, dtype=np.float64).reshape(-1) for a in broadcast)


def midpoint_angles(n_phi: int) -> FloatArray:
    """Return ``phi_j=(j+1/2)2pi/N_phi`` on a complete ring.

    This grid is distinct from the endpoint-excluding Fourier-solver grid used
    in ``multipoles.py``.
    """
    n = int(n_phi)
    if n < 8:
        raise ValueError("n_phi must be at least eight")
    return (np.arange(n, dtype=np.float64) + 0.5) * (2.0*np.pi/float(n))


@dataclass(frozen=True)
class RingGeometry:
    """Adopted-center and true-center coordinates for one ring.

    Arrays with halo dependence have shape ``(n_halo, n_phi)``. ``omega`` is
    the signed rotation from the adopted local radial basis to the true-center
    basis; only its sine/cosine enter the spin-2 rotation, so it is not wrapped
    into a special angular interval.
    """

    phi: FloatArray
    radius_over_r200c: float
    d_over_r200c: FloatArray
    phi_offset: FloatArray
    true_radius_over_r200c: FloatArray
    true_phi: FloatArray
    omega: FloatArray
    x_true: FloatArray

    def __post_init__(self) -> None:
        phi = _angle_array(self.phi)
        if self.true_radius_over_r200c.shape != self.true_phi.shape or self.true_phi.shape != self.omega.shape or self.x_true.shape != self.omega.shape:
            raise ValueError("ring geometry arrays must have the same two-dimensional shape")
        if self.true_radius_over_r200c.ndim != 2 or self.true_radius_over_r200c.shape[1] != phi.size:
            raise ValueError("ring geometry arrays must have shape (n_halo, n_phi)")
        if np.any(~np.isfinite(self.true_radius_over_r200c)) or np.any(self.true_radius_over_r200c < 0.0):
            raise ValueError("true radii must be finite and non-negative")
        for name in ("true_phi", "omega", "x_true"):
            if np.any(~np.isfinite(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if np.any(self.x_true <= 0.0):
            raise ValueError("x_true must be strictly positive; exact center crossings require an explicit policy")


@dataclass(frozen=True)
class RingFields:
    """Linear lensing fields on an adopted-center ring.

    The fields are not reduced-shear or magnification observables. They should
    normally be interpreted as far-background fields until a source weight is
    explicitly applied by ``observables.py``.
    """

    kappa: FloatArray
    gamma_plus: FloatArray
    gamma_cross: FloatArray
    geometry: RingGeometry | None = None

    def __post_init__(self) -> None:
        k = np.asarray(self.kappa, dtype=np.float64)
        gp = np.asarray(self.gamma_plus, dtype=np.float64)
        gx = np.asarray(self.gamma_cross, dtype=np.float64)
        if k.shape != gp.shape or k.shape != gx.shape or k.ndim != 2:
            raise ValueError("field arrays must have equal shape (n_halo, n_phi)")
        if np.any(~np.isfinite(k)) or np.any(~np.isfinite(gp)) or np.any(~np.isfinite(gx)):
            raise ValueError("field arrays must be finite")
        if self.geometry is not None and self.geometry.x_true.shape != k.shape:
            raise ValueError("geometry and field arrays have inconsistent shapes")


@dataclass(frozen=True)
class CircularNFWFieldEvaluator:
    """True-center spherical NFW evaluator preserving a supplied normalization."""

    x_floor: float | None = None

    def evaluate(self, x: ArrayLike, phi: ArrayLike, kappa_s: ArrayLike) -> RingFields:
        radii = np.asarray(x, dtype=np.float64)
        if np.any(~np.isfinite(radii)) or np.any(radii <= 0.0):
            if self.x_floor is None:
                raise ValueError("x must be positive and finite")
            floor = _positive_scalar(self.x_floor, "x_floor")
            radii = np.maximum(radii, floor)
        amplitude = np.asarray(kappa_s, dtype=np.float64)
        if np.any(~np.isfinite(amplitude)) or np.any(amplitude < 0.0):
            raise ValueError("kappa_s must be finite and non-negative")
        lensing = nfw_lensing(radii, amplitude)
        return RingFields(
            kappa=np.asarray(lensing.kappa, dtype=np.float64),
            gamma_plus=np.asarray(lensing.gamma_t, dtype=np.float64),
            gamma_cross=np.zeros_like(lensing.gamma_t, dtype=np.float64),
        )


@dataclass(frozen=True)
class ModeFieldEvaluator:
    """Evaluate true-center fields from precomputed lensing multipoles."""

    modes: LensingModes
    radial_interpolation: str

    def _interpolated_coefficients(self, x: FloatArray, array: FloatArray) -> FloatArray:
        flat = x.reshape(-1)
        if flat.min() < self.modes.radius[0] or flat.max() > self.modes.radius[-1]:
            raise ValueError("x values lie outside the mode grid; the public path does not silently clip")
        if self.radial_interpolation == "x":
            grid = self.modes.radius
            points = flat
        elif self.radial_interpolation == "logx":
            grid = np.log(self.modes.radius)
            points = np.log(flat)
        else:
            raise ValueError("radial_interpolation must be 'x' or 'logx'")
        coeff = np.empty((flat.size, array.shape[1]), dtype=np.float64)
        for m in range(array.shape[1]):
            coeff[:, m] = np.interp(points, grid, array[:, m])
        return coeff.reshape(x.shape + (array.shape[1],))

    def _field_from_modes(self, x: FloatArray, phi: FloatArray, cosine: FloatArray, sine: FloatArray) -> FloatArray:
        ac = self._interpolated_coefficients(x, cosine)
        bs = self._interpolated_coefficients(x, sine)
        m = np.arange(cosine.shape[1], dtype=np.float64)
        phase = phi[..., None]*m
        return np.sum(ac*np.cos(phase) + bs*np.sin(phase), axis=-1)

    def evaluate(self, x: ArrayLike, phi: ArrayLike, kappa_s: ArrayLike) -> RingFields:
        radii = np.asarray(x, dtype=np.float64)
        angles = np.asarray(phi, dtype=np.float64)
        if radii.shape != angles.shape or np.any(~np.isfinite(radii)) or np.any(radii <= 0.0) or np.any(~np.isfinite(angles)):
            raise ValueError("x and phi must have the same finite positive-radius shape")
        amplitude = np.asarray(kappa_s, dtype=np.float64)
        if np.any(~np.isfinite(amplitude)) or np.any(amplitude < 0.0):
            raise ValueError("kappa_s must be finite and non-negative")
        return RingFields(
            kappa=self._field_from_modes(radii, angles, self.modes.kappa_cos, self.modes.kappa_sin)*amplitude,
            gamma_plus=self._field_from_modes(radii, angles, self.modes.gamma_plus_cos, self.modes.gamma_plus_sin)*amplitude,
            gamma_cross=self._field_from_modes(radii, angles, self.modes.gamma_cross_cos, self.modes.gamma_cross_sin)*amplitude,
        )


def displaced_ring_geometry(
    *,
    radius_over_r200c: float,
    concentration: ArrayLike,
    projected_scale_factor: ArrayLike,
    d_over_r200c: ArrayLike = 0.0,
    phi_offset: ArrayLike = 0.0,
    phi: ArrayLike | None = None,
    n_phi: int = 256,
) -> RingGeometry:
    """Return true-center coordinates and dimensionless radial arguments."""
    radius = _positive_scalar(radius_over_r200c, "radius_over_r200c")
    c, scale, d, angle = _halo_arrays(concentration, projected_scale_factor, d_over_r200c, phi_offset)
    if np.any(c <= 0.0) or np.any(scale <= 0.0) or np.any(d < 0.0):
        raise ValueError("concentration/scale must be positive and offsets non-negative")
    angles = midpoint_angles(n_phi) if phi is None else _angle_array(phi)
    adopted_x = radius*np.cos(angles)[None, :]
    adopted_y = radius*np.sin(angles)[None, :]
    true_x = adopted_x - d[:, None]*np.cos(angle)[:, None]
    true_y = adopted_y - d[:, None]*np.sin(angle)[:, None]
    true_radius = np.hypot(true_x, true_y)
    true_phi = np.arctan2(true_y, true_x)
    omega = true_phi - angles[None, :]
    x_true = c[:, None]*true_radius/scale[:, None]
    return RingGeometry(
        phi=angles,
        radius_over_r200c=radius,
        d_over_r200c=d,
        phi_offset=angle,
        true_radius_over_r200c=true_radius,
        true_phi=true_phi,
        omega=omega,
        x_true=x_true,
    )


def rotate_spin2(gamma_plus_true: ArrayLike, gamma_cross_true: ArrayLike, omega: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Rotate true-center shear components into the adopted local radial frame."""
    gp, gx, w = np.broadcast_arrays(
        np.asarray(gamma_plus_true, dtype=np.float64),
        np.asarray(gamma_cross_true, dtype=np.float64),
        np.asarray(omega, dtype=np.float64),
    )
    if np.any(~np.isfinite(gp)) or np.any(~np.isfinite(gx)) or np.any(~np.isfinite(w)):
        raise ValueError("spin-2 rotation inputs must be finite")
    cos2 = np.cos(2.0*w)
    sin2 = np.sin(2.0*w)
    return np.asarray(gp*cos2 - gx*sin2), np.asarray(gp*sin2 + gx*cos2)


def evaluate_ring_fields(
    field_evaluator: CircularNFWFieldEvaluator | ModeFieldEvaluator,
    *,
    concentration: ArrayLike,
    projected_scale_factor: ArrayLike,
    kappa_s_projected: ArrayLike,
    radius_over_r200c: float,
    d_over_r200c: ArrayLike = 0.0,
    phi_offset: ArrayLike = 0.0,
    phi: ArrayLike | None = None,
    n_phi: int = 256,
) -> RingFields:
    """Evaluate far-background linear fields on a complete adopted-center ring.

    ``field_evaluator`` supplies true-center fields as functions of ``x`` and
    true-center polar angle. The supplied ``kappa_s_projected`` is broadcast to
    the halo axis and applied before the spin-2 rotation.
    """
    c, scale, amplitude, d, angle = _halo_arrays(
        concentration, projected_scale_factor, kappa_s_projected, d_over_r200c, phi_offset
    )
    if np.any(amplitude < 0.0) or np.any(~np.isfinite(amplitude)):
        raise ValueError("kappa_s_projected must be finite and non-negative")
    geometry = displaced_ring_geometry(
        radius_over_r200c=radius_over_r200c,
        concentration=c,
        projected_scale_factor=scale,
        d_over_r200c=d,
        phi_offset=angle,
        phi=phi,
        n_phi=n_phi,
    )
    true_fields = field_evaluator.evaluate(geometry.x_true, geometry.true_phi, amplitude[:, None])
    gamma_plus, gamma_cross = rotate_spin2(true_fields.gamma_plus, true_fields.gamma_cross, geometry.omega)
    return RingFields(
        kappa=true_fields.kappa,
        gamma_plus=gamma_plus,
        gamma_cross=gamma_cross,
        geometry=geometry,
    )
