"""Volume-preserving triaxial-to-elliptical projection (specification §4).

Intrinsic axes: major/intermediate/minor. p=b/a and s=c/a, 0<s<=p<=1.
LOS vectors are expressed in this intrinsic frame. No sampling occurs here.
The scalar and batch paths preserve the historical eigensolver and analytic
2x2-eigenvalue conventions respectively; they share the same geometry.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _axis_ratios(p: ArrayLike, s: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    p_arr, s_arr = np.broadcast_arrays(np.asarray(p, dtype=np.float64), np.asarray(s, dtype=np.float64))
    if np.any(~np.isfinite(p_arr)) or np.any(~np.isfinite(s_arr)) or np.any((s_arr <= 0) | (s_arr > p_arr) | (p_arr > 1)):
        raise ValueError("axis ratios must be finite and satisfy 0 < s <= p <= 1")
    return p_arr, s_arr


def _unit_los(los: ArrayLike) -> NDArray[np.float64]:
    n = np.asarray(los, dtype=np.float64)
    if n.ndim < 1 or n.shape[-1] != 3 or np.any(~np.isfinite(n)):
        raise ValueError("los must be finite, with last dimension 3")
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norm)) or np.any(norm <= 0):
        raise ValueError("los must have finite non-zero norm")
    return n/norm


def _basis_for_unit_los(n: NDArray[np.float64]) -> NDArray[np.float64]:
    ref = np.eye(3)[np.argmin(np.abs(n), axis=-1)]
    e1 = np.cross(n, ref)
    e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
    e2 = np.cross(n, e1)
    e2 /= np.linalg.norm(e2, axis=-1, keepdims=True)
    return np.stack((e1, e2), axis=-1)


def volume_preserving_stretches(p: ArrayLike, s: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return A,B,C with B/A=p, C/A=s and ABC=1; inputs broadcast."""
    pp, ss = _axis_ratios(p, s)
    a = (pp*ss)**(-1.0/3.0)
    return np.asarray(a), np.asarray(pp*a), np.asarray(ss*a)


def intrinsic_shape_matrix(p: float, s: float) -> NDArray[np.float64]:
    """Scalar-halo matrix S=diag(A^-2,B^-2,C^-2)."""
    a, b, c = volume_preserving_stretches(p, s)
    if a.ndim != 0:
        raise ValueError("intrinsic_shape_matrix requires scalar p and s")
    return np.diag([float(a)**-2, float(b)**-2, float(c)**-2])


def sky_basis_from_los(los: ArrayLike, spin_angle: float = 0.0) -> NDArray[np.float64]:
    """Orthonormal 3x2 sky basis, with cross(e1,e2)=n; spin in radians.

    The reference axis has the smallest absolute component of the LOS, as in
    the historical calculation. It need not vary continuously with direction.
    """
    n = _unit_los(los)
    if n.shape != (3,):
        raise ValueError("sky_basis_from_los requires one length-3 LOS")
    angle = np.asarray(spin_angle, dtype=np.float64)
    if angle.ndim != 0 or not np.isfinite(angle):
        raise ValueError("spin_angle must be a finite scalar")
    basis = _basis_for_unit_los(n)
    if float(angle) != 0.0:
        cc, ss = np.cos(angle), np.sin(angle)
        basis = basis @ np.array([[cc, -ss], [ss, cc]])
    return basis


@dataclass(frozen=True)
class TriaxialProjection:
    """A scalar projected halo; matrices refer to the returned sky basis.

    The projected major axis is the eigenvector of the smaller eigenvalue.
    Its angle is defined modulo pi, and is physically arbitrary for a circular
    projection. The supplied/returned arrays must be treated as numerical data.
    """

    p: float
    s: float
    los: NDArray[np.float64]
    sky_basis: NDArray[np.float64]
    shape_matrix: NDArray[np.float64]
    projected_matrix: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    major_axis_angle_rad: float
    q_perp: float
    b_los: float
    scale_factor: float

    def circular_reference_parameters(
        self, r_s: ArrayLike, kappa_s: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return r_s,perp=b_los^-1/2*r_s and kappa_s,perp=b_los*kappa_s.

        The input kappa_s may refer to any one source plane. Removing ellipticity
        for an offset-only control keeps THESE projected parameters fixed.
        """
        rs, ks = np.broadcast_arrays(np.asarray(r_s, dtype=np.float64), np.asarray(kappa_s, dtype=np.float64))
        if np.any(~np.isfinite(rs)) or np.any(rs <= 0) or np.any(~np.isfinite(ks)) or np.any(ks < 0):
            raise ValueError("r_s must be finite positive; kappa_s finite non-negative")
        return np.asarray(self.scale_factor*rs), np.asarray(self.b_los*ks)


@dataclass(frozen=True)
class BatchProjection:
    """Scalar projected parameters for an explicitly supplied batch of halos."""

    q_perp: NDArray[np.float64]
    b_los: NDArray[np.float64]
    scale_factor: NDArray[np.float64]


def project_triaxial(
    p: float, s: float, los: ArrayLike, spin_angle: float = 0.0,
) -> TriaxialProjection:
    """Scalar Schur-complement projection, using a symmetric eigensolver."""
    shape = intrinsic_shape_matrix(p, s)
    n = _unit_los(los)
    if n.shape != (3,):
        raise ValueError("project_triaxial requires one length-3 LOS")
    basis = sky_basis_from_los(n, spin_angle)
    xi = float(n @ shape @ n)
    v = basis.T @ shape @ n
    projected = basis.T @ shape @ basis - np.outer(v, v)/xi
    projected = 0.5*(projected+projected.T)
    eig, vec = np.linalg.eigh(projected)
    if np.any(~np.isfinite(eig)) or np.any(eig <= 0):
        raise ValueError("projected matrix is numerically non-positive; shape is unsupported")
    major = vec[:, 0]
    boost = xi**-0.5
    return TriaxialProjection(
        float(p), float(s), n, basis, shape, projected, eig,
        float(np.arctan2(major[1], major[0])), float(np.sqrt(eig[0]/eig[1])),
        boost, boost**-0.5,
    )


def project_triaxial_batch(p: ArrayLike, s: ArrayLike, los: ArrayLike) -> BatchProjection:
    """Project N supplied halos using the historical trace/discriminant path.

    p and s have shape (N,), and los has shape (N,3). Scalar p/s may broadcast
    over that batch; no random numbers are consumed. This is not a sampler.
    """
    n = _unit_los(los)
    if n.ndim != 2 or n.shape[0] == 0:
        raise ValueError("los must have non-empty shape (N,3)")
    pp, ss = _axis_ratios(p, s)
    try:
        pp, ss = np.broadcast_to(pp, (len(n),)), np.broadcast_to(ss, (len(n),))
    except ValueError as exc:
        raise ValueError("p and s must be scalar or have shape (N,)") from exc
    a, b, c = volume_preserving_stretches(pp, ss)
    ds = np.column_stack((a**-2, b**-2, c**-2))
    basis = _basis_for_unit_los(n)
    e1, e2 = basis[:, :, 0], basis[:, :, 1]
    sn = ds*n
    xi = np.einsum("ni,ni->n", n, sn)
    b11 = np.einsum("ni,ni->n", e1, ds*e1)
    b22 = np.einsum("ni,ni->n", e2, ds*e2)
    b12 = np.einsum("ni,ni->n", e1, ds*e2)
    v1 = np.einsum("ni,ni->n", e1, sn)
    v2 = np.einsum("ni,ni->n", e2, sn)
    p11, p22, p12 = b11-v1*v1/xi, b22-v2*v2/xi, b12-v1*v2/xi
    trace = p11+p22
    discriminant = np.sqrt(np.maximum((p11-p22)**2 + 4.0*p12*p12, 0.0))
    eig_lo, eig_hi = 0.5*(trace-discriminant), 0.5*(trace+discriminant)
    if np.any(~np.isfinite(eig_lo)) or np.any(eig_lo <= 0) or np.any(~np.isfinite(eig_hi)):
        raise ValueError("projected matrix is numerically non-positive; shape is unsupported")
    boost = xi**-0.5
    return BatchProjection(np.sqrt(eig_lo/eig_hi), boost, boost**-0.5)


def projected_radius(xy: ArrayLike, result: TriaxialProjection) -> NDArray[np.float64]:
    """sqrt(X^T P X), in the same length units as input sky-basis coordinates."""
    points = np.asarray(xy, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 2 or np.any(~np.isfinite(points)):
        raise ValueError("xy must be finite with last dimension 2")
    squared = np.einsum("...i,ij,...j->...", points, result.projected_matrix, points)
    if np.any(squared < 0):
        raise ValueError("negative projected quadratic form")
    return np.asarray(np.sqrt(squared))


def area_preserving_radius(xy_major_frame: ArrayLike, q_perp: float) -> NDArray[np.float64]:
    """zeta=sqrt(q*X_major^2+X_minor^2/q), in the input length units.

    Coordinates MUST be in the projected principal-axis frame, not an arbitrary
    sky basis. The circular reference is recovered by setting q_perp=1.
    """
    points = np.asarray(xy_major_frame, dtype=np.float64)
    q = np.asarray(q_perp, dtype=np.float64)
    if q.ndim != 0 or not np.isfinite(q) or not 0 < float(q) <= 1:
        raise ValueError("q_perp must be a finite scalar in (0,1]")
    if points.ndim < 1 or points.shape[-1] != 2 or np.any(~np.isfinite(points)):
        raise ValueError("xy_major_frame must be finite with last dimension 2")
    return np.asarray(np.sqrt(q*points[..., 0]**2 + points[..., 1]**2/q))
