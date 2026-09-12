"""Selection masks and radial-grid conventions for the population calculation.

This module is deliberately independent of halo generation and field solvers. It
implements the sampled local safety quantity, the outer-contiguous retention
rule, publication radial-grid helpers, and small summary utilities used by the
population-field layer.  It does not decide which retained-domain cut belongs to
a final paper table or figure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

DEFAULT_N_RADIAL = 36
DEFAULT_RADIAL_MIN = 0.02
DEFAULT_RADIAL_MAX = 1.0
DEFAULT_KEY_RADIUS_TARGETS = (0.2, 0.3, 0.5, 1.0)
DEFAULT_SAFETY_THRESHOLD = 0.1


def _finite_array(values: ArrayLike, name: str) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")
    return arr


def publication_radial_grid(
    *,
    n_radial: int = DEFAULT_N_RADIAL,
    r_min: float = DEFAULT_RADIAL_MIN,
    r_max: float = DEFAULT_RADIAL_MAX,
) -> FloatArray:
    """Return the publication scaled-radius grid, ``geomspace(0.02,1,36)``."""
    n = int(n_radial)
    r0 = float(r_min)
    r1 = float(r_max)
    if n < 2 or not np.isfinite(r0) or not np.isfinite(r1) or not (0.0 < r0 < r1):
        raise ValueError("invalid publication radial-grid parameters")
    return np.geomspace(r0, r1, n, dtype=np.float64)


def key_radius_indices(
    radial_grid: ArrayLike,
    targets: ArrayLike = DEFAULT_KEY_RADIUS_TARGETS,
) -> NDArray[np.int64]:
    """Indices nearest to requested key radii by absolute distance in R/r200c."""
    radii = _finite_array(radial_grid, "radial_grid").reshape(-1)
    if np.any(radii <= 0.0) or np.any(np.diff(radii) <= 0.0):
        raise ValueError("radial_grid must be positive and strictly increasing")
    requested = _finite_array(targets, "targets").reshape(-1)
    if np.any(requested <= 0.0):
        raise ValueError("key-radius targets must be positive")
    indices = [int(np.argmin(np.abs(radii - target))) for target in requested]
    return np.asarray(indices, dtype=np.int64)


def key_radius_values(radial_grid: ArrayLike, targets: ArrayLike = DEFAULT_KEY_RADIUS_TARGETS) -> FloatArray:
    """Actual grid radii selected by :func:`key_radius_indices`."""
    radii = _finite_array(radial_grid, "radial_grid").reshape(-1)
    return np.asarray(radii[key_radius_indices(radii, targets)], dtype=np.float64)


def lambda_minus_minimum(
    kappa: ArrayLike,
    gamma_plus: ArrayLike,
    gamma_cross: ArrayLike,
    *,
    axis_phi: int = -1,
) -> FloatArray:
    """Sampled ring minimum of ``1-kappa-|gamma|`` along the azimuth axis."""
    kap, gp, gx = np.broadcast_arrays(
        np.asarray(kappa, dtype=np.float64),
        np.asarray(gamma_plus, dtype=np.float64),
        np.asarray(gamma_cross, dtype=np.float64),
    )
    if kap.ndim < 1 or np.any(~np.isfinite(kap)) or np.any(~np.isfinite(gp)) or np.any(~np.isfinite(gx)):
        raise ValueError("field arrays must be finite and have an azimuth axis")
    return np.asarray(np.min(1.0 - kap - np.hypot(gp, gx), axis=axis_phi), dtype=np.float64)


def local_safety_mask(lambda_min: ArrayLike, *, threshold: float = DEFAULT_SAFETY_THRESHOLD) -> BoolArray:
    """Boolean local safety mask for ``lambda_min >= threshold``."""
    value = _finite_array(lambda_min, "lambda_min")
    cut = float(threshold)
    if not np.isfinite(cut):
        raise ValueError("threshold must be finite")
    return np.asarray(value >= cut, dtype=bool)


def outer_contiguous_mask(safe_local: ArrayLike, *, radius_axis: int = -1) -> BoolArray:
    """Cumulative logical AND from largest radius inward.

    A radius is retained only when it and every larger sampled radius satisfy the
    local safety condition.  The input is converted to boolean but is otherwise
    not modified.
    """
    safe = np.asarray(safe_local, dtype=bool)
    if safe.ndim < 1 or safe.shape[radius_axis] < 1:
        raise ValueError("safe_local must have a non-empty radius axis")
    moved = np.moveaxis(safe, radius_axis, -1)
    retained = np.logical_and.accumulate(moved[..., ::-1], axis=-1)[..., ::-1]
    return np.moveaxis(retained, -1, radius_axis)


def represented_fraction(safe_outer: ArrayLike, *, halo_axis: int = -2) -> FloatArray:
    """Fraction of halos retained by the supplied outer-contiguous mask."""
    safe = np.asarray(safe_outer, dtype=bool)
    if safe.ndim < 1 or safe.shape[halo_axis] < 1:
        raise ValueError("safe_outer must contain a non-empty halo axis")
    return np.asarray(np.mean(safe, axis=halo_axis), dtype=np.float64)


@dataclass(frozen=True)
class ValueSummary:
    """Finite-value summary of a selected one-dimensional sample."""

    n_selected: int
    n_finite: int
    mean: float
    p16: float
    p50: float
    p84: float
    p97p5: float

    @property
    def finite_fraction(self) -> float:
        return np.nan if self.n_selected == 0 else self.n_finite / float(self.n_selected)


def summarize_selected_values(values: ArrayLike, mask: ArrayLike) -> ValueSummary:
    """Summarize finite values under a boolean mask using NumPy's quantile rule."""
    val = np.asarray(values, dtype=np.float64).reshape(-1)
    sel = np.asarray(mask, dtype=bool).reshape(-1)
    if val.shape != sel.shape:
        raise ValueError("values and mask must have matching flattened shape")
    selected = val[sel]
    finite = selected[np.isfinite(selected)]
    if finite.size == 0:
        return ValueSummary(int(selected.size), 0, np.nan, np.nan, np.nan, np.nan, np.nan)
    q = np.percentile(finite, [16.0, 50.0, 84.0, 97.5])
    return ValueSummary(
        n_selected=int(selected.size),
        n_finite=int(finite.size),
        mean=float(np.mean(finite)),
        p16=float(q[0]),
        p50=float(q[1]),
        p84=float(q[2]),
        p97p5=float(q[3]),
    )
