"""Centered projected-shape response analysis for the paper.

This module implements the numerical analysis underlying the centered-shape
leading-response diagnostic.  It consumes already generated population key
samples; it does not evaluate population lensing fields, impose the Table-2
common-domain cut, fit a normalization, plot Figure 3, or run Section 6
nonzero-offset diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .nfw import nfw_f, nfw_g
from .population_fields import PopulationFieldProducts

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

CENTERED_SHAPE_SIGMA_D = 0.0
CENTERED_SHAPE_SOURCE_Z = 1.0
CENTERED_SHAPE_SOURCE_LABEL = "z1"
CENTERED_SHAPE_RADIUS = 0.2924408238028535
DEFAULT_SHAPE_RESPONSE_BINS = 10
_RADIUS_ATOL = 5.0e-15

# f(1+t), through t^6: same audited near-unity coefficients as nfw.py.
_F_NEAR_ONE = (1/3, -2/5, 13/35, -20/63, 61/231, -94/429, 1181/6435)

SHAPE_RESPONSE_COLUMNS = (
    "grid_index",
    "M200c_Msun_h",
    "z_l",
    "z_s",
    "sigma_d",
    "R_over_R200c",
    "bin_index",
    "n_bin",
    "epsilon_perp_sq_median",
    "delta_g_median_percent",
    "delta_g_leading_prediction_median_percent",
    "delta_g_second_order_prediction_median_percent",
    "leading_response_coefficient_median_percent",
)


def _as_1d_float(values: ArrayLike, name: str) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0 or np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} must be a non-empty finite one-dimensional array")
    return arr


def _same_shape(*arrays: ArrayLike) -> tuple[FloatArray, ...]:
    converted = tuple(np.asarray(a, dtype=np.float64).reshape(-1) for a in arrays)
    shape = converted[0].shape
    if any(a.shape != shape for a in converted):
        raise ValueError("all arrays must have the same flattened shape")
    return converted


def _require_finite(values: FloatArray, name: str) -> None:
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{name} must be finite")


def _horner(x: FloatArray, coefficients: Sequence[float]) -> FloatArray:
    out = np.zeros_like(x, dtype=np.float64) + float(coefficients[-1])
    for coefficient in reversed(coefficients[:-1]):
        out = out*x + float(coefficient)
    return out


def nfw_f_derivative(x: ArrayLike) -> FloatArray:
    """Derivative df/dx for the projected NFW surface-density shape.

    The general branch uses the analytic relation
    ``f'=(1/x - 3*x*f)/(x**2 - 1)``.  Around x=1 the derivative is evaluated
    from the audited near-unity series instead of using numerical
    differentiation or a cancellation-prone expression.
    """
    xx = _as_1d_float(x, "x")
    if np.any(xx <= 0.0):
        raise ValueError("x must be strictly positive")
    out = np.empty_like(xx)
    near = np.abs(xx - 1.0) <= 1.0e-3
    if np.any(near):
        t = xx[near] - 1.0
        derivative_coefficients = tuple(i*_F_NEAR_ONE[i] for i in range(1, len(_F_NEAR_ONE)))
        out[near] = _horner(t, derivative_coefficients)
    if np.any(~near):
        xg = xx[~near]
        fg = nfw_f(xg)
        out[~near] = (1.0/xg - 3.0*xg*fg)/(xg*xg - 1.0)
    return np.asarray(out, dtype=np.float64)


def nfw_j4_shape(x: ArrayLike) -> FloatArray:
    """Return the dimensionless J4 shape entering the coherent quadrupole response."""
    xx = _as_1d_float(x, "x")
    if np.any(xx <= 0.0):
        raise ValueError("x must be strictly positive")
    f = nfw_f(xx)
    g = nfw_g(xx)
    return np.asarray(xx**2*(xx**2 - 1.0)*f - 0.5*xx**2 + 2.0*g, dtype=np.float64)


def leading_shape_coefficient_Aepsilon(x: ArrayLike, kappa_s_projected: ArrayLike) -> FloatArray:
    """Leading centered-shape coefficient A_epsilon for reduced tangential shear.

    The returned coefficient is dimensionless.  Multiplying it by
    ``epsilon_perp**2`` gives the leading fractional reduced-shear residual.
    The caller supplies the finite-source projected ``kappa_s`` appropriate to
    the selected source plane.
    """
    xx, ks = _same_shape(x, kappa_s_projected)
    if np.any(~np.isfinite(xx)) or np.any(~np.isfinite(ks)) or np.any(xx <= 0.0) or np.any(ks < 0.0):
        raise ValueError("x must be finite positive and kappa_s_projected finite non-negative")
    f = nfw_f(xx)
    g = nfw_g(xx)
    fp = nfw_f_derivative(xx)
    kappa0 = 2.0*ks*f
    gamma0 = 4.0*ks*g/xx**2 - kappa0
    kappa1 = 2.0*ks*xx*fp
    kappa4_bar = 8.0*ks*nfw_j4_shape(xx)/xx**4
    D0 = 1.0 - kappa0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        coeff = -kappa1*(kappa1 - 2.0*kappa0 + 3.0*kappa4_bar)/(2.0*D0*gamma0) + kappa1**2/(2.0*D0**2)
    return np.asarray(coeff, dtype=np.float64)


@dataclass(frozen=True)
class ShapeResponseInput:
    """Arrays for one grid in the centered-shape response analysis."""

    grid_index: int
    mass_msun_h: float
    z_l: float
    source_z: float
    sigma_d: float
    radius_over_r200c: float
    epsilon_perp: FloatArray
    c200c: FloatArray
    projected_scale_factor: FloatArray
    kappa_s_perp_infinity: FloatArray
    source_weight: float
    safe_outer_infinity: BoolArray
    delta_g_plus_fractional: FloatArray
    delta_g_plus_second_order: FloatArray | None = None

    def __post_init__(self) -> None:
        arrays = _same_shape(
            self.epsilon_perp,
            self.c200c,
            self.projected_scale_factor,
            self.kappa_s_perp_infinity,
            self.delta_g_plus_fractional,
        )
        eps, c, scale, ks, dg = arrays
        mask = np.asarray(self.safe_outer_infinity, dtype=bool).reshape(-1)
        if mask.shape != eps.shape:
            raise ValueError("safe_outer_infinity must match the halo arrays")
        _require_finite(eps, "epsilon_perp")
        _require_finite(c, "c200c")
        _require_finite(scale, "projected_scale_factor")
        _require_finite(ks, "kappa_s_perp_infinity")
        if np.any(c <= 0.0) or np.any(scale <= 0.0) or np.any(ks < 0.0):
            raise ValueError("c200c and projected_scale_factor must be positive; kappa_s_perp_infinity must be non-negative")
        if self.delta_g_plus_second_order is not None:
            second = np.asarray(self.delta_g_plus_second_order, dtype=np.float64).reshape(-1)
            if second.shape != eps.shape:
                raise ValueError("delta_g_plus_second_order must match the halo arrays")
            object.__setattr__(self, "delta_g_plus_second_order", second)
        object.__setattr__(self, "epsilon_perp", eps)
        object.__setattr__(self, "c200c", c)
        object.__setattr__(self, "projected_scale_factor", scale)
        object.__setattr__(self, "kappa_s_perp_infinity", ks)
        object.__setattr__(self, "delta_g_plus_fractional", dg)
        object.__setattr__(self, "safe_outer_infinity", mask)
        for name in ("mass_msun_h", "z_l", "source_z", "radius_over_r200c", "source_weight"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        sigma = float(self.sigma_d)
        if not np.isfinite(sigma) or sigma < 0.0:
            raise ValueError("sigma_d must be finite and non-negative")
        object.__setattr__(self, "sigma_d", sigma)

    @property
    def size(self) -> int:
        return int(self.epsilon_perp.size)

    @property
    def retained_count(self) -> int:
        return int(np.count_nonzero(self.safe_outer_infinity))

    @property
    def x_nfw(self) -> FloatArray:
        return np.asarray(self.radius_over_r200c*self.c200c/self.projected_scale_factor, dtype=np.float64)

    @property
    def kappa_s_perp_source(self) -> FloatArray:
        return np.asarray(self.source_weight*self.kappa_s_perp_infinity, dtype=np.float64)


def _find_float_index(values: ArrayLike, target: float, *, name: str, atol: float = 1.0e-14) -> int:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    matches = np.where(np.isclose(arr, float(target), rtol=0.0, atol=atol))[0]
    if matches.size != 1:
        raise ValueError(f"expected exactly one {name} entry matching {target}; found {matches.size}")
    return int(matches[0])


def centered_shape_input_from_product(
    product: PopulationFieldProducts,
    *,
    source_index: int = 0,
    source_z: float = CENTERED_SHAPE_SOURCE_Z,
    sigma_d: float = CENTERED_SHAPE_SIGMA_D,
    radius_over_r200c: float = CENTERED_SHAPE_RADIUS,
) -> ShapeResponseInput:
    """Extract the centered-shape inputs from one population-field product."""
    key = product.key_samples
    sigma_index = _find_float_index(key["sigma_d"], sigma_d, name="sigma_d")
    radius_index = _find_float_index(key["R_over_r200c"], radius_over_r200c, name="key radius", atol=_RADIUS_ATOL)
    weights = np.asarray(key["source_weights"], dtype=np.float64).reshape(-1)
    sidx = int(source_index)
    if sidx < 0 or sidx >= weights.size:
        raise ValueError("source_index is outside the stored source axis")
    dg = np.asarray(key["delta_g_plus_fractional"], dtype=np.float64)[sidx, sigma_index, :, radius_index]
    second = None
    if "delta_g_plus_second_order" in key:
        second = np.asarray(key["delta_g_plus_second_order"], dtype=np.float64)[sidx, sigma_index, :, radius_index]
    return ShapeResponseInput(
        grid_index=int(product.grid_index),
        mass_msun_h=float(product.mass_msun_h),
        z_l=float(product.z_l),
        source_z=float(source_z),
        sigma_d=float(sigma_d),
        radius_over_r200c=float(radius_over_r200c),
        epsilon_perp=np.asarray(key["epsilon_perp"], dtype=np.float64),
        c200c=np.asarray(key["c200c"], dtype=np.float64),
        projected_scale_factor=np.asarray(key["projected_scale_factor"], dtype=np.float64),
        kappa_s_perp_infinity=np.asarray(key["kappa_s_perp_infinity"], dtype=np.float64),
        source_weight=float(weights[sidx]),
        safe_outer_infinity=np.asarray(key["safe_outer"], dtype=bool)[sigma_index, :, radius_index],
        delta_g_plus_fractional=dg,
        delta_g_plus_second_order=second,
    )


@dataclass(frozen=True)
class QuantileBinning:
    """Quantile-edge binning result for retained epsilon_perp^2 values."""

    edges: FloatArray
    bin_index: NDArray[np.int64]
    selected_mask: BoolArray

    @property
    def n_bins(self) -> int:
        return int(self.edges.size - 1)


def quantile_bins_epsilon2(
    epsilon_perp: ArrayLike,
    selected_mask: ArrayLike,
    *,
    n_bins: int = DEFAULT_SHAPE_RESPONSE_BINS,
) -> QuantileBinning:
    """Bin retained halos by epsilon_perp^2 using quantile edges.

    Internal-edge ties are assigned with ``searchsorted(..., side='right')``,
    matching the Figure-3 builder convention.
    """
    eps = _as_1d_float(epsilon_perp, "epsilon_perp")
    mask = np.asarray(selected_mask, dtype=bool).reshape(-1)
    if mask.shape != eps.shape:
        raise ValueError("selected_mask must match epsilon_perp")
    n = int(n_bins)
    if n < 1:
        raise ValueError("n_bins must be positive")
    eps2 = eps**2
    valid = mask & np.isfinite(eps2)
    if np.count_nonzero(valid) < n:
        raise ValueError("not enough selected halos for the requested number of bins")
    edges = np.asarray(np.quantile(eps2[valid], np.linspace(0.0, 1.0, n+1)), dtype=np.float64)
    if np.any(~np.isfinite(edges)) or np.any(np.diff(edges) < 0.0):
        raise ValueError("invalid quantile edges")
    bins = np.full(eps.shape, -1, dtype=np.int64)
    # Assignment is based on the retained halos only.  Clipping protects the
    # final maximum value from roundoff beyond the last edge.
    assigned = np.searchsorted(edges[1:-1], eps2[valid], side="right")
    bins[valid] = np.clip(assigned, 0, n-1)
    return QuantileBinning(edges=edges, bin_index=bins, selected_mask=valid)


def centered_shape_response_rows(
    data: ShapeResponseInput,
    *,
    n_bins: int = DEFAULT_SHAPE_RESPONSE_BINS,
) -> tuple[dict[str, float | int], ...]:
    """Return the binned centered-shape response table for one grid."""
    eps2 = data.epsilon_perp**2
    coeff = leading_shape_coefficient_Aepsilon(data.x_nfw, data.kappa_s_perp_source)
    leading = coeff*eps2
    selected = data.safe_outer_infinity & np.isfinite(data.delta_g_plus_fractional) & np.isfinite(leading)
    if data.delta_g_plus_second_order is not None:
        selected = selected & np.isfinite(data.delta_g_plus_second_order)
    binning = quantile_bins_epsilon2(data.epsilon_perp, selected, n_bins=n_bins)
    rows: list[dict[str, float | int]] = []
    for b in range(binning.n_bins):
        mask = binning.bin_index == b
        n_bin = int(np.count_nonzero(mask))
        if n_bin == 0:
            eps_med = dg_med = pred_med = coeff_med = second_med = np.nan
        else:
            eps_med = float(np.median(eps2[mask]))
            dg_med = float(100.0*np.median(data.delta_g_plus_fractional[mask]))
            pred_med = float(100.0*np.median(leading[mask]))
            coeff_med = float(100.0*np.median(coeff[mask]))
            second_med = float("nan")
            if data.delta_g_plus_second_order is not None:
                second_med = float(100.0*np.median(data.delta_g_plus_second_order[mask]))
        rows.append({
            "grid_index": int(data.grid_index),
            "M200c_Msun_h": float(data.mass_msun_h),
            "z_l": float(data.z_l),
            "z_s": float(data.source_z),
            "sigma_d": float(data.sigma_d),
            "R_over_R200c": float(data.radius_over_r200c),
            "bin_index": int(b),
            "n_bin": int(n_bin),
            "epsilon_perp_sq_median": eps_med,
            "delta_g_median_percent": dg_med,
            "delta_g_leading_prediction_median_percent": pred_med,
            "delta_g_second_order_prediction_median_percent": second_med,
            "leading_response_coefficient_median_percent": coeff_med,
        })
    return tuple(rows)


def centered_shape_response_rows_from_product(
    product: PopulationFieldProducts,
    *,
    source_index: int = 0,
    n_bins: int = DEFAULT_SHAPE_RESPONSE_BINS,
) -> tuple[dict[str, float | int], ...]:
    """Compute the centered-shape response rows from one product's key samples."""
    return centered_shape_response_rows(
        centered_shape_input_from_product(product, source_index=source_index),
        n_bins=n_bins,
    )


def centered_shape_response_rows_from_products(
    products: Iterable[PopulationFieldProducts],
    *,
    source_index: int = 0,
    n_bins: int = DEFAULT_SHAPE_RESPONSE_BINS,
) -> tuple[dict[str, float | int], ...]:
    """Concatenate centered-shape rows from multiple grid products."""
    rows: list[dict[str, float | int]] = []
    for product in sorted(products, key=lambda p: int(p.grid_index)):
        rows.extend(centered_shape_response_rows_from_product(product, source_index=source_index, n_bins=n_bins))
    return tuple(rows)


def retained_counts_by_grid(rows: Iterable[Mapping[str, object]]) -> dict[int, int]:
    """Return total binned counts by grid from centered-shape rows."""
    counts: dict[int, int] = {}
    for row in rows:
        grid = int(row["grid_index"])
        counts[grid] = counts.get(grid, 0) + int(row["n_bin"])
    return counts


def write_shape_response_csv(rows: Iterable[Mapping[str, object]], path: str | Path) -> None:
    """Write centered-shape response rows using the Figure-3-compatible schema."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHAPE_RESPONSE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in SHAPE_RESPONSE_COLUMNS})


def read_shape_response_csv(path: str | Path) -> tuple[dict[str, str], ...]:
    """Read a centered-shape response CSV as dictionaries of strings."""
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def compact_shape_response_metrics(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Return concise validation metrics for reports."""
    row_list = [dict(row) for row in rows]
    counts = retained_counts_by_grid(row_list)
    if row_list:
        pred = np.asarray([float(r["delta_g_leading_prediction_median_percent"]) for r in row_list], dtype=np.float64)
        direct = np.asarray([float(r["delta_g_median_percent"]) for r in row_list], dtype=np.float64)
    else:
        pred = direct = np.asarray([], dtype=np.float64)
    return {
        "n_rows": len(row_list),
        "grid_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "max_abs_direct_percent": float(np.max(np.abs(direct))) if direct.size else float("nan"),
        "max_abs_leading_prediction_percent": float(np.max(np.abs(pred))) if pred.size else float("nan"),
    }
